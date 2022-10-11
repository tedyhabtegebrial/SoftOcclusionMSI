import torch
import torch.nn as nn
from .spt_utils import Utils as SptUtils

class SphereSweeper(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.spt_utils = SptUtils(configs)
        self.unit_rays = self.spt_utils._get_unit_rays_on_sphere()
        self.sphere_radii = self._get_sphere_radii()
    
    def _get_sphere_radii(self, b_size=1):
        sphere_radii = 1 / \
            torch.linspace(1/self.configs.near, 1 /
                            self.configs.far, self.configs.num_spheres)
        sphere_radii = sphere_radii.view(1, self.configs.num_spheres)
        sphere_radii = sphere_radii.expand(b_size, self.configs.num_spheres)
        return sphere_radii

    def forward(self, inp_erp, ssw_pose):
        return self.sweep_erp(inp_erp, ssw_pose)
    
    def get_inverse_pose(self, input_pose):
        r, t = input_pose[:, :3, :3], (input_pose[:, :3, 3]).view(-1, 3, 1)
        r_inv = torch.inverse(r)
        t_inv = torch.bmm(r_inv, -1*t)
        return r_inv, t_inv

    # def apply_rigid_transform()

    def sweep_erp(self, inp_erp, ssw_pose):
        configs = self.configs
        b, c, h, w = inp_erp.shape
        # ssw_pose should be from inp_erp to sphere swweping camera pose
        device = inp_erp.device
        unit_rays = self.unit_rays.to(device).view(1, 1, h, w, 3).repeat(b, configs.num_spheres, 1, 1, 1).unsqueeze(-1)  # B, D, H, W, 3
        sphere_radii = self.sphere_radii.to(device).view(
            1, configs.num_spheres, 1, 1, 1).unsqueeze(-1)
        scaled_rays = unit_rays * sphere_radii
        # transform scaled rays to the input camera pose
        # get cam pose form ssw location to input
        r_inv, t_inv = self.get_inverse_pose(ssw_pose)
        r_inv = r_inv.view(-1, 1, 1, 1, 3, 3)
        t_inv = t_inv.view(-1, 1, 1, 1, 3, 1)
        scaled_rays_inp = torch.matmul(r_inv, scaled_rays) + t_inv
        scaled_rays_inp_norm = self.spt_utils.normalize_3d_vectors(scaled_rays_inp)
        spherical_coords_inp = self.spt_utils.cartesian_2_spherical(
            scaled_rays_inp_norm, normalized=True)  # [B, num_spheres, height, width, 2]
        uv_coords_src = self.spt_utils.spherical_2_equi(
            spherical_coords_inp, height=h, width=w)
        input_img = inp_erp.view(b, 1, 3, h, w).repeat(1, configs.num_spheres, 1, 1, 1)
        # print(input_img.shape, uv_coords_src.shape)
        # exit()
        ssv = self.spt_utils.sample_equi(input=input_img,
                                              grid=uv_coords_src)
        return ssv 
        

class AlphaComposition(nn.Module):
    """This class implements alpha compostion.
    Accepts input warped images and their alpha channels.
    We perform A-over-B composition in back-to-front manner.......

    Input Tensors:
        src_imgs = [B, D, 3, H, W]
        alpha = [B, D, 1, H, W]

    Output Tensor:
        composite = [B, 3, H, W]
    """

    def __init__(self, configs=None):
        super(AlphaComposition, self).__init__()
        self.configs = configs

    def forward(self, src_imgs, alpha_imgs, return_weighted_feats=False, return_intermediate=False):
        alpha_imgs = alpha_imgs.clamp(min=0.0001)
        ones = torch.ones_like(alpha_imgs[:, 0, ...]).unsqueeze(1)
        weight_1 = torch.cumprod(
            torch.cat([ones, 1-alpha_imgs[:, :-1, :, :, :]], dim=1), dim=1)
        weighted_imgs = weight_1*alpha_imgs*src_imgs
        if return_weighted_feats:
            num_l = weighted_imgs.shape[1]
            weighted_feats = [weighted_imgs[:, l, :, :, :]
                              for l in range(num_l)]
            comb = torch.sum(weighted_imgs, dim=1)
            return comb, weighted_feats
        elif return_intermediate:
            comb = torch.sum(weighted_imgs, dim=1)
            layer_contribs = weight_1*alpha_imgs
            return comb, layer_contribs
        else:
            comb = torch.sum(weighted_imgs, dim=1)
            return comb


class MSIRenderer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.spt_utils = SptUtils(configs)
        self.unit_rays = self.spt_utils._get_unit_rays_on_sphere()
        self.sphere_radii = self._get_sphere_radii()
        self.alpha_compose = AlphaComposition()

    def _get_sphere_radii(self, b_size=1):
        sphere_radii = 1 / \
            torch.linspace(1/self.configs.near, 1 /
                           self.configs.far, self.configs.num_spheres)
        sphere_radii = sphere_radii.view(1, self.configs.num_spheres)
        sphere_radii = sphere_radii.expand(b_size, self.configs.num_spheres)
        return sphere_radii

    def forward(self, alpha, color, target_pose):
        b, d, c, h, w = color.shape
        r_mats, t_vecs = self.get_r_t(target_pose)
        intesection_pts, ray_dirs = self._get_intersection_points(
            r_mats, t_vecs, h, w, self.sphere_radii)
        spherical_coords = self.spt_utils.cartesian_2_spherical(
            intesection_pts, normalized=False)
        uv_coords = self.spt_utils.spherical_2_equi(
            spherical_coords, height=h, width=w)
        msi_color = self.spt_utils.sample_equi(
            input=color, grid=uv_coords)
        msi_alpha = self.spt_utils.sample_equi(
            input=alpha, grid=uv_coords)
        novel_view = self.alpha_compose(msi_color, msi_alpha)
        return novel_view

    def _get_intersection_points(self, r_mats, t_vecs, height, width, sphere_radii):
        b_size = r_mats.shape[0]
        ray_direction_trg = self.unit_rays.to(r_mats.device).repeat(b_size, 1, 1, 1)
        #   # B, H, W, 3
        # print(ray_direction_trg.shape)
        # # print(r_mats.device, t_vecs.device)
        # exit()
        if b_size != ray_direction_trg.shape[0]:
            ray_direction_trg = (ray_direction_trg[0]).unsqueeze(0)
        ray_direction_src = torch.matmul(r_mats.view(
            b_size, 1, 1, 3, 3), ray_direction_trg.unsqueeze(-1)).squeeze(-1)
        ray_origin_src = t_vecs.view(b_size, 1, 1, 3).expand(
            ray_direction_src.shape)  # [B, H, W, 3, 1]

        intersections = self.spt_utils.safe_ray_sphere_intersection(
            ray_origin_src, ray_direction_src, sphere_radii.view(1, self.configs.num_spheres).to(r_mats.device).repeat(b_size, 1))
        return intersections.view(b_size, self.configs.num_spheres, height, width, 3), ray_direction_src.view_as(ray_direction_trg)

    def get_r_t(self, input_pose):
        r, t = input_pose[:, :3, :3], (input_pose[:, :3, 3]).view(-1, 3, 1)
        return r, t
