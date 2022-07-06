import torch
from src.utils import Utils
from src.alpha_compose import AlphaComposition

class SOMSIRenderer(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.no_rotation = False
        self.utils = Utils(configs)
        self.alpha_compose = AlphaComposition(configs)

    def forward(self, alphas, feats, r_mats, t_vecs, radii, beta=None):
        return self.render(r_mats, t_vecs, alphas, feats, radii, beta)

    def render(self, r_mats, t_vecs, alphas, feats, sphere_radii, beta=None):
        result = {}
        feats_per_layer = feats.shape[2]
        num_layers = feats.shape[1]
        b_size, num_spheres, _, height, width = alphas.shape
    
        # Ray-Sphere Intersection Points
        intersection_points_src, ray_directions = self._get_intersection_points(r_mats, t_vecs, height, width, sphere_radii)
        ray_directions = ray_directions.view(
            b_size, height, width, 3).permute(0, 3, 1, 2)
        intersection_points_src_norm = self.utils.normalize_3d_vectors(intersection_points_src)
        spherical_coords = self.utils.cartesian_2_spherical(intersection_points_src_norm, normalized=True) # [B, num_spheres, height, width, 2]
        # UV Locations of each Ray-Sphere intersection point
        uv_coords_src = self.utils.spherical_2_equi(spherical_coords, height=height, width=width)
        # Read Alphas and betas for each Ray-Sphere intersection point
        w_alphas = self.utils.sample_equi(input=alphas, grid=uv_coords_src)
        warped_beta = self.utils.sample_equi(input=beta.view(b_size, num_spheres,num_layers, height,  width),
                                              grid=uv_coords_src)
        warped_beta = warped_beta.view(
            b_size, num_spheres, num_layers, height,  width)

        # Alpha composite betas
        warped_beta_2d = self.alpha_compose(warped_beta, w_alphas)
        # Alpha composite 2D Pixel locations of ray intersection points
        ## First normalize
        uv_coords_src_normalized = self.utils.normalize_grid(uv_coords_src)
        ## Alpha composite
        sampling_grid_src = self.alpha_compose(
            uv_coords_src_normalized.permute(0, 1, 4, 2, 3), w_alphas).permute(0, 2, 3, 1)
        # Sample Layered Features
        sampled_l_feats = self.utils.sample_equi(input=feats.view(
            b_size, num_layers*feats_per_layer, height,  width), grid=sampling_grid_src, normalize_grid=False)
        sampled_l_feats = sampled_l_feats.view(b_size, num_layers, feats_per_layer, height,  width)
        # Normalize betas to sum to one
        warped_beta_2d_sum = torch.sum(warped_beta_2d, dim=1, keepdim=True) + 0.00000001
        warped_beta_2d = warped_beta_2d / warped_beta_2d_sum
        # Use betas to combine sampled features
        weighted_feats = warped_beta_2d.unsqueeze(2) * sampled_l_feats
        novel_view_feats = torch.sum(weighted_feats, dim=1, keepdim=False)
        result['nv_feats'] = novel_view_feats
        # return novel view features, note that the returned features should be decoded
        return result

    def _get_intersection_points(self, r_mats, t_vecs, height, width, sphere_radii):
        b_size = r_mats.shape[0]
        ray_direction_trg = self._get_unit_rays_on_sphere(r_mats.device, b_size) # B, H, W, 3
        ray_direction_trg = ray_direction_trg
        if b_size != ray_direction_trg.shape[0]:
            ray_direction_trg = (ray_direction_trg[0]).unsqueeze(0)
        if self.no_rotation:
            ray_direction_src = ray_direction_trg
        else:
            ray_direction_src = torch.matmul(r_mats.view(b_size, 1, 1, 3, 3), ray_direction_trg.unsqueeze(-1)).squeeze(-1)
        ray_origin_src = t_vecs.view(b_size, 1, 1, 3).expand(ray_direction_src.shape) # [B, H, W, 3, 1]

        intersections = self.utils.safe_ray_sphere_intersection(ray_origin_src, ray_direction_src, sphere_radii.view(b_size, self.configs.num_spheres))
        return intersections.view(b_size, self.configs.num_spheres, height, width, 3), ray_direction_src.view_as(ray_direction_trg)

    def _get_unit_rays_on_sphere(self, device='cpu', b_size=1):
        return self.utils._get_unit_rays_on_sphere(device, b_size)
