import torch
import torch.nn as nn
from .spt import Utils as spt

class ImageWarping(nn.Module):
    def __init__(self, config):
        super(ImageWarping, self).__init__()
        self.height = config.height
        self.width = config.width
        self.device = config.device
        self.spt_utils = spt(height=self.height, width=self.width, dataset=config.dataset, device=self.device)

    def get_sphere_ray_intersections(self, src_pose, tgt_pose, radii, transform=None):
        """
        Calculates erp locations of intersections between rays and spheres.
        Rays are defined to originate from 'tgt_pose' and intersect spheres around 'src_pose'.
        Number and size of spheres depend on 'radii'.

        Parameters
        ----------
        src_pose : torch.Tensor[B, 3, 3]
            Pose of spheres to be intersected.
        tgt_pose : torch.Tensor[B, 3, 1] or torch.Tensor[B, 3]
            Pose of originating rays.
        radii : torch.Tensor[D]
            Radii of spheres.
        Returns
        -------
        erp_intersections : torch.Tensor[B, D, H, W, 2]
            Erp locations of intersections between rays and spheres.
        """
        r_src, t_src = src_pose
        r_tgt, t_tgt = tgt_pose
        if not (transform == None):
            r_trans, t_trans = transform
        r_tgt_src, t_tgt_src = self.spt_utils.get_rel_pose(src_pose=[r_tgt.view(-1,3,3), t_tgt.view(-1,3,1)], trg_pose=[r_src.view(-1,3,3), t_src.view(-1,3,1)])

        batch_size, *_ = src_pose[0].shape
        erp_tgt = self.spt_utils.get_xy_coords(batch_size=batch_size, device=self.device)
        sph_tgt = self.spt_utils.equi_2_spherical(erp_tgt, radius=1)
        xyz_tgt = self.spt_utils.spherical_2_cartesian(sph_tgt)

        if not (transform == None):
            r_origin = -1* torch.bmm(torch.inverse(r_trans), t_trans) + t_tgt_src
            r_origin = r_origin.view(batch_size, 1, 1, 3).expand(batch_size, self.height, self.width, 3)
            r_direction = self.spt_utils.transform_3d_points(xyz_tgt, rot_mats=torch.inverse(r_trans), t_vecs=torch.zeros_like(t_tgt_src))
            r_direction = self.spt_utils.transform_3d_points(r_direction, rot_mats=r_tgt_src, t_vecs=torch.zeros_like(t_tgt_src))
        else:
            r_origin = t_tgt_src.view(batch_size, 1, 1, 3).expand(batch_size, self.height, self.width, 3)
            r_direction = self.spt_utils.transform_3d_points(xyz_tgt, rot_mats=r_tgt_src, t_vecs=torch.zeros_like(t_tgt_src))
        intersections = self.spt_utils.safe_ray_sphere_intersection(ray_origin=r_origin, direction=r_direction, sphere_radii=(radii.unsqueeze(0)))
        sph_intersections = self.spt_utils.cartesian_2_spherical(intersections, normalized=False)
        erp_intersections = self.spt_utils.spherical_2_equi(sph_intersections)

        return erp_intersections

    def warp_image(self, msi, sph_ray_intersec):
        """
        Warps MSI according to sphere ray intersection locations.

        Parameters
        ----------
        msi : torch.Tensor[B, D, F, H, W] or torch.Tensor[B, D, H, W]
            MSI to be warped.
        sph_ray_intersec : torch.Tensor[B, D, H, W, 2]
            Erp intersection locations between rays originating from target pose and spheres around 'msi' pose.
        Returns
        -------
        warped_img : torch.Tensor[B, D, F, H, W]
            MSI image warped to target pose.
        """
        msi = msi.unsqueeze(2) if len(msi.shape) == 4 else msi
        warped_img = self.spt_utils.sample_equi(input=msi, grid=sph_ray_intersec)
        return warped_img