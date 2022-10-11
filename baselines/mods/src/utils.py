import torch
import torch.nn as nn
from .spt import Utils as spt
from src.rendering import Rendering
import math

class Utils(nn.Module):
    """
    Spherical View Synthesis utils.
    """
    def __init__(self, config):
        super(Utils, self).__init__()
        self.height = config.height
        self.width = config.width
        self.dataset = config.dataset
        self.device = config.device
        self.spt_utils = spt(height=self.height, width=self.width, dataset=self.dataset, device=self.device)
        self.renderer = Rendering(config=config)

    def get_radii(self, near_depth, far_depth, num_layers):
        """
        Radii for sphere sweep volume.

        Parameters
        ----------
        near_depth : float
            Depth of nearest sphere.
        far_depth : float
            Depth of farthest spehre.
        num_layers : int
            Number of spheres.

        Returns
        -------
        radii : torch.Tensor[D]
            Depths of spheres.
        """
        radii = 1.0/torch.linspace(1/near_depth, 1/far_depth, num_layers)
        radii = radii.to(self.device)
        return radii
    
    def get_center_pose(self, pose1, pose2):
        r1, t1 = pose1
        _, t2 = pose2
        t_c = (t1 + t2)/2

        return r1, t_c

    def get_transform(self, factor=1.0, batch_size=1):
        """
        Creates a random transformation. Transformation is in the range [-1.7°, 1.7°] for rotation and [-0.01, 0.01] for translation.
        
        Parameters
        ----------
        factor : float
            Constant factor multiplied to the random transform.
        batch_size : Integer
            batch size.

        Returns
        -------
        r_trans : torch.Tensor[B, 3, 3]
            Rotation matrix of random transform.
        t_trans : torch.Tensor[B, 3, 1]
            Translation vector of random transform.
        """
        #paper: factor of 1.0 -> (x,y,z)+/-0.01m, (theta, phi, psi)+/-1.7°
        rnd = torch.randint(low=0, high=2, size=(2,3)).to(self.device)
        rnd[rnd==0] = -1

        t_trans = rnd[0,:] * 0.01 * factor
        
        r_vals = rnd[1,:] * 0.0296706 * factor #0.0296706 ~= 1.7°
        a = r_vals[0]; b = r_vals[1]; r = r_vals[2]
        r_trans = torch.Tensor([[a.cos()*b.cos(), a.cos()*b.sin()*r.sin()-a.sin()*r.cos(), a.cos()*b.sin()*r.cos()+a.sin()*r.sin()],
                                [a.sin()*b.cos(), a.sin()*b.sin()*r.sin()+a.cos()*r.cos(), a.sin()*b.sin()*r.cos()-a.cos()*r.sin()],
                                [-b.sin(), b.cos()*r.sin(), b.cos()*r.cos()]]).to(self.device)

        return r_trans.view(1,3,3).expand(batch_size,3,3), t_trans.view(1,3,1).expand(batch_size,3,1)
    
    def get_sphere_sweep_volume(self, view_ref, view_input, pose_sph_ref, pose_sph_inp, radii, transform=None):
        """
        Creates a sphere sweep volume.

        Parameters
        ----------
        view_ref : torch.Tensor[B, RGB, H, W]
            Reference RGB image.
        view_input : torch.Tensor[B, RGB, H, W]
            Input RGB image.
        pose_sph_ref : Tuple/list of relative rotation and translation
            Camera pose Sphere -> Reference.
        pose_sph_inp : Tuple/list of relative rotation and translation
            Camera pose Sphere -> Input.
        radii : torch.Tensor[D]
            List of radii.
        device : string, optional
            Device to be used. The default is 'cpu'.

        Returns
        -------
        sph_vol : torch.Tensor[B, D, 6, H, W]
            Sphere volume consisting of reference and transformed input view.
        """
        batch_size, *_ = view_ref.shape
        sph_vol = torch.empty(0).to(self.device)
        r_sph_ref, t_sph_ref = pose_sph_ref
        r_sph_inp, t_sph_inp = pose_sph_inp
        
        erp_locs_ref = self.spt_utils.get_xy_coords(batch_size=batch_size, device=self.device)
        for radius in radii:
            sph_locs_ref = self.spt_utils.equi_2_spherical(erp_locs_ref, radius=radius)
            xyz_locs_ref1 = self.spt_utils.spherical_2_cartesian(sph_locs_ref)
            if not (transform == None):
                r_trans, t_trans = transform
                xyz_locs_ref1 = self.spt_utils.transform_3d_points(xyz_locs_ref1, rot_mats=r_trans, t_vecs=t_trans)
            xyz_locs_ref2 = xyz_locs_ref1.clone()

            xyz_locs_input1 = self.spt_utils.transform_3d_points(xyz_locs_ref1, rot_mats=r_sph_ref, t_vecs=t_sph_ref)
            norm_xyz_input1 = self.spt_utils.normalize_3d_vectors(xyz_locs_input1)
            sph_locs_input1 = self.spt_utils.cartesian_2_spherical(norm_xyz_input1, normalized=True)
            erp_locs_input1 = self.spt_utils.spherical_2_equi(sph_locs_input1)
            view_input_ref_si_1 = self.spt_utils.sample_equi(input=view_ref, grid=erp_locs_input1)

            xyz_locs_input2 = self.spt_utils.transform_3d_points(xyz_locs_ref2, rot_mats=r_sph_inp, t_vecs=t_sph_inp)
            norm_xyz_input2 = self.spt_utils.normalize_3d_vectors(xyz_locs_input2)
            sph_locs_input2 = self.spt_utils.cartesian_2_spherical(norm_xyz_input2, normalized=True)
            erp_locs_input2 = self.spt_utils.spherical_2_equi(sph_locs_input2)
            view_input_ref_si_2 = self.spt_utils.sample_equi(input=view_input, grid=erp_locs_input2)
            
            sph_vol = torch.cat([sph_vol, torch.cat([view_input_ref_si_1.view(batch_size, 1, 3, self.height, self.width),
                                                    view_input_ref_si_2.view(batch_size, 1, 3, self.height, self.width)], dim=2)], dim=1)
        
        return sph_vol
    
    def get_predicted_and_disp(self, alphas, blend_wgts, bg_img, fg_img, radii, ref_pose, tgt_pose, transform=None):
        """
        Render predicted view and depth image.

        Parameters
        ----------
        alpha_imgs : torch.Tensor[B, D, 1, H, W]
            Alpha images.
        blend_wgts : torch.Tensor[B, D, 1, H, W] or torch.Tensor[B, D, H, W]
            Blending weights.
        bg_img : torch.Tensor[B, D, F, H, W] or torch.Tensor[B, 1, F, H, W] or torch.Tensor[B, F, H, W]
            Background image.
        fg_img : torch.Tensor[B, D, F, H, W] or torch.Tensor[B, 1, F, H, W] or torch.Tensor[B, F, H, W]
            Foreground image.
        radii : torch.Tensor[D]
            List of radii.
        ref_pose : torch.Tensor[B, 3, 3]
            Pose of spheres to be intersected.
        tgt_pose : torch.Tensor[B, 3, 1] or torch.Tensor[B, 3]
            Pose of originating rays.

        Returns
        -------
        pred_view : torch.Tensor[B, 3, H, W]
            Predicted view.
        disp : torch.Tensor[B, 1, H, W]
            Depth image.
        """
        r_ref, t_ref = ref_pose
        r_tgt, t_tgt = tgt_pose

        msi_img = self.renderer.image_blending(bg_img, fg_img, blend_wgts)

        if not (transform == None):
            sph_ray_intersection = self.renderer.get_sphere_ray_intersections(src_pose=[r_ref, t_ref], tgt_pose=[r_tgt, t_tgt], radii=radii, transform=transform)
        else:
            sph_ray_intersection = self.renderer.get_sphere_ray_intersections(src_pose=[r_ref, t_ref], tgt_pose=[r_tgt, t_tgt], radii=radii)

        msi_img_novel = self.renderer.image_warping(msi=msi_img, sph_ray_intersec=sph_ray_intersection)
        msi_alpha_novel = self.renderer.image_warping(msi=alphas, sph_ray_intersec=sph_ray_intersection)

        pred_view = self.renderer.alpha_compose(msi_img_novel, msi_alpha_novel)
        disp = self.get_depth_from_alpha(msi_alpha_novel, radii.view(1, radii.shape[0], 1, 1, 1))
        return pred_view, disp
    
    def get_depth_from_alpha(self, alphas, radii):
        depth = self.renderer.alpha_compose(src_imgs=radii, alpha_imgs=alphas)
        disp = 1/depth
        return disp