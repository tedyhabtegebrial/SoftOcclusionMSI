import torch
import torch.nn as nn
import torch.nn.functional as F

class ComputeHomographies(nn.Module):
    def __init__(self, configs):
        super(ComputeHomographies, self).__init__()
        self.configs = configs
        self.num_spheres = configs.num_spheres

    def __call__(self, r_mats, t_vecs, cam_info):
        print(r_mats.shape, t_vecs.shape)
        b_size = r_mats.shape[0]
        device = r_mats.device
        near, far = cam_info['near'], cam_info['far']
        if self.configs.invz:
            depth_proposals = torch.linspace(
                1 / near, 1 / far, self.num_spheres)
            depth_proposals = 1 / depth_proposals
        else:
            depth_proposals = torch.linspace(near, far, self.num_spheres)
        depth_proposals = depth_proposals.view(self.num_spheres, 1, 1).to(device)
        # Reference camera pose: from world-to-camera
        r_s, t_s = cam_info['r_s'], cam_info['t_s']
        r_s, t_s = r_s.view(-1, 3, 3), t_s.view(-1, 3, 1)
        r_s, t_s = r_s.expand(b_size, 3, 3), t_s.expand(b_size, 3, 1)
        # new camera poses: from reference to target
        t_s2t = torch.bmm(r_s.permute(0,2,1), -1*t_s) + t_vecs
        r_s2t = torch.bmm(r_mats, r_s.permute(0, 2, 1))
        r_t2s = torch.inverse(r_s2t.clone())
        # target camera k-matrix
        k_t = torch.Tensor([[cam_info['fx'], 0, cam_info['cx']],
                            [0, cam_info['fy'], cam_info['cy']],
                            [0, 0, 1], ]).to(device)
        # source camera k-matrix
        k_s = k_t.clone()
        k_s[0, 2] += self.configs.offset
        k_s[1, 2] += self.configs.offset
        #
        k_t = k_t.view(1, 3, 3).expand(b_size, 3, 3)
        k_s = k_s.view(1, 3, 3).expand(b_size, 3, 3)
        #
        k_t_inv = torch.inverse(k_t)
        k_s_inv = torch.inverse(k_s)
        
        #
        
        # expand tensors to fit multi-plane specification
        def expand(x): return x.unsqueeze(1).repeat(repeats=(1, self.num_spheres, 1, 1))
        k_t, k_t_inv = expand(k_t), expand(k_t_inv)
        k_s, k_s_inv = expand(k_s), expand(k_s_inv)
        t_s2t, r_s2t = expand(t_s2t), expand(r_s2t)
        hmats = self.calculate_h_mats(k_s, k_t, r_s2t, t_s2t, depth_proposals)
        return hmats, r_t2s

    def calculate_h_mats(self, k_s, k_t, r_mats, t_vec, depth_proposals):
        device_ = k_s.device
        batch_size = r_mats.shape[0]
        num_dep = depth_proposals.shape[0]

        r_mats = r_mats.contiguous().view(-1, 3, 3)
        t_vec = t_vec.contiguous().view(-1, 3, 1)
        kmat = k_s.view(-1, 1, 3, 3).contiguous()
        kinv = torch.stack([torch.inverse(k) for k in k_t]).contiguous()
        # print(kinv.shape, kmat.shape, r_mats.shape, t_vec.shape)
        # exit()
        kinv, kmat = kinv.view(-1, 3, 3), kmat.view(-1, 3, 3)
        n = torch.Tensor([0, 0, 1]).view(1, 1, 3).expand(r_mats.shape[0], 1, 3)
        n = n.to(device_).float()
        depth_proposals = depth_proposals.view(-1, 1, 1)
        num_1 = torch.bmm(torch.bmm(torch.bmm(r_mats.permute(
            0, 2, 1), t_vec), n), r_mats.permute(0, 2, 1))
        den_1 = -depth_proposals - \
            torch.bmm(torch.bmm(n, r_mats.permute(0, 2, 1)), t_vec)
        h_mats = torch.bmm(
            torch.bmm(kmat, (r_mats.permute(0, 2, 1) + (num_1 / den_1))), kinv)
        h_mats = h_mats.view(batch_size, num_dep, 3, 3)
        return h_mats        


class MPIGrid(nn.Module):
    def __init__(self, configs):
        super(MPIGrid, self).__init__()
        self.configs = configs

    def create_grid(self, b, d, h, w, device):
        x_locs = torch.linspace(0.5, w - 0.5, w).to(device)
        x_locs = x_locs.view(1, 1, w, 1).expand(1, h, w, 1)
        y_locs = torch.linspace(0.5, h - 0.5, h).to(device)
        y_locs = y_locs.view(1, h, 1, 1).expand(1, h, w, 1)
        ones_ = torch.ones_like(x_locs)
        grid = torch.cat([x_locs, y_locs, ones_], dim=3).view(1, 1, h, w, 3)
        grid = grid.repeat(b, d, 1, 1, 1)
        return grid.unsqueeze(-1)

    def forward(self, hmats, inp_shape, targ_shape):
        device = hmats.device
        b_size = hmats.shape[0]
        num_spheres = hmats.shape[1]
        h, w = targ_shape[:2]
        # get a regular grid of shape [b, d, h, w, 3, 1]
        grid_hom = self.create_grid(b_size, num_spheres, h, w, device)
        hmats_exp = hmats.view(b_size, num_spheres, 1, 1, 3, 3)
        # project points from target to source
        pts_src = torch.matmul(hmats_exp, grid_hom).squeeze(-1)
        pts_x, pts_y, pts_z = torch.split(
            pts_src, split_size_or_sections=1, dim=-1)
        pts_x, pts_y = pts_x/pts_z, pts_y/pts_z
        # normalize x and y locations
        pts_x = (pts_x / inp_shape[1])
        pts_y = (pts_y / inp_shape[0])
        # concatenate x and y locations
        normalized_grid = torch.cat([pts_x, pts_y], dim=-1)
        return normalized_grid.clamp(min=0, max=1), grid_hom[:, 0, ...]

class MPIWarper(nn.Module):
    def __init__(self, configs):
        super(MPIWarper, self).__init__()
        self.configs = configs
        
    def forward(self, input_tens, grid):
        """Forward
            given a MPI features and MPI samping grids it retuns warped MPI features
        """
        b, d, c, h, w = input_tens.shape
        _b, _d, h_g, w_g, _c = grid.shape
        # grid_x, grid_y = torch.split(grid, split_size_or_sections=1, dim=-1)
        # grid_x = 2*(grid_x - 0.5)
        # grid_y = 2*(grid_x - 0.5)
        norm_grid = 2*(grid - 0.5)
        sampled_feats = F.grid_sample(input_tens.view(b*d, c, h, w), norm_grid.view(b*d, h_g, w_g, 2),
                                      mode='bilinear', align_corners=False)
        return sampled_feats.view(b, d, c, h_g, w_g)