import os
import h5py
import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms as T
# from ray_utils import *
import torch.nn.functional as F
from .ray_utils import get_sphere_rays, get_sphere_ray_directions
from .spt_utils import Utils

class ResidentialDataset(Dataset):
    def __init__(self, split, opts):
        root_dir = opts.root_dir
        img_wh = opts.img_wh
        self.spt_utils = Utils(copy.deepcopy(opts))
        self.root_dir = root_dir
        self.near, self.far = 0.5, 60
        self.split = split
        self.opts = opts
        self.img_wh = img_wh
        self.read_meta()
        self.white_back = True

    def _adjust_world(self, pose, ref_idx):
        mid_idx = ref_idx
        ref_pose = (pose[mid_idx]).view(1, 4, 4).expand_as(pose)
        pose = torch.bmm(torch.inverse(ref_pose), pose)
        return pose

    def _load_ricoh(self):
        val_index = 1
        ref_idx = 4
        scene_number = self.opts.scene_number
        w, h = self.img_wh
        file_name = os.path.join(self.opts.root_dir, 'residential', f'{scene_number}.h5')
        with h5py.File(file_name, 'r') as f:
            color = torch.from_numpy(f['color'][:]).float().contiguous()
            color = color/255
            color = color.permute(0, 3, 1, 2)
            color = F.interpolate(color, size=(h, w), align_corners=False, mode='bilinear')
            color = color.permute(0, 2, 3, 1)
            pose = torch.from_numpy(f['pose'][:]).float().contiguous()
            pose = self._adjust_world(pose, ref_idx)[..., :3, :4]
            train_indices = list(range(9))
            train_indices.pop(val_index)
            train_pose = pose[train_indices].contiguous()
            train_color = color[train_indices].contiguous()
            val_pose = pose[val_index].contiguous()
            val_color = color[val_index].contiguous()
        return train_pose, train_color, val_pose, val_color

    def read_meta(self):
        train_pose, train_color, val_pose, val_color = self._load_ricoh()
        self.bounds = np.array([self.near, self.far])        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_sphere_ray_directions(self.spt_utils) # (h, w, 3)
        self.train_poses = train_pose
        self.all_train_rays = []
        self.all_train_rgbs = train_color.view(-1, 3)
        for frame in range(self.train_poses.shape[0]):
            c2w = self.train_poses[frame]
            rays_o, rays_d = get_sphere_rays(self.directions, c2w) # both (h*w, 3)
            self.all_train_rays += [torch.cat([rays_o, rays_d,
                                            self.near*torch.ones_like(rays_o[:, :1]),
                                            self.far*torch.ones_like(rays_o[:, :1])],
                                            1)] # (h*w, 8)
        self.all_train_rays = torch.cat(self.all_train_rays, 0)
        self.val_poses = [val_pose]
        self.all_val_rays = []
        self.all_val_rgbs = val_color.view(-1, 3)
        for frame in range(len(self.val_poses)):
            c2w = self.val_poses[frame]
            rays_o, rays_d = get_sphere_rays(
                self.directions, c2w)  # both (h*w, 3)
            self.all_val_rays += [torch.cat([rays_o, rays_d,
                                            self.near *
                                            torch.ones_like(rays_o[:, :1]),
                                            self.far*torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)
        self.all_val_rays = torch.cat(self.all_val_rays, 0)

    def __len__(self):
        if self.split == 'train':
            return len(self.all_train_rays)
        else:
            return 1

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_train_rays[idx],
                      'rgbs': self.all_train_rgbs[idx]}

        else: # create data for each image separately
            img = self.all_val_rgbs
            c2w = self.val_poses[0][:3, :4]
            rays_o, rays_d = get_sphere_rays(self.directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1)
            # valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w}
        return sample
