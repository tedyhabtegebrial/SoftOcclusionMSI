import os
import h5py
import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2 as cv
import torch.nn.functional as F
from torchvision import transforms as T
# from ray_utils import *
from .ray_utils import get_sphere_rays, get_sphere_ray_directions
from .spt_utils import Utils

class ReplicaDataset(Dataset):
    def __init__(self, split, opts):
        root_dir = opts.root_dir
        img_wh = opts.img_wh
        self.spt_utils = Utils(copy.deepcopy(opts))
        self.root_dir = root_dir
        self.near, self.far = 0.6, 50
        self.split = split
        self.opts = opts
        # assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh

        self.read_meta()
        self.white_back = True

    def _read_imgs(self, files):
        height, width = self.opts.img_wh[1], self.opts.img_wh[0]
        images_list = []
        for f in files:
            img = np.asarray(cv.imread(f)[:, :, :3]
                             [..., [2, 1, 0]], dtype=np.float32)
            img_tens = torch.from_numpy(img).permute(2, 0, 1) / 255.0
            img_tens = F.interpolate(img_tens.unsqueeze(0), size=(
                height, width), mode='bilinear', align_corners=False)
            images_list.append(img_tens.squeeze(0))
        images_list = torch.stack(images_list)
        images_list = images_list.view(-1, 3, height, width)
        images_list = images_list.permute(0, 2, 3, 1).contiguous()
        return images_list.squeeze(0)

    def _get_pose(self, idx_list, ref_idx):
        pose_list = []
        h_ref, w_ref = ref_idx//9, ref_idx % 9
        r_mat = torch.eye(3)
        for idx in idx_list:
            h_idx, w_idx = idx//9, idx % 9
            pose = torch.FloatTensor(
                [0, self.baseline*(w_ref - w_idx), self.baseline*(h_ref - h_idx)]).view(3, 1)
            pose = torch.cat([r_mat, pose], dim=1)
            pose_list.append(pose)
        pose_list = torch.stack(pose_list)
        return pose_list

    def _load_replica(self):
        self.baseline = 0.1  # Hardcoded for this dataset
        scene_number = self.opts.scene_number
        base_name = os.path.join(
            os.path.abspath(self.opts.root_dir), f'replica/scene_{str(scene_number).zfill(2)}')
        files = []
        # Load image file names
        for h in range(0, 9):
            for w in range(0, 9):
                fname = os.path.join(
                    base_name, f'image_{h}_{w}.png')
                files.append(fname)
        # train_idxs = [0, 1, 3, 4, 5, 9, 12, 15, 19, 20, 21, 24]
        train_idxs = [0, 2, 6, 8, 18, 26, 40, 54, 62, 72, 74, 80]
        ref_idx = train_idxs.index(40)
        ref_img = self._read_imgs([files[ref_idx]])
        if self.split == 'train':
            idxs = train_idxs
            files = [files[i] for i in idxs]
        elif self.split == 'val':
            idxs = [1, 5, 7, 37, 41, 43, 71, 77, 79]
            files = [files[i] for i in idxs]
        else:
            # Test
            idxs = [4, 20, 22, 24, 36, 38, 42, 44, 56, 58, 60, 76]
            files = [files[i] for i in idxs]

        poses = self._get_pose(idxs, 40)
        self.files = files
        images = self._read_imgs(files)
        ref_img = ref_img.squeeze()
        return poses, images

    def read_meta(self):
        # train_pose, train_color, val_pose, val_color = self._load_replica()
        self.poses, self.color = self._load_replica()
        self.bounds = np.array([self.near, self.far])
        self.directions = get_sphere_ray_directions(
            self.spt_utils)  # (h, w, 3)
        self.all_rays = []
        self.all_rgbs = self.color.view(-1, 3)
        for frame in range(self.poses.shape[0]):
            c2w = self.poses[frame]
            rays_o, rays_d = get_sphere_rays(
                self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d,
                                         self.near *
                                         torch.ones_like(rays_o[:, :1]),
                                         self.far*torch.ones_like(rays_o[:, :1])],
                                        1)]
        self.all_rays = torch.cat(self.all_rays, 0)

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        else:
            return len(self.poses)

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        else:
            img = self.color[idx].view(-1, 3)
            c2w = self.poses[idx][:3, :4]
            rays_o, rays_d = get_sphere_rays(self.directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                             1)  # (H*W, 8)
            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w}
        return sample
