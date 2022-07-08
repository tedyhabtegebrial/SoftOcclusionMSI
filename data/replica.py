import os

import numpy as np
import cv2 as cv

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class ReplicaLoader(Dataset):
    def __init__(self, configs):
        super(ReplicaLoader, self).__init__()
        self.configs = configs
        self.baseline = 0.1
        self.poses, self.images, self.ref_img = self._load_replica()

    def _get_pose(self, idx_list, ref_idx):
        pose_list = []
        h_ref, w_ref = ref_idx//9, ref_idx%9
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
        self.split = self.configs.mode
        scene_number = self.configs.scene_number
        base_name = os.path.join(
            self.configs.dataset_path, f'replica/scene_{str(scene_number).zfill(2)}')
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
        return poses, images, ref_img


    def __len__(self):
        if self.configs.mode == 'train':
            return 100 * len(self.images)
        else:
            return len(self.images)

    def __getitem__(self, index):
        if self.configs.mode == 'train':
            index = index % len(self.images)
        data = {'color': (self.images[index]).squeeze(),
                'pose': (self.poses[index]).view(3, 4), 'ref_rgb': self.ref_img.squeeze()}
        return data

    def _read_imgs(self, files):
        height, width = self.configs.height, self.configs.width
        images_list = []
        for f in files:
            img = np.asarray(cv.imread(f)[:, :, :3]
                             [..., [2, 1, 0]], dtype=np.float32)
            img_tens = torch.from_numpy(img).permute(2, 0, 1) / 255.0
            img_tens = 2*(img_tens - 0.5)
            img_tens = F.interpolate(img_tens.unsqueeze(0), size=(
                height, width), mode='bilinear', align_corners=True)
            images_list.append(img_tens.squeeze())
        images_list = torch.stack(images_list)
        return images_list
