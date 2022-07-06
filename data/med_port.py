import os
import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class MedPortLoader(Dataset):
    def __init__(self, configs):
        super(MedPortLoader, self).__init__()
        self.configs = configs
        near = 2.5
        far = 50.0
        print(f'Recommended near = {near} and far = {far}')
        folder = os.path.join(os.path.abspath(
            self.configs.dataset_path), 'medieval_port/tonemapped')
        self.split = configs.mode
        if configs.num_inputs==9:
            # This is what we used for the CVPR submission
            train_idxs = [0, 2, 4, 10, 12, 14, 20, 22, 24]
        else:
            raise NotImplementedError(f"please choose input views for num_inputs=={configs.num_inputs}")
        # The MSI will be defined at Cam 12
        ref_idx = 12
        ref_img = os.path.join(
            folder, f'{str(ref_idx).zfill(8)}.jpg')
        if self.split == 'train':
            idxs = train_idxs
            files = [os.path.join(
                folder, f'{str(i).zfill(8)}.jpg') for i in idxs]
        elif self.split=='demo':
            idxs = list(range(len(os.listdir(folder))))
            files = [os.path.join(
                folder, f'{str(i).zfill(8)}.jpg') for i in idxs]
        elif self.split=='val':
            # Val split
            idxs = [1, 3, 7, 11, 13, 17, 21, 23]
            files = [os.path.join(
                folder, f'{str(i).zfill(8)}.jpg') for i in idxs]
        else:
            # Test split
            idxs = [6, 8, 16, 18, 5, 9, 15, 19]
            files = [os.path.join(
                folder, f'{str(i).zfill(8)}.jpg') for i in idxs]
        self.poses = self._get_pose(idxs, ref_idx)
        self.imgs = self._read_imgs(files)
        self.ref_rgb = self._read_imgs([ref_img])

    def _get_pose(self, idx_list, ref_idx):
        pose_list = []
        h_ref, w_ref = ref_idx//5, ref_idx%5
        r_mat = torch.eye(3)
        for idx in idx_list:
            h_idx, w_idx = idx//5, idx%5
            pose = torch.FloatTensor([0, 0.2*(w_ref - w_idx), 0.2*(h_ref - h_idx)]).view(3, 1)
            pose = torch.cat([r_mat, pose], dim=1)
            pose_list.append(pose)
        pose_list = torch.stack(pose_list)
        return pose_list

    def __len__(self):
        if self.split == 'train':
            return len(self.imgs)*100
        else:
            return len(self.imgs)

    def __getitem__(self, index):
        if self.configs.split=='train':
            index = index % len(self.imgs)
        imgs = self.imgs[index]
        poses = self.poses[index]
        ref_rgb = self.ref_rgb.squeeze()
        data = {'color': (imgs).squeeze(),
                'pose': poses, 'ref_rgb': ref_rgb}
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
                height, width), mode='bilinear', align_corners=False)
            images_list.append(img_tens.squeeze())
        images_list = torch.stack(images_list)
        return images_list
