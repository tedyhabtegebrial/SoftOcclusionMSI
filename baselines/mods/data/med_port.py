import os
import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as F
import random

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from .utils import sample_triple
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


class MedPortLoader(Dataset):
    def __init__(self, configs):
        super(MedPortLoader, self).__init__()
        self.configs = configs
        scene_number = 0
        near = [2.5, 2.0][scene_number]
        far = [50.0, 5][scene_number]
        print(f'Recommended near = {near} and far = {far}')
        folder = os.path.join(self.configs.dataset_path, 'tonemapped')
        self.split = configs.mode
        # if configs.num_inputs==9:
        #     # train_idxs = [0, 1, 3, 4, 5, 9, 12, 15, 19, 20, 21, 24]
        self.train_indices = [0, 2, 4, 10, 12, 14, 20, 22, 24]
        # elif configs.num_inputs == 5:
        #     # train_idxs = [0, 4, 5, 9, 12, 15, 20, 24]
        #     self.train_idxs = [0, 4, 12, 20, 24]
        # elif configs.num_inputs == 3:
        #     # train_idxs = [0, 4, 12, 20, 24]
        #     self.train_idxs = [0, 12, 24]
        # else:
        #     assert False, 'please choos num_inputs from [13, 8, 5]'
        self.val_indices = [6, 8, 16, 18, 5, 9, 15, 19]
        ref_idx = 12
        files = [os.path.join(
            folder, f'{str(i).zfill(8)}.jpg') for i in range(25)]
        self.all_pose = self._get_pose(list(range(25)), ref_idx)
        self.all_img = self._read_imgs(files)

    def get_inv_transform_location(self, trg_pose, return_identity=False):
        if return_identity:
            return trg_pose
        x_rot = random.uniform(-1.7, 1.7)
        y_rot = random.uniform(-1.7, 1.7)
        z_rot = random.uniform(-1.7, 1.7)
        x_t = random.uniform(-0.01, 0.01)
        y_t = random.uniform(-0.01, 0.01)
        z_t = random.uniform(-0.01, 0.01)
        t_vecs = torch.FloatTensor([x_t, y_t, z_t]).view(3, 1)
        r_mats = torch.from_numpy(Rot.from_euler(
            'zyx', [z_rot, y_rot, x_rot], degrees=True).as_matrix()).float()
        # From Inv-transpose location to target camera
        p_invtrans_2_trg = torch.cat([r_mats, t_vecs], dim=1)
        p_invtrans_2_trg = torch.cat(
            [p_invtrans_2_trg, torch.FloatTensor([0, 0, 0, 1]).view(1, 4)], dim=0)
        # from Inv-transpose location to world
        trg_pose_2_w = torch.cat(
            [trg_pose, torch.FloatTensor([0, 0, 0, 1]).view(1, 4)], dim=0)
        p_invtrans_2_wld = torch.mm(trg_pose_2_w, p_invtrans_2_trg)
        return p_invtrans_2_wld[:3, :]
        # r_1 = (p1[:3, :3]).numpy()
        # r_2 = (p2[:3, :3]).numpy()
        # key_rots = Rot.from_matrix([r_1, r_2])
        # key_times = [0, 1]
        # slerp = Slerp(key_times, key_rots)
        # t = random.uniform(0, 1)
        # interp_rots = torch.from_numpy(slerp([t]).as_matrix().reshape(3,3))
        # t_1, t_2 = (p1[:3, 3]).view(3,1), (p2[:3, 3]).view(3,1)
        # t_center = t*t_1 + (1-t)*t_2
        # pose_intrp = torch.cat([interp_rots, t_center], dim=1)
        # return pose_intrp

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
            return 100*len(self.train_indices)
        else:
            return len(self.val_indices)

    def __getitem__(self, index):
        if self.split == 'train':
            ref_idx, inp_idx, trg_idx = sample_triple(self.train_indices)
        else:
            ref_idx, inp_idx, trg_idx = 0, 24, self.val_indices[index]
        inp_pose = self.all_pose[inp_idx]
        inp_img = self.all_img[inp_idx]
        trg_pose = self.all_pose[trg_idx]
        trg_img = self.all_img[trg_idx]
        ref_pose = self.all_pose[ref_idx]
        ref_img = self.all_img[ref_idx]
        pose_ti = self.get_inv_transform_location(trg_pose)
        data = {'pose_ti': pose_ti,
                'inp_pose': inp_pose,
                'ref_pose': ref_pose,
                'trg_pose': trg_pose,
                'inp_img': inp_img,
                'ref_img': ref_img,
                'trg_img': trg_img,
                }
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
        # images_list = images_list.view(-1, 3, height, width)
        return images_list

if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.dataset = 'd3dkit'
    configs = Configs()
    configs.scene_number = 1
    configs.height = 512
    configs.width = 1024
    # configs.dataset_path = '/netscratch/teddy/dense3dkit/spherical_light_fields/datasets/'
    configs.dataset_path = '/home/habtegebrial/Desktop/repos/datasets/dense3dkit/spherical_light_fields/datasets'
    configs.mode='train'    
    dataset = D3DKitLoader(configs)
    print('Dataset length: ', len(dataset))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    # for itr, data in enumerate(loader):
    #     print(itr, data['index'])
    # loader = iter(loader)
    # data = next(loader)
    # print(data['pose'])
