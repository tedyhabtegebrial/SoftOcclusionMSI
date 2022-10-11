import os
import csv
from pathlib import Path
from numpy.lib.function_base import interp
import torch
import h5py
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import json
import cv2 as cv
from .utils import sample_triple
import torch.nn.functional as F
import random
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

# scenes with min depth>0.5
# Selcted
selected_scenes = [2, 1, 7, 131, 78, 8, 9, 29, 40, 41, 46, 47, 55, 63, 68, 81, 88, 115]
# All scenes
all_scenes = [1, 2, 6, 7, 8, 9, 10, 11, 13, 14, 19, 20, 23, 27, 29,
                31, 32, 33, 34, 39, 40, 41, 43, 45, 46, 47, 49, 52, 
                53, 55, 56, 58, 63, 64, 65, 67, 68, 70, 73, 74, 76,
                77, 78, 79, 80, 81, 82, 83, 85, 86, 88, 89, 90, 91,
                92, 93, 94, 95, 97, 100, 101, 102, 103, 106, 107, 110,
                111, 114, 115, 116, 119, 120, 122, 123, 131]
class ResidentialLoader(Dataset):
    """docstring for RicohLoader."""


    
    def __init__(self, configs):
        super(ResidentialLoader, self).__init__()
        self.configs = configs
        split = configs.mode
        self.split = split
        # get data and save it
        val_index = 1
        ref_idx = 4
        # scene_number = selected_scenes[configs.scene_number]
        scene_number = configs.scene_number
        file_name = os.path.join(configs.dataset_path, f'residential/{scene_number}.h5')
        with h5py.File(file_name, 'r') as f:
            color = torch.from_numpy(f['color'][:]).float().cpu()
            color = color/255
            color = 2 * (color - 0.5)
            color = color.permute(0, 3, 1, 2)
            h_w = [self.configs.height, self.configs.width]
            self.all_color = F.interpolate(color.cpu(), size=h_w,
                                  align_corners=False, mode='bilinear')
            pose = torch.from_numpy(f['pose'][:]).float()
            self.all_pose = self.adjust_world(pose, ref_idx)
            train_indices = list(range(9))
            train_indices.pop(val_index)
            self.train_indices = train_indices
            self.val_indices = [val_index]
    
    def read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as fid:
            reader = csv.reader(fid)
            for line in reader:
                lines.extend(line)
        lines = list(map(float, lines))
        lines = list(map(int, lines))
        return lines
    
    def get_inv_transform_location(self, trg_pose, return_identity=False):
        if return_identity:
            return trg_pose
        x_rot = random.uniform(-1.7, 1.7)
        y_rot = random.uniform(-1.7, 1.7)
        z_rot = random.uniform(-1.7, 1.7)
        x_t = random.uniform(-0.01, 0.01)
        y_t = random.uniform(-0.01, 0.01)
        z_t = random.uniform(-0.01, 0.01)
        t_vecs = torch.FloatTensor([x_t, y_t, z_t]).view(3,1)
        r_mats = torch.from_numpy(Rot.from_euler('zyx', [z_rot, y_rot, x_rot], degrees=True).as_matrix()).float()
        # From Inv-transpose location to target camera
        p_invtrans_2_trg = torch.cat([r_mats, t_vecs], dim=1)
        p_invtrans_2_trg = torch.cat([p_invtrans_2_trg, torch.FloatTensor([0, 0, 0, 1]).view(1,4)], dim=0)
        # from Inv-transpose location to world
        trg_pose_2_w = torch.cat([trg_pose, torch.FloatTensor([0, 0, 0, 1]).view(1, 4)], dim=0)
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

    def adjust_world(self, pose, ref_idx):
        mid_idx = ref_idx
        ref_pose = (pose[mid_idx]).view(1, 4, 4).expand_as(pose)
        pose = torch.bmm(torch.inverse(ref_pose), pose)
        return pose

    def __len__(self):
        if self.split == 'train':
            return 100*len(self.train_indices)
        else:
            return len(self.val_indices)

    def __getitem__(self, index):
        if self.split == 'train':
            # index = index % len(self.train_indices)
            ref_idx, inp_idx, trg_idx = sample_triple(self.train_indices)
        else:
            ref_idx, inp_idx, trg_idx = 0, 8, self.val_indices[0]
        inp_pose = self.all_pose[inp_idx]
        inp_img = self.all_color[inp_idx]
        trg_pose = self.all_pose[trg_idx]
        trg_img = self.all_color[trg_idx]
        ref_pose = self.all_pose[ref_idx]
        ref_img = self.all_color[ref_idx]
        p1, p2, _ = sample_triple(self.train_indices)
        # pose for inverse transform regularization location to world
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


if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.exp_name = 'multi_view_rendering'
            self.logging_dir = '/home/habtegebrial/Desktop/repos/spherical_expts'
            self.dataset = 'ricoh'
            self.mode='train'
            self.dataset_path = '/home/habtegebrial/Desktop/repos/datasets/ricoh/spherical/ResidentialHouse/lightfield.h5'
            # dataset settings
            self.width = 512
            self.height = 384
            self.scene_number = 1
            self.val_index = 1
            self.load_depth = False
    configs = Configs()
    loader = RicohLoader(configs)
    data = loader.__getitem__(10)
    for k, v in data.items():
        print(k, v.shape)
