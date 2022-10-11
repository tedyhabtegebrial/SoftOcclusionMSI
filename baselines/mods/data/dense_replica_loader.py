import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

import os
import tqdm
import torch
import json
from PIL import Image
import numpy as np
from itertools import islice
import random
import torch.nn.functional as F
import random
import copy
import cv2 as cv
from .utils import sample_triple
# random.seed(1234)
# np.random.seed(1234)
# torch.random.manual_seed(1234)
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp



class DenseReplicaLoader(Dataset):
    def __init__(self, configs):
        super(DenseReplicaLoader, self).__init__()
        # scenes_map = {0: 9, 1: 4, 4: 8, 2: 3, 5: 5, 6: 4, 7: 0,
        #             8: 5, 10: 3, 11: 8, 13: 5, 14: 1, 15: 0, 16: 3}
        self.configs = configs
        self.baseline = 0.2  # Hardcoded for this dataset
        self._load_replica()

    def _get_pose(self, idx_list, ref_idx):
        pose_list = []
        h_ref, w_ref = ref_idx//5, ref_idx % 5
        r_mat = torch.eye(3)
        for idx in idx_list:
            h_idx, w_idx = idx//5, idx % 5
            pose = torch.FloatTensor(
                [0, self.baseline*(w_ref - w_idx), self.baseline*(h_ref - h_idx)]).view(3, 1)
            pose = torch.cat([r_mat, pose], dim=1)
            pose_list.append(pose)
        pose_list = torch.stack(pose_list)
        return pose_list

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
    def _load_replica(self):
        self.split = self.configs.mode
        # {room_id:scene_id}
        scenes_map = {0: 9, 1: 4, 4: 8, 2: 3, 5: 5, 6: 4, 7: 0,
                      8: 5, 10: 3, 11: 8, 13: 5, 14: 1, 15: 0, 16: 3}
        room_id = list(scenes_map.keys())[self.configs.scene_number]
        scene_id = scenes_map[room_id]
        base_name = os.path.join(self.configs.dataset_path, f'episode_{str(room_id).zfill(5)}', str(scene_id).zfill(5))
        files = []
        # since we skip every other image in the LF
        for h in range(0, 9, 2):
            for w in range(0, 9, 2):
                fname = os.path.join(
                    base_name, f'example_{str(scene_id).zfill(10)}.color_{h}_{w}.png')
                files.append(fname)
        self.train_indices = [0, 1, 3, 4, 5, 9, 12, 15, 19, 20, 21, 24]
        # if self.configs.num_inputs == 12:
        # elif self.configs.num_inputs == 8:
        #     train_idxs = [0, 4, 5, 9, 12, 15, 20, 24]
        # elif self.configs.num_inputs == 5:
        #     train_idxs = [0, 4, 12, 20, 24]
        ref_idx = 12
        self.val_indices = [2, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 22]
        self.all_pose = self._get_pose(list(range(25)), ref_idx)
        self.all_img = self._read_imgs(files)

    def __len__(self):
        if self.configs.mode == 'train':
            return 100*len(self.train_indices)
        else:
            return len(self.val_indices)

    def __getitem__(self, index):
        if self.configs.mode == 'train':
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
                height, width), mode='bilinear', align_corners=True)
            images_list.append(img_tens.squeeze())
        images_list = torch.stack(images_list)
        return images_list

if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.dataset = 'd3dkit'
    configs = Configs()
    configs.scene_number = 1
    configs.height = 256
    configs.width = 512
    configs.num_inputs = 12
    # configs.dataset_path = '/netscratch/teddy/dense3dkit/spherical_light_fields/datasets/'
    configs.dataset_path = '/data/teddy/Datasets/dense_lf_1024_512/replica/'
    configs.mode = 'train'
    loader_train = DenseReplicaLoader(configs)
    # print('Dataset length: ', len(dataset))
    # print('Train')
    # loader_train = DataLoader(dataset, batch_size=1, shuffle=False)
    # print(loader_train.poses)
    configs.mode = 'test'
    # print('Test')
    loader_test = DenseReplicaLoader(configs)
    # print(loader_test.poses)
    # for itr, data in enumerate(loader):
    #     print(itr, data['index'])
    # loader = iter(loader)
    # data = next(loader)
    # print(data['pose'])
