from distutils.command.config import config
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

import torch.nn.functional as F

from thirdparty.nex_utils.mpi_utils import OrbiterDataset
from thirdparty.nex_utils.utils import getDatasetScale
from thirdparty.nex_utils.utils import is_deepview, prepareDataloaders


def loadDataset(args):
    dpath = os.path.join(args.dataset_path, args.dataset_type, args.scene_name)
    if args.scale == -1:
        args.scale = getDatasetScale(dpath, args.deepview_width, args.width)
    if is_deepview(dpath) and args.ref_img == '':
        with open(dpath + "/ref_image.txt", "r") as fi:
            args.ref_img = str(fi.readline().strip())
    render_style = 'llff' if args.nice_llff else 'shiny' if args.nice_shiny else ''
    dataset =  OrbiterDataset(dpath, ref_img=args.ref_img, scale=args.scale,
                          dmin=args.dmin,
                          dmax=args.dmax,
                          invz=args.invz,
                          render_style=render_style,
                          offset=args.offset,
                          cv2resize=args.cv2resize)
    sampler_train, sampler_val, dataloader_train, dataloader_val = prepareDataloaders(
        dataset,
        dpath,
        random_split=args.random_split,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers)
    return dataloader_train, dataloader_val


class NexMPILoader(Dataset):
    def __init__(self, configs):
        super(NexMPILoader, self).__init__()
        dpath = os.path.join(configs.dataset_path, configs.dataset_type, configs.scene_name)
        # with open(os.path.join(dpath, 'planes.txt')) as fid:
        #     data = map(float, [i for i in fid.readline()[:-1].split(' ')])
        #     # self.near, self.far, self.invz, self.offset = data
        #     self.mpi_dict = {'near': data[0], 'far': data[1],
        #                 'invz': bool(data[2]), 'offset': data[3]}
        #     configs.__dict__.update(self.mpi_dict)
        train_dataset, val_dataset = loadDataset(dpath, configs)
        if configs.split=='train':
            self.dataset = train_dataset
        else:
            self.dataset = val_dataset
        self.height, self.width = train_dataset.dataset.ref_img.shape[1:]
        
    
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
        # data_dict = self.dataset.__getitem__(idx)
        # data_dict.update(self.mpi_dict)
        # return data_dict



if __name__ == '__main__':


    class Args:
        def __init__(self):
            self.scale = -1
            self.ref_img = ''
            self.width = 1008
            self.deepview_width = 800
            self.nice_llff = False
            self.nice_shiny = True
            self.dmin = -1
            self.dmax = -1
            self.invz = False
            self.offset = 200
            self.cv2resize = True
            self.random_split = False
            self.train_ratio = 0.875
            self.num_workers = 8
            self.split = 'val'
            self.dataset_path = "nex_data/shiny"

            self.dataset = 'food'
    args = Args()
    dataset = NexMPILoader(args)
    print(len(dataset))
