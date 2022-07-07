import os
import torch
import h5py
from torch.utils.data import Dataset
import torch.nn.functional as F


class ResidentialAreaLoader(Dataset):
    """docstring for RicohLoader."""

    def __init__(self, configs):
        super(ResidentialAreaLoader, self).__init__()
        self.configs = configs
        self.split = configs.mode
        # get data and save it
        val_index = 1
        ref_idx = 4
        scene_number = configs.scene_number
        file_name = os.path.join(configs.dataset_path, 'residential', f'{scene_number}.h5')
        
        with h5py.File(file_name, 'r') as f:
            color = torch.from_numpy(f['color'][:]).float().cpu()
            color = color/255
            color = 2 * (color - 0.5)
            color = color.permute(0, 3, 1, 2)
            color = F.interpolate(color.cpu(), size=(self.configs.height, self.configs.width),
                                  align_corners=False, mode='bilinear')
            pose = torch.from_numpy(f['pose'][:]).float()
            pose = self.adjust_world(pose, ref_idx)
            train_indices = list(range(9))
            train_indices.pop(val_index)
            val_index = val_index
            self.ref_rgb = color[ref_idx]
            self.train_pose = pose[train_indices]
            self.train_color = color[train_indices]
            self.val_pose = pose[val_index]
            self.val_color = color[val_index]

    def adjust_world(self, poses, ref_idx):
        """
        Given camera poses and a reference camera index.
        Input poses are defined as: from camera to world.
        Returns new poses that are defined from the each camera to the reference camera.
        """
        mid_idx = ref_idx
        ref_pose = (poses[mid_idx]).view(1, 4, 4).expand_as(poses)
        poses_wrt_ref = torch.bmm(torch.inverse(ref_pose), poses)
        return poses_wrt_ref

    def __len__(self):
        if self.split == 'train':
            return 100*(self.train_color.shape[0])
        else:
            return 1

    def __getitem__(self, index):
        if self.split == 'train':
            index = index%self.train_color.shape[0]
            pose = self.train_pose[index]
            color = self.train_color[index]
        else:
            index = 0
            pose = self.val_pose
            color = self.val_color

        return {'color': color, 'ref_rgb': self.ref_rgb, 'pose':pose}

if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.dataset = 'residential'
            self.mode='train'
            self.dataset_path = 'somsi_data/residential/'
            self.width = 640
            self.height = 320
            self.scene_number = 0
    configs = Configs()
    loader = ResidentialAreaLoader(configs)
    data = loader.__getitem__(10)
    for k, v in data.items():
        print(k, v.shape)

