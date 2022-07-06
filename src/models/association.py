import torch
import torch.nn as nn
import torch.nn.functional as F


class Alpha2Assoc(torch.nn.Module):
    def __init__(self, configs):
        '''
            This class computes association between planes and layers.
            number of planes is inferred from the input alpha
            number of layers is given as an input argument
        '''
        super(Alpha2Assoc, self).__init__()
        self.num_layers = configs.num_layers

    def forward(self, alpha_imgs):
        alpha = alpha_imgs
        alpha = alpha.clamp(min=0.0001)
        ones = torch.ones_like(alpha[:, 0, ...]).unsqueeze(1)
        vis_list = [torch.zeros_like(alpha[:, 0, ...]).unsqueeze(1)]
        occ_list = [torch.ones_like(alpha[:, 0, ...]).unsqueeze(1)]
        for l in range(self.num_layers):
            alpha = alpha * occ_list[-1]
            vis = torch.cumprod(torch.cat([ones, 1-alpha[:, :-1, :, :, :]], dim=1), dim=1)
            vis_list.append(vis*(occ_list[-1]))
            occ_list.append(1 - vis)
        visibility = torch.cat(vis_list[1:], dim=2)
        # visibility = torch.sum(visibility, dim=2)
        return visibility

if __name__ == '__main__':
    h, w = 256, 256
    num_layers = 4
    num_planes = 32
    class Obj:
        def __init__(self, num_layers): self.num_layers = num_layers
    configs = Obj(num_layers)
    alpha = torch.rand(1, num_planes, 1, h, w).float()
    alpha_2_assoc = Alpha2Assoc(configs)
    assoc = alpha_2_assoc(alpha)
    print(assoc.shape)