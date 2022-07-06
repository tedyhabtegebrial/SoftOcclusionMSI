import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(12345)


class AlphaComposition(nn.Module):
    """This class implements alpha compostion.
    Accepts input warped images and their alpha channels.
    We perform A-over-B composition in back-to-front manner.......

    Input Tensors:
        src_imgs = [B, D, 3, H, W]
        alpha = [B, D, 1, H, W]

    Output Tensor:
        composite = [B, 3, H, W]
    """

    def __init__(self, configs=None):
        super(AlphaComposition, self).__init__()
        self.configs = configs

    def forward(self, src_imgs, alpha_imgs, return_weighted_feats=False, return_intermediate=False):
        # if self.configs.render_mode=='render_3d':
        #     ones = torch.ones_like(alpha_imgs[:, 0, ...]).unsqueeze(1)
        #     weight_1 = torch.cumprod(torch.cat([ones, 1-alpha_imgs[:, :-1, :, :, :]], dim=1), dim=1)
        #     alpha_1 = torch.cat([alpha_imgs[:, :-1, :, :, :], ones], dim=1)
        #     weighted_imgs = torch.sum(weight_1*alpha_1*src_imgs,dim=1)
        #     return weighted_imgs

        # else:
        alpha_imgs = alpha_imgs.clamp(min=0.0001)
        ones = torch.ones_like(alpha_imgs[:, 0, ...]).unsqueeze(1)
        weight_1 = torch.cumprod(
            torch.cat([ones, 1-alpha_imgs[:, :-1, :, :, :]], dim=1), dim=1)
        weighted_imgs = weight_1*alpha_imgs*src_imgs
        if return_weighted_feats:
            num_l = weighted_imgs.shape[1]
            weighted_feats = [weighted_imgs[:, l, :, :, :] for l in range(num_l)]
            comb = torch.sum(weighted_imgs, dim=1)
            return comb, weighted_feats
        elif return_intermediate:
            comb = torch.sum(weighted_imgs, dim=1)
            layer_contribs = weight_1*alpha_imgs
            return comb, layer_contribs
        else:
            comb = torch.sum(weighted_imgs, dim=1)
            return comb

    def forward_old(self, src_imgs, alpha_imgs):
        # alpha_img=[B,D,1,H,W]
        # src_img=[B,D,C,H,W]
        # print('Alpha imgs:', src_imgs.data.min(), src_imgs.data.max())
        # print('Alpha alps:', alpha_imgs.data.min(), alpha_imgs.data.max())
        b_size, num_d, _c, h, w = src_imgs.shape
        src_imgs = torch.split(src_imgs, split_size_or_sections=1, dim=1)
        alpha_imgs = torch.split(alpha_imgs, split_size_or_sections=1, dim=1)
        comp_rgb = src_imgs[-1]
        # * alpha_imgs[-1]
        for d in reversed(range(num_d - 1)):
            comp_rgb = src_imgs[d] * alpha_imgs[d] + \
                (1.0 - alpha_imgs[d]) * comp_rgb
        return comp_rgb.squeeze(1)

if __name__=='__main__':
    device = 'cuda:0'
    alpha = torch.rand(1,32,1,256,256).to(device)
    col = torch.rand(1,32,3,256,256).to(device)
    alpha_compose = AlphaComposition()
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(10):
        with torch.no_grad():
            out2 = alpha_compose.forward_old(col, alpha)
    torch.cuda.synchronize()
    t_1_1 = time.time()-t1
    t1 = time.time()
    for i in range(10):
        with torch.no_grad():
            out1 = alpha_compose(col, alpha)
    torch.cuda.synchronize()
    t_2_2 = time.time()-t1
    print(t_1_1/t_2_2)
    print(out1.shape)
    print(out2.shape)
    print(torch.max(torch.abs(out1-out2)))
    print(torch.min(torch.abs(out1-out2)))
    print(torch.mean(torch.abs(out1-out2)))
