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

    def forward(self, src_imgs, alpha_imgs):
        alpha_imgs = alpha_imgs.clamp(min=0.0001)
        ones = torch.ones_like(alpha_imgs[:, 0, ...]).unsqueeze(1)
        weight_1 = torch.cumprod(
            torch.cat([ones, 1-alpha_imgs[:, :-1, :, :, :]], dim=1), dim=1)
        weighted_imgs = weight_1*alpha_imgs*src_imgs
        comb = torch.sum(weighted_imgs, dim=1)
        return comb


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
