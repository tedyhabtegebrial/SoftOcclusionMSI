import torch
import torch.nn as nn

class AlphaComposition(nn.Module):
    """
    Alpha composition of sphere volume.

    Parameters
    ----------
    src_imgs : torch.Tensor[B, D, F, H, W]
        Source RGB images.
    alpha_imgs : torch.Tensor[B, D, 1, H, W]
        Alpha images.

    Returns
    -------
    comp_rgb : torch.Tensor[B, F, H, W]
        Alpha composited image.
    """
    def __init__(self):
        super(AlphaComposition, self).__init__()

    def forward(self, src_imgs, alpha_imgs):
        alpha_imgs = alpha_imgs.unsqueeze(2) if len(alpha_imgs.shape) == 4 else alpha_imgs
        _, num_d, *_ = src_imgs.shape

        src_imgs = torch.split(src_imgs, split_size_or_sections=1, dim=1)
        alpha_imgs = torch.split(alpha_imgs, split_size_or_sections=1, dim=1)
        comp_rgb = src_imgs[-1] #* alpha_imgs[-1] #with alpha has holes
        for d in reversed(range(num_d - 1)):
            comp_rgb = src_imgs[d] * alpha_imgs[d] + \
                (1.0 - alpha_imgs[d]) * comp_rgb
        return comp_rgb.squeeze(1)