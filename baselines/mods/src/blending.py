import torch.nn as nn

class ImageBlending(nn.Module):
    """
    Blending of background and foreground images.

    Parameters
    ----------
    bg_img : torch.Tensor[B, D, F, H, W] or torch.Tensor[B, 1, F, H, W] or torch.Tensor[B, F, H, W]
        Background image.
    fg_imgs : torch.Tensor[B, D, F, H, W] or torch.Tensor[B, 1, F, H, W] or torch.Tensor[B, F, H, W]
        Foreground image.
    blend_wgts : torch.Tensor[B, D, 1, H, W] or torch.Tensor[B, D, H, W]
        Blending weights.

    Returns
    -------
    blend_img : torch.Tensor[B, D, F, H, W]
        Blended image.
    """
    def __init__(self, config):
        super(ImageBlending, self).__init__()
        self.height = config.height
        self.width = config.width

    def forward(self, bg_img, fg_imgs, blend_wgts):
        bg_img = bg_img.unsqueeze(1) if len(bg_img.shape) == 4 else bg_img
        fg_imgs = fg_imgs.unsqueeze(1) if len(fg_imgs.shape) == 4 else fg_imgs
        blend_wgts = blend_wgts.unsqueeze(2) if len(blend_wgts.shape) == 4 else blend_wgts

        blend_img = bg_img * blend_wgts + fg_imgs * (1 - blend_wgts)
        return blend_img