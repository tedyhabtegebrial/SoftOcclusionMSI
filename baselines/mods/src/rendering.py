import torch.nn as nn
from .alpha_compose import AlphaComposition
from .blending import ImageBlending
from .warping import ImageWarping

class Rendering(nn.Module):
    """
    Rendering functions.
    """
    def __init__(self, config):
        super(Rendering, self).__init__()
        self._alpha_comp = AlphaComposition()
        self._img_blend = ImageBlending(config=config)
        self._img_warp = ImageWarping(config=config)

    def alpha_compose(self, src_imgs, alpha_imgs):
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
        return self._alpha_comp(src_imgs, alpha_imgs)

    def image_blending(self, bg_img, fg_imgs, blend_wgts):
        """
        Blending of a single background and multiple foreground images.

        Parameters
        ----------
        bg_img : torch.Tensor[B, 1, F, H, W] or torch.Tensor[B, F, H, W]
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
        return self._img_blend(bg_img, fg_imgs, blend_wgts)

    def get_sphere_ray_intersections(self, src_pose, tgt_pose, radii, transform=None):

        return self._img_warp.get_sphere_ray_intersections(src_pose, tgt_pose, radii, transform=transform)

    def image_warping(self, msi, sph_ray_intersec):

        return self._img_warp.warp_image(msi, sph_ray_intersec)