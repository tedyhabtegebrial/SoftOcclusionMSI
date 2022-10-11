import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import math
import lpips
# from .lpips.lpips import LPIPS


class Loss(nn.Module):
    """
    Class with a selection of losses.
    """
    def __init__(self, config):
        super(Loss, self).__init__()
        self.height = config.height
        self.width = config.width
        self.dataset = config.dataset
        self.device = config.device
        # self.spt_utils = spt(height=self.height, width=self.width, dataset=self.dataset, device=self.device)
        self.lpips_vgg = None

    def _get_spherical_weighting(self, r=1, fix_const=False, const=1, do_sph_wgts=True):
        """
        Returns weights according to the area that each pixel occupies on the sphere.

        Parameters
        ----------
        r : float
            Radius of the sphere.
        fix_const : boolean
            Use a fixed constant instead of calculating the constant.
        const : float
            Constant to be used if 'fix_const'=True.

        Returns
        -------
        area : torch.Tensor[1, 1, H, W]
            Spherical (area) weights.
        """
        if do_sph_wgts:
            phi = torch.linspace(0, math.pi, self.height).view(1, self.height, 1).expand(1, self.height, self.width).to(self.device)
            area = (self.width*self.height) / torch.abs(torch.sin(phi)).sum() * torch.abs(torch.sin(phi))
            return area.view(1, 1, self.height, self.width)
        else:
            return torch.ones([1, 1 , self.height, self.width]).to(self.device)

    def loss_psnr(self, pred_view, tgt_view):
        """
        Calculates the peak signal-to-noise ratio.

        Parameters
        ----------
        pred_view : torch.Tensor[1, 3, H, W]
            Predicted image with values in range [0,1].
        tgt_view : torch.Tensor[1, 3, H, W]
            Groundtruth image with values in range [0,1].

        Returns
        -------
        loss : torch.Tensor[1]
            PSNR between pred_view and tgt_view.
        """
        assert not torch.min(pred_view) < 0 and not torch.min(tgt_view) < 0

        return 20 * torch.log10(1 / torch.sqrt(F.mse_loss(pred_view, tgt_view)))

    def loss_ssim(self, pred_view, tgt_view):
        """
        Calculates the Structural Similarity Index Measure.

        Parameters
        ----------
        pred_view : torch.Tensor[1, 3, H, W]
            Predicted image.
        tgt_view : torch.Tensor[1, 3, H, W]
            Groundtruth image.

        Returns
        -------
        loss : torch.Tensor[1]
            SSIM between pred_view and tgt_view.
        """
        return kornia.losses.ssim(pred_view, tgt_view, max_val=1, window_size=3, reduction="mean")
    
    def loss_l2(self, pred_view, tgt_view):
        return torch.mean((pred_view - tgt_view) **2)

    def loss_erp_l2(self, pred_view, tgt_view, sph_wgts=None, sph_wgt_const=1, do_sph_wgts=True):
        """
        Calculates the L2 loss between two images with spherical weighting.

        Parameters
        ----------
        pred_view : torch.Tensor[B, 3, H, W]
            Predicted image.
        tgt_view : torch.Tensor[B, 3, H, W]
            Groundtruth image.
        sph_wgts : torch.Tensor[B, 1, H, W]
            Spherical weights.

        Returns
        -------
        loss : torch.Tensor[1]
            Erp L2 loss between 'pred_view' and 'tgt_view'.
        """
        batch_size, *_ = pred_view.shape
        sph_wgts = self._get_spherical_weighting(r=1, fix_const=False, const=sph_wgt_const, do_sph_wgts=do_sph_wgts).expand(batch_size, 1, self.height, self.width) if sph_wgts == None else sph_wgts
        return torch.mean((sph_wgts * (pred_view - tgt_view)) **2)

    def loss_perceptual(self, pred_view, tgt_view, sph_wgts=None, sph_wgt_const=1, do_sph_wgts=True):
        """
        Calculates the perceptual LPIPS loss between two images with spherical weighting.

        Parameters
        ----------
        pred_view : torch.Tensor[B, 3, H, W]
            Predicted image in range [-1, 1].
        tgt_view : torch.Tensor[B, 3, H, W]
            Groundtruth image in range [-1, 1].
        sph_wgts : torch.Tensor[B, 1, H, W]
            Spherical weights.

        Returns
        -------
        loss : torch.Tensor[1]
            Erp LPIPS loss between 'pred_view' and 'tgt_view'.
        """
        if self.lpips_vgg is None:
            self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device)
        # batch_size, *_ = pred_view.shape
        # sph_wgts = self._get_spherical_weighting(r=1, fix_const=False, const=sph_wgt_const, do_sph_wgts=do_sph_wgts).expand(batch_size, 1, self.height, self.width) if sph_wgts == None else sph_wgts
        return torch.mean(self.lpips_vgg(pred_view, tgt_view, normalize=False))
