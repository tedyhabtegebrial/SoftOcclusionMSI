import torch
import torch.nn as nn
from .spt import Utils as spt

class VCoord(nn.Module):
    """
    V-coords for angular awareness.

    Parameters
    ----------
    batch_size : int
    height : int
    width : int
    dataset : string
    device : string

    Returns
    -------
    v_coords : torch.Tensor[B, H, W, 1]
    """
    def __init__(self, config):
        super(VCoord, self).__init__()
        self.device = config.device
        self.spt_utils = spt(height=config.height, width=config.width, dataset=config.dataset, device=self.device)

    def forward(self, batch_size=1):
        erp = self.spt_utils.get_xy_coords(batch_size=batch_size, device=self.device)
        sph = self.spt_utils.equi_2_spherical(erp)
        _theta, self.phi, _rad = torch.split(sph, split_size_or_sections=1, dim=-1)
        v_coords = torch.sin(self.phi)
        return torch.abs(v_coords)