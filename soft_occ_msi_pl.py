import os
import json
import math
import copy
import torch
from itertools import chain
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia
from torch.utils.data import DataLoader
from data import get_dataset
from src.models import SOMSIRenderer
from src.models import basis
from src.models.networks import SOMSINetwork
from src.models.networks import DecodeFeatures
from src.models.basis import BasisFunction
from src.utils import Utils


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class SoftOccMSIPl(pl.LightningModule):
    def __init__(self, configs):
        super(SoftOccMSIPl, self).__init__()
        self.configs = configs
        self.scene_net = SOMSINetwork(configs)
        self.renderer = SOMSIRenderer(configs)
        self.decode_features = DecodeFeatures(configs)
        self.sphere_radii = None
        self.ssim_metric = kornia.losses.SSIM(3, max_val=255.0)
        self.basis_fun = BasisFunction(configs)
        self.spt_utils = Utils(configs)

    def _get_sphere_radii(self, b_size):
        if self.sphere_radii is None:
            sphere_radii = 1/torch.linspace(1/self.configs.near, 1/self.configs.far, self.configs.num_spheres)
            sphere_radii = sphere_radii.view(1, self.configs.num_spheres).to(self.device)
        else:
            sphere_radii = self.sphere_radii.clone()
        sphere_radii = sphere_radii.expand(b_size, self.configs.num_spheres)
        return sphere_radii

    def get_spherical_weight(self, input):
        b, c, h, w = input.shape
        device = input.device
        if (self.sph_weight is None) or (input.shape[0]!=b):
            y_coords = torch.linspace(0.5, h-0.5, h).view(1, 1, -1, 1).to(device)
            phi = math.pi*(y_coords/(h))
            sin_phi = torch.sin(phi)
            self.sph_weight = sin_phi.expand(
                b, 1, h, w)
        return self.sph_weight.clone()

    def get_nv_feats(self, batch, return_intermediate=False):
        ref_img = batch['ref_rgb']
        b_size, _c, h, w = ref_img.shape
        num_s = self.configs.num_spheres
        num_l = self.configs.num_layers
        scene_data = self.scene_net(ref_img)

        alpha_1 = scene_data['alpha']
        coeff = scene_data['coefs']
        # call the basis function
        r_mats = batch['pose'][:, :3, :3]
        t_vecs = (batch['pose'][:, :3, 3]).view(b_size, 3, 1)
        unit_rays = self.spt_utils._get_unit_rays_on_sphere(alpha_1.device)[:b_size, ...]
        unit_rays_ref = torch.matmul(r_mats.view(b_size, 1, 1, 3, 3),
                                    unit_rays.unsqueeze(-1)).squeeze(-1)
        unit_rays_ref = unit_rays_ref.permute(0, 3, 1, 2)
        # If num of basis is 1, basis function return a tensor of ones
        # Which basically by passes the reflectance modelling
        basis_fun = self.basis_fun(unit_rays_ref)
        coeff = coeff.view(b_size, num_l, self.configs.num_basis,
                            self.configs.feats_per_layer, h, w)
        basis_fun = basis_fun.view(b_size, num_l, self.configs.num_basis, 1, h, w)
        assoc = scene_data['assoc'].view(b_size, num_s, num_l, h, w)
        feats_1 = torch.sum(coeff * basis_fun, 2, keepdim=False)
        alpha_1 = alpha_1.unsqueeze(2).contiguous()
        _b, _, c, h, w = feats_1.shape
        feats = feats_1.expand(b_size, num_l, c, h, w).contiguous()
        alpha = alpha_1.expand(b_size, num_s, 1, h, w).contiguous()
        
        result = self.renderer(alpha, feats, r_mats,
                            t_vecs, self._get_sphere_radii(b_size), assoc)
        nv_col = self.decode_features(result['nv_feats'])
        result['nv_col'] = nv_col
        # ref_view_feats = feats[:, 0, :, :, :]
        # result['ref_view_col'] = self.decode_features(ref_view_feats)
        return result

    def forward(self, batch):
        return self.get_nv_feats(batch)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        lambda_mse = self.configs.lambda_mse
        # lambda_ssim = self.configs.lambda_ssim
        result = self.get_nv_feats(batch)
        nv_col = result['nv_col']
        loss = {}
        loss['mse_loss'] = lambda_mse * F.mse_loss(nv_col, batch['color'])
        # loss['ssim_loss'] = lambda_ssim * kornia.losses.ssim(nv_col, batch['color'], window_size=3)
        self.log('train/mse', loss['mse_loss'], prog_bar=True)
        # self.log('train/ssim', loss['ssim_loss'], prog_bar=True)
        with torch.no_grad():
            if self.global_step%100==0:
                psnr = self.psnr(nv_col, batch['color'])
                self.log('train/psnr', psnr, prog_bar=True)
        self.log('lr', get_learning_rate(self.optimizer), prog_bar=True)
        return sum([v for v in loss.values()])

    def validation_step(self, batch, batch_idx):
        result = self.get_nv_feats(batch)
        nv_col = result['nv_col']
        loss = F.mse_loss(nv_col, batch['color'])
        psnr = self.psnr(nv_col, batch['color'])
        ssim = self.ssim(nv_col, batch['color'])
        if batch_idx==0:
            self.logger.experiment.add_images('val/0_fake_view', self.to_image(nv_col), self.global_step)
            self.logger.experiment.add_images('val/1_real_view', self.to_image(batch['color']), self.global_step)
        log  = {'val/mse': loss,
                'val/ssim': ssim,
                'val/psnr': psnr}
        return log

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        avg_outs = {}
        for k in keys:
            avg_outs[k] = torch.stack([x[k] for x in outputs]).mean()
            self.logger.experiment.add_scalar(k, avg_outs[k], self.global_step)
            self.log(k, avg_outs[k], prog_bar=True)
        return avg_outs

    def test_step(self, batch, batch_idx):
        res_dict = self.forward(batch)
        nv_col = res_dict['nv_col']
        psnr = self.psnr(nv_col, batch['color'])
        ssim = self.ssim(nv_col, batch['color'])
        test_metrics = {}
        test_metrics['ssim'] = ssim.view(-1, 1)
        test_metrics['psnr'] = psnr.view(-1, 1)
        test_images = {'pred': nv_col, 'real':batch['color']}
        return {'metrics':test_metrics, 'images':test_images}

    def test_epoch_end(self, test_results):
        images = [res['images'] for res in test_results]
        metrics = [res['metrics'] for res in test_results]
        pred, real = [], []
        ssim, psnr = [], []
        for im,mt in zip(images, metrics):
            pred.append(im['pred'])
            real.append(im['real'])
            ssim.append(mt['ssim'])
            psnr.append(mt['psnr'])
        pred, real = list(map(lambda x: torch.cat(x), [pred, real]))
        ssim, psnr = list(map(lambda x: torch.cat(x), [ssim, psnr]))
        self.test_results = {'ssim':ssim, 'psnr':psnr, 'real':real, 'fake': pred}

    def train_dataloader(self):
        configs = copy.deepcopy(self.configs)
        configs.__dict__['mode'] = 'train'
        train_set = get_dataset(configs)
        train_loader = DataLoader(train_set,
                batch_size=self.configs.batch_size,
                num_workers=self.configs.batch_size,
                shuffle=True,
                drop_last=False)
        return train_loader

    def val_dataloader(self):
        configs = copy.deepcopy(self.configs)
        configs.__dict__['mode'] = 'val'
        val_set = get_dataset(configs)
        val_loader = DataLoader(val_set,
                batch_size=1,
                num_workers=1,
                pin_memory=True)
        return val_loader        
    
    def test_dataloader(self):
        configs = copy.deepcopy(self.configs)
        configs.__dict__['mode'] = 'test'
        val_set = get_dataset(configs)
        val_loader = DataLoader(val_set,
                    shuffle=False,
                    batch_size=1,
                    num_workers=configs.batch_size,
                    pin_memory=True)
        return val_loader

    def demo_dataloader(self):
        configs = copy.deepcopy(self.configs)
        configs.__dict__['mode'] = 'demo'
        val_set = get_dataset(configs)
        val_loader = DataLoader(val_set,
                batch_size=1,
                num_workers=1,
                pin_memory=True)
        return val_loader        

    def configure_optimizers(self):
        if self.configs.use_sgd:
            self.optimizer = torch.optim.SGD(chain(self.scene_net.parameters(), 
                                self.basis_fun.parameters(), self.decode_features.parameters()), lr=self.configs.learning_rate)
        else:
            print('chaning optimizers')
            self.optimizer = torch.optim.Adam(
                chain(self.scene_net.parameters(),
                        self.basis_fun.parameters(), 
                        self.decode_features.parameters()), 
                        lr=self.configs.learning_rate)
        if self.configs.no_scheduler:
            return self.optimizer
        else:
            #step, gamma = [2, 10, 20, 40], 0.5
            step, gamma = self.configs.decay_step, self.configs.decay_gamma
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=step, gamma=gamma)
            return [self.optimizer], [scheduler]
    
    def to_image(self, input):
        return (1+input.clamp(min=-1, max=1)) / 2.0
 
    def psnr(self, pred, target):
        # def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
        #    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

        with torch.no_grad():
            pred, target  = self.to_image(pred), self.to_image(target)
            pred = pred.clamp(min=0, max=1)
            target = target.clamp(min=0, max=1)
            psnr = -10*torch.log10(F.mse_loss(pred, target))
            # psnr = kornia.losses.psnr_loss(pred, target, max_val=255.0)
        return psnr
    
    def ssim(self, pred, target):
        with torch.no_grad():
            pred, target  = self.to_image(pred), self.to_image(target)
            # pred, target = pred.mul(255), target.mul(255)
            ssim = 1-2*kornia.losses.ssim(pred, target, window_size=3, max_val=1.0, reduction='mean')
        return ssim