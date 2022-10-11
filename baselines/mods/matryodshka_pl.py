import copy
from itertools import chain
import torch

import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import kornia


from src.matryodshka import MatryODSHKhaNet
from src.loss import Loss
from src.sphere_sweeper import SphereSweeper, MSIRenderer
from data import get_dataset

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MODSPL(pl.LightningModule):
    def __init__(self, configs):
        super(MODSPL, self).__init__()
        self.configs = configs
        self.mods_net = MatryODSHKhaNet(configs)
        # metrics
        self.ssim_metric = kornia.losses.SSIM(3, max_val=255.0)
        self.sphere_sweeper = SphereSweeper(configs)
        self.msi_renderer = MSIRenderer(configs)
        self.loss_util = Loss(configs)
    def run_mods(self, batch):
        b, c, h, w = batch['ref_img'].shape
        ref_ssv = self.sphere_sweeper(batch['ref_img'], batch['ref_pose'])
        inp_ssv = self.sphere_sweeper(batch['inp_img'], batch['inp_pose'])
        ssv = torch.cat([ref_ssv, inp_ssv], dim=1).view(b, -1, h, w)
        alpha, beta = self.mods_net(ssv)
        alpha = alpha.unsqueeze(2)
        color = beta.unsqueeze(2)*ref_ssv + (1-beta.unsqueeze(2))*inp_ssv
        novel_view = self.msi_renderer(alpha, color, batch['trg_pose'])
        return novel_view, alpha, color

    def adapt_pose_to_target(self, pose_inp, new_wrld_inp):
        # give pose_inp: from cam to world, new_wrld_inp: from ne_wrld to world
        # return from first camera to the second
        device = pose_inp.device
        b_size = pose_inp.shape[0]
        ones = torch.FloatTensor([0, 0, 0, 1]).view(1, 1, 4).repeat(b_size, 1, 1).to(device)
        pose = torch.cat([pose_inp[:, :3, :4], ones], dim=1)
        new_wrld = torch.cat([new_wrld_inp[:, :3, :4], ones], dim=1)
        # print(new_wrld.shape)
        pose_new = torch.bmm(torch.inverse(new_wrld), pose)
        return pose_new[:, :3, :]
    def forward(self, batch):
        novel_view, alpha, color = self.run_mods(batch)
        return novel_view

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        configs = self.configs
        # configs = self.configs
        novel_view, alpha, color = self.run_mods(batch)
        loss_p = configs.lambda_perceptual*self.loss_util.loss_perceptual(pred_view=novel_view, tgt_view=batch['trg_img'],
                                                                          sph_wgt_const=configs.sph_wgt_const, do_sph_wgts=configs.do_sph_wgts)
        loss_l2 = configs.lambda_l2*self.loss_util.loss_erp_l2(pred_view=novel_view, tgt_view=batch['trg_img'],
                                                               sph_wgt_const=configs.sph_wgt_const, do_sph_wgts=configs.do_sph_wgts)

        loss = loss_l2 + loss_p
        self.log('tr/l2', loss_l2, prog_bar=True)
        self.log('tr/p', loss_p, prog_bar=True)
        self.log('tr/p+l2', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        nv_col, _a, _c = self.run_mods(batch)
        psnr = self.psnr(nv_col, batch['trg_img'])
        ssim = self.ssim(nv_col, batch['trg_img'])
        if batch_idx == 0:
            self.logger.experiment.add_images(
                'val/0_fake_view', self.to_image(nv_col), self.global_step)
            self.logger.experiment.add_images(
                'val/1_real_view', self.to_image(batch['trg_img']), self.global_step)
        log = {'val/mse': F.mse_loss(nv_col, batch['trg_img']),
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
        configs.__dict__['mode'] = 'test'
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

    def configure_optimizers(self):
        # epochs = self.configs.epochs
        # self.optimizer = torch.optim.Adam(
        #     chain(self.scene_net.parameters(),
        #             self.basis_fun.parameters(),
        #             self.decode_features.parameters()),
        #     lr=self.configs.lr)
        self.optimizer = torch.optim.Adam(self.mods_net.parameters(), lr=self.configs.lr)
        return self.optimizer

    def to_image(self, input):
        return (1+input.clamp(min=-1, max=1)) / 2.0

    def psnr(self, pred, target):
        # def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
        #    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

        with torch.no_grad():
            pred, target = self.to_image(pred), self.to_image(target)
            pred = pred.clamp(min=0, max=1)
            target = target.clamp(min=0, max=1)
            psnr = -10*torch.log10(F.mse_loss(pred, target))
            # psnr = kornia.losses.psnr_loss(pred, target, max_val=255.0)
        return psnr

    def ssim(self, pred, target):
        with torch.no_grad():
            pred, target = self.to_image(pred), self.to_image(target)
            # pred, target = pred.mul(255), target.mul(255)
            ssim = 1-2 * \
                kornia.losses.ssim(pred, target, window_size=3,
                                   max_val=1.0, reduction='mean')
        return ssim
