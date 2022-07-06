import os
import json

import torch
import numpy as np
import cv2 as cv

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger
seed_everything(999)

from soft_occ_msi_pl import SoftOccMSIPl
from options import argparser

def main(argparser):
    configs = argparser.parse_args()
    somsi_model = SoftOccMSIPl(configs)
    pre_trained_dict = torch.load(configs.ckpt_path, map_location='cpu')
    somsi_model.load_state_dict(pre_trained_dict['state_dict'], strict=True)
    logger = TestTubeLogger(save_dir=configs.logging_path,
                            name=configs.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False,
                            )
    trainer = Trainer(max_epochs=configs.epochs,
                      logger=logger,
                      check_val_every_n_epoch=1,
                      weights_summary='top',
                      progress_bar_refresh_rate=1,
                      gpus=configs.ngpus,
                      num_nodes=configs.nnodes,
                      accelerator='ddp' if configs.ngpus > 1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if configs.ngpus == 1 else None,
                      deterministic=True,
                      terminate_on_nan=True,
                      gradient_clip_val=10)

    if not os.path.exists(os.path.join(configs.logging_path, configs.exp_name)):
        os.makedirs(os.path.join(configs.logging_path, configs.exp_name), exist_ok=True)
    trainer.test(somsi_model)
    test_results = somsi_model.test_results
    # Save Test Results
    ## SSIM and PSNR
    ssim = test_results['ssim']
    psnr = test_results['psnr']
    ssim_float = [round(s.item(), 5) for s in ssim.view(-1)]
    psnr_float = [round(p.item(), 5) for p in psnr.view(-1)]
    ssim_avg = torch.mean(ssim).item()
    psnr_avg = torch.mean(psnr).item()
    test_dir = os.path.join(configs.logging_path, configs.exp_name, 'test_dir')
    os.makedirs(test_dir, exist_ok=True)
    test_summary_file = os.path.join(test_dir, 'test_summary.json')
    if trainer.global_rank == 0:
        with open(test_summary_file, 'w') as fid:
            json.dump(obj={'ssim': ssim_float, 'psnr': psnr_float,
                            'mean_ssim': round(ssim_avg, 5), 'mean_psnr': round(psnr_avg, 5)},
                            fp=fid, indent=4, sort_keys=True)
    ## Prediction and ground truth images
    for itr, (pred, targ) in enumerate(zip(test_results['fake'], test_results['real'])):
        pred, targ = map(lambda x: x.add(1.0).div(2.0).mul(255.0).cpu(), (pred, targ))
        pred, targ = map(lambda x: x.permute(1, 2, 0)[..., [2,1,0]], (pred, targ))
        pred, targ = map(lambda x: x.numpy().astype(np.uint8), (pred, targ))
        if trainer.global_rank == 0:
            cv.imwrite(test_dir+f'/{str(itr).zfill(4)}_prediction.png', pred)
            cv.imwrite(test_dir+f'/{str(itr).zfill(4)}_ground_truth.png', targ)
    if trainer.global_rank == 0:
        print(f'Please find results in {test_dir}')
        print("Summary: \n")
        os.system(f"cat {test_summary_file}")
        print("\n")
if __name__ == '__main__':
    main(argparser)
