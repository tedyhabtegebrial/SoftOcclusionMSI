import numpy as np
import torch
import random
random.seed(999)
torch.random.manual_seed(999)
np.random.seed(999)



import sys

import os
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.config import argparser
from matryodshka_pl import MODSPL
from data import get_dataset


from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, seed_everything
seed_everything(999)
from pytorch_lightning.loggers import TestTubeLogger

def train(argparser):
    conf = argparser.parse_args()
    mods_network = MODSPL(conf)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(conf.out_path, conf.exp_name),
                        filename='ckpts-{epoch:04d}',
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=5)
    logger = TestTubeLogger(save_dir=conf.out_path,
                            name=conf.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False,
                            )
    trainer = Trainer(max_epochs=conf.num_epoch,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      check_val_every_n_epoch=1,
                      weights_summary='top',
                      progress_bar_refresh_rate=1,
                      gpus=conf.ngpus,
                      num_nodes=conf.nnodes,
                      accelerator='ddp' if conf.ngpus > 1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if conf.ngpus == 1 else None,
                      deterministic=True,
                      terminate_on_nan=True,
                      gradient_clip_val=10)
    if trainer.global_rank == 0:
        print(f'Find logs here: {os.path.join(conf.out_path, conf.exp_name)}')
        if not os.path.exists(os.path.join(conf.out_path, conf.exp_name)):
            os.makedirs(os.path.join(conf.out_path,
                        conf.exp_name), exist_ok=True)
            # os.makedirs(conf.out_path, exist_ok=True)
        with open(os.path.join(conf.out_path, conf.exp_name, 'hparams.json'), 'w') as fid:
            json.dump(obj=conf.__dict__, fp=fid, indent=4, sort_keys=True)
    trainer.fit(mods_network)

if __name__=='__main__':
    train(argparser)
