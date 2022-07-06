import os
import json
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger
seed_everything(999)

from options import argparser
from soft_occ_msi_pl import SoftOccMSIPl


def main(argparser):
    configs = argparser.parse_args()
    somsi_model = SoftOccMSIPl(configs)
    checkpoint_callback = \
            ModelCheckpoint(dirpath=os.path.join(configs.logging_path, configs.exp_name),
                    filename='ckpts-{epoch:04d}',
                    monitor='val/psnr',
                    mode='max',
                    save_top_k=5)
    logger = TestTubeLogger(save_dir=configs.logging_path,
                            name=configs.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False,
                            )
    trainer = Trainer(max_epochs=configs.epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      check_val_every_n_epoch=1,
                      weights_summary='top',
                      progress_bar_refresh_rate=1,
                      gpus=configs.ngpus,
                      num_nodes=configs.nnodes,
                      accelerator='ddp' if configs.ngpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if configs.ngpus==1 else None,
                      deterministic= True,
                      terminate_on_nan=True,
                      gradient_clip_val=10)

    if trainer.global_rank==0:
        if not os.path.exists(os.path.join(configs.logging_path, configs.exp_name)):
            os.makedirs(os.path.join(configs.logging_path, configs.exp_name), exist_ok=True)
        with open(os.path.join(configs.logging_path, configs.exp_name, 'hparams.json'), 'w') as fid:
            json.dump(obj=configs.__dict__, fp=fid, indent=4, sort_keys=True)
    trainer.fit(somsi_model)

if __name__ == '__main__':
    main(argparser)
