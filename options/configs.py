import argparse

# Data related settings
argparser = argparse.ArgumentParser()
argparser.add_argument('--slurm', action='store_true', help='pass when using slurm')
argparser.add_argument('--dataset', default='replica', help='choose dataset from [replica, ricoh]')
argparser.add_argument('--dataset_path', default='/netscratch/teddy/spherical_lf_tar_files/', type=str)
argparser.add_argument('--mode', default='train', type=str)
argparser.add_argument('--exp_name', default='experiment_v1', type=str)
argparser.add_argument('--logging_path', default='/netscratch/teddy/LayeredMSI_expts/', type=str)

# Dataset settings
argparser.add_argument('--num_inputs', default=8, type=int)
argparser.add_argument('--scene_number', default=0, type=int,
                        help='scene number for datasets with multiple scenes')

argparser.add_argument('--local_rank', type=int, default=1)
argparser.add_argument('--nnodes', type=int, default=1)
argparser.add_argument('--ngpus', type=int, default=1)

# Training related settings

argparser.add_argument('--batch_size', default=4, type=int)
argparser.add_argument('--split', default='train', type=str)
argparser.add_argument('--epochs', default=20, type=int)
argparser.add_argument('--input_channels', default=3, type=int, help='number of input channels')
argparser.add_argument('--network_width', default=32, type=int, help='this controls the number of channels in the base encoder')

argparser.add_argument('--height', type=int, default=256)
argparser.add_argument('--width', type=int, default=512)

# Network related settings
argparser.add_argument('--num_spheres', default=64, type=int, help='number of spheres to represent the scene')
argparser.add_argument('--far', default=40, type=float,
                       help='near sphere radius | 40 for replica')
argparser.add_argument('--near', default=0.6, type=float, help='far sphere radius| .6 for replica')
argparser.add_argument('--num_layers', default=3, type=int, help='number occlusion layers')
argparser.add_argument('--feats_per_layer', default=3, type=int, help='number features per layer in the scene appearance')
argparser.add_argument('--lambda_mse', type=float, default=1,
                       help='weight mse reconstruction loss')

# Positional Encoding details
argparser.add_argument('--num_octaves', default=4, type=int, help='number of octaves in positional encoding')
argparser.add_argument('--add_noise', action='store_true', help='if set to true, positional encoding will get x, y and noise')
argparser.add_argument('--with_pixel_input', action='store_true',
                       help='if set to true, positional encoding and rgb will be used as input')


argparser.add_argument('--no_scheduler', action='store_true', help='set to turn off scheduler')
argparser.add_argument('--use_sgd', action='store_true',
                       help='use sgd')
argparser.add_argument('--learning_rate', default=0.0004, type=float,
                       help='generator learning rate')
argparser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                    help='scheduler decay step')
argparser.add_argument('--decay_gamma', type=float, default=0.1,
                    help='learning rate decay amount')
# 
argparser.add_argument('--num_blocks', type=int, default=7)
argparser.add_argument('--convs_per_block', type=int, default=4,
                       help='convs per block')
argparser.add_argument('--max_chans', default=256,
                       type=int, help='max number of chans')
argparser.add_argument('--min_chans', default=128, type=int,
                       help='max number of chans, the final value for this will be min_chans*(num_spheres//32)')

# View Dependent Effects
argparser.add_argument('--basis_layers', type=int, default=8,
                    help='number of layers in basis functions')
argparser.add_argument('--num_basis', type=int, default=1,
                    help='number basis functions, if 1, not reflectance coefs will be used')

# Pre-trained model
argparser.add_argument('--ckpt_path', type=str, default='',
                       help='pre-trained ckpt')


if __name__=='__main__':
    args = argparser.parse_args()
    for k in sorted(args.__dict__.keys()):
        print(k, args.__dict__[k])
