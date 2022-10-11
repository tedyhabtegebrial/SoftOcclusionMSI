import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--dataset_path', default='/netscratch/rogge/erp_matryodshka/data/ricoh/residential_house.h5',
                       type=str, help='path for the dataset')
argparser.add_argument('--out_path', default='/netscratch/teddy/cvpr21/experiments/mods/ricoh/unknown_exp/train/', type=str, help='path for the experiment output')
# argparser.add_argument('--test_model', default='/netscratch/teddy/cvpr21/experiments/mods/ricoh/unknown_exp/train/iter100000_matryodshka.pth', type=str, help='path of the model to use for test')
argparser.add_argument('--ngpus', default=1,
                       type=int, help='gpus per node')
argparser.add_argument('--nnodes', default=1,
                       type=int, help='num of nodes')

argparser.add_argument('--mode', default='train', type=str, help='train, test, val')
argparser.add_argument('--dataset', default='ricoh', type=str, help='name of dataset')
argparser.add_argument('--exp_name', default='deleteme', type=str, help='experiment name')
argparser.add_argument('--scene_number', default=0, type=int, help='selection of scene')
argparser.add_argument('--scene_list', default=[], action='append', type=int, help='selection of multiple scenes with ricoh dataset')
argparser.add_argument('--num_inputs', default=8, type=int, help='number of images of a selected scene')
argparser.add_argument('--height', default=256, type=int, help='height of the images')
argparser.add_argument('--width', default=512, type=int, help='width of the images')

argparser.add_argument('--device', default='gpu', type=str, help='use cuda or cpu')
argparser.add_argument('--batch_size', default=1, type=int, help='number of batches')
argparser.add_argument('--num_epoch', default=10, type=int, help='number of epochs')
argparser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
argparser.add_argument('--num_spheres', default=32, type=int, help='number of spheres in the sphere volume')
argparser.add_argument('--near', default=1, type=float, help='minimum depth of the sphere volume')
argparser.add_argument('--far', default=60, type=float, help='maximum depth of the sphere volume')
argparser.add_argument('--do_sph_wgts', action='store_true', help='if set, applies spherical weighting')
argparser.add_argument('--sph_wgt_const', default=1, type=float, help='constant applied to spherical weighting')
argparser.add_argument('--lambda_perceptual', default=1,
                       type=float, help='lambda for perceptual loss')
argparser.add_argument('--lambda_erp_l2', default=1, type=float, help='lambda for l2 loss with spherical weighting (will be ignored if lambda_per is set)')
argparser.add_argument('--lambda_l2', default=1, type=float, help='lambda for l2 loss (will be ignored if lambda_per or lambda_erp_l2 is set)')
argparser.add_argument('--lambda_ti', default=0, type=float, help='lambda for transform inverse')
