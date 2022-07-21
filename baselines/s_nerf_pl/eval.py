import os
import cv2
import time

from collections import defaultdict
from tqdm import tqdm
import imageio
import json
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
from opt import get_opts
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True

@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh
    root_dir = os.path.join(args.logging_dir, args.exp_name)
    dataset = dataset_dict[args.dataset_name]('test', args)
    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    ckpt_file = os.path.abspath(args.ckpt_path)
    load_ckpt(nerf_coarse, ckpt_file, model_name='nerf_coarse')
    nerf_coarse.cuda().eval()

    models = {'coarse': nerf_coarse}
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    if args.N_importance > 0:
        nerf_fine = NeRF()
        load_ckpt(nerf_fine, ckpt_file, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models['fine'] = nerf_fine

    imgs, depth_maps, psnrs, ssims = [], [], [], []
    dir_name = os.path.join(root_dir, 'evaluations')
    os.makedirs(dir_name, exist_ok=True)
    imgs_folder = os.path.join(dir_name, 'evaluations')
    os.makedirs(imgs_folder, exist_ok=True)
    t_acc = 0
    counter = 0 
    for i in tqdm(range(len(dataset))):
        torch.cuda.synchronize()
        t_begin = time.time()
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        t_diff = time.time() - t_begin
        t_acc += t_diff
        counter += 1
        img_pred = results[f'rgb_{typ}'].view(h, w, 3)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt.permute(2, 0, 1).unsqueeze(0), img_pred.permute(2, 0, 1).unsqueeze(0)).item()]
            ssims += [metrics.ssim(img_gt.permute(2, 0, 1).unsqueeze(0),
                                   img_pred.permute(2, 0, 1).unsqueeze(0)).item()]
        img_pred = np.clip(img_pred.cpu().numpy(), 0, 1)
        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(imgs_folder, f'{i:03d}.png'), img_pred_)

    imageio.mimsave(os.path.join(
        imgs_folder, f'novel_views.gif'), imgs, fps=30)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)
        print(f'Mean PSNR : {mean_psnr:.2f}')
        print(f'Mean SSIM : {mean_ssim:.3f}')
        print(f'TIME ==== : {(t_acc / counter):.3f}')
        with open(os.path.join(root_dir, 'summary.json'),'w') as fid:
            json.dump({'ssim': mean_ssim, 'psnr': mean_psnr}, fid, indent=4)
