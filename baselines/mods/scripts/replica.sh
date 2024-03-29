python train.py \
    --ngpu=1 \
    --nnodes=1 \
    --scene_number=0 \
    --out_path=/home/habtegebrial/Desktop/Work/Papers/CVPR22/code_release/exp_logs/replica/scene_00 \
    --dataset_path=../../somsi_data/ \
    --exp_name=no_ti_reg \
    --do_sph_wgts \
    --sph_wgt_const=1 \
    --lambda_perceptual=1 \
    --lambda_l2=1 \
    --dataset=replica \
    --height=320 \
    --width=640 \
    --batch_size=1 \
    --num_epoch=200 \
    --num_inputs=12 \
    --lr=0.002 \
    --num_spheres=64 \
    --near=0.60 --far=60 \
