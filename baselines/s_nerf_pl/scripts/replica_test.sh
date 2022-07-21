python -m eval \
        --num_gpus 1 \
        --dataset_name replica \
        --num_input 12 \
        --N_samples 64 --N_importance 64 \
        --img_wh 640 320 --noise_std 0 \
        --num_epochs 1000 --batch_size 4096 \
        --optimizer adam --lr 5e-4 \
        --lr_scheduler linear \
        --scene_number 0 \
        --logging_dir=exp_logs/snerf/replica/scene_00 \
        --exp_name=nerf \
        --use_disp \
        --root_dir=../../somsi_data \
        --ckpt_path=../../ckpts/nerf/replica/model_scene_00.ckpt \

