python -m train \
        --dataset_name d3dkit \
        --num_input 9 \
        --use_disp \
        --N_samples 64 --N_importance 64 \
        --img_wh 640 320 --noise_std 0 \
        --num_epochs 1000 --batch_size 4096 \
        --optimizer adam --lr 5e-4 \
        --lr_scheduler linear \
        --scene_number 0 \
        --logging_dir=exp_logs/snerf/medieval_port \
        --root_dir=../../somsi_data \
        --exp_name=nerf
