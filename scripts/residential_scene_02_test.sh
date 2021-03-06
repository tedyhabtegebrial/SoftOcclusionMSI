python -m test \
        --ngpus=1 \
        --nnodes=1 \
        --with_pixel_input \
        --dataset=residential \
        --logging_path=exp_logs \
        --dataset_path=somsi_data \
        --num_basis=1 \
        --batch_size=1 \
        --epochs=300 \
        --decay_step 60 --decay_gamma 0.8 \
        --learning_rate=0.001 \
        --near=0.60 --far=20 \
        --num_blocks=7 \
        --convs_per_block=4 \
        --exp_name=feats_3 \
        --scene_number=2 \
        --height=320 \
        --width=640 \
        --feats_per_layer=24 \
        --ckpt_path=ckpts/residential_area/scene_02/320x640/feats_24.ckpt \
        # --ckpt_path=ckpts/residential/scene_00/320x640/residential_feats_12.ckpt
