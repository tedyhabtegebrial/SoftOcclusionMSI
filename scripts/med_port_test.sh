python -m test \
        --ngpus=1 \
        --nnodes=1 \
        --with_pixel_input \
        --dataset=d3dkit \
        --logging_path=exp_logs \
        --dataset_path=somsi_data \
        --num_basis=1 \
        --num_inputs=9 \
        --batch_size=1 \
        --epochs=300 \
        --decay_step 60 --decay_gamma 0.8 \
        --learning_rate=0.001 \
        --num_spheres=64 \
        --near=2.5 --far=1000 \
        --lambda_mse=1 \
        --num_blocks=7 \
        --convs_per_block=4 \
        --max_chans=256 \
        --min_chans=128 \
        --exp_name=feats_3 \
        --height=768 \
        --width=1536 \
        --feats_per_layer=3 \
        --ckpt_path=ckpts/medieval_port_768x1536/medieval_port_feats_3.ckpt
