export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
REAL_DATASET_DIR=/storage/dataset/vae_eval/testvideo
EXP_NAME=release-test
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=288
CKPT=/storage/lcm/WF-VAE/results/Middle888
SUBSET_SIZE=0

accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    scripts/recon_video.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --num_workers 8 \
    --from_pretrained ${CKPT} \
    --model_name WFVAE \
    # --output_origin \
    # --crop_size ${RESOLUTION}
