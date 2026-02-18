export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
REAL_DATASET_DIR=/storage/lcm/WF-VAE/video_gen
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
CKPT=results/WF-VAE-L-16Chn
SUBSET_SIZE=1000

accelerate launch \
    --config_file examples/accelerate_configs/debug.yaml \
    scripts/vae_var_test.py \
    --batch_size 8 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --crop_size ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --num_workers 8 \
    --from_pretrained ${CKPT} \
    --model_name WFVAE