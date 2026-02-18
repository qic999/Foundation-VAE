unset https_proxy
export WANDB_PROJECT=WFVAE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXP_NAME=WFVAE

torchrun \
    --nnodes=1 --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=12135 \
    train_ddp.py \
    --exp_name ${EXP_NAME} \
    --video_path /storage/lcm/WF-VAE/video_gen \
    --eval_video_path /storage/dataset/vae_eval/OpenMMLab___Kinetics-400/raw/Kinetics-400/videos_val/ \
    --model_name WFVAE \
    --model_config examples/wfvae-large-16chn.json \
    --resolution 256 \
    --num_frames 25 \
    --batch_size 1 \
    --lr 0.00001 \
    --epochs 4 \
    --disc_start 0 \
    --save_ckpt_step 5000 \
    --eval_steps 1000 \
    --eval_batch_size 1 \
    --eval_num_frames 33 \
    --eval_sample_rate 1 \
    --eval_subset_size 1000 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --wavelet_loss \
    --wavelet_weight 0.1