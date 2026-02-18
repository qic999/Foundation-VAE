pip install beartype PyWavelets


CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/full_ct_3d_with_body_mask2.yaml -t --gpus 0, --resume ./logs/full_ct_3d_with_body_mask2

CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/full_ct_3d_with_body_mask_finetune.yaml -t --gpus 0, --resume ./logs/full_ct_3d_with_body_mask_finetune2

