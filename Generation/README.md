# Generation

This folder contains the latent diffusion training and inference pipeline for CT generation with Foundation VAE.

Released generation assets:
[https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Generation](https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Generation)

## 1. Environment Setup

Create and activate your environment first (example):
```bash
conda create -n foundation_vae_gen python=3.10 -y
conda activate foundation_vae_gen

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --user
pip install torchmetrics==0.7.3
pip install taming-transformers-rom1504 --user
pip install clip kornia --user
pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib --user
pip install PyWavelets --user
```

## 2. Training
2D training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/latent-diffusion/full_ct_2d_with_body_mask.yaml -t --gpus 0,1,2,3, --resume ./logs/full_ct_2d_with_body_mask
```
3D training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/latent-diffusion/full_ct_3d_with_body_mask.yaml -t --gpus 0,1,2,3, --resume ./logs/full_ct_3d_with_body_mask
```

## 3. Inference
```bash
python inference_CT_Gen_3d.py
```

## 4. Notes
- Please make sure dataset paths and checkpoint paths are correctly set in config files before training/inference.
- We recommend using the same GPU setup and CUDA/PyTorch versions for reproducibility.
- Generated CT volumes can be used for downstream classification and segmentation.
