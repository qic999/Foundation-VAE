# Reconstruction

This module provides CT reconstruction inference with multiple VAE backbones.

Released reconstruction results:
[https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Reconstruction](https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Reconstruction)

## 1. Supported Methods

- `CV-VAE`
- `iv-vae`
- `LeanVAE`
- `VideoVAEplus`
- `Wan`
- `WF-VAE`

## 2. Data

Please prepare your CT volumes (e.g., `.nii.gz`) and place them in your local input folder.
You can also directly use our released reconstruction assets from Hugging Face.

## 3. Environment

Different methods may require different dependencies.
Please create a dedicated conda environment for each method and install dependencies from each method folder.

Example:

```bash
conda create -n wan python=3.10 -y
conda activate wan
cd Reconstruction/WAN
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# if provided
pip install -r requirements.txt
```
Repeat similarly for iv-vae, LeanVAE, VideoVAEplus, Wan, and WF-VAE.

## 4. Inference by Method
Run inference inside each method directory.

### 4.1 CV-VAE
```bash
conda activate cvvae
cd Reconstruction/CV-VAE
python ct_recon_window.py
```

### 4.2 iv-vae
```bash
conda activate ivvae
cd Reconstruction/iv-vae
python recon_ct_window.py
```

### 4.3 LeanVAE
```bash
conda activate leanvae
cd Reconstruction/LeanVAE
python recon_ct_window.py
# or
python recon_ct_window_175_250.py
```

### 4.4 VideoVAEplus
```bash
conda activate videovaeplus
cd Reconstruction/VideoVAEplus
python recon_ct_window.py
```

### 4.5 Wan
```bash
conda activate wan
cd Reconstruction/Wan
python recon_ct_window_wan2.1.py
python recon_ct_window_wan2.2.py
```

### 4.6 WF-VAE
```bash
conda activate wfvae
cd Reconstruction/WF-VAE
python scripts/ct_recon_window.py
```

