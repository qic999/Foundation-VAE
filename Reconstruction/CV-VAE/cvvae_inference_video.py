from models.modeling_vae import CVVAEModel

import torch
import os
from einops import rearrange

from fire import Fire
import nibabel as nib
import numpy as np

def main(video_path, save_path):
    vae3d_path = 'AILab-CVC/CV-VAE'

    vae3d = CVVAEModel.from_pretrained(vae3d_path,subfolder="vae3d",torch_dtype=torch.float16)
    vae3d.requires_grad_(False)
    vae3d = vae3d.cuda()
    os.makedirs(save_path,exist_ok=True)

    video = nib.load(video_path)
    affine = video.affine
    video = video.get_fdata()  
    video = np.clip(video, -175,250)
    video = (video+175.0)/425.0
    # breakpoint()
    video = torch.from_numpy(video)
    video = video.permute(-1,0,1)[:,None]
    video = video.repeat(1,3,1,1)

    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).half()

    video = video * 2.0 - 1.0

    video = video.cuda()

    print(f'Shape of input video: {video.shape}')
    latent = vae3d.encode(video).latent_dist.sample()

    print(f'Shape of video latent: {latent.shape}') # 4 8 8

    results = vae3d.decode(latent).sample

    results = rearrange(results.squeeze(0), 'c t h w -> c h w t')
    results = results.mean(0)


    results = (torch.clamp(results,-1.0,1.0) + 1.0) / 2.0

    results = results*425.0 - 175.0

    results = results.cpu().numpy()
    breakpoint()
    name = os.path.basename(video_path).split('.')[0]
    data_path = os.path.join(save_path, str(f'{name}_recon.nii.gz'))
    data_nii = nib.Nifti1Image(results.astype(np.int16), affine)
    nib.save(data_nii, data_path)
    

if __name__ == '__main__':
    Fire(main)
