from models.vae3d import IV_VAE
from decord import VideoReader, cpu
import torch
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
from fire import Fire
import nibabel as nib
import numpy as np

@torch.no_grad()
def main(video_path, save_path, z_dim=16, dim=96):
    # {ivvae_z4_dim64, ivvae_z8_dim64, ivvae_z16_dim64, ivvae_z16_dim96} 
    # ivvae_z4_dim64,
    # ivvae_z8_dim64,
    # ivvae_z16_dim64, 4 8 8
    # ivvae_z16_dim96,
    vae3d = IV_VAE(z_dim, dim).to(torch.bfloat16)
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

    # video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).half()
    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).to(torch.bfloat16)
    # video=video[:,:,:65,:,:]
    video = video * 2.0 - 1.0

    video = video.cuda()

    print(f'Shape of input video: {video.shape}')
    latent = vae3d.encode(video) 

    print(f'Shape of video latent: {latent.shape}') # 4 8 8
    
    results = vae3d.decode(latent)

    results = rearrange(results.squeeze(0), 'c t h w -> c h w t')
    results = results.mean(0)


    results = (torch.clamp(results,-1.0,1.0) + 1.0) / 2.0

    results = results*425.0 - 175.0

    results = results.to('cpu', dtype=torch.int16).numpy()
    print(f'Shape of video latent: {results.shape}')
    # breakpoint()
    name = os.path.basename(video_path).split('.')[0]
    data_path = os.path.join(save_path, str(f'{name}_recon.nii.gz'))
    data_nii = nib.Nifti1Image(results.astype(np.int16), affine)
    nib.save(data_nii, data_path)

if __name__ == '__main__':
    Fire(main)