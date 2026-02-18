import torch
import numpy as np

from wan.modules.vae2_2 import Wan2_2_VAE

import torch
import os
from einops import rearrange

import nibabel as nib
import numpy as np

def main(video_path, save_path):
    model_id = "Wan-AI/Wan2.2-TI2V-5B"
    checkpoint_dir='/home/v-qichen3/debug/Wan2.2-TI2V-5B'
    vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, 'Wan2.2_VAE.pth'))

    os.makedirs(save_path,exist_ok=True)

    video = nib.load(video_path)
    affine = video.affine
    video = video.get_fdata()  
    video = np.clip(video, -1000,1000)
    video = video/1000.0
    # breakpoint()
    video = torch.from_numpy(video)
    video = video.permute(-1,0,1)[:,None]
    video = video.repeat(1,3,1,1)

    video = rearrange(video,'t c h w -> c t h w').float()

    video = video.cuda()
    with torch.no_grad():
        latent = vae.encode([video])
        results = vae.decode(latent)

    results = rearrange(results[0], 'c t h w -> c h w t')
    results = results.mean(0)

    results = torch.clamp(results,-1.0,1.0)

    results = results*1000.0

    results = results.cpu().numpy()
    # breakpoint()
    name = os.path.basename(video_path).split('.')[0]
    data_path = os.path.join(save_path, str(f'{name}_recon.nii.gz'))
    data_nii = nib.Nifti1Image(results.astype(np.int16), affine)
    nib.save(data_nii, data_path)
    

if __name__ == '__main__':
    video_path='/home/v-qichen3/blob/qichen_blob/data/CT/Task07_Pancreas/imagesTr/pancreas_001.nii.gz'
    save_path='.'
    main(video_path, save_path)
