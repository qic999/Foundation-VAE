import os
import argparse
import torch
from LeanVAE import LeanVAE
from decord import VideoReader, cpu
import torch
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
import tqdm
import numpy as np
import torch.nn.functional as F
import nibabel as nib


def main(args, model, video_path, save_path):
    use_half = args.fp16
    device = args.device

    if args.tile_inference:
        model.set_tile_inference(True)
        model.chunksize_enc = args.chunksize_enc if args.chunksize_enc else 5
        model.chunksize_dec = args.chunksize_dec if args.chunksize_dec else 5

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
    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).float()
    video=video[:,:,:65,:,:]
    regular_size = 2.0
    video = (video * 2.0 - 1.0) / regular_size
    
    with torch.no_grad():
        
        video = video.to(device)

        x, x_rec=  model.inference(video)

    results = rearrange(x_rec.squeeze(0), 'c t h w -> c h w t')
    # breakpoint()
    results = results.mean(0)


    results = (torch.clamp(results*regular_size,-1.0,1.0) + 1.0) / 2.0

    results = results*425.0 - 175.0

    results = results.to('cpu', dtype=torch.int16).numpy()
    print(f'Shape of video latent: {results.shape}')
    # breakpoint()
    name = os.path.basename(video_path).split('.')[0]
    data_path = os.path.join(save_path, str(f'{name}_recon.nii.gz'))
    data_nii = nib.Nifti1Image(results.astype(np.int16), affine)
    nib.save(data_nii, data_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--input_video', type=str, default='./input_videos')
    parser.add_argument('--reconstruct_video', type=str, default='./reconstruct_videos')
    parser.add_argument('--sequence_length', type=int, default=17)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--tile_inference', action='store_true')
    parser.add_argument('--chunksize_enc', type=int, default=None)
    parser.add_argument('--chunksize_dec', type=int, default=None)


    args = parser.parse_args()
    vae  = LeanVAE.load_from_checkpoint(args.ckpt_path, strict=False)
    os.makedirs(args.reconstruct_video ,exist_ok=True)
    vae = vae.half().to(args.device) if args.fp16 else vae.to(args.device)

    main(args, vae, args.input_video, args.reconstruct_video)
         

