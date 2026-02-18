import argparse

import cv2
import numpy as np
import numpy.typing as npt
import torch
from decord import VideoReader, cpu
from torchvision.transforms import Lambda, Compose
from einops import rearrange
import nibabel as nib
import numpy as np

import sys
import os

sys.path.append(".")
from causalvideovae.model import *
from causalvideovae.dataset.transform import ToTensorVideo, CenterCropResizeVideo



def main(args: argparse.Namespace):
    # Set device and data type for computation
    device = args.device
    data_type = torch.bfloat16

    # Load the specified VAE model
    model_cls = ModelRegistry.get_model(args.model_name)
    vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(device).to(data_type)

    # Enable tiling mode if specified (useful for large video)
    if args.enable_tiling:
        vae.enable_tiling()

    vae.eval()
    vae = vae.to(device, dtype=data_type)

    os.makedirs(args.rec_path,exist_ok=True)

    video = nib.load(args.video_path)
    affine = video.affine
    video = video.get_fdata()  
    video = np.clip(video, -175,250)
    video = (video+175.0)/425.0
    # breakpoint()
    video = torch.from_numpy(video)
    video = video.permute(-1,0,1)[:,None]
    video = video.repeat(1,3,1,1)

    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).float()
    video = video[:,:,:65,:,:]
    video = video * 2.0 - 1.0

    # breakpoint()
    with torch.no_grad():
        # Preprocess the input video

        x_vae = video.to(device, dtype=data_type)

        # Encode the video into latent space
        latents = vae.encode(x_vae).latent_dist.sample()
        latents = latents.to(data_type)
        # Decode the latent vectors back to reconstructed video
        video_recon = vae.decode(latents).sample

        print("recon shape", video_recon.shape)
    # breakpoint()
    results = rearrange(video_recon.squeeze(0), 'c t h w -> c h w t')
    results = results.mean(0)


    results = (torch.clamp(results,-1.0,1.0) + 1.0) / 2.0

    results = results*425.0 - 175.0

    results = results.to('cpu', dtype=torch.int16).numpy()
    # breakpoint()
    name = os.path.basename(args.video_path).split('.')[0]
    data_path = os.path.join(args.rec_path, str(f'{name}_recon.nii.gz'))
    data_nii = nib.Nifti1Image(results.astype(np.int16), affine)
    nib.save(data_nii, data_path)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser()

    # Video input and output paths
    parser.add_argument(
        "--video_path", type=str, default="", help="Path to the input video file"
    )
    parser.add_argument(
        "--rec_path", type=str, default="", help="Path to save the reconstructed video"
    )

    # Model settings
    parser.add_argument(
        "--model_name", type=str, default="vae", help="Name of the model to use"
    )
    parser.add_argument(
        "--from_pretrained",
        type=str,
        default="",
        help="Path or identifier of the pretrained model",
    )



    # Device and memory settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (e.g., 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        help="Enable tiling for large image processing",
    )

    # Parse the command-line arguments and run the main function
    args = parser.parse_args()
    main(args)
