import os
import torch
import argparse
import logging
from decord import VideoReader, cpu
from glob import glob
from omegaconf import OmegaConf
import numpy as np
import imageio
from tqdm import tqdm
from utils.common_utils import instantiate_from_config
from src.modules.t5 import T5Embedder
import torchvision

from einops import rearrange
import nibabel as nib

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Video VAE Inference Script")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the folder containing input videos.",
    )
    parser.add_argument(
        "--out_root", type=str, required=True, help="Path to save reconstructed videos."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help="Number of frames per chunk for processing.",
    )

    return parser.parse_args()



def process_in_chunks(
    video_data,
    model,
    chunk_size,
    text_embeddings=None,
    text_attn_mask=None,
    device="cuda:0",
):
    try:
        assert chunk_size % 4 == 0, "Chunk size must be a multiple of 4."
        num_frames = video_data.size(2)

        output_chunks = []

        start = 0

        while start < num_frames:
            end = min(start + chunk_size, num_frames)
            chunk = video_data[:, :, start:end, :, :]

            with torch.no_grad():
                chunk = chunk.to(device)
                recon_chunk, _ = model.forward(chunk, sample_posterior=False)
                recon_chunk = recon_chunk.cpu().float()
            output_chunks.append(recon_chunk)
            start += chunk_size

        ret = torch.cat(output_chunks, dim=2)

        return ret
    except Exception as e:
        logging.error(f"Error processing chunks: {e}")
        return None


def main():
    """Main function for video VAE inference."""
    args = parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    config = OmegaConf.load(args.config_path)

    # Initialize model
    model = instantiate_from_config(config.model)

    model = model.to(args.device)
    model.eval()

    video = nib.load(args.data_root)
    affine = video.affine
    video = video.get_fdata()  
    video = np.clip(video, -175,250)
    video = (video+175.0)/425.0
    # breakpoint()
    video = torch.from_numpy(video)
    video = video.permute(-1,0,1)[:,None]
    video = video.repeat(1,3,1,1)

    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).float()
    video = video[:,:,:64,:,:]
    video = video * 2.0 - 1.0

    # Process each video

    with torch.no_grad():

        video_recon = process_in_chunks(
            video, model, args.chunk_size, device=args.device
        )
    # breakpoint()
    results = rearrange(video_recon.squeeze(0), 'c t h w -> c h w t')
    results = results.mean(0)


    results = (torch.clamp(results,-1.0,1.0) + 1.0) / 2.0

    results = results*425.0 - 175.0

    results = results.to('cpu', dtype=torch.int16).numpy()
    # breakpoint()
    name = os.path.basename(args.data_root).split('.')[0]
    data_path = os.path.join(args.out_root, str(f'{name}_recon.nii.gz'))
    data_nii = nib.Nifti1Image(results.astype(np.int16), affine)
    nib.save(data_nii, data_path)



if __name__ == "__main__":
    main()
