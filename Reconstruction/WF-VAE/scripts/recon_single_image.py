import argparse

import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import ToTensor, Compose, Resize, Lambda
from PIL import Image

import sys

sys.path.append(".")
from causalvideovae.model import *


def preprocess(video_data: torch.Tensor, short_size: int = 128) -> torch.Tensor:
    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: 2.0 * x - 1.0),
            Resize(size=short_size),
        ]
    )
    outputs = transform(video_data)
    outputs = outputs.unsqueeze(0).unsqueeze(2)
    return outputs


def main(args: argparse.Namespace):
    # Set device and data type for computation
    device = args.device
    data_type = torch.bfloat16

    # Load the specified VAE model
    model_cls = ModelRegistry.get_model(args.model_name)
    vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(device).to(data_type)

    vae.eval()
    vae = vae.to(device, dtype=data_type)

    with torch.no_grad():
        # Preprocess the input video
        x_vae = preprocess(Image.open(args.image_path), args.short_size)
        x_vae = x_vae.to(device, dtype=data_type)

        # Encode the video into latent space
        latents = vae.encode(x_vae).latent_dist.sample()
        latents = latents.to(data_type)
        # Decode the latent vectors back to reconstructed video
        image_recon = vae.decode(latents).sample

    x = image_recon[0, :, 0, :, :]
    x = x.squeeze()
    x = x.detach().float().cpu().numpy()
    x = np.clip(x, -1, 1)
    x = (x + 1) / 2
    x = (255 * x).astype(np.uint8)
    x = rearrange(x, "c h w -> h w c")
    image = Image.fromarray(x)
    image.save(args.rec_path)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser()

    # Video input and output paths
    parser.add_argument(
        "--image_path", type=str, default="", help="Path to the input image file"
    )
    parser.add_argument(
        "--rec_path", type=str, default="", help="Path to save the reconstructed image"
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

    # Video parameters
    parser.add_argument(
        "--short_size", type=int, default=336, help="Short size of the image"
    )

    # Device and memory settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (e.g., 'cuda', 'cpu')",
    )

    # Parse the command-line arguments and run the main function
    args = parser.parse_args()
    main(args)
