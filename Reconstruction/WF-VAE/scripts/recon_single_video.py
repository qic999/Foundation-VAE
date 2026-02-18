import argparse

import cv2
import numpy as np
import numpy.typing as npt
import torch
from decord import VideoReader, cpu
from torchvision.transforms import Lambda, Compose
from einops import rearrange

import sys

sys.path.append(".")
from causalvideovae.model import *
from causalvideovae.dataset.transform import ToTensorVideo, CenterCropResizeVideo


def array_to_video(
    image_array: npt.NDArray, fps: float = 30.0, output_file: str = "output_video.mp4"
) -> None:
    height, width, _ = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))

    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()


def custom_to_video(
    x: torch.Tensor, fps: float = 2.0, output_file: str = "output_video.mp4"
) -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = rearrange(x, "c t h w -> t h w c").float().numpy()
    x = (255 * x).astype(np.uint8)

    array_to_video(x, fps=fps, output_file=output_file)
    return


def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    s = 0
    e = sample_frames_len
    print(
        f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
        video_path,
        total_frames,
    )

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data


def preprocess(
    video_data: torch.Tensor, height: int = 128, width: int = 128
) -> torch.Tensor:
    transform = Compose(
        [
            ToTensorVideo(),
            CenterCropResizeVideo((height, width)),
            Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )

    video_outputs = transform(video_data)
    video_outputs = torch.unsqueeze(video_outputs, 0)

    return video_outputs


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

    with torch.no_grad():
        # Preprocess the input video
        x_vae = preprocess(
            read_video(args.video_path, args.num_frames, args.sample_rate),
            args.height,
            args.width,
        )
        x_vae = x_vae.to(device, dtype=data_type)
        # breakpoint()
        # Encode the video into latent space
        latents = vae.encode(x_vae).latent_dist.sample()
        latents = latents.to(data_type)
        # Decode the latent vectors back to reconstructed video
        video_recon = vae.decode(latents).sample

        print("recon shape", video_recon.shape)

    # Save the reconstructed video to a file
    custom_to_video(video_recon[0], fps=args.fps, output_file=args.rec_path)


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

    # Video parameters
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the output video"
    )
    parser.add_argument(
        "--height", type=int, default=336, help="Height of the processed video frames"
    )
    parser.add_argument(
        "--width", type=int, default=336, help="Width of the processed video frames"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=100,
        help="Number of frames to extract from the input video",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=1,
        help="Frame sampling rate for the input video",
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
