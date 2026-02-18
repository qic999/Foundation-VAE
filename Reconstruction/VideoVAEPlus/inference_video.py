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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


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
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[720, 1280],
        help="Resolution to process videos (height, width).",
    )
    return parser.parse_args()


def data_processing(video_path, resolution):
    """Load and preprocess video data."""
    try:
        video_reader = VideoReader(video_path, ctx=cpu(0))
        video_resolution = video_reader[0].shape

        # Rescale resolution to match specified limits
        resolution = [
            min(video_resolution[0], resolution[0]),
            min(video_resolution[1], resolution[1]),
        ]
        video_reader = VideoReader(
            video_path, ctx=cpu(0), width=resolution[1], height=resolution[0]
        )

        video_length = len(video_reader)
        vid_fps = video_reader.get_avg_fps()
        frame_indices = list(range(0, video_length))
        frames = video_reader.get_batch(frame_indices)
        assert (
            frames.shape[0] == video_length
        ), f"Frame mismatch: {len(frames)} != {video_length}"

        frames = (
            torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        )  # [t, h, w, c] -> [c, t, h, w]
        frames = (frames / 255 - 0.5) * 2  # Normalize to [-1, 1]
        return frames, vid_fps
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")
        return None, None


def save_video(tensor, save_path, fps: float):
    """Save video tensor to a file."""
    try:
        tensor = torch.clamp((tensor + 1) / 2, 0, 1) * 255
        arr = tensor.detach().cpu().squeeze().to(torch.uint8)
        c, t, h, w = arr.shape
            
        torchvision.io.write_video(save_path, arr.permute(1, 2, 3, 0), fps=fps, options={'codec': 'libx264', 'crf': '15'})
        logging.info(f"Video saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving video {save_path}: {e}")


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
        padding_frames = 0
        output_chunks = []

        # Pad video to make the frame count divisible by 4
        if num_frames % 4 != 0:
            padding_frames = 4 - (num_frames % 4)
            padding = video_data[:, :, -1:, :, :].repeat(1, 1, padding_frames, 1, 1)
            video_data = torch.cat((video_data, padding), dim=2)
            num_frames = video_data.size(2)

        start = 0

        while start < num_frames:
            end = min(start + chunk_size, num_frames)
            chunk = video_data[:, :, start:end, :, :]

            with torch.no_grad():
                chunk = chunk.to(device)
                if text_embeddings is not None and text_attn_mask is not None:
                    recon_chunk, _ = model.forward(
                        chunk,
                        text_embeddings=text_embeddings,
                        text_attn_mask=text_attn_mask,
                        sample_posterior=False,
                    )
                else:
                    recon_chunk, _ = model.forward(chunk, sample_posterior=False)
                recon_chunk = recon_chunk.cpu().float()
            output_chunks.append(recon_chunk)
            start += chunk_size

        ret = torch.cat(output_chunks, dim=2)
        if padding_frames > 0:
            ret = ret[:, :, :-padding_frames, :, :]
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
    is_t5 = getattr(model, "caption_guide", False)
    model = model.to(args.device)
    model.eval()

    # Initialize text embedder if T5 is used
    text_embedder = None
    if is_t5:
        text_embedder = T5Embedder(
            device=args.device, model_max_length=model.t5_model_max_length
        )

    # Get all videos
    all_videos = sorted(glob(os.path.join(args.data_root, "*.mp4")))
    if not all_videos:
        logging.error(f"No videos found in {args.data_root}")
        return

    # Process each video
    for video_path in tqdm(all_videos, desc="Processing videos", unit="video"):
        logging.info(f"Processing video: {video_path}")
        frames, vid_fps = data_processing(video_path, args.resolution)
        if frames is None:
            continue

        video_name = os.path.basename(video_path).split(".")[0]
        frames = torch.unsqueeze(frames, dim=0)  # Add batch dimension

        with torch.no_grad():
            if is_t5:
                # Load caption if available
                text_path = os.path.join(args.data_root, f"{video_name}.txt")
                try:
                    with open(text_path, "r") as f:
                        caption = [f.read()]
                except Exception as e:
                    logging.warning(f"Caption file not found for {video_name}: {e}")
                    caption = [""]

                text_embedding, text_attn_mask = text_embedder.get_text_embeddings(
                    caption
                )
                text_embedding = text_embedding.to(args.device, dtype=model.dtype)
                text_attn_mask = text_attn_mask.to(args.device, dtype=model.dtype)

                video_recon = process_in_chunks(
                    frames,
                    model,
                    args.chunk_size,
                    text_embedding,
                    text_attn_mask,
                    device=args.device,
                )
            else:
                video_recon = process_in_chunks(
                    frames, model, args.chunk_size, device=args.device
                )

            if video_recon is not None:
                save_path = os.path.join(
                    args.out_root, f"{video_name}_reconstructed.mp4"
                )
                save_video(video_recon, save_path, vid_fps)


if __name__ == "__main__":
    main()
