import os
import torch
import logging
from glob import glob
import argparse
from omegaconf import OmegaConf
from utils.common_utils import instantiate_from_config
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Image Inference Script")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the folder containing input images.",
    )
    parser.add_argument(
        "--out_root", type=str, required=True, help="Path to save reconstructed images."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for image processing."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., 'cpu', 'cuda:0').",
    )
    return parser.parse_args()


def data_processing(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform(img)
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        return None


def save_img(tensor, save_path):
    try:
        tensor = (tensor + 1) / 2  # Denormalize
        tensor = tensor.clamp(0, 1).detach().cpu()
        to_pil = transforms.ToPILImage()
        img = to_pil(tensor)
        img.save(save_path, format="JPEG")
        logging.info(f"Image saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving image to {save_path}: {e}")


def process_batch(image_list, img_name_list, model, device, out_root):
    try:
        frames = torch.stack(image_list)  # [batch_size, c, h, w]
        frames = frames.unsqueeze(1)  # [batch_size, 1, c, h, w]
        frames = frames.permute(0, 2, 1, 3, 4)  # [batch_size, c, 1, h, w]

        with torch.no_grad():
            frames = frames.to(device)
            dec, _ = model.forward(frames, sample_posterior=False, mask_temporal=True)
            dec = dec.squeeze(2)  # [batch_size, c, h, w]

        for i in range(len(image_list)):
            output_img = dec[i]
            save_img(output_img, os.path.join(out_root, img_name_list[i] + ".jpeg"))
    except Exception as e:
        logging.error(f"Error processing batch: {e}")


def main():
    args = parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    config = OmegaConf.load(args.config_path)
    model = instantiate_from_config(config.model)
    model = model.to(args.device)
    model.eval()

    # Load all image paths
    all_images = sorted(glob(os.path.join(args.data_root, "*jpeg")))
    if not all_images:
        logging.error(f"No images found in {args.data_root}")
        return

    batch_size = args.batch_size
    image_list = []
    img_name_list = []

    logging.info(f"Starting inference on {len(all_images)} images...")

    for img_path in all_images:
        img = data_processing(img_path)  # [c, h, w]
        if img is None:
            logging.warning(f"Skipping invalid image {img_path}")
            continue

        img_name = os.path.basename(img_path).split(".")[0]
        image_list.append(img)
        img_name_list.append(img_name)

        # Process a batch when full
        if len(image_list) == batch_size:
            logging.info(f"Processing batch of {batch_size} images...")
            process_batch(image_list, img_name_list, model, args.device, args.out_root)

            # Clear lists for next batch
            image_list = []
            img_name_list = []

    # Process any remaining images
    if len(image_list) > 0:
        logging.info(f"Processing remaining {len(image_list)} images...")
        process_batch(image_list, img_name_list, model, args.device, args.out_root)

    logging.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
