import os
import numpy as np
import argparse
import math
from glob import glob
from skimage.metrics import structural_similarity as compare_ssim
import imageio
import lpips
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Argument parser
parser = argparse.ArgumentParser(
    description="Calculate PSNR, SSIM, and LPIPS between two sets of images."
)
parser.add_argument(
    "--root1",
    "-r1",
    type=str,
    required=True,
    help="Directory for the first set of images.",
)
parser.add_argument(
    "--root2",
    "-r2",
    type=str,
    required=True,
    help="Directory for the second set of images.",
)
parser.add_argument("--ssim", action="store_true", default=False, help="Compute SSIM.")
parser.add_argument("--psnr", action="store_true", default=False, help="Compute PSNR.")
parser.add_argument(
    "--lpips", action="store_true", default=False, help="Compute LPIPS."
)

args = parser.parse_args()

# Define metric functions


def compute_psnr(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_ssim(img1, img2):
    return compare_ssim(img1, img2, data_range=img1.max() - img1.min(), channel_axis=-1)


def compute_lpips(img1, img2, loss_fn):
    img1_tensor = (
        torch.from_numpy(img1 / 255.0)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to("cuda:0")
    )
    img2_tensor = (
        torch.from_numpy(img2 / 255.0)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to("cuda:0")
    )

    img1_tensor = img1_tensor * 2 - 1  # Normalize to [-1, 1]
    img2_tensor = img2_tensor * 2 - 1

    return loss_fn(img1_tensor, img2_tensor).item()


def read_image(file_path):
    try:
        return imageio.imread(file_path)
    except Exception as e:
        logging.error(f"Error reading image {file_path}: {e}")
        return None


def save_results(results, root1, root2, output_file="metrics.txt"):
    with open(output_file, "a") as f:
        f.write("\n")
        f.write(f"Root1: {root1}\n")
        f.write(f"Root2: {root2}\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")
    logging.info(f"Results saved to {output_file}")


def main():
    # Load image paths
    all_images1 = sorted(glob(os.path.join(args.root1, "*jpeg")))
    all_images2 = sorted(glob(os.path.join(args.root2, "*jpeg")))

    assert len(all_images1) == len(
        all_images2
    ), f"Number of files mismatch: {len(all_images1)} in {args.root1}, {len(all_images2)} in {args.root2}"

    # Metrics storage
    metric_psnr = []
    metric_ssim = []
    metric_lpips = []

    lpips_model = None
    if args.lpips:
        lpips_model = lpips.LPIPS(net="alex").to("cuda:0")
        logging.info("Initialized LPIPS model (AlexNet).")

    # Compute metrics for each pair of images
    for i, (img1_path, img2_path) in enumerate(
        tqdm(
            zip(all_images1, all_images2),
            total=len(all_images1),
            desc="Processing images",
        )
    ):
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        if img1 is None or img2 is None:
            logging.warning(f"Skipping pair: {img1_path}, {img2_path}")
            continue

        if args.psnr:
            try:
                psnr_value = compute_psnr(img1, img2)
                metric_psnr.append(psnr_value)
            except Exception as e:
                logging.error(f"Error computing PSNR for {img1_path}, {img2_path}: {e}")

        if args.ssim:
            try:
                ssim_value = compute_ssim(img1, img2)
                metric_ssim.append(ssim_value)
            except Exception as e:
                logging.error(f"Error computing SSIM for {img1_path}, {img2_path}: {e}")

        if args.lpips:
            try:
                lpips_value = compute_lpips(img1, img2, lpips_model)
                metric_lpips.append(lpips_value)
            except Exception as e:
                logging.error(
                    f"Error computing LPIPS for {img1_path}, {img2_path}: {e}"
                )

    results = {}
    if args.psnr and metric_psnr:
        results["PSNR"] = sum(metric_psnr) / len(metric_psnr)
    if args.ssim and metric_ssim:
        results["SSIM"] = sum(metric_ssim) / len(metric_ssim)
    if args.lpips and metric_lpips:
        results["LPIPS"] = sum(metric_lpips) / len(metric_lpips)

    # Print and save results
    logging.info(f"Results: {results}")
    save_results(results, args.root1, args.root2)


if __name__ == "__main__":
    main()
