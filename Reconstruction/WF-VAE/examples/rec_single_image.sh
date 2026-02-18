CUDA_VISIBLE_DEVICES=1 python scripts/recon_single_image.py \
    --model_name WFVAE \
    --from_pretrained "results/WF-VAE-L-16Chn" \
    --image_path assets/gt_5544.jpg \
    --rec_path rec.jpg \
    --device cuda \
    --short_size 512 