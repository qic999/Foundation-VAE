CUDA_VISIBLE_DEVICES=0 python scripts/recon_single_ct.py \
    --model_name WFVAE \
    --from_pretrained "chestnutlzj/WF-VAE-L-16Chn" \
    --video_path /home/v-qichen3/data/CT/Dataset100_Liver/imagesTr/liver_0_0000.nii.gz \
    --rec_path ./debug \
    --device cuda \
    --enable_tiling