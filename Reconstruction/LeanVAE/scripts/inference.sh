python leanvae_inference_ct.py --ckpt_path "./ckpts/LeanVAE-dim4.ckpt" \
                      --device "cuda:0" \
                      --input_video /home/v-qichen3/data/CT/Dataset100_Liver/imagesTr/liver_0_0000.nii.gz \
                      --reconstruct_video ./debug
                      #FOR Tile Inference
                      #--tile_inference 
                      #--chunksize_enc 5  
                      #--chunksize_dec 5   
                     
