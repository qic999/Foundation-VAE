import glob
import nibabel as nib
import numpy as np
import os
import torch

from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ldm.util import instantiate_from_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--save_path",
    type=str,
    default=None,
)

args = parser.parse_args()
ddim_steps=args.time_steps
# breakpoint()

config = OmegaConf.load('./configs/latent-diffusion/mask_generation.yaml')
data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()


save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

val_dataset = data.datasets['validation']
batch_size = 1
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
val_num = len(val_dataset)
save_gt = True

for idx, data in tqdm(enumerate(valloader)):

    if idx >= val_num:
        break

    name = data['name'][0]
    volume_data = data['volume_data']

    
    window_length = 16
    h = 1
    slice_num =volume_data.shape[1]
    result = torch.zeros((batch_size, slice_num, 4, 64, 64)).cuda()
    

    upper_iters = (slice_num-h) // (window_length-h)+1 if (slice_num-h)%(window_length-h) != 0 else (slice_num-h) // (window_length-h)
    print('upper_iters', upper_iters)
    # breakpoint()
    for i in range(upper_iters):
        print('i', i)
        input_data={}
        if i == upper_iters-1:
            input_data['name'] = data['name']
            input_data['volume_data'] = data['volume_data'][:,-window_length:].to(device)
            input_data['masked_data'] = data['masked_data'][:,-window_length:].to(device)
            input_data['tumor_mask'] = data['tumor_mask'][:,-window_length:].to(device)
        else:
            input_data['volume_data'] = data['volume_data'][:, i*window_length-i*h:(i+1)*window_length-i*h].to(device)
            input_data['masked_data'] = data['masked_data'][:, i*window_length-i*h:(i+1)*window_length-i*h].to(device)
            input_data['tumor_mask'] = data['tumor_mask'][:, i*window_length-i*h:(i+1)*window_length-i*h].to(device)
    
        with torch.no_grad():
            _, c  = model.get_input(input_data, model.first_stage_key)
            
            if i == 0:
                samples_i, _ = model.sample_log(cond=c, batch_size=window_length, ddim=True, eta=1., ddim_steps=ddim_steps)
            else:
                samples_i, _ = model.sample_log(cond=c, batch_size=window_length, ddim=True, eta=1., ddim_steps=ddim_steps, previous=x_minus1)
            # breakpoint()
            samples_i = rearrange(samples_i, '(b z) c h w -> b z c h w', z=window_length)

            if i == upper_iters-1:
                result[:, -window_length+h:] = samples_i[:,h:,...]
            else:
                if i == 0:
                    result[:, :window_length] = samples_i
                else:
                    result[:, i*window_length-i*h+h:(i+1)*window_length-i*h] = samples_i[:, h:]
                x_minus1 = samples_i[:, -h:,...]
    # breakpoint()
    result = rearrange(result, 'b z c h w -> (b z) c h w')
    x_result = torch.zeros((result.shape[0],3,512,512))
    # breakpoint()
    dec_unit = 16
    num_dec_iter = slice_num // dec_unit + 1 if slice_num % dec_unit != 0 else slice_num // dec_unit
    for i in range(num_dec_iter):
        if i == num_dec_iter - 1:
            x_result[-dec_unit:] = model.decode_first_stage(result[-dec_unit:])
        x_result[i*dec_unit:(i+1)*dec_unit] = model.decode_first_stage(result[i*dec_unit:(i+1)*dec_unit])
    x_result[x_result>1.0] = 1.0
    x_result[x_result<-1.0] = -1.0
    x_result = (x_result+1)/2
    x_result = rearrange(x_result, '(b z) c h w -> b z c h w', z=slice_num)
    x_result_ = x_result[0].mean(axis=1).detach().cpu().numpy()
    # x_result = x_result[0,:,0,...].detach().cpu().numpy()

    x_result = x_result_.transpose(2,1,0)
    # x_result = np.rot90(x_result, k=1, axes=(0,1))
    # x_result = np.flip(x_result,axis=(0,1))
    # import imageio as io
    # io.imsave('exp.png', (x_result[:,:,400]*255).astype(np.uint8))

    # breakpoint()
    ref_root = '/storage/chenqi/data/CT/Task03_Liver/imagesTr'
    ref_nii = os.path.join(ref_root, name+'.nii.gz')
    affine = nib.load(ref_nii).affine

    x_result = x_result*425.0 - 175.0
    data_path = os.path.join(save_path, str(f'{name}.nii.gz'))
    data_nii = nib.Nifti1Image(x_result.astype(np.int16), affine)

    nib.save(data_nii, data_path)

    # breakpoint()

