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

def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``

    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

    return ornt_transf, ornt_init, ornt_fin

def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)

    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    return nib.orientations.apply_orientation(data_array, ornt_transf)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--time_steps",
    type=int,
    default=20,
)
args = parser.parse_args()
ddim_steps=args.time_steps
# breakpoint()

logdir = 'logs/full_ct_2d_with_body_mask'
ckpt = os.path.join(logdir, "checkpoints", "epoch=000068.ckpt")

configs_file = "configs/latent-diffusion/full_ct_2d_with_body_mask.yaml"
configs = OmegaConf.load(configs_file)
model = instantiate_from_config(configs.model)
model.init_from_ckpt(ckpt)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)
model = model.to(device)

config = OmegaConf.load('./configs/latent-diffusion/full_ct_2d_with_body_mask.yaml')
data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()

save_path = f'3d_results_step{ddim_steps}_train_latest'
save_path = os.path.join(logdir, save_path)
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

    name=data['name'][0].split('.')[0]
    volume_data = data['volume_data']
    volume_seg = data['volume_seg']
    # breakpoint()
    
    window_length = 65
    latent_lenght=17
    h = 1
    slice_num =volume_data.shape[2]
    result = torch.zeros((batch_size, latent_lenght, 16, 64, 64)).cuda()
    

    upper_iters = (slice_num-h) // (window_length-h)+1 if (slice_num-h)%(window_length-h) != 0 else (slice_num-h) // (window_length-h)
    print('upper_iters', upper_iters)
    # breakpoint()
    for i in range(upper_iters):
        print('i', i)
        input_data={}
        if i == upper_iters-1:
            input_data['name'] = data['name']
            input_data['volume_data'] = data['volume_data'][:, :, -window_length:].to(device)
            input_data['volume_seg'] = data['volume_seg'][:, :, -window_length:].to(device)
            input_data['input_text'] = data['input_text']
        else:
            input_data['volume_data'] = data['volume_data'][:, :, i*window_length-i*h:(i+1)*window_length-i*h].to(device)
            input_data['volume_seg'] = data['volume_seg'][:, :, i*window_length-i*h:(i+1)*window_length-i*h].to(device)
            input_data['input_text'] = data['input_text']
        # breakpoint()
        with torch.no_grad():
            _, c  = model.get_input(input_data, model.first_stage_key)
            # breakpoint()
            if i == 0:
                samples_i, _ = model.sample_log(cond=c, batch_size=latent_lenght, ddim=True, eta=1., ddim_steps=ddim_steps)
            else:
                samples_i, _ = model.sample_log(cond=c, batch_size=latent_lenght, ddim=True, eta=1., ddim_steps=ddim_steps, previous=x_minus1)

            samples_i = rearrange(samples_i, '(b z) c h w -> b z c h w', z=latent_lenght)

            if i == upper_iters-1:
                result[:, -latent_lenght+h:] = samples_i[:,h:,...]
            else:
                if i == 0:
                    result[:, :latent_lenght] = samples_i
                else:
                    result[:, i*latent_lenght-i*h+h:(i+1)*latent_lenght-i*h] = samples_i[:, h:]
                x_minus1 = samples_i[:, -h:,...]
    # breakpoint()
    # result = rearrange(result, 'b z c h w -> (b z) c h w')
    result = result.permute(0,2,1,3,4)
    x_result = torch.zeros((3,slice_num,512,512))
    dec_unit = 65
    num_dec_iter = slice_num // dec_unit + 1 if slice_num % dec_unit != 0 else slice_num // dec_unit
    for i in range(num_dec_iter):
        if i == num_dec_iter - 1:
            x_result[:,-dec_unit:] = model.decode_first_stage(result[:,-dec_unit:])[0]
        x_result[:,i*dec_unit:(i+1)*dec_unit] = model.decode_first_stage(result[:,i*dec_unit:(i+1)*dec_unit])[0]
    x_result = x_result*2
    x_result[x_result>1.0] = 1.0
    x_result[x_result<-1.0] = -1.0
    x_result = (x_result+1)/2
    # breakpoint()

    x_result_ = x_result.mean(axis=0).detach().cpu().numpy()
    # x_result = x_result[0,:,0,...].detach().cpu().numpy()
    # breakpoint()
    x_result = x_result_.transpose(1,2,0)
    # x_result = np.rot90(x_result, k=1, axes=(0,1))
    # x_result = np.flip(x_result,axis=(0,1))
    # import imageio as io
    # io.imsave('exp.png', (x_result[:,:,400]*255).astype(np.uint8))

    # breakpoint()
    ref_root = '/sd/shuhan/CT-RATE/dataset/valid_fixed'
    ref_nii = os.path.join(ref_root, name.split('_')[0]+'_'+name.split('_')[1], name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2],name+'.nii.gz')
    affine = nib.load(ref_nii).affine

    x_result = x_result*2000.0 - 1000.0
    data_path = os.path.join(save_path, str(f'{name}.nii.gz'))
    data_nii = nib.Nifti1Image(x_result.astype(np.int16), affine)

    nib.save(data_nii, data_path)

    breakpoint()

