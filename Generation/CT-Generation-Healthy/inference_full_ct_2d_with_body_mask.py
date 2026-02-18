import glob
import nibabel as nib
import numpy as np
import os
import torch
from skimage import io
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ldm.util import instantiate_from_config


logdir = 'logs/full_ct_2d_with_body_mask/'
ckpt = os.path.join(logdir, "checkpoints", "epoch=000999.ckpt")

configs_file = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))[1]
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

save_path = 'inference'
save_path = os.path.join(logdir, save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

val_dataset = data.datasets['validation']
batch_size = 1
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
val_num = len(val_dataset)
save_gt = True
# val_num = 10
# breakpoint()
for idx, data in tqdm(enumerate(valloader)):
    # if idx >= val_num:
    #     break    
    name=data['name'][0].split('.')[0]
    data['volume_data'] = data['volume_data'].to(device)
    data['volume_seg'] = data['volume_seg'].to(device)
    input_text = data['input_text']
    # .to(device)

    x, c  = model.get_input(data, model.first_stage_key)
    # breakpoint()
    samples, _ = model.sample_log(cond=c, batch_size=17, ddim=True, eta=1., ddim_steps=100)
    samples=samples.permute(1,0,2,3)

    res = model.decode_first_stage(samples[None,:])
    res = res*2
    # breakpoint()
    res[res>1.0] = 1.0
    res[res<-1.0] = -1.0
    res = (res+1)/2
    # breakpoint()
    res = res[0].mean(axis=0).detach().cpu().numpy()
    # slice_ct=(slice_ct+1)/2
    # slice_seg=(slice_seg+1)/2
    for z_slice in range(res.shape[0]):
        # input_img = np.clip(np.repeat(slice_ct[0,0].detach().cpu().numpy(), 3, 0),0,1) * 255
        res_slice = np.repeat(res[z_slice][None,], 3, 0).transpose(1,2,0)* 255
        # slice_seg = np.clip(np.repeat(slice_seg[0,0].detach().cpu().numpy(), 3, 0),0,1) * 255
        
        io.imsave(os.path.join(save_path, str(name) + f'_sample_{z_slice}.png'), res_slice.astype(np.uint8))
        # io.imsave(os.path.join(save_path, str(name) + f'_input_img_{z_slice}.png'), input_img.astype(np.uint8))
        # io.imsave(os.path.join(save_path, str(name) + f'_seg_{z_slice}.png'), slice_seg.astype(np.uint8))

    break


