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

logdir = 'logs/full_ct_3d_with_body_mask'
ckpt = os.path.join(logdir, "checkpoints", "epoch=000070.ckpt")

configs_file = "configs/latent-diffusion/full_ct_3d_with_body_mask.yaml"
configs = OmegaConf.load(configs_file)
model = instantiate_from_config(configs.model)
model.init_from_ckpt(ckpt)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)
model = model.to(device)

config = OmegaConf.load('./configs/latent-diffusion/full_ct_3d_with_body_mask.yaml')
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
    
    import math
    # ---- 必要参数 ----
    window_length  = 65           # 输入窗口长度（切片数）
    latent_lenght  = 17           # 每窗原始输出长度（latent 轴）
    overleap_lengh = 1            # ★ 输出侧希望的重叠（用于融合）
    slice_num      = volume_data.shape[2]
    device = next(model.parameters()).device

    # === 仅保留后16个（丢弃第0个） ===
    keep_len = max(0, latent_lenght - 1)

    # 1) 输出重叠 -> 输入重叠（按比例换算；用 keep_len）
    overlap_in = int(round(overleap_lengh * window_length / max(1, keep_len)))
    overlap_in = max(0, min(window_length - 1, overlap_in))     # 保证 0 <= overlap_in < window_length

    # 2) 输入步长 & upper_iters（首窗从0开始，末窗贴尾）
    step_in = window_length - overlap_in
    extra = max(0, slice_num - window_length)
    upper_iters = 1 + (extra + step_in - 1) // step_in          # 等价于 ceil(extra/step_in)
    print(f"[plan] overlap_in={overlap_in}, step_in={step_in}, upper_iters={upper_iters}")

    # 3) 输出侧拼接步长（latent轴，用 keep_len）
    stride_out = max(1, keep_len - overleap_lengh)
    total_len  = keep_len + (upper_iters - 1) * stride_out
    print(f"[plan] stride_out={stride_out}, total_len={total_len}")

    B = batch_size
    C, H, W = 16, 64, 64

    result_sum = torch.zeros((B, total_len, C, H, W), device=device, dtype=torch.float32)
    weight_sum = torch.zeros((1, total_len, 1, 1, 1), device=device, dtype=torch.float32)

    # 1D 高斯权重（长度=keep_len）；sigma 可随重叠自适应
    z = torch.arange(keep_len, device=device, dtype=torch.float32)
    center = (keep_len - 1) / 2.0 if keep_len > 0 else 0.0
    sigma  = max(overleap_lengh / 2.0, 1.0)    # 重叠越大，sigma 越大；下限1.0避免过尖
    g = torch.exp(-0.5 * ((z - center) / sigma) ** 2) if keep_len > 0 else torch.ones(1, device=device)
    g = g / g.max() if keep_len > 0 else g
    g = g.view(1, keep_len, 1, 1, 1)

    x_minus1 = None
    cond_h   = 0   # 是否启用历史条件

    for i in range(upper_iters):
        if i < upper_iters - 1:
            st = i * step_in
            st = max(0, min(st, slice_num - window_length))
            ed = st + window_length
            vol_i = data['volume_data'][:, :, st:ed].to(device)
            seg_i = data['volume_seg'][:,  :, st:ed].to(device)
        else:
            vol_i = data['volume_data'][:, :, -window_length:].to(device)
            seg_i = data['volume_seg'][:,  :, -window_length:].to(device)

        input_data = {
            'name': data.get('name', None),
            'volume_data': vol_i, 'volume_seg': seg_i,
            'input_text': data['input_text']
        }

        with torch.no_grad():
            _, c = model.get_input(input_data, model.first_stage_key)
            if x_minus1 is None or cond_h == 0:
                samples_i, _ = model.sample_log(cond=c, batch_size=latent_lenght, ddim=True, eta=1., ddim_steps=ddim_steps)
            else:
                samples_i, _ = model.sample_log(cond=c, batch_size=latent_lenght, ddim=True, eta=1., ddim_steps=ddim_steps, previous=x_minus1)

            # (b*z, c, h, w) -> (b, z, c, h, w)
            samples_i = rearrange(samples_i, '(b z) c h w -> b z c h w', z=latent_lenght)

            # === 关键改动：只拼接后 keep_len 个（丢弃第0个）===
            kept = samples_i[:, 1:, ...]     # shape: (B, keep_len, C, H, W)

            gstart = i * stride_out
            gend   = gstart + keep_len
            result_sum[:, gstart:gend] += kept * g
            weight_sum[:, gstart:gend] += g

            # 历史条件（若启用），通常取“保留段”的尾部
            x_minus1 = kept[:, -cond_h:, ...] if cond_h > 0 else None

    # 融合
    result = result_sum / torch.clamp_min(weight_sum, 1e-8)


    # result = rearrange(result, 'b z c h w -> (b z) c h w')
    result = result.permute(0,2,1,3,4)
    x_result = torch.zeros((3,slice_num,512,512))
    dec_unit = 65
    res_unit = window_length-dec_unit
    num_dec_iter = slice_num // dec_unit + 1 if slice_num % dec_unit != 0 else slice_num // dec_unit
    for i in range(num_dec_iter):
        if i == num_dec_iter - 1:
            x_result[:,-dec_unit:] = model.decode_first_stage(result[:,:,-latent_lenght:])[0][:,res_unit:]
        else:
            # breakpoint()
            x_result[:,i*dec_unit:(i+1)*dec_unit] = model.decode_first_stage(result[:,:,i*latent_lenght:(i+1)*latent_lenght])[0][:,res_unit:]
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

