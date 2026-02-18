import torch
import numpy as np
from wan.modules.vae2_2 import Wan2_2_VAE
import os
from einops import rearrange
import nibabel as nib
from glob import glob
from tqdm import tqdm
import traceback

def sliding_window_encode(video, vae, window_size=65):
    """
    使用滑窗方式对视频进行编码，返回所有窗口的latent
    
    Args:
        video: 输入视频tensor, shape: (c, t, h, w)
        vae: VAE模型
        window_size: 滑窗大小
    
    Returns:
        latent列表或单个latent
    """
    c, t, h, w = video.shape
    original_t = t
    
    # 如果视频长度小于等于窗口大小，padding后直接编码
    if t <= window_size:
        if t < window_size:
            pad_size = window_size - t
            video = torch.nn.functional.pad(
                video, 
                (0, 0, 0, 0, 0, pad_size),
                mode='replicate'
            )
        with torch.no_grad():
            latent = vae.encode([video])
        return latent, original_t
    
    # 如果视频长度大于窗口大小，使用滑窗处理
    else:
        all_latents = []
        num_windows = (t + window_size - 1) // window_size
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, t)
            actual_window_size = end_idx - start_idx
            
            window = video[:, start_idx:end_idx, :, :]
            
            if actual_window_size < window_size:
                pad_size = window_size - actual_window_size
                window = torch.nn.functional.pad(
                    window, 
                    (0, 0, 0, 0, 0, pad_size), 
                    mode='replicate'
                )
            
            with torch.no_grad():
                latent = vae.encode([window])
            # breakpoint()
            all_latents.append(latent[0].cpu().numpy())
        
        return all_latents, original_t

def process_single_file(video_path, save_path, vae, window_size=65, skip_existing=True):
    """
    处理单个nii.gz文件，只保存latent到npy文件
    """
    try:
        name = os.path.basename(video_path).replace('.nii.gz', '')
        latent_path = os.path.join(save_path, f'{name}_latent.npy')
        
        if skip_existing and os.path.exists(latent_path):
            print(f"  ⊙ File already exists, skipping: {latent_path}")
            return 'skipped'
        
        # 加载和预处理视频
        video_nib = nib.load(video_path)
        video_data = video_nib.get_fdata()
        
        original_z_size = video_data.shape[2]
        print(f"  Original shape: {video_data.shape}")
        
        # 预处理
        video_data = np.clip(video_data, -1000, 1000)
        video_data = video_data / 1000.0
        
        video = torch.from_numpy(video_data)
        video = video.permute(2, 0, 1)[:, None]
        video = video.repeat(1, 3, 1, 1)
        video = rearrange(video, 't c h w -> c t h w').float()
        video = video.cuda()
        
        print(f"  Video tensor shape: {video.shape}")
        
        # 编码
        latent, original_t = sliding_window_encode(video, vae, window_size=window_size)
        
        # 保存latent
        if isinstance(latent, list):
            latent_np = np.array(latent)
        else:
            latent_np = latent.cpu().numpy()
        
        np.save(latent_path, latent_np)
        print(f"  ✓ Latent saved to: {latent_path}")
        print(f"  ✓ Latent shape: {latent_np.shape}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {video_path}: {str(e)}")
        traceback.print_exc()
        return False

def main_dir(input_dir, output_dir, window_size=65, checkpoint_dir=None, skip_existing=True):
    """
    处理目录下的所有nii.gz文件，只保存latent
    """
    if checkpoint_dir is None:
        checkpoint_dir = '/datasdc16T/qichen/debug/'
    
    print("Loading VAE model...")
    vae = Wan2_2_VAE(vae_pth=os.path.join(checkpoint_dir, 'Wan2.2_VAE.pth'))
    print("Model loaded successfully!\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_nii_files = glob(os.path.join(input_dir, '*.nii.gz'))
    
    if not all_nii_files:
        print(f"No .nii.gz files found in {input_dir}")
        return
    
    print(f"Found {len(all_nii_files)} .nii.gz files in {input_dir}")
    
    # 过滤出需要处理的文件
    nii_files_to_process = []
    already_exist_files = []
    
    if skip_existing:
        for video_path in all_nii_files:
            name = os.path.basename(video_path).replace('.nii.gz', '')
            latent_path = os.path.join(output_dir, f'{name}_latent.npy')
            if os.path.exists(latent_path):
                already_exist_files.append(os.path.basename(video_path))
            else:
                nii_files_to_process.append(video_path)
        
        print(f"Already processed (skipping): {len(already_exist_files)} files")
        print(f"To be processed: {len(nii_files_to_process)} files")
    else:
        nii_files_to_process = all_nii_files
    
    if not nii_files_to_process:
        print("\nAll files have been processed! Nothing to do.")
        return
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Window size: {window_size}\n")
    
    successful = 0
    failed = 0
    failed_files = []
    
    for idx, video_path in enumerate(tqdm(nii_files_to_process, desc="Processing files")):
        filename = os.path.basename(video_path)
        print(f"\n[{idx+1}/{len(nii_files_to_process)}] Processing: {filename}")
        
        result = process_single_file(video_path, output_dir, vae, window_size, skip_existing=False)
        if result == True:
            successful += 1
        else:
            failed += 1
            failed_files.append(filename)
    
    print("\n" + "="*50)
    print("Processing Complete!")
    print(f"Total files: {len(all_nii_files)}")
    print(f"Skipped: {len(already_exist_files)}")
    print(f"Processed: {len(nii_files_to_process)}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    print("="*50)

if __name__ == '__main__':
    input_dir = '/datasdc16T/qichen/blob/qichen_blob/data/CT/Task07_Pancreas/imagesTr'
    output_dir = './latent_results_wan2.2_pancreas'
    
    main_dir(input_dir, output_dir, window_size=65, skip_existing=True)