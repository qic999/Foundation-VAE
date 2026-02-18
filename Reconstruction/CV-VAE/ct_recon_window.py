import torch
import numpy as np
import os
from einops import rearrange
import nibabel as nib
from glob import glob
from tqdm import tqdm
import traceback
from models.modeling_vae import CVVAEModel

def sliding_window_encode_decode(video, vae3d, window_size=64):
    """
    使用滑窗方式对视频进行编码和解码（无重叠）
    适配CV-VAE模型
    
    Args:
        video: 输入视频tensor, shape: (b, c, t, h, w)
        vae3d: CV-VAE模型
        window_size: 滑窗大小
    
    Returns:
        重建后的视频tensor
    """
    b, c, t, h, w = video.shape
    device = video.device
    original_t = t  # 保存原始时间维度长度
    
    # 如果视频长度小于窗口大小，需要padding到window_size
    if t < window_size:
        pad_size = window_size - t
        # 正确的padding顺序: (left, right, top, bottom, front, back) for 5D tensor
        video = torch.nn.functional.pad(
            video, 
            (0, 0, 0, 0, 0, pad_size),  # 只在时间维度padding
            mode='replicate'
        )
        # 处理padding后的视频
        with torch.no_grad():
            latent = vae3d.encode(video).latent_dist.sample()
            results = vae3d.decode(latent).sample
        # 返回原始长度的部分
        return results[:, :, :original_t, :, :]
    
    # 如果视频长度等于窗口大小，直接处理
    elif t == window_size:
        with torch.no_grad():
            latent = vae3d.encode(video).latent_dist.sample()
            results = vae3d.decode(latent).sample
        return results
    
    # 如果视频长度大于窗口大小，使用滑窗处理
    else:
        # 初始化输出tensor - 在相同设备上
        output = torch.zeros((b, c, t, h, w), device=device, dtype=video.dtype)
        
        # 计算窗口数量
        num_windows = (t + window_size - 1) // window_size  # 向上取整
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, t)
            actual_window_size = end_idx - start_idx
            
            # 提取当前窗口
            window = video[:, :, start_idx:end_idx, :, :]
            
            # 如果最后一个窗口小于window_size，需要padding
            if actual_window_size < window_size:
                # 使用边缘复制的方式进行padding
                pad_size = window_size - actual_window_size
                padded_window = torch.nn.functional.pad(
                    window, 
                    (0, 0, 0, 0, 0, pad_size),  # 只在时间维度padding
                    mode='replicate'
                )
            else:
                padded_window = window
            
            # 编码和解码当前窗口
            with torch.no_grad():
                latent = vae3d.encode(padded_window).latent_dist.sample()
                window_recon = vae3d.decode(latent).sample
            
            # 只保留实际的窗口大小部分
            output[:, :, start_idx:end_idx, :, :] = window_recon[:, :, :actual_window_size, :, :]
        
        return output

def process_single_file(video_path, save_path, vae3d, window_size=64, skip_existing=True):
    """
    处理单个nii.gz文件，对z维度小于window_size的数据进行padding处理
    
    Args:
        video_path: 输入视频路径
        save_path: 输出保存路径
        vae3d: CV-VAE模型
        window_size: 滑窗大小
        skip_existing: 是否跳过已存在的文件
    
    Returns:
        True if successful, False otherwise, 'skipped' if file exists
    """
    try:
        # 检查目标文件是否已存在
        name = os.path.basename(video_path).replace('.nii.gz', '')
        data_path = os.path.join(save_path, f'{name}.nii.gz')
        
        if skip_existing and os.path.exists(data_path):
            print(f"  ⊙ File already exists, skipping: {data_path}")
            return 'skipped'
        
        # 加载和预处理视频
        video_nib = nib.load(video_path)
        affine = video_nib.affine
        video_data = video_nib.get_fdata()
        
        # 记录原始z维度大小
        original_z_size = video_data.shape[2]
        
        print(f"  Original shape: {video_data.shape}")
        print(f"  Z dimension: {original_z_size}")
        
        # 预处理 - 使用CV-VAE的预处理方式
        video_data = np.clip(video_data, -1000, 1000)
        video_data = (video_data + 1000.0) / 2000.0
        
        video = torch.from_numpy(video_data)
        video = video.permute(2, 0, 1)[:, None]  # (z, 1, x, y)
        video = video.repeat(1, 3, 1, 1)  # (z, 3, x, y)
        video = rearrange(video, 't c h w -> c t h w').unsqueeze(0).half()  # 使用half precision
        video = video * 2.0 - 1.0  # 归一化到[-1, 1]
        
        # 移动到GPU
        video = video.cuda()
        
        print(f"  Video tensor shape: {video.shape}")
        
        # 使用滑窗方式进行编码和解码
        results = sliding_window_encode_decode(video, vae3d, window_size=window_size)
        
        # 验证输出的时间维度与原始输入一致
        assert results.shape[2] == original_z_size, \
            f"Output z dimension {results.shape[2]} doesn't match original {original_z_size}"
        
        print(f"  Reconstructed shape: {results.shape}")
        
        # 后处理 - 使用CV-VAE的后处理方式
        results = rearrange(results.squeeze(0), 'c t h w -> c h w t')
        results = results.mean(0)  # 对通道维度取平均
        results = (torch.clamp(results, -1.0, 1.0) + 1.0) / 2.0
        results = results * 2000.0 - 1000.0
        results = results.cpu().numpy()
        
        # 验证最终输出形状
        assert results.shape == (video_data.shape[0], video_data.shape[1], original_z_size), \
            f"Final output shape {results.shape} doesn't match expected shape"
        
        # 保存结果
        data_nii = nib.Nifti1Image(results.astype(np.int16), affine)
        nib.save(data_nii, data_path)
        
        print(f"  ✓ Saved to: {data_path}")
        print(f"  ✓ Final shape: {results.shape}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {video_path}: {str(e)}")
        traceback.print_exc()
        return False

def main_dir(input_dir, output_dir, model_path='AILab-CVC/CV-VAE', window_size=64, skip_existing=True):
    """
    处理目录下的所有nii.gz文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        model_path: CV-VAE模型路径，默认'AILab-CVC/CV-VAE'
        window_size: 滑窗大小，默认64
        skip_existing: 是否跳过已存在的文件，默认True
    """
    # 初始化CV-VAE模型
    print(f"Loading CV-VAE model from {model_path}...")
    vae3d = CVVAEModel.from_pretrained(model_path, subfolder="vae3d", torch_dtype=torch.float16)
    vae3d.requires_grad_(False)
    vae3d = vae3d.cuda()
    vae3d.eval()
    print("Model loaded successfully!\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有nii.gz文件
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
            data_path = os.path.join(output_dir, f'{name}.nii.gz')
            if os.path.exists(data_path):
                already_exist_files.append(os.path.basename(video_path))
            else:
                nii_files_to_process.append(video_path)
        
        print(f"Already processed (skipping): {len(already_exist_files)} files")
        print(f"To be processed: {len(nii_files_to_process)} files")
        
        if already_exist_files and len(already_exist_files) <= 10:
            print("\nSkipped files:")
            for f in already_exist_files[:10]:
                print(f"  ⊙ {f}")
            if len(already_exist_files) > 10:
                print(f"  ... and {len(already_exist_files) - 10} more")
    else:
        nii_files_to_process = all_nii_files
        print(f"Processing all {len(nii_files_to_process)} files (skip_existing=False)")
    
    if not nii_files_to_process:
        print("\nAll files have been processed! Nothing to do.")
        return
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Window size: {window_size}")
    print(f"Model: {model_path}\n")
    
    # 处理统计
    successful = 0
    failed = 0
    failed_files = []
    
    # 处理文件
    for idx, video_path in enumerate(tqdm(nii_files_to_process, desc="Processing files")):
        filename = os.path.basename(video_path)
        print(f"\n[{idx+1}/{len(nii_files_to_process)}] Processing: {filename}")
        
        result = process_single_file(video_path, output_dir, vae3d, 
                                    window_size=window_size, skip_existing=False)
        
        if result == True:
            successful += 1
        else:
            failed += 1
            failed_files.append(filename)
    
    # 打印最终统计
    print("\n" + "="*50)
    print("Processing Complete!")
    print(f"Total files in directory: {len(all_nii_files)}")
    if skip_existing and already_exist_files:
        print(f"Previously processed (skipped): {len(already_exist_files)}")
    print(f"Processed in this run: {len(nii_files_to_process)}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print("="*50)

def main_dir_batch(input_dir, output_dir, model_path='AILab-CVC/CV-VAE', window_size=64,
                  batch_size=4, skip_existing=True):
    """
    批量处理版本：处理目录下的所有nii.gz文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        model_path: CV-VAE模型路径
        window_size: 滑窗大小，默认64
        batch_size: 批处理大小
        skip_existing: 是否跳过已存在的文件，默认True
    """
    # 初始化CV-VAE模型
    print(f"Loading CV-VAE model from {model_path}...")
    vae3d = CVVAEModel.from_pretrained(model_path, subfolder="vae3d", torch_dtype=torch.float16)
    vae3d.requires_grad_(False)
    vae3d = vae3d.cuda()
    vae3d.eval()
    print("Model loaded successfully!\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有nii.gz文件
    all_nii_files = glob(os.path.join(input_dir, '*.nii.gz'))
    
    if not all_nii_files:
        print(f"No .nii.gz files found in {input_dir}")
        return
    
    print(f"Found {len(all_nii_files)} .nii.gz files")
    
    # 过滤出需要处理的文件
    nii_files_to_process = []
    already_exist_files = []
    
    if skip_existing:
        for video_path in all_nii_files:
            name = os.path.basename(video_path).replace('.nii.gz', '')
            data_path = os.path.join(output_dir, f'{name}.nii.gz')
            if os.path.exists(data_path):
                already_exist_files.append(os.path.basename(video_path))
            else:
                nii_files_to_process.append(video_path)
        
        print(f"Already processed (skipping): {len(already_exist_files)} files")
        print(f"To be processed: {len(nii_files_to_process)} files")
    else:
        nii_files_to_process = all_nii_files
        print(f"Processing all {len(nii_files_to_process)} files (skip_existing=False)")
    
    if not nii_files_to_process:
        print("\nAll files have been processed! Nothing to do.")
        return
    
    print(f"Processing with batch_size={batch_size}, window_size={window_size}")
    print(f"Model: {model_path}\n")
    
    successful = 0
    failed = 0
    failed_files = []
    
    # 批量处理文件
    for idx, video_path in enumerate(tqdm(nii_files_to_process, desc="Processing files")):
        filename = os.path.basename(video_path)
        
        try:
            name = os.path.basename(video_path).replace('.nii.gz', '')
            data_path = os.path.join(output_dir, f'{name}.nii.gz')
            
            # 加载视频
            video_nib = nib.load(video_path)
            affine = video_nib.affine
            video_data = video_nib.get_fdata()
            
            # 记录原始z维度
            original_z_size = video_data.shape[2]
            
            # 预处理 - 使用CV-VAE的预处理方式
            video_data = np.clip(video_data, -1000, 1000)
            video_data = (video_data + 1000.0) / 2000.0
            
            video = torch.from_numpy(video_data)
            video = video.permute(2, 0, 1)[:, None]
            video = video.repeat(1, 3, 1, 1)
            video = rearrange(video, 't c h w -> c t h w').unsqueeze(0).half()
            video = video * 2.0 - 1.0
            video = video.cuda()
            
            b, c, t, h, w = video.shape
            
            # 处理z维度小于window_size的情况
            if t < window_size:
                # Padding到window_size
                pad_size = window_size - t
                padded_video = torch.nn.functional.pad(
                    video, (0, 0, 0, 0, 0, pad_size), mode='replicate'
                )
                with torch.no_grad():
                    latent = vae3d.encode(padded_video).latent_dist.sample()
                    results = vae3d.decode(latent).sample
                # 取原始长度
                results = results[:, :, :original_z_size, :, :]
            
            elif t == window_size:
                # 直接处理
                with torch.no_grad():
                    latent = vae3d.encode(video).latent_dist.sample()
                    results = vae3d.decode(latent).sample
            
            else:
                # 批量滑窗处理
                output = torch.zeros((b, c, t, h, w), device=video.device, dtype=video.dtype)
                num_windows = (t + window_size - 1) // window_size
                
                for batch_start in range(0, num_windows, batch_size):
                    batch_end = min(batch_start + batch_size, num_windows)
                    windows = []
                    window_infos = []
                    
                    for i in range(batch_start, batch_end):
                        start_idx = i * window_size
                        end_idx = min(start_idx + window_size, t)
                        actual_size = end_idx - start_idx
                        
                        window = video[:, :, start_idx:end_idx, :, :]
                        
                        if actual_size < window_size:
                            pad_size = window_size - actual_size
                            window = torch.nn.functional.pad(
                                window, (0, 0, 0, 0, 0, pad_size), mode='replicate'
                            )
                        
                        windows.append(window)
                        window_infos.append((start_idx, end_idx, actual_size))
                    
                    # 拼接批次并处理
                    if windows:
                        batch_windows = torch.cat(windows, dim=0)
                        with torch.no_grad():
                            latents = vae3d.encode(batch_windows).latent_dist.sample()
                            recons = vae3d.decode(latents).sample
                        
                        # 分割结果并放回对应位置
                        for j, (start_idx, end_idx, actual_size) in enumerate(window_infos):
                            output[:, :, start_idx:end_idx, :, :] = recons[j:j+1, :, :actual_size, :, :]
                
                results = output
            
            # 验证形状
            assert results.shape[2] == original_z_size, \
                f"Output z dimension {results.shape[2]} doesn't match original {original_z_size}"
            
            # 后处理
            results = rearrange(results.squeeze(0), 'c t h w -> c h w t')
            results = results.mean(0)
            results = (torch.clamp(results, -1.0, 1.0) + 1.0) / 2.0
            results = results * 2000.0 - 1000.0
            results = results.cpu().numpy()
            
            # 保存
            data_nii = nib.Nifti1Image(results.astype(np.int16), affine)
            nib.save(data_nii, data_path)
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            traceback.print_exc()
            failed += 1
            failed_files.append(filename)
    
    # 打印统计
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"Total files in directory: {len(all_nii_files)}")
    if skip_existing and already_exist_files:
        print(f"Previously processed (skipped): {len(already_exist_files)}")
    print(f"Processed in this run: {len(nii_files_to_process)}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    
    if failed_files:
        print("\nFailed files:", failed_files)

def process_single_file_simple(video_path, save_path, model_path='AILab-CVC/CV-VAE'):
    """
    简单版本：处理单个文件，不使用滑窗（用于调试）
    """
    print(f"Processing {video_path}...")
    
    # 初始化模型
    vae3d = CVVAEModel.from_pretrained(model_path, subfolder="vae3d", torch_dtype=torch.float16)
    vae3d.requires_grad_(False)
    vae3d = vae3d.cuda()
    vae3d.eval()
    
    os.makedirs(save_path, exist_ok=True)
    
    # 加载数据
    video = nib.load(video_path)
    affine = video.affine
    video = video.get_fdata()
    video = np.clip(video, -1000, 2000)
    video = (video + 1000.0) / 2000.0
    
    video = torch.from_numpy(video)
    video = video.permute(-1, 0, 1)[:, None]
    video = video.repeat(1, 3, 1, 1)
    video = rearrange(video, 't c h w -> c t h w').unsqueeze(0).half()
    video = video * 2.0 - 1.0
    video = video.cuda()
    
    print(f'Shape of input video: {video.shape}')
    
    with torch.no_grad():
        latent = vae3d.encode(video).latent_dist.sample()
        print(f'Shape of video latent: {latent.shape}')
        results = vae3d.decode(latent).sample
    
    results = rearrange(results.squeeze(0), 'c t h w -> c h w t')
    results = results.mean(0)
    results = (torch.clamp(results, -1.0, 1.0) + 1.0) / 2.0
    results = results * 2000.0 - 1000.0
    results = results.cpu().numpy()
    
    print(f'Shape of output: {results.shape}')
    
    name = os.path.basename(video_path).split('.')[0]
    data_path = os.path.join(save_path, f'{name}.nii.gz')
    data_nii = nib.Nifti1Image(results.astype(np.int16), affine)
    nib.save(data_nii, data_path)
    print(f"Saved to {data_path}")

if __name__ == '__main__':
    # 配置参数
    input_dir = '/home/v-qichen3/blob/qichen_blob/data/CT/Task06_Lung/imagesTr'
    output_dir = './reconstructed_results_cvvae_lung'
    model_path = 'AILab-CVC/CV-VAE'  # 或本地路径
    
    # 使用基础版本处理目录（默认跳过已存在的文件）
    main_dir(input_dir, output_dir, model_path=model_path, 
            window_size=65, skip_existing=True)
    
    # 如果想要重新处理所有文件（不跳过已存在的）
    # main_dir(input_dir, output_dir, model_path=model_path,
    #         window_size=64, skip_existing=False)
    
    # 或者使用批量处理版本
    # main_dir_batch(input_dir, output_dir, model_path=model_path,
    #               window_size=64, batch_size=4, skip_existing=True)
    
    # 处理单个文件（简单版本，无滑窗）
    # process_single_file_simple(
    #     video_path='./path/to/single/file.nii.gz',
    #     save_path='./output',
    #     model_path=model_path
    # )