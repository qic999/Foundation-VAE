import torch
import numpy as np
import os
from einops import rearrange
import nibabel as nib
from glob import glob
from tqdm import tqdm
import traceback
from LeanVAE import LeanVAE

class LeanVAEWrapper:
    """Wrapper class for LeanVAE to provide consistent interface"""
    def __init__(self, ckpt_path, device='cuda:0', fp16=False, tile_inference=False, 
                 chunksize_enc=5, chunksize_dec=5):
        self.device = device
        self.fp16 = fp16
        self.regular_size = 2.0
        
        # Load model
        self.model = LeanVAE.load_from_checkpoint(ckpt_path, strict=False)
        self.model = self.model.half().to(device) if fp16 else self.model.to(device)
        self.model.eval()
        
        # Set tile inference if needed
        if tile_inference:
            self.model.set_tile_inference(True)
            self.model.chunksize_enc = chunksize_enc
            self.model.chunksize_dec = chunksize_dec
    
    def process_window(self, video_tensor):
        """
        Process a single video window through LeanVAE
        
        Args:
            video_tensor: Input tensor shape (c, t, h, w)
        
        Returns:
            Reconstructed tensor shape (c, t, h, w)
        """
        # Add batch dimension
        video = video_tensor.unsqueeze(0)
        
        # Normalize to LeanVAE's expected range
        video = (video * 2.0 - 1.0) / self.regular_size
        
        # Convert to half precision if needed
        if self.fp16:
            video = video.half()
        
        with torch.no_grad():
            video = video.to(self.device)
            _, x_rec = self.model.inference(video)
        
        # Remove batch dimension
        x_rec = x_rec.squeeze(0)
        
        # Denormalize
        x_rec = (torch.clamp(x_rec * self.regular_size, -1.0, 1.0) + 1.0) / 2.0
        
        return x_rec

def sliding_window_encode_decode(video, vae_wrapper, window_size=65):
    """
    使用滑窗方式对视频进行编码和解码（无重叠）- LeanVAE版本
    
    Args:
        video: 输入视频tensor, shape: (c, t, h, w), normalized to [0, 1]
        vae_wrapper: LeanVAEWrapper实例
        window_size: 滑窗大小
    
    Returns:
        重建后的视频tensor
    """
    c, t, h, w = video.shape
    device = video.device
    original_t = t  # 保存原始时间维度长度
    
    # 如果视频长度小于窗口大小，需要padding到window_size
    if t < window_size:
        pad_size = window_size - t
        video = torch.nn.functional.pad(
            video, 
            (0, 0, 0, 0, 0, pad_size),  # (left, right, top, bottom, front, back)
            mode='replicate'
        )
        # 处理padding后的视频
        results = vae_wrapper.process_window(video)
        # 返回原始长度的部分
        return results[:, :original_t, :, :]
    
    # 如果视频长度等于窗口大小，直接处理
    elif t == window_size:
        results = vae_wrapper.process_window(video)
        return results
    
    # 如果视频长度大于窗口大小，使用滑窗处理
    else:
        # 初始化输出tensor
        output = torch.zeros((c, t, h, w), device=device)
        
        # 计算窗口数量
        num_windows = (t + window_size - 1) // window_size  # 向上取整
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, t)
            actual_window_size = end_idx - start_idx
            
            # 提取当前窗口
            window = video[:, start_idx:end_idx, :, :]
            
            # 如果最后一个窗口小于window_size，需要padding
            if actual_window_size < window_size:
                # 使用边缘复制的方式进行padding
                pad_size = window_size - actual_window_size
                padded_window = torch.nn.functional.pad(
                    window, 
                    (0, 0, 0, 0, 0, pad_size), 
                    mode='replicate'
                )
            else:
                padded_window = window
            
            # 编码和解码当前窗口
            window_recon = vae_wrapper.process_window(padded_window)
            
            # 只保留实际的窗口大小部分
            output[:, start_idx:end_idx, :, :] = window_recon[:, :actual_window_size, :, :]
        
        return output

def process_single_file(video_path, save_path, vae_wrapper, window_size=65, skip_existing=True):
    """
    处理单个nii.gz文件，对z维度小于65的数据进行padding处理
    
    Args:
        video_path: 输入视频路径
        save_path: 输出保存路径
        vae_wrapper: LeanVAEWrapper实例
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
        
        # 预处理 - 使用LeanVAE的归一化方式
        video_data = np.clip(video_data, -1000.0, 1000.0)
        video_data = (video_data + 1000.0) / 2000.0  # 归一化到[-1, 1]

        
        video = torch.from_numpy(video_data)
        video = video.permute(2, 0, 1)[:, None]  # (z, 1, x, y)
        video = video.repeat(1, 3, 1, 1)  # (z, 3, x, y)
        video = rearrange(video, 't c h w -> c t h w').float()
        
        # 移动到GPU
        video = video.cuda()
        
        print(f"  Video tensor shape: {video.shape}")
        
        # 使用滑窗方式进行编码和解码（内部会处理padding）
        results = sliding_window_encode_decode(video, vae_wrapper, window_size=window_size)
        
        # 验证输出的时间维度与原始输入一致
        assert results.shape[1] == original_z_size, \
            f"Output z dimension {results.shape[1]} doesn't match original {original_z_size}"
        
        print(f"  Reconstructed shape: {results.shape}")

        # 后处理
        results = rearrange(results, 'c t h w -> c h w t')
        results = results.mean(0)  # 对通道维度取平均
        
        # 反归一化回原始范围
        results = results * 2000.0 - 1000.0
        results = torch.clamp(results, -1000.0, 1000.0)
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

def main_dir(input_dir, output_dir, ckpt_path, window_size=65, device='cuda:0', 
             fp16=False, tile_inference=False, chunksize_enc=5, chunksize_dec=5, 
             skip_existing=True):
    """
    处理目录下的所有nii.gz文件 - LeanVAE版本
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        ckpt_path: LeanVAE模型checkpoint路径
        window_size: 滑窗大小，默认65
        device: 设备，默认'cuda:0'
        fp16: 是否使用半精度
        tile_inference: 是否使用tile推理
        chunksize_enc: 编码器chunk大小
        chunksize_dec: 解码器chunk大小
        skip_existing: 是否跳过已存在的文件，默认True
    """
    # 初始化LeanVAE模型
    print("Loading LeanVAE model...")
    vae_wrapper = LeanVAEWrapper(
        ckpt_path=ckpt_path,
        device=device,
        fp16=fp16,
        tile_inference=tile_inference,
        chunksize_enc=chunksize_enc,
        chunksize_dec=chunksize_dec
    )
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
    print(f"FP16: {fp16}")
    print(f"Tile inference: {tile_inference}\n")
    
    # 处理统计
    successful = 0
    failed = 0
    failed_files = []
    
    # 处理文件
    for idx, video_path in enumerate(tqdm(nii_files_to_process, desc="Processing files")):
        filename = os.path.basename(video_path)
        print(f"\n[{idx+1}/{len(nii_files_to_process)}] Processing: {filename}")
        
        result = process_single_file(video_path, output_dir, vae_wrapper, window_size, skip_existing=False)
        
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

def main_dir_batch(input_dir, output_dir, ckpt_path, window_size=65, batch_size=4,
                   device='cuda:0', fp16=False, tile_inference=False, 
                   chunksize_enc=5, chunksize_dec=5, skip_existing=True):
    """
    批量处理版本：处理目录下的所有nii.gz文件，支持z维度padding - LeanVAE版本
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        ckpt_path: LeanVAE模型checkpoint路径
        window_size: 滑窗大小，默认65
        batch_size: 批处理大小
        device: 设备
        fp16: 是否使用半精度
        tile_inference: 是否使用tile推理
        chunksize_enc: 编码器chunk大小
        chunksize_dec: 解码器chunk大小
        skip_existing: 是否跳过已存在的文件，默认True
    """
    # 初始化LeanVAE模型
    print("Loading LeanVAE model...")
    vae_wrapper = LeanVAEWrapper(
        ckpt_path=ckpt_path,
        device=device,
        fp16=fp16,
        tile_inference=tile_inference,
        chunksize_enc=chunksize_enc,
        chunksize_dec=chunksize_dec
    )
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
    print(f"FP16: {fp16}, Tile inference: {tile_inference}\n")
    
    successful = 0
    failed = 0
    failed_files = []
    
    # 处理文件
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
            
            # 预处理 - 使用LeanVAE的归一化
            video_data = np.clip(video_data, -1000, 1000)
            video_data = (video_data + 1000.0) / 2000.0 

            
            video = torch.from_numpy(video_data)
            video = video.permute(2, 0, 1)[:, None]
            video = video.repeat(1, 3, 1, 1)
            video = rearrange(video, 't c h w -> c t h w').float()
            video = video.cuda()
            
            c, t, h, w = video.shape
            
            # 处理不同z维度的情况
            if t < window_size:
                # Padding到window_size
                pad_size = window_size - t
                padded_video = torch.nn.functional.pad(
                    video, (0, 0, 0, 0, 0, pad_size), mode='replicate'
                )
                results = vae_wrapper.process_window(padded_video)
                results = results[:, :original_z_size, :, :]
            
            elif t == window_size:
                # 直接处理
                results = vae_wrapper.process_window(video)
            
            else:
                # 批量滑窗处理
                output = torch.zeros((c, t, h, w), device=video.device)
                num_windows = (t + window_size - 1) // window_size
                
                for batch_start in range(0, num_windows, batch_size):
                    batch_end = min(batch_start + batch_size, num_windows)
                    
                    for i in range(batch_start, batch_end):
                        start_idx = i * window_size
                        end_idx = min(start_idx + window_size, t)
                        actual_size = end_idx - start_idx
                        
                        window = video[:, start_idx:end_idx, :, :]
                        
                        if actual_size < window_size:
                            pad_size = window_size - actual_size
                            window = torch.nn.functional.pad(
                                window, (0, 0, 0, 0, 0, pad_size), mode='replicate'
                            )
                        
                        # 处理单个窗口
                        window_recon = vae_wrapper.process_window(window)
                        output[:, start_idx:end_idx, :, :] = window_recon[:, :actual_size, :, :]
                
                results = output
            
            # 验证形状
            assert results.shape[1] == original_z_size, \
                f"Output z dimension {results.shape[1]} doesn't match original {original_z_size}"


            # 后处理
            results = rearrange(results, 'c t h w -> c h w t')
            results = results.mean(0)
            
            # 反归一化
            results = results * 2000.0 - 1000.0
            results = torch.clamp(results, -1000, 1000)
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

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process medical images with LeanVAE')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing .nii.gz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for reconstructed files')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to LeanVAE checkpoint')
    parser.add_argument('--window_size', type=int, default=65,
                        help='Sliding window size (default: 65)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for batch processing mode (default: 4)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--tile_inference', action='store_true',
                        help='Use tile inference')
    parser.add_argument('--chunksize_enc', type=int, default=5,
                        help='Encoder chunk size for tile inference')
    parser.add_argument('--chunksize_dec', type=int, default=5,
                        help='Decoder chunk size for tile inference')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Use batch processing mode')
    parser.add_argument('--no_skip_existing', action='store_true',
                        help='Process all files even if they already exist')
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip_existing
    
    if args.batch_mode:
        # 使用批量处理版本
        main_dir_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            ckpt_path=args.ckpt_path,
            window_size=args.window_size,
            batch_size=args.batch_size,
            device=args.device,
            fp16=args.fp16,
            tile_inference=args.tile_inference,
            chunksize_enc=args.chunksize_enc,
            chunksize_dec=args.chunksize_dec,
            skip_existing=skip_existing
        )
    else:
        # 使用基础版本
        main_dir(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            ckpt_path=args.ckpt_path,
            window_size=args.window_size,
            device=args.device,
            fp16=args.fp16,
            tile_inference=args.tile_inference,
            chunksize_enc=args.chunksize_enc,
            chunksize_dec=args.chunksize_dec,
            skip_existing=skip_existing
        )



# Basic usage
# python recon_ct_window.py --input_dir /home/v-qichen3/blob/qichen_blob/data/CT/Task06_Lung/imagesTr --output_dir reconstructed_results_leanvae_lung --ckpt_path ckpts/LeanVAE-dim16.ckpt
