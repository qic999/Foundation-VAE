import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm
import copy
from collections import defaultdict

def merge_list_elements(data):
    """
    data: list of lists，每个元素长度为7
    返回: 合并后的list，仍然是7列
    """
    merged = defaultdict(lambda: [[] for _ in range(5)])

    for item in data:
        key = (item[0], item[1])  # 前两个元素作为key
        for i in range(5):  # 后面五列分别合并
            merged[key][i].append(item[2+i])

    # 重新组装结果
    result = []
    for (a, b), rest in merged.items():
        result.append([a, b] + rest)
    return result
    

def build_multiclass_mask(
    disease_files,
    disease_mask_channels,
    disease_class_ids,
    *,
    channel_axis=0,
    threshold=0.5,
    out_dtype=np.float32,
):
    """
    将 (file, channel, class_id) 成对列表合并为单通道多类别mask（无重叠假设）。
    背景=0，其它为对应 class_id。

    参数
    ----
    disease_files : List[str]
        NIfTI 文件路径列表
    disease_mask_channels : List[int]
        通道索引列表（与 disease_files 一一对应）
    disease_class_ids : List[int]
        每个 (file, channel) 对应的最终类别 id
    channel_axis : int
        通道所在维度（默认 0）
    threshold : float
        二值化阈值；mask > threshold 视为前景
    out_dtype : numpy dtype
        输出标签图的数据类型，默认 uint16
    """
    
    if not (len(disease_files) == len(disease_mask_channels) == len(disease_class_ids)):
        raise ValueError("disease_files、disease_mask_channels、disease_class_ids 长度必须一致")

    # 用第一份确定空间形状
    first_img = nib.load(str(disease_files[0]))
    first_data = first_img.get_fdata(dtype=np.float32)
    sample_mask = np.take(first_data, int(disease_mask_channels[0]), axis=channel_axis)
    spatial_shape = sample_mask.shape

    label_map = np.zeros(spatial_shape, dtype=out_dtype)

    for fpath, ch, cid in zip(disease_files, disease_mask_channels, disease_class_ids):
        img = nib.load(str(fpath))
        data = img.get_fdata(dtype=np.float32)
        # breakpoint()
        mask = np.take(data, int(ch), axis=channel_axis)

        if mask.shape != spatial_shape:
            raise ValueError(
                f"{fpath} 的 mask 形状 {mask.shape} 与期望 {spatial_shape} 不一致，请先重采样"
            )

        # 二值化并赋类标
        label_map[mask > threshold] = cid

    return label_map

def resize_array(array, current_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    
    
    new_shape = [original_shape[0], 512, 512]
    scaling_factors = [new_shape[i] / original_shape[i] for i in range(len(original_shape))]
    resized_spacing = [current_spacing[i] / scaling_factors[i] for i in range(len(original_shape))]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    # breakpoint()
    return resized_array, resized_spacing

def resize_mask(array, current_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]

    new_shape = [original_shape[0], 512, 512]

    resized_array = F.interpolate(array, size=new_shape, mode='nearest').cpu().numpy()
    # breakpoint()
    return resized_array

class CTReportDataset(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        percent = 80
        num_files = int((len(self.samples) * percent) / 100)
        #num_files = 2286
        self.samples = self.samples[:num_files]
        print(len(self.samples))
        self.count = 0


        #self.resize_dim = resize_dim
        #self.resize_transform = transforms.Resize((resize_dim, resize_dim))
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        self.sample_length=65

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            # breakpoint()
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']

        return accession_to_text


    def prepare_samples(self):
        samples = []
        import json

        with open('/sd/qichen/full_ct_gen/CT-RATE/disease_mask_json/disease_train_single_prompt_checked_label.json', 'r') as f:
            items = [json.loads(line) for line in f]

        # 2. 提取所有 volume_path
        volume_paths = [item['volume_path']
                        for item in items
                        if 'volume_path' in item]

        effusion_mask_paths = [item['disease_mask']
                        for item in items
                        if 'disease_mask' in item]

        organ_mask_paths = [item['organ_mask']
                        for item in items
                        if 'organ_mask' in item]

        disease_findings_list = [item['disease_findings']
                        for item in items
                        if 'disease_findings' in item]

        disease_mask_channels = [item['disease_mask_channel']
                        for item in items
                        if 'disease_mask_channel' in item]
        disease_labels = [item['disease_label']
                        for item in items
                        if 'disease_label' in item]
        disease_classes =  [item['disease_class']
                        for item in items
                        if 'disease_class' in item]
        
        for (ct_file, organ_mask_file, disease_mask_file, disease_findings, disease_mask_channel, disease_label, disease_class) in tqdm.tqdm(zip(volume_paths, organ_mask_paths, effusion_mask_paths, disease_findings_list, disease_mask_channels, disease_labels, disease_classes)):

            accession_number = organ_mask_file.split("/")[-1]

            ct_file = '/sd/shuhan/CT-RATE/'+ct_file
            organ_file = '/sd/shuhan/CT-RATE/'+organ_mask_file
            disease_file = '/sd/shuhan/CT-RATE/'+disease_mask_file

            impression_text = disease_findings

            samples.append((ct_file, organ_file, disease_file, impression_text, disease_mask_channel, disease_label, disease_class))

        samples = merge_list_elements(samples)

        with open('/sd/qichen/full_ct_gen/CT-RATE/train_no_disease_2_checked.json', 'r') as f:
            items = [json.loads(line) for line in f]

        # 2. 提取所有 volume_path
        volume_paths = [item['volume_path']
                        for item in items
                        if 'volume_path' in item]
        organ_masks = [item['organ_mask']
                        for item in items
                        if 'organ_mask' in item]
        for nii_file, organ_mask in tqdm.tqdm(zip(volume_paths, organ_masks)):

            accession_number = nii_file.split("/")[-1]

            seg_file = '/sd/shuhan/CT-RATE/'+ organ_mask
            nii_file = '/sd/shuhan/CT-RATE/'+nii_file

            if accession_number not in self.accession_to_text:
                continue

            impression_text = self.accession_to_text[accession_number]

            if impression_text == "Not given.":
                impression_text=""

            input_text_concat = ""
            for text in impression_text:
                input_text_concat = input_text_concat + str(text)
            input_text_concat = impression_text[0]
            input_text = f'{impression_text}'
            samples.append((nii_file, seg_file, None, input_text_concat, None, None, None))

        return samples

    def __len__(self):
        return len(self.samples)


    def nii_img_to_tensor(self, ct_file, organ_file, disease_file, disease_mask_channel, disease_label, disease_class, transform):
        if disease_file == None:

            nii_img = nib.load(str(ct_file))
            img_data = nii_img.get_fdata()

            df = pd.read_csv("/sd/shuhan/CT-RATE/metadata/all_metadata.csv") #select the metadata
            file_name = ct_file.split("/")[-1]
            row = df[df['VolumeName'] == file_name]
            slope = float(row["RescaleSlope"].iloc[0])
            intercept = float(row["RescaleIntercept"].iloc[0])
            xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
            z_spacing = float(row["ZSpacing"].iloc[0])

            nii_seg = nib.load(str(organ_file))
            mask_data = nii_seg.get_fdata()

            current = (z_spacing, xy_spacing, xy_spacing)

            img_data = img_data.transpose(2, 0, 1)
            tensor = torch.tensor(img_data)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            img_data, target_spacing = resize_array(tensor, current)
            img_data = img_data[0][0]

            mask_data = mask_data.transpose(2, 0, 1)
            tensor = torch.tensor(mask_data)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            mask_data = resize_mask(tensor, current)
            mask_data = mask_data[0][0]
            # breakpoint()
            assert mask_data.shape == img_data.shape
            fg_mask = (mask_data>0).astype(np.uint8)
            mask_data = (((mask_data ) / 300)).astype(np.float32) * 2 -1
            
            hu_min, hu_max = -1000, 1000
            img_data = np.clip(img_data, hu_min, hu_max)

            bg_np = np.ones_like(img_data) * -1000
            
            img_data = img_data*fg_mask + bg_np*(1-fg_mask)
            
            
            img_data = (((img_data ) / 1000)).astype(np.float32)
            start_id = np.random.randint(0, img_data.shape[0]-self.sample_length)
            img_data = img_data[start_id:start_id+self.sample_length]
            mask_data = mask_data[start_id:start_id+self.sample_length]

            # slices=[]
            img_data = img_data/2.0
            mask_data = mask_data/2.0

            img_data = torch.tensor(img_data)
            mask_data = torch.tensor(mask_data)


            img_data = img_data.unsqueeze(0)
            mask_data = mask_data.unsqueeze(0)

            return img_data, mask_data, target_spacing, file_name
        else:
            ct_nii = nib.load(str(ct_file))
            img_data = ct_nii.get_fdata()

            df = pd.read_csv("/sd/shuhan/CT-RATE/metadata/all_metadata.csv") #select the metadata
            file_name = ct_file.split("/")[-1]
            row = df[df['VolumeName'] == file_name]
            slope = float(row["RescaleSlope"].iloc[0])
            intercept = float(row["RescaleIntercept"].iloc[0])
            xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
            z_spacing = float(row["ZSpacing"].iloc[0])

            current = (z_spacing, xy_spacing, xy_spacing)
            img_data = img_data.transpose(2, 0, 1)
            tensor = torch.tensor(img_data)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            img_data, target_spacing = resize_array(tensor, current)
            img_data = img_data[0][0]

            organ_nii = nib.load(str(organ_file))
            organ_mask = organ_nii.get_fdata()
            organ_mask = organ_mask.transpose(2, 0, 1)
            tensor = torch.tensor(organ_mask)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            organ_mask = resize_mask(tensor, current)
            organ_mask = organ_mask[0][0]

            # disease_nii = nib.load(str(disease_file))
            disease_mask = build_multiclass_mask(disease_file,disease_mask_channel,disease_class)
            # disease_mask = disease_nii.get_fdata()[int(disease_mask_channel)]
            disease_mask = disease_mask.transpose(2, 0, 1)
            tensor = torch.tensor(disease_mask)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            disease_mask = resize_mask(tensor, current)
            disease_mask = disease_mask[0][0]

            assert organ_mask.shape == img_data.shape

            num_slices = img_data.shape[0]

            # 找到 mask>0 的切片索引
            valid_indices = np.where(disease_mask.sum(axis=(1, 2)) > 0)[0]

            if len(valid_indices) > 0:
                # 只保留能完整取 sample_length 的起点
                valid_start_ids = valid_indices[valid_indices <= num_slices - self.sample_length]

                if len(valid_start_ids) > 0:
                    start_id = np.random.choice(valid_start_ids)
                else:
                    # 如果没有合法起点就退化到随机，但保证能取 sample_length
                    start_id = np.random.randint(0, num_slices - self.sample_length + 1)
            else:
                # 如果 mask 全部为0，则完全随机
                start_id = np.random.randint(0, num_slices - self.sample_length + 1)

            end_id = start_id + self.sample_length  # 保证长度严格等于 sample_length
            img_data = img_data[start_id:end_id]
            organ_mask = organ_mask[start_id:end_id]
            disease_mask = disease_mask[start_id:end_id]
            # img_data = img_data[start_id]
            # organ_mask = organ_mask[start_id]
            # disease_mask = disease_mask[start_id]

            organ_disease_mask = copy.deepcopy(organ_mask)
            # organ_disease_mask[disease_mask>0] = disease_class+100
            organ_disease_mask[disease_mask!=0] = disease_mask[disease_mask != 0]+100

            organ_mask = (((organ_mask ) / 300)).astype(np.float32) * 2 -1
            organ_disease_mask = (((organ_disease_mask ) / 300)).astype(np.float32) * 2 -1

            hu_min, hu_max = -1000, 1000
            img_data = np.clip(img_data, hu_min, hu_max)
            img_data = (((img_data ) / 1000)).astype(np.float32)

            img_data = torch.tensor(img_data)
            organ_mask = torch.tensor(organ_mask)
            organ_disease_mask = torch.tensor(organ_disease_mask)

            img_data = img_data/2.0
            organ_mask = organ_mask/2.0
            organ_disease_mask = organ_disease_mask/2.0

            img_data = img_data.unsqueeze(0)
            organ_mask = organ_mask.unsqueeze(0)
            organ_disease_mask = organ_disease_mask.unsqueeze(0)


            # img_data=img_data.repeat(3,1,1,1)
            # organ_mask=organ_mask.repeat(3,1,1,1)
            # organ_disease_mask=organ_disease_mask.repeat(3,1,1,1)


            # return img_data, organ_mask, organ_disease_mask, target_spacing, file_name
            # return img_data, organ_mask, target_spacing, file_name
            return img_data, organ_disease_mask, target_spacing, file_name



    def __getitem__(self, index):
        ct_file,  organ_file, disease_file, input_text, disease_mask_channel, disease_label, disease_class = self.samples[index]
        video_tensor, volume_seg, spacing, file_name = self.nii_to_tensor(ct_file, organ_file, disease_file, disease_mask_channel, disease_label, disease_class)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        # return video_tensor, input_text
        example = {}
        example['name'] = file_name
        example['volume_data'] = video_tensor.float()
        example['volume_seg'] = volume_seg.float()
        example['spacing'] = spacing
        example['input_text'] = input_text
        return example