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
import tqdm
import nibabel as nib

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

class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, labels = "labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
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
            accession_to_text[row['Names']] = row["Text_prompts"]
        return accession_to_text


    def prepare_samples(self):
        samples = []
        # patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        for nii_file in tqdm.tqdm(self.accession_to_text.keys()):

            name = nii_file.split("/")[-1]
            
            nii_file = os.path.join('/sd/shuhan/CT-RATE/dataset/train_fixed', name.split('_')[0]+'_'+name.split('_')[1], name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2],name)
            seg_file = os.path.join('/sd/shuhan/CT-RATE/organ_mask_whole/train',name)
            # nii_file = os.path.join('/sd/shuhan/CT-RATE/dataset/train_fixed', name.split('_')[0]+'_'+name.split('_')[1], name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2],name)
            # seg_file = os.path.join('/sd/shuhan/CT-RATE/organ_mask_whole/train',name)
            if not os.path.exists(seg_file):
                continue

            impression_text = self.accession_to_text[name]
            text_final = ""
            for text in list(impression_text):
                text = str(text)
                if text == "Not given.":
                    text = ""

                text_final = text_final + text

            samples.append((nii_file, seg_file, text_final))
            self.paths.append(nii_file)
            # breakpoint()
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, seg_file, transform):
        # print('path', path)
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        # try:
        #     nii_img = nib.load(str(path))
        #     img_data = nii_img.get_fdata()
        # except Exception as e:
        #     print(f"[LoadError] Failed on: {path} -> {e}")

        df = pd.read_csv("/sd/shuhan/CT-RATE/metadata/all_metadata.csv") #select the metadata
        file_name = path.split("/")[-1]
        row = df[df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        nii_seg = nib.load(str(seg_file))
        mask_data = nii_seg.get_fdata()

        current = (z_spacing, xy_spacing, xy_spacing)

        # breakpoint()
        # img_data = slope * img_data + intercept

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
        mask_data = (((mask_data ) / 400)).astype(np.float32) * 2 -1

        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        bg_np = np.ones_like(img_data) * -1000
        
        img_data = img_data*fg_mask + bg_np*(1-fg_mask)
        
        
        img_data = (((img_data ) / 1000)).astype(np.float32)
        start_id = np.random.randint(0, img_data.shape[0]-self.sample_length)
        # img_data = img_data[start_id:start_id+self.sample_length]
        # mask_data = mask_data[start_id:start_id+self.sample_length]

        # slices=[]

        img_data = img_data/2.0
        mask_data = mask_data/2.0

        img_data = torch.tensor(img_data)
        mask_data = torch.tensor(mask_data)


        img_data = img_data.unsqueeze(0)
        mask_data = mask_data.unsqueeze(0)
        # example = {}
        # example['name'] = file_name
        # example['volume_data'] = tensor
        # # example['organ_mask'] = volume_seg
        # example['spacing'] = target_spacing
        return img_data, mask_data, target_spacing, file_name

    def __getitem__(self, index):
        nii_file, seg_file, input_text = self.samples[index]
        video_tensor, volume_seg, spacing, file_name = self.nii_to_tensor(nii_file, seg_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  
        name_acc = nii_file.split("/")[-2]

        # return video_tensor, input_text, onehotlabels, name_acc
        example = {}
        example['name'] = file_name
        example['volume_data'] = video_tensor.float()
        example['volume_seg'] = volume_seg.float()
        example['spacing'] = spacing
        example['input_text'] = input_text
        return example
