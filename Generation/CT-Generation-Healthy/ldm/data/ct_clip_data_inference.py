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
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text


    def prepare_samples(self):
        samples = []
        # patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        import json

        with open('/sd/qichen/full_ct_gen/CT-RATE/valid_no_disease_2_checked.json', 'r') as f:
            items = [json.loads(line) for line in f]

        # 2. 提取所有 volume_path
        volume_paths = [item['volume_path']
                        for item in items
                        if 'volume_path' in item]
        organ_masks = [item['organ_mask']
                for item in items
                if 'organ_mask' in item]

        for nii_file, organ_mask in tqdm.tqdm(zip(volume_paths, organ_masks)):

        # for patient_folder in tqdm.tqdm(patient_folders):
        #     accession_folders = glob.glob(os.path.join(patient_folder, '*'))

        #     for accession_folder in accession_folders:
        #         nii_files = glob.glob(os.path.join(accession_folder, '*.npz'))

                # for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]
                    
                    seg_file = '/sd/shuhan/CT-RATE/'+organ_mask
                    nii_file = '/sd/shuhan/CT-RATE/'+nii_file
                    # accession_number = accession_number.replace(".npz", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]
                    text_final = ""
                    for text in list(impression_text):
                        text = str(text)
                        if text == "Not given.":
                            text = ""

                        text_final = text_final + text

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        samples.append((nii_file, seg_file, text_final, onehotlabels[0]))
                        self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)

    # def nii_img_to_tensor(self, path, transform):
    #     img_data = np.load(path)['arr_0']
    #     img_data= np.transpose(img_data, (1, 2, 0))
    #     img_data = img_data*1000
    #     hu_min, hu_max = -1000, 200
    #     img_data = np.clip(img_data, hu_min, hu_max)

    #     img_data = (((img_data+400 ) / 600)).astype(np.float32)
    #     slices=[]

    #     tensor = torch.tensor(img_data)
    #     # Get the dimensions of the input tensor
    #     target_shape = (480,480,240)
    #     # Extract dimensions
    #     h, w, d = tensor.shape

    #     # Calculate cropping/padding values for height, width, and depth
    #     dh, dw, dd = target_shape
    #     h_start = max((h - dh) // 2, 0)
    #     h_end = min(h_start + dh, h)
    #     w_start = max((w - dw) // 2, 0)
    #     w_end = min(w_start + dw, w)
    #     d_start = max((d - dd) // 2, 0)
    #     d_end = min(d_start + dd, d)

    #     # Crop or pad the tensor
    #     tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    #     pad_h_before = (dh - tensor.size(0)) // 2
    #     pad_h_after = dh - tensor.size(0) - pad_h_before

    #     pad_w_before = (dw - tensor.size(1)) // 2
    #     pad_w_after = dw - tensor.size(1) - pad_w_before

    #     pad_d_before = (dd - tensor.size(2)) // 2
    #     pad_d_after = dd - tensor.size(2) - pad_d_before

    #     tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)


    #     tensor = tensor.permute(2, 0, 1)

    #     tensor = tensor.unsqueeze(0)

    #     return tensor

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
        # # Get the dimensions of the input tensor
        # target_shape = (480,480,240)

        # # Extract dimensions
        # h, w, d = tensor.shape

        # # Calculate cropping/padding values for height, width, and depth
        # dh, dw, dd = target_shape
        # h_start = max((h - dh) // 2, 0)
        # h_end = min(h_start + dh, h)
        # w_start = max((w - dw) // 2, 0)
        # w_end = min(w_start + dw, w)
        # d_start = max((d - dd) // 2, 0)
        # d_end = min(d_start + dd, d)

        # # Crop or pad the tensor
        # tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        # pad_h_before = (dh - tensor.size(0)) // 2
        # pad_h_after = dh - tensor.size(0) - pad_h_before

        # pad_w_before = (dw - tensor.size(1)) // 2
        # pad_w_after = dw - tensor.size(1) - pad_w_before

        # pad_d_before = (dd - tensor.size(2)) // 2
        # pad_d_after = dd - tensor.size(2) - pad_d_before

        # tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        # tensor = tensor.permute(2, 0, 1)



        img_data = img_data.unsqueeze(0)
        mask_data = mask_data.unsqueeze(0)
        # example = {}
        # example['name'] = file_name
        # example['volume_data'] = tensor
        # # example['organ_mask'] = volume_seg
        # example['spacing'] = target_spacing
        return img_data, mask_data, target_spacing, file_name

    def __getitem__(self, index):
        nii_file, seg_file, input_text, onehotlabels = self.samples[index]
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
        example['onehotlabels'] = onehotlabels
        return example
