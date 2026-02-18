import os
import numpy as np
from skimage.transform import resize
import cv2
from torch.utils.data import Dataset
import copy
# from batchgenerators.utilities.file_and_folder_operations import load_json

import random
import itertools

def get_random_classes_from_mask(mask):
    # 分析 mask 中有哪些类别
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]  # 去除背景类（假设背景类为0）
    
    # 获取所有可能的类别组合
    all_combinations = []
    for i in range(1, len(unique_classes) + 1):
        all_combinations.extend(itertools.combinations(unique_classes, i))
    
    # 随机选择一个组合
    selected_combination = random.choice(all_combinations)
    return selected_combination

def get_filtered_mask(mask, selected_classes):
    # 创建只包含选定类别的 mask
    filtered_mask = np.isin(mask, selected_classes) * mask
    return filtered_mask
    
class slice_base(Dataset):
    def __init__(self,
                 data_root,
                 data_name,
                 data_repeat=1,
                 phase='train',
                 planner='nnUNetPlans_3d_fullres',
                 data_file=None
                 ):

        self.data_root = data_root
        # planner = 'nnUNetResEncUNetLPlans_torchres_3d_fullres' # nnUNetResEncUNetLPlans_torchres_3d_fullres nnUNetPlans_3d_fullres
        data_root = os.path.join(data_root, data_name , planner) 
        # self.list = [os.path.join(data_name, planner ,f) for f in os.listdir(data_root) if f.endswith('.npz')]

        data_splits = load_json(data_file)
        file_list = data_splits[0][phase]
        # breakpoint()
        self.list = [os.path.join(data_name, planner , i+'.npz') for i in file_list]
        
        self._length = len(self.list)
        self._data_repeat = data_repeat
        self.class_num = 28

    def __len__(self):
        return self._length * self._data_repeat

    def __getitem__(self, i):
        i = i % self._length
        
        npz_file = os.path.join(self.data_root, self.list[i])
        data_npz = np.load(npz_file)['data'][0] # np.load(npz_file)['seg']
        seg_npz = np.load(npz_file)['seg'][0]
        # print(npz_file, data_npz.shape, seg_npz.shape)
        # breakpoint()
        pos_id = np.random.randint(0, data_npz.shape[0])
        slice_data = data_npz[pos_id]
        slice_seg = seg_npz[pos_id]

        # breakpoint()
        slice_data[slice_data <= -175] = -175
        slice_data[slice_data >= 250] = 250
        slice_data = (slice_data+175)/425.0

        # print(np.unique(slice_seg))
        if len(np.unique(slice_seg)) > 1:
            selected_classes = get_random_classes_from_mask(slice_seg)
            slice_seg = get_filtered_mask(slice_seg, selected_classes)
        # print(np.unique(slice_seg))

        # breakpoint()

        # slice_data = resize(slice_data, (512,512))
        # slice_seg = resize(slice_seg.astype(np.float32), (512,512), order=0, anti_aliasing=False)
        # breakpoint()

        # tumor_mask = (slice_seg==2).astype(np.float32)
        slice_seg = slice_seg.astype(np.float32)
        foreground_mask = (slice_seg>0).astype(np.float32)
        masked_data = (1-foreground_mask)*slice_data

        slice_data = slice_data[:,:,None].astype(np.float32)
        masked_data = masked_data[:,:,None].astype(np.float32)
        
        # breakpoint()
        # tumor_mask = resize(tumor_mask, (64,64), order=0)
        slice_seg = cv2.resize(slice_seg,(64,64),interpolation=cv2.INTER_NEAREST)
        slice_seg = slice_seg[:,:,None].astype(np.float32)
        if (slice_data.shape[0] != 512) or (slice_data.shape[1] != 512):
            print(self.list[i],slice_data.shape,masked_data.shape,slice_seg.shape)
        # print(np.unique(slice_seg))
        # breakpoint()
        example = {}
        example['name'] = self.list[i].split('/')[-1].split('.')[0]
        # example['pos_id'] = pos_id
        example['slice_data'] = slice_data * 2 - 1
        example['masked_data'] = masked_data * 2 - 1
        example['tumor_mask'] = slice_seg
        # breakpoint()
        return example

class slice_train(slice_base):
    def __init__(self, phase='train', **kwargs):
        super().__init__(**kwargs)


class slice_val(slice_base):
    def __init__(self, phase='val', **kwargs):
        super().__init__(phase=phase, **kwargs)

    # def __len__(self):
    #     return 2 if super().__len__() // 10000 < 2 else super().__len__() // 10000

class slice_test(slice_base):
    def __init__(self, data_file='data_splits/test.txt', **kwargs):
        super().__init__(**kwargs)
    