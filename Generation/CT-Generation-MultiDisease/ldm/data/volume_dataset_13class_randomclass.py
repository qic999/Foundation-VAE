import os
import numpy as np
from skimage.transform import resize
import cv2
from torch.utils.data import Dataset
import copy
import random
import itertools
from batchgenerators.utilities.file_and_folder_operations import load_json

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

class volume_base(Dataset):
    def __init__(self,
                 data_root,
                 data_name,
                 data_file,
                 data_repeat=1,
                 phase='train'
                 ):

        self.data_root = data_root
        planner = 'nnUNetPlans_3d_fullres' # nnUNetResEncUNetLPlans_torchres_3d_fullres nnUNetPlans_3d_fullres
        data_root = os.path.join(data_root, data_name , planner) 
        # self.list = [os.path.join(data_name, planner ,f) for f in os.listdir(data_root) if f.endswith('.npz')]

        data_splits = load_json(data_file)
        file_list = data_splits[0][phase]
        # breakpoint()
        self.list = [os.path.join(data_name, planner , i+'.npz') for i in file_list]
        
        self._length = len(self.list)
        self.class_num = 13
        
        self.sample_length=16
        self.sample=True
        self._data_repeat = data_repeat


    def __len__(self):
        return self._length * self._data_repeat

    def __getitem__(self, i):
        i = i % self._length
        
        npz_file = os.path.join(self.data_root, self.list[i])
        data_npz = np.load(npz_file)['data'][0] # np.load(npz_file)['seg']
        seg_npz = np.load(npz_file)['seg'][0]
        
        # breakpoint()

        if self.sample:
            start_id = np.random.randint(0, data_npz.shape[0]-self.sample_length)
            volume_data = data_npz[start_id:start_id+self.sample_length]
            volume_seg = seg_npz[start_id:start_id+self.sample_length]

        # breakpoint()


        volume_data[volume_data <= -175] = -175
        volume_data[volume_data >= 250] = 250
        volume_data = (volume_data+175)/425.0

        if len(np.unique(volume_seg)) > 1:
            selected_classes = get_random_classes_from_mask(volume_seg)
            volume_seg = get_filtered_mask(volume_seg, selected_classes)


        volume_seg = volume_seg.astype(np.float32)
        foreground_mask = (volume_seg>0).astype(np.float32)
        masked_data = (1-foreground_mask)*volume_data

        volume_data = volume_data[:,:,:,None].astype(np.float32)
        masked_data = masked_data[:,:,:,None].astype(np.float32)
        
        # breakpoint()
        volume_seg = resize(volume_seg, (volume_seg.shape[0],64,64), order=0, anti_aliasing=False)
        # print(np.unique(resize(volume_seg, (volume_seg.shape[0],64,64), order=0, anti_aliasing=False)))
        volume_seg = volume_seg[:,:,:,None].astype(np.float32)

        example = {}
        example['name'] = self.list[i].split('/')[-1].split('.')[0]
        # example['pos_id'] = pos_id
        example['volume_data'] = volume_data * 2 - 1
        example['masked_data'] = masked_data * 2 - 1
        example['tumor_mask'] = volume_seg
        # breakpoint()
        return example

class volume_train(volume_base):
    def __init__(self, phase='train', **kwargs):
        super().__init__( phase=phase, **kwargs)


class volume_val(volume_base):
    def __init__(self, phase='val', **kwargs):
        super().__init__(phase=phase, **kwargs)

    # def __len__(self):
    #     return 2 if super().__len__() // 10000 < 2 else super().__len__() // 10000

class volume_test(volume_base):
    def __init__(self, data_file='data_splits/test.txt', **kwargs):
        super().__init__(**kwargs)
    