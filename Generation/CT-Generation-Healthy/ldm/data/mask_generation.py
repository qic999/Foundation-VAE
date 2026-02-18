import os
import numpy as np
from skimage.transform import resize
import cv2
from torch.utils.data import Dataset
import copy

class volume_base(Dataset):
    def __init__(self,
                 data_root,
                 data_name,
                 data_file,
                 data_repeat=1
                 ):

        self.data_root = data_root
        planner = 'nnUNetResEncUNetLPlans_torchres_3d_fullres' # nnUNetResEncUNetLPlans_torchres_3d_fullres nnUNetPlans_3d_fullres
        data_root = os.path.join(data_root, data_name , planner) 
        file_list = []
        for line in open(data_file):
            name = line.strip().split()[1].split('.nii.gz')[0]
            name = name.split('/')[-1]
            file_list.append(name)

        self.list = [os.path.join(data_name, planner , i+'.npz') for i in file_list]
        
        self._length = len(self.list)
        self.sample_length=16
        self.sample=False
        self._data_repeat = data_repeat


    def __len__(self):
        return self._length * self._data_repeat

    def __getitem__(self, i):
        i = i % self._length
        
        npz_file = os.path.join(self.data_root, self.list[i])
        seg_npz = np.load(npz_file)['seg'][0]

        volume_seg = volume_seg.astype(np.float32)

        return volume_seg

class volume_train(volume_base):
    def __init__(self, data_file='data_split/liver/train.txt', **kwargs):
        super().__init__(data_file=data_file, **kwargs)


class volume_val(volume_base):
    def __init__(self, data_file='data_split/liver/eval.txt', **kwargs):
        super().__init__(data_file=data_file, **kwargs)

    # def __len__(self):
    #     return 2 if super().__len__() // 10000 < 2 else super().__len__() // 10000

class volume_test(volume_base):
    def __init__(self, data_file='data_splits/test.txt', **kwargs):
        super().__init__(**kwargs)
    