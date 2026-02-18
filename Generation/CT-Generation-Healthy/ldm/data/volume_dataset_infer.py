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
        # self.list = [os.path.join(data_name, planner ,f) for f in os.listdir(data_root) if f.endswith('.npz')]
        file_list = []
        for line in open(data_file):
            name = line.strip().split()[1].split('.nii.gz')[0]
            name = name.split('/')[-1]
            file_list.append(name)
        # breakpoint()
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
        data_npz = np.load(npz_file)['data'][0] # np.load(npz_file)['seg']
        seg_npz = np.load(npz_file)['seg'][0]
        # breakpoint()

        if self.sample:
            start_id = np.random.randint(0, data_npz.shape[0]-self.sample_length)
            volume_data = data_npz[start_id:start_id+self.sample_length]
            volume_seg = seg_npz[start_id:start_id+self.sample_length]
        else:
            volume_data=data_npz
            volume_seg = seg_npz
        # breakpoint()


        volume_data[volume_data <= -175] = -175
        volume_data[volume_data >= 250] = 250
        volume_data = (volume_data+175)/425.0


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
    