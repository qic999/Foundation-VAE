import os
import numpy as np
# from skimage.transform import resize
import cv2
from torch.utils.data import Dataset
import copy

class slice_base(Dataset):
    def __init__(self,
                 data_root,
                 data_name,
                #  data_file,
                 data_repeat=1
                 ):

        self.data_root = data_root
        planner = 'nnUNetResEncUNetLPlans_torchres_3d_fullres' # nnUNetResEncUNetLPlans_torchres_3d_fullres nnUNetPlans_3d_fullres
        data_root = os.path.join(data_root, data_name , planner) 
        self.list = [os.path.join(data_name, planner ,f) for f in os.listdir(data_root) if f.endswith('.npz')]
        self.list.sort()
        # breakpoint()
        self.list = self.list[:5]
        # file_list = []
        # with open(data_file, 'r') as f:
        #     file_list = f.readlines()
        # file_list = [i.split('\n')[0] for i in file_list]
        # self.list = [os.path.join(data_name, planner , i+'.npz') for i in file_list]
        # breakpoint()
        self._length = len(self.list)
        self._data_repeat = data_repeat


    def __len__(self):
        return self._length * self._data_repeat

    def __getitem__(self, i):
        i = i % self._length
        
        npz_file = os.path.join(self.data_root, self.list[i])
        data_npz = np.load(npz_file)['data'][0] # np.load(npz_file)['seg']
        seg_npz = np.load(npz_file)['seg'][0]

        # pos_id = np.random.randint(0, data_npz.shape[0])
        # pos_id = data_npz.shape[0]//2
        # slice_data = data_npz[pos_id]
        # slice_seg = seg_npz[pos_id]

        # slice_data = np.pad(slice_data,((0,0),(8,8),(8,8)),'constant')
        # breakpoint()
        data_npz[data_npz <= -175] = -175
        data_npz[data_npz >= 250] = 250
        # breakpoint()
        data_npz = (data_npz+175)/425.0

        data_npz = data_npz[:,:,:,None].astype(np.float32)

        # breakpoint()
        example = {}
        example['name'] = self.list[i].split('/')[-1].split('.')[0]
        # example['pos_id'] = pos_id
        example['slice_data'] = data_npz * 2 - 1

        # breakpoint()
        return example

class slice_train(slice_base):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)


class slice_val(slice_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # def __len__(self):
    #     return 2 if super().__len__() // 10000 < 2 else super().__len__() // 10000

class slice_test(slice_base):
    def __init__(self, data_file='data_splits/test.txt', **kwargs):
        super().__init__(**kwargs)
    