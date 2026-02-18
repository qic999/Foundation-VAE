import os
import numpy as np
# from skimage.transform import resize
import cv2
from torch.utils.data import Dataset
import copy
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p


class slice_base(Dataset):
    def __init__(self,
                 data_root,
                 data_name,
                 data_repeat=1,
                 planner='nnUNetPlans_3d_fullres',
                 data_file='data_split/liver/train.txt',
                 phase='train',
                 ):

        self.data_root = data_root
        # planner = 'nnUNetResEncUNetLPlans_torchres_3d_fullres' # nnUNetResEncUNetLPlans_torchres_3d_fullres nnUNetPlans_3d_fullres
        data_root = os.path.join(data_root, data_name , planner) 
        # self.list = [os.path.join(data_name, planner ,f) for f in os.listdir(data_root) if f.endswith('.npz')]

        file_list = []
        if 'liver' in data_file:
            for line in open(data_file):
                name = line.strip().split()[1].split('.nii.gz')[0]
                name = name.split('/')[-1]
                file_list.append(name)
        else:
            file_list = load_json(data_file)[0][phase]

        self.list = [os.path.join(data_name, planner , i+'.npz') for i in file_list]
        
        self._length = len(self.list)
        self._data_repeat = data_repeat


    def __len__(self):
        return self._length * self._data_repeat

    def __getitem__(self, i):
        i = i % self._length
        
        npz_file = os.path.join(self.data_root, self.list[i])
        data_npz = np.load(npz_file)['data'][0] # np.load(npz_file)['seg']
        seg_npz = np.load(npz_file)['seg'][0]
        # print('data_npz', data_npz.shape)
        # print('seg_npz', seg_npz.shape)
        # breakpoint()
        pos_id = np.random.randint(0, data_npz.shape[0])
        # pos_id = data_npz.shape[0]//2
        slice_data = data_npz[pos_id]
        slice_seg = seg_npz[pos_id]

        # slice_data = np.pad(slice_data,((0,0),(8,8),(8,8)),'constant')
        # breakpoint()
        slice_data[slice_data <= -175] = -175
        slice_data[slice_data >= 250] = 250
        # breakpoint()
        slice_data = (slice_data+175)/425.0

        # tumor_mask = (slice_seg==2).astype(np.float32)
        slice_seg = slice_seg.astype(np.float32)
        foreground_mask = (slice_seg>0).astype(np.float32)
        masked_data = (1-foreground_mask)*slice_data

        slice_data = slice_data[:,:,None].astype(np.float32)
        masked_data = masked_data[:,:,None].astype(np.float32)
        
        # tumor_mask = resize(tumor_mask, (64,64), order=0)
        slice_seg = cv2.resize(slice_seg,(64,64),interpolation=cv2.INTER_NEAREST)
        slice_seg = slice_seg[:,:,None].astype(np.float32)
        # print(np.unique(slice_seg))
        # breakpoint()
        if (slice_data.shape[0] != 512) or (slice_data.shape[1] != 512):
            print(self.list[i],slice_data.shape,masked_data.shape,slice_seg.shape)
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
        super().__init__(phase=phase, **kwargs)


class slice_val(slice_base):
    def __init__(self, phase='val', **kwargs):
        super().__init__(phase=phase, **kwargs)

    # def __len__(self):
    #     return 2 if super().__len__() // 10000 < 2 else super().__len__() // 10000

class slice_test(slice_base):
    def __init__(self, data_file='data_splits/test.txt', **kwargs):
        super().__init__(**kwargs)
    