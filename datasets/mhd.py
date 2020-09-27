import torch.utils.data as data
import SimpleITK as sitk
import numpy as np
import os
from glob import glob

def normalizePlanes(npzarray):
    maxHU = 1000.
    minHU = -1000.

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray

class MHDData(data.Dataset):
    
    def __init__(self, data_dir, is_train, transform=None):
        self.data_dir = os.path.join(data_dir, 'MHDdata')
        self.transform = transform
        self.image_dir = os.path.join(self.data_dir, '1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866.mhd')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        mhd = sitk.ReadImage(self.image_dir)
        img = sitk.GetArrayFromImage(mhd)
        img = normalizePlanes(img)
        if self.transform:
            img = self.transform(img)
            img = img.permute(1,2,0)
            # img = img[:,:,10:30,100:200,200:300]
        return img