import os 
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

from torchvision import transforms as T
import torchvision.transforms.functional as TF
from random import random


class BasicDataset(Dataset):
    def __init__(
        self,
        path_root,
        task_kwargs,
        data_strengthen=False
    ):
        super().__init__()
        self.task_names = task_kwargs['task_names']
        
        self.path_root = path_root
        if not os.path.exists(self.path_root):
            print(self.path_root)
            raise Exception(f"[!] dataset is not exited")
        
        self.image_file_name = sorted(os.listdir(os.path.join(self.path_root, 'DAPI')))

        self.data_strengthen = data_strengthen
        self.transform = T.Compose([
            T.ToTensor(),
            # T.Normalize(mean=0.5, std=0.5) # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
        ])

    def __len__(self):
        return len(self.image_file_name)

    def __getitem__(self, index):
        
        file_name = self.image_file_name[index]
        img_CD8 = Image.open(os.path.join(self.path_root, 'CD8', file_name)).convert('RGB') # RGB | L
        img_CD45RO = Image.open(os.path.join(self.path_root, 'CD45RO', file_name)).convert('RGB')
        img_CD68 = Image.open(os.path.join(self.path_root, 'CD68', file_name)).convert('RGB')
        img_DAPI = Image.open(os.path.join(self.path_root, 'DAPI', file_name)).convert('RGB')
        img_Vimentin = Image.open(os.path.join(self.path_root, 'Vimentin', file_name)).convert('RGB')

        if self.data_strengthen:
            if random() > 0.7:
                img_CD8 = TF.vflip(img_CD8)
                img_CD45RO = TF.vflip(img_CD45RO)
                img_CD68 = TF.vflip(img_CD68)
                img_DAPI = TF.vflip(img_DAPI)
                img_Vimentin = TF.vflip(img_Vimentin)

            if random() > 0.7:
                img_CD8 = TF.hflip(img_CD8)
                img_CD45RO = TF.hflip(img_CD45RO)
                img_CD68 = TF.hflip(img_CD68)
                img_DAPI = TF.hflip(img_DAPI)
                img_Vimentin = TF.hflip(img_Vimentin)
        
        sample = {}
        sample['image'] = self.transform(img_DAPI)
        if 'gen1' in self.task_names:
            sample['gen1'] = self.transform(img_CD8)
        if 'gen2' in self.task_names:
            sample['gen2'] = self.transform(img_CD45RO)
        if 'gen3' in self.task_names:
            sample['gen3'] = self.transform(img_CD68)
        if 'gen4' in self.task_names:
            sample['gen4'] = self.transform(img_Vimentin)

        return sample