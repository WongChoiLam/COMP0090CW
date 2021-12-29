import h5py
import torch
from torch.utils.data import Dataset

class Oxpet_Dataset(Dataset):
    def __init__(self, img_path, binary_path, bboxes_path, masks_path, require_binary = True, require_bbox = True, require_masks = True):
        self.img_path = img_path
        self.require_binary = require_binary
        self.require_bbox = require_bbox
        self.require_masks = require_masks
        if self.require_binary:
            self.binary_path = binary_path
        if self.require_bbox:
            self.bboxes_path = bboxes_path
        if self.require_masks:
            self.masks_path = masks_path

    def __len__(self):
        with h5py.File(self.img_path,"r") as f:
            key = list(f.keys())[0]
            return len(f[key])

    def __getitem__(self, idx):
        img = None
        binary = None
        bboxes = None
        masks = None

        with h5py.File(self.img_path,"r") as f:
            key = list(f.keys())[0]
            img = torch.FloatTensor(f[key][idx]).permute(2,0,1)/255
        
        if self.require_binary:
            with h5py.File(self.binary_path,"r") as f:
                key = list(f.keys())[0]
                binary = torch.FloatTensor(f[key][idx])
        
        if self.require_bbox:
            with h5py.File(self.bboxes_path,"r") as f:
                key = list(f.keys())[0]
                bboxes = torch.FloatTensor(f[key][idx])

        if self.require_masks:
            with h5py.File(self.masks_path,"r") as f:
                key = list(f.keys())[0]
                masks = torch.FloatTensor(f[key][idx]).permute(2,0,1)/255
                masks[masks != 0] = 1
        
        result = [img]
        if self.require_binary:
            result.append(binary)
        if self.require_bbox:
            result.append(bboxes)
        if self.require_masks:
            result.append(masks)
        return result

class Oxpet_Dataset_RAM(Dataset):
    def __init__(self, img_path, binary_path, bboxes_path, masks_path, require_binary = True, require_bbox = True, require_masks = True):
        self.img_path = img_path
        self.require_binary = require_binary
        self.require_bbox = require_bbox
        self.require_masks = require_masks
        with h5py.File(img_path,"r") as f:
            key = list(f.keys())[0]
            self.img = f[key][()]
            self.length = len(f[key])

        if self.require_binary:
            with h5py.File(binary_path,"r") as f:
                self.binary = f[list(f.keys())[0]][()]

        if self.require_bbox:
            with h5py.File(bboxes_path,"r") as f:
                self.bbox = f[list(f.keys())[0]][()]
        if self.require_masks:
            with h5py.File(masks_path,"r") as f:
                self.mask = f[list(f.keys())[0]][()]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = None
        binary = None
        bboxes = None
        masks = None
        img = torch.FloatTensor(self.img[idx]).permute(2,0,1)/255
        
        if self.require_binary:
            binary = torch.FloatTensor(self.binary[idx])
        
        if self.require_bbox:
            bboxes = torch.FloatTensor(self.bbox[idx])

        if self.require_masks:
            masks = torch.FloatTensor(self.mask[idx]).permute(2,0,1)/255
            masks[masks != 0] = 1
        
        result = [img]
        if self.require_binary:
            result.append(binary)
        if self.require_bbox:
            result.append(bboxes)
        if self.require_masks:
            result.append(masks)
        return result

class Oxpet_Dataset_True(Dataset):
    def __init__(self, img_path, binary_path, bboxes_path, masks_path, require_binary = True, require_bbox = True, require_masks = True):
        self.img_path = img_path
        self.require_binary = require_binary
        self.require_bbox = require_bbox
        self.require_masks = require_masks
        self.img = h5py.File(img_path,"r")

        if self.require_binary: self.binary = h5py.File(binary_path,"r")
        if self.require_bbox: self.bbox = h5py.File(bboxes_path,"r")
        if self.require_masks:  self.mask = h5py.File(masks_path,"r")

        # with h5py.File(img_path,"r") as f:
        #     key = list(f.keys())[0]
        #     self.img = f[key][()]
        #     self.length = len(f[key])

        # if self.require_binary:
        #     with h5py.File(binary_path,"r") as f:
        #         self.binary = f[list(f.keys())[0]][()]

        # if self.require_bbox:
        #     with h5py.File(bboxes_path,"r") as f:
        #         self.bbox = f[list(f.keys())[0]][()]
        # if self.require_masks:
        #     with h5py.File(masks_path,"r") as f:
        #         self.mask = f[list(f.keys())[0]][()]

    def __len__(self):
        key = list(self.img.keys())[0]
        self.length = len(self.img[key])
        return self.length

    def __getitem__(self, idx):
        img = None
        binary = None
        bboxes = None
        masks = None

        key = list(self.img.keys())[0]
        img = torch.FloatTensor(self.img[key][idx]).permute(2,0,1)/255.0
        
        if self.require_binary:
            key = list(self.binary.keys())[0]
            binary = torch.FloatTensor(self.binary[key][idx])
        
        if self.require_bbox:
            key = list(self.bbox.keys())[0]
            bboxes = torch.FloatTensor(self.bbox[key][idx])

        if self.require_masks:
            key = list(self.mask.keys())[0]
            masks = torch.FloatTensor(self.mask[key][idx]).permute(2,0,1)/255.0
        
        result = [img]
        if self.require_binary:
            result.append(binary)
        if self.require_bbox:
            result.append(bboxes)
        if self.require_masks:
            result.append(masks)
        return result

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import numpy as np
    import os
    import time

    a = time.time()
    training_data = Oxpet_Dataset(os.path.join("datasets-oxpet", "train","images.h5"),os.path.join("datasets-oxpet", "train","binary.h5"),os.path.join("datasets-oxpet", "train","bboxes.h5"),os.path.join("datasets-oxpet", "train","masks.h5"))
    ox_dataloader = DataLoader(training_data, batch_size=1000, shuffle= True,num_workers=4)
    b = time.time()
    print(b-a)
    a = time.time()
    training_data_RAM = Oxpet_Dataset_RAM(os.path.join("datasets-oxpet", "train","images.h5"),os.path.join("datasets-oxpet", "train","binary.h5"),os.path.join("datasets-oxpet", "train","bboxes.h5"),os.path.join("datasets-oxpet", "train","masks.h5"))
    ox_dataloader_RAM = DataLoader(training_data_RAM, batch_size=50, shuffle= True,num_workers=0)
    b = time.time()
    print(b-a)
    c = time.time()
    for i, data in enumerate(ox_dataloader_RAM, 0):
        pass
    d = time.time()
    print(f'{b-a}, {d-c}')