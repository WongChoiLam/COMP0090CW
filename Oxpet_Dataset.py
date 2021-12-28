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
            img = torch.FloatTensor(f[key][idx]).permute(2,0,1)
        
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
                masks = torch.FloatTensor(f[key][idx]).permute(2,0,1)
        
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

    training_data = Oxpet_Dataset(os.path.join("datasets-oxpet", "train","images.h5"),os.path.join("datasets-oxpet", "train","binary.h5"),os.path.join("datasets-oxpet", "train","bboxes.h5"),os.path.join("datasets-oxpet", "train","masks.h5"))
    print(training_data.__getitem__(0)[1].shape)
    # ox_dataloader = DataLoader(training_data, batch_size=32, shuffle= True,num_workers=4)