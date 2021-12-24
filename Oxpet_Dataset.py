import h5py
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import numpy as np
# import os

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
        with h5py.File(self.binary_path,"r") as f:
            key = list(f.keys())[0]
            return len(f[key])

    def __getitem__(self, idx):
        img = None
        binary = None
        bboxes = None
        masks = None

        with h5py.File(self.img_path,"r") as f:
            key = list(f.keys())[0]
            img = f[key][idx]
        
        if self.require_binary:
            with h5py.File(self.binary_path,"r") as f:
                key = list(f.keys())[0]
                binary = f[key][idx]
        
        if self.require_bbox:
            with h5py.File(self.bboxes_path,"r") as f:
                key = list(f.keys())[0]
                bboxes = f[key][idx]
        if self.require_masks:
            with h5py.File(self.masks_path,"r") as f:
                key = list(f.keys())[0]
                masks = f[key][idx]
        
        result = [img]
        if self.require_binary:
            result.append(binary)
        if self.require_bbox:
            result.append(bboxes)
        if self.require_masks:
            result.append(masks)
        return result

# training_data = Oxpet_Dataset(os.path.join("datasets-oxpet", "train","images.h5"),os.path.join("datasets-oxpet", "train","binary.h5"),os.path.join("datasets-oxpet", "train","bboxes.h5"),os.path.join("datasets-oxpet", "train","masks.h5"), require_binary=False, require_masks=False)
# print(len(training_data.__getitem__(0)))
# ox_dataloader = DataLoader(training_data, batch_size=32, shuffle= True,num_workers=4)