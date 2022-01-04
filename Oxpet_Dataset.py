import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class Oxpet_Dataset(Dataset):
    def __init__(self, img_path, binary_path, bboxes_path, masks_path, require_binary = True, require_bbox = True, require_masks = True):
        self.require_binary = require_binary
        self.require_bbox = require_bbox
        self.require_masks = require_masks
        
        if self.require_binary: self.binary = h5py.File(binary_path,"r")
        if self.require_bbox: self.bbox = h5py.File(bboxes_path,"r")
        if self.require_masks:  self.mask = h5py.File(masks_path,"r")

        self.img = h5py.File(img_path,"r")
        self.length = len(list(self.img))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img = torch.tensor(self.img[f'{idx}'][()].astype('float32')).permute(2,0,1)/255.0
        result = [img]

        if self.require_binary:
            binary = torch.tensor(self.binary[f'{idx}'][()].astype('int64'))
            result.append(binary)
        
        if self.require_bbox:
            bboxes = torch.tensor(self.bbox[f'{idx}'][()].astype('float32'))
            result.append(bboxes)

        if self.require_masks:
            masks = torch.tensor(self.mask[f'{idx}'][()].astype('int64')).permute(2,0,1)
            result.append(masks)
        
        return result

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import numpy as np
    import os
    import time
    
    training_data = Oxpet_Dataset(os.path.join("datasets-oxpet-rewritten", "train","images.h5"),os.path.join("datasets-oxpet-rewritten", "train","binary.h5"),os.path.join("datasets-oxpet-rewritten", "train","bboxes.h5"),os.path.join("datasets-oxpet-rewritten", "train","masks.h5"),False,False)
    # training_data = Oxpet_Dataset(os.path.join("datasets-oxpet", "train","images.h5"),os.path.join("datasets-oxpet", "train","binary.h5"),os.path.join("datasets-oxpet", "train","bboxes.h5"),os.path.join("datasets-oxpet", "train","masks.h5"),False,False,False)
    ox_dataloader = DataLoader(training_data, batch_size=16, shuffle= True,num_workers=8)
    # training_data.__getitem__(0)
    for i in range(10):
        a = time.time()
        for i, data in enumerate(ox_dataloader):
            pass
        b = time.time()
        print(b-a)
