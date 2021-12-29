import h5py
import torch
from torch.utils.data import Dataset

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