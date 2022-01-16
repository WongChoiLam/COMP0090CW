import h5py
import torch
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

        img = torch.FloatTensor(self.img[f'{idx}'][()]).permute(2,0,1)/255.0
        result = [img]

        if self.require_binary:
            binary = torch.LongTensor(self.binary[f'{idx}'][()])
            result.append(binary)
        
        if self.require_bbox:
            bboxes = torch.FloatTensor(self.bbox[f'{idx}'][()])
            result.append(bboxes)

        if self.require_masks:
            masks = torch.LongTensor(self.mask[f'{idx}'][()]).permute(2,0,1)
            result.append(masks)
        
        return result
