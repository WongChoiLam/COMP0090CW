import h5py
import torch
from torch.utils.data import Dataset

class Oxpet_Dataset(Dataset):
    def __init__(self, img_path, binary_path, bboxes_path, masks_path, require_binary = True, require_bbox = True, require_masks = True):
        self.img_path = img_path
        self.require_binary = require_binary
        self.require_bbox = require_bbox
        self.require_masks = require_masks
        self.img = h5py.File(img_path,"r")['/images'][()]
        self.length = h5py.File(img_path,"r")['/images'].__len__()

        if self.require_binary: self.binary = h5py.File(binary_path,"r")['/binary'][()]
        if self.require_bbox: self.bbox = h5py.File(bboxes_path,"r")['/bboxes'][()]
        if self.require_masks: self.mask = h5py.File(masks_path,"r")['/masks'][()]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img = torch.tensor(self.img[idx][()].astype('float32')).permute(2,0,1)/255.0
        result = [img]
        
        if self.require_binary:
            binary = torch.tensor(self.binary[idx][()].astype('int64'))
            result.append(binary)

        if self.require_bbox:
            bboxes = torch.tensor(self.bbox[idx][()].astype('float32'))
            result.append(bboxes)

        if self.require_masks:
            masks = torch.tensor(self.mask[idx][()].astype('int64')).permute(2,0,1)
            result.append(masks)
        
        return result