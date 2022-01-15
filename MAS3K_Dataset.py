import h5py
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from torchvision.utils import save_image

class MAS3K_Dataset(Dataset):
    def __init__(self, img_path, masks_path, require_masks = True):
        self.require_masks = require_masks
        
        if self.require_masks:  
            self.mask = h5py.File(masks_path,"r")

        self.img = h5py.File(img_path,"r")
        self.length = len(list(self.img))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img = torch.FloatTensor(self.img[f'{idx}'][()])
        result = [img]

        if self.require_masks:
            masks = torch.LongTensor(self.mask[f'{idx}'][()])
            result.append(masks)
        
        return result


def to_h5py(filePaths, name, mask):

    toTensor = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            ])

    count = 0
    if mask:
        with h5py.File(name,'w') as f:
            for filePath in filePaths:
                for childPath in os.listdir(filePath):
                    imgPath = os.path.join(filePath, childPath)
                    img = Image.open(imgPath).convert('1')
                    img_tensor = toTensor(img)
                    f.create_dataset(f'{count}', data=img_tensor)
                    count += 1
            f.close()
    else:
        with h5py.File(name,'w') as f:
            for filePath in filePaths:
                for childPath in os.listdir(filePath):
                    imgPath = os.path.join(filePath, childPath)
                    img = Image.open(imgPath).convert('RGB')
                    img_tensor = toTensor(img)
                    f.create_dataset(f'{count}', data=img_tensor)
                    count += 1
            f.close()

    return 0

if __name__ == '__main__':
    import time
    import os

    start = time.time()

    TrainIfilePath = os.path.join('MAS3K','train','Image')
    TestIfilePath = os.path.join('MAS3K','test','Image')
    TrainMfilePath = os.path.join('MAS3K','train','Mask')
    TestMfilePath = os.path.join('MAS3K','test','Mask')

    ISavePath = os.path.join('MAS3K','TrainImage.h5')
    MSavePath = os.path.join('MAS3K','TrainMask.h5')

    to_h5py([TrainIfilePath, TestIfilePath], ISavePath, False)
    to_h5py([TrainMfilePath, TestMfilePath], MSavePath, True)
    
    end = time.time()
    print(f'Conversion has been done in {end-start:.2f} seconds')