import torch
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset
import PIL.Image
import os
import h5py 


class CityScapesDataset(Dataset):
    """
    Read h5 file to tensors
    """
    def __init__(self, dataset_path, split):
        img_path = os.path.join(dataset_path, 'images', f'{split}.h5')
        mask_path = os.path.join(dataset_path, 'masks', f'{split}.h5')
        self.img = h5py.File(img_path, 'r')
        self.mask = h5py.File(mask_path, 'r')
        self.length = len(list(self.img))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_image = torch.FloatTensor(self.img[f'{idx}'][()]).permute(2,0,1)
        target_image = torch.LongTensor(self.mask[f'{idx}'][()])
        return input_image, target_image

    
def CityScapes_to_h5py(dataset_path):
    """
    Convert CityScapes data to h5py
    """
    img_dir = os.path.join(dataset_path, 'leftImg8bit')
    print(img_dir)
    mask_dir = os.path.join(dataset_path, 'gtFine')
    print(mask_dir)
    os.makedirs(f'{dataset_path}-rewritten', exist_ok=True)
    rewritten = f'{dataset_path}-rewritten'
    os.makedirs(os.path.join(rewritten, 'images'), exist_ok=True)
    os.makedirs(os.path.join(rewritten, 'masks'), exist_ok=True)
    crop = T.CenterCrop(256)
    
    for split in os.listdir(img_dir):
        # subdir of img_dir: 'train', 'test', 'val'
        i = 0
        with h5py.File(os.path.join(rewritten, 'images', f'{split}.h5'), "w") as f:
            for city in os.listdir(os.path.join(img_dir, split)):
                for file in os.listdir(os.path.join(img_dir, split, city)):
                    f.create_dataset(f"{i}", data=np.asarray(crop(PIL.Image.open(os.path.join(img_dir, split, city, file)))/255), compression='gzip')
                    i += 1
                    
    for split in os.listdir(mask_dir):
        # subdir of mask_dir: 'train', 'test', 'val'
        i = 0
        with h5py.File(os.path.join(rewritten, 'masks', f'{split}.h5'), "w") as f:
            for city in os.listdir(os.path.join(mask_dir, split)):
                for file in os.listdir(os.path.join(mask_dir, split, city)):
                    file_name = file.split('.')[0]
                    if file_name[-8:] == "labelIds":
                        f.create_dataset(f"{i}", data=np.asarray(crop(PIL.Image.open(os.path.join(mask_dir, split, city, file)))), compression='gzip')
                        i += 1 

                
if __name__ == '__main__':
    
    import time
    start = time.time()
    CityScapes_to_h5py('dataset-cityscapes')
    end = time.time()
    print(f'Conversion has been done in {end-start:.2f} seconds')