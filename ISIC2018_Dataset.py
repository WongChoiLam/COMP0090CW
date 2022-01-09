import torch
import numpy as np
from torch.utils.data import Dataset
import PIL.Image
import os
import h5py 

class ISIC2018_Dataset(Dataset):
    def __init__(self, img_path, mask_path):
        self.img = h5py.File(img_path, 'r')
        self.mask = h5py.File(mask_path, 'r')
        self.length = len(list(self.img))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        input_image = torch.FloatTensor(self.img[f'ISIC_{idx:07d}'][()]).permute(2,0,1)
        target_image = torch.LongTensor(self.mask[f'ISIC_{idx:07d}_segmentation'][()])
        return input_image, target_image

def ISIC_to_h5py(dataset_path):
    '''
    Convert ISIC data to h5py
    '''

    # ISIC data file names are not contineous, so order it first
    for directory in os.listdir(f'{dataset_path}'):

        # Find the number of files in total
        max_counter = 0
        for file in os.listdir(os.path.join(dataset_path,directory)):
            if file.endswith('txt'):
                continue
            max_counter += 1

        # Replacee file name
        real_counter = 0
        file_counter = 0
        while real_counter < max_counter:
            if os.path.exists(os.path.join(dataset_path,directory,f'ISIC_{file_counter:07d}.jpg')):
                os.rename(os.path.join(dataset_path,directory,f'ISIC_{file_counter:07d}.jpg'),os.path.join(dataset_path,directory,f'ISIC_{real_counter:07d}.jpg'))
                real_counter += 1
                file_counter += 1
            elif os.path.exists(os.path.join(dataset_path,directory,f'ISIC_{file_counter:07d}_segmentation.png')):
                os.rename(os.path.join(dataset_path,directory,f'ISIC_{file_counter:07d}_segmentation.png'),os.path.join(dataset_path,directory,f'ISIC_{real_counter:07d}_segmentation.png'))
                real_counter += 1
                file_counter += 1
            else:
                file_counter += 1

    # Convert ordered ISIC data to h5py
    os.makedirs(f'{dataset_path}-rewritten',exist_ok=True)
    for directory in os.listdir(f'{dataset_path}'):
        with h5py.File(os.path.join(f'{dataset_path}-rewritten',f'{directory}.h5'), "w") as f:
            for file in os.listdir(os.path.join(dataset_path,directory)):
                if file.endswith('txt'):
                    continue
                filename = file.split('.')[0]
                f.create_dataset(f'{filename}', data=np.asarray(PIL.Image.open(os.path.join(dataset_path,directory,file)).resize((256,256)))/255, compression='gzip')

if __name__ == '__main__':
    
    import time
    start = time.time()
    ISIC_to_h5py('ISIC')
    end = time.time()
    print(f'Conversion has been done in {end-start:.2f} seconds')