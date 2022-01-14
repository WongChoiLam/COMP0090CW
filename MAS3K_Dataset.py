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
            masks = torch.LongTensor(self.mask[f'{idx}'][()] / 255)
            result.append(masks)
        
        return result


def to_h5py(filePaths, name, mask):

    toTensor = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            ])

    count = 0
    if mask:
        with h5py.File(name,'a') as f:
            for filePath in filePaths:
                for childPath in os.listdir(filePath):
                    imgPath = os.path.join(filePath, childPath)
                    img = Image.open(imgPath).convert('1')
                    img_tensor = toTensor(img)
                    f.create_dataset(f'{count}', data=img_tensor)
                    count += 1
            f.close()
    else:
        with h5py.File(name,'a') as f:
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
    from torch.utils.data import DataLoader
    import os
    import time
    import matplotlib.pyplot as plt

    TrainIfilePath = 'MAS3K\\train\Image'
    TrainMfilePath = 'MAS3K\\train\Mask'
    TestIfilePath = 'MAS3K\\test\Image'
    TestMfilePath = 'MAS3K\\test\Mask'

    # TrainIfilePath = 'MAS3K\\TrainImage.h5'
    # TrainMfilePath = 'MAS3K\\TrainMask.h5'
    # TestIfilePath = 'MAS3K\\TestImage.h5'
    # TestMfilePath = 'MAS3K\\TestMask.h5'

    to_h5py([TrainIfilePath, TestIfilePath], 'MAS3K\\TrainImage.h5', False)
    to_h5py([TrainMfilePath, TestMfilePath], 'MAS3K\\TrainMask.h5', True)

    # training_data = MAS3K_Dataset(TrainIfilePath, TrainMfilePath, True)
    # training_data = Oxpet_Dataset(os.path.join("datasets-oxpet", "train","images.h5"),os.path.join("datasets-oxpet", "train","binary.h5"),os.path.join("datasets-oxpet", "train","bboxes.h5"),os.path.join("datasets-oxpet", "train","masks.h5"),False,False,False)
    # MAS3K_dataloader = DataLoader(training_data, batch_size=4, shuffle= False)

    # with h5py.File('datasets-oxpet-rewritten\\train\images.h5','r') as f:
    #     for key in f.keys():
            # print(key)
            # print(f[key].shape)
        #     break
        # f.close()

    # with h5py.File('MAS3K\\TrainImage.h5','r') as f:
    #     for key in f.keys():
    #         # print(key)
    #         # print(f[key][0])
    #         # print(f[key].shape)
    #         break
    #     f.close()

    # with h5py.File('MAS3K\\TrainMask.h5','r') as f:
    #     for key in f.keys():
    #         # print(key)
    #         for i in range(f[key].shape[1]):
    #             print(f[key][0][i])
    #         # print(f[key].shape)
    #         break
    #     f.close()

    # toTensor = transforms.Compose([
    #         transforms.ToTensor(),
    #         ])
    # img = Image.open('MAS3K\\train\Image\MAS_Arthropod_Crab_Cam_363.jpg')
    # img = toTensor(img)
    # print(img.shape)
    # for i in range(img.shape[1]):
    #     for j in range(img.shape[2]):
    #         print(img[0][i][j])
    #         print(img[1][i][j])
    #         print(img[2][i][j])

    # data = training_data.__getitem__(1)
    # for i in range(256):
    #     for j in range(256):
    #         if data[1][0][i][j] != 0:
    #             print(data[1][0][i][j])
    # print(data[0].shape)
    # print(data[1].shape)
    # plt.imshow(data[0].permute(1,2,0))
    # plt.savefig('test.png')
    # print(data[1])
    # print(data[2])
    # plt.imshow(data[3].permute(1,2,0).squeeze())
    # plt.savefig('test1.png')

    # for i in range(1):
    #     a = time.time()
    #     for i, data in enumerate(MAS3K_dataloader):
    #         pass
    #     b = time.time()
    #     print(b-a)
    
