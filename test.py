from UNet import UNet
from Oxpet_Dataset import Oxpet_Dataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

if __name__ == '__main__':

    ## cifar-10 dataset
    batch_size = 1

    testset = Oxpet_Dataset(os.path.join("datasets-oxpet-rewritten", "test","images.h5"),os.path.join("datasets-oxpet-rewritten", "test","binary.h5"),os.path.join("datasets-oxpet-rewritten", "test","bboxes.h5"),os.path.join("datasets-oxpet-rewritten", "test","masks.h5"), require_binary=False, require_bbox=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle= True,num_workers=0)
    dataiter = iter(testloader)

    ## load the trained model
    model = UNet(1)
    model.load_state_dict(torch.load('saved_model.pt'))


    ## inference
    images, labels = dataiter.next()

    outputs = torch.sigmoid(model(images))
    output = outputs[0].cpu().reshape(256,256).detach()
    output[output>=0.5] = 1
    output[output<0.5] = 0
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(images[0].permute(1,2,0))
    plt.subplot(1,3,2)
    plt.imshow(output,cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(labels.reshape(256,256),cmap='gray')
    plt.show()