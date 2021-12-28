from UNet import UNet
from Oxpet_Dataset import Oxpet_Dataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

if __name__ == '__main__':

    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1

    testset = Oxpet_Dataset(os.path.join("datasets-oxpet", "test","images.h5"),os.path.join("datasets-oxpet", "test","binary.h5"),os.path.join("datasets-oxpet", "test","bboxes.h5"),os.path.join("datasets-oxpet", "test","masks.h5"), require_binary=False, require_bbox=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle= True,num_workers=4)
    dataiter = iter(testloader)


    ## load the trained model
    model = UNet()
    model.load_state_dict(torch.load('saved_model.pt'))


    ## inference
    images, labels = dataiter.next()

    outputs = model(images)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(outputs)
    plt.subplot(1,2,2)
    plt.imshow(labels)
    plt.savefig('res.png')