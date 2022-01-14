from UNet import UNet
from Oxpet_Dataset import Oxpet_Dataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

if __name__ == '__main__':

    ## cifar-10 dataset
    batch_size = 8

    testset = Oxpet_Dataset(os.path.join("datasets-oxpet-rewritten", "test","images.h5"),os.path.join("datasets-oxpet-rewritten", "test","binary.h5"),os.path.join("datasets-oxpet-rewritten", "test","bboxes.h5"),os.path.join("datasets-oxpet-rewritten", "test","masks.h5"), require_binary=False, require_bbox=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle= True)
    dataiter = iter(testloader)

    ## load the baseline model
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=2)
    model.load_state_dict(torch.load('BaseLine.pt'))
    model.eval()

    # load the transfer learning model
    # model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    # model.classifier = DeepLabHead(2048, 2)
    # model.load_state_dict(torch.load('Transfer.pt'))
    # model.eval()

    ## inference
    images, labels = dataiter.next()
    outputs = torch.softmax(model(images)['out'],dim=1)
    outputs = torch.argmax(outputs,dim=1)

    plt.figure(figsize=(20,10))
    for i in range(batch_size):
        plt.subplot(batch_size,3,3*i+1)
        plt.imshow(images[i].permute(1,2,0))
        if i == 0: plt.title('Origin')
        plt.subplot(batch_size,3,3*i+2)
        plt.imshow(outputs[i].reshape(256,256),cmap='gray')
        if i == 0: plt.title('BaseLine')
        plt.subplot(batch_size,3,3*i+3)
        plt.imshow(labels[i].reshape(256,256),cmap='gray')
        if i == 0: plt.title('target')
    plt.show()