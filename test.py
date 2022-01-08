from UNet import UNet
from Oxpet_Dataset import Oxpet_Dataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import torchvision

if __name__ == '__main__':

    ## cifar-10 dataset
    batch_size = 8

    testset = Oxpet_Dataset(os.path.join("datasets-oxpet-rewritten", "test","images.h5"),os.path.join("datasets-oxpet-rewritten", "test","binary.h5"),os.path.join("datasets-oxpet-rewritten", "test","bboxes.h5"),os.path.join("datasets-oxpet-rewritten", "test","masks.h5"), require_binary=False, require_bbox=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle= True)
    dataiter = iter(testloader)

    ## load the trained model
    model_ce = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=2)
    model_ce.load_state_dict(torch.load('saved_model-ce.pt'))
    model_ce.eval()

    model_dice = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=1)
    model_dice.load_state_dict(torch.load('saved_model-dice.pt'))
    model_dice.eval()

    for _ in range(3):
        ## inference
        images, labels = dataiter.next()
        outputs_ce = torch.softmax(model_ce(images)['out'],dim=1)
        outputs_ce = torch.argmax(outputs_ce,dim=1)

        outputs_dice = torch.sigmoid(model_dice(images)['out'])
        outputs_dice = outputs_dice.cpu().detach()
        outputs_dice[outputs_dice>=0.5] = 1
        outputs_dice[outputs_dice<0.5] = 0

        plt.figure(figsize=(20,10))
        for i in range(batch_size):
            plt.subplot(batch_size,4,4*i+1)
            plt.imshow(images[i].permute(1,2,0))
            if i == 0: plt.title('Origin')
            plt.subplot(batch_size,4,4*i+2)
            plt.imshow(outputs_ce[i].reshape(256,256),cmap='gray')
            if i == 0: plt.title('CE')
            plt.subplot(batch_size,4,4*i+3)
            plt.imshow(outputs_dice[i].reshape(256,256),cmap='gray')
            if i == 0: plt.title('Dice')
            plt.subplot(batch_size,4,4*i+4)
            plt.imshow(labels[i].reshape(256,256),cmap='gray')
            if i == 0: plt.title('target')
        plt.show()