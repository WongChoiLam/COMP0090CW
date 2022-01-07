import os
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from Evaluation import Evaluation_mask
from Oxpet_Dataset import Oxpet_Dataset

class DiceLoss(nn.Module):

    'DiceLoss is adapted from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch'
    
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.argmax(inputs,dim=1)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


def deeplabv3_resnet50_fit(trainset):
    # initialize model
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=2)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # runing machine
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # fit
    for epoch in range(2):
        batch_size = 8
        trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
        transforms = torch.nn.Sequential(
            T.RandomHorizontalFlip(p=0.5),
        )
        for i, train_data in enumerate(trainloader, 0):
            inputs, labels = train_data
            labels = labels.squeeze()
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)   
            # data augmentation
            inputs = transforms(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)['out']             
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()   
        print(loss)  
    return model

dataset = Oxpet_Dataset(
    os.path.join("datasets-oxpet-rewritten", "train","images.h5"),
    os.path.join("datasets-oxpet-rewritten", "train","binary.h5"), 
    os.path.join("datasets-oxpet-rewritten", "train","bboxes.h5"),
    os.path.join("datasets-oxpet-rewritten", "train","masks.h5"), 
    require_binary=False,
    require_bbox=False,
    require_masks=True)
dataset_test = Oxpet_Dataset(
    os.path.join("datasets-oxpet-rewritten", "test","images.h5"),
    os.path.join("datasets-oxpet-rewritten", "test","binary.h5"), 
    os.path.join("datasets-oxpet-rewritten", "test","bboxes.h5"),
    os.path.join("datasets-oxpet-rewritten", "test","masks.h5"), 
    require_binary=False,
    require_bbox=False,
    require_masks=True)

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    data_test_loader = DataLoader(dataset_test,batch_size=8,shuffle=True)
    images,targets = next(iter(data_test_loader))
    images,targets = images.to(device),targets.to(device)
    
    model = deeplabv3_resnet50_fit(dataset)
    model = model.to(device)
    model.eval()
    
    res = model(images)['out']
    res = res.argmax(1)

    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")
   
    precision,recall,Accuracy,F_1 = Evaluation_mask(model,images, targets)     
    print(Accuracy)  
    # print(images[2].shape)
    # r1 = Image.fromarray(images[2].cpu().numpy())
    # r2 = Image.fromarray(res[2].byte().cpu().numpy()).resize((256, 256))
    # r2.putpalette(colors)

    # plt.imshow(r1)
    # plt.imshow(r2)
    # plt.show()