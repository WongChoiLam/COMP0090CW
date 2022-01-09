import os
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as T
import utils
from torch.utils.data import DataLoader
from Oxpet_Dataset import Oxpet_Dataset
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def deeplabv3_resnet50_fit(trainset, model_name):

    batch_size = 8
    trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
    transforms = torch.nn.Sequential(
            T.RandomHorizontalFlip(p=1),
        )
    # initialize model
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)

    # Transfer Learning
    # model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad=False
    # model.classifier = DeepLabHead(2048, 2)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # runing machine
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # fit
    for epoch in range(2):
        running_loss = 0
        for i, train_data in enumerate(trainloader, 0):
            inputs, labels = train_data
            labels = labels.squeeze()

            inputs, labels = inputs.to(device), labels.to(device)  

            # data augmentation
            p = torch.rand(1)
            if p >=0.5:
                inputs = transforms(inputs)
                labels = transforms(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize

            outputs = model(inputs)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()   
            running_loss += loss.item()
        print(running_loss/trainset.__len__())
    torch.save(model.state_dict(), f'{model_name}.pt')
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
    
    model = deeplabv3_resnet50_fit(dataset,'BaseLine')
    model = model.to(device)
    model.eval()

    # model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=2).to(device)
    # model.load_state_dict(torch.load('BaseLine.pt'))
    # model.eval()
    
    # res = model(images)['out']
    # res = res.argmax(1)

    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")

    precision,recall,accuracy,F_1 = 0,0,0,0
    for i, data in enumerate(data_test_loader):
        images,targets = data
        images,targets = images.to(device),targets.to(device)
        p,r,a,f = utils.Evaluation_mask(model,images, targets.squeeze())
        precision += p
        recall += r
        accuracy += a
        F_1 += f

    print(f'accuracy={accuracy/(i+1)}')
    print(f'recall={recall/(i+1)}')
    print(f'precision={precision/(i+1)}')
    print(f'f1={F_1/(i+1)}')

    # print(images[2].shape)
    # r1 = Image.fromarray(images[2].cpu().numpy())
    # r2 = Image.fromarray(res[2].byte().cpu().numpy()).resize((256, 256))
    # r2.putpalette(colors)

    # plt.imshow(r1)
    # plt.imshow(r2)
    # plt.show()