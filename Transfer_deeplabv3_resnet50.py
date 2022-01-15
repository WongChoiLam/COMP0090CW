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
import csv

# Due to pytorch's pre-trained model manual https://pytorch.org/vision/stable/models.html
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def train(trainset, val_set, batch_size, num_epochs, device, model_name):

    trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
    validloader = DataLoader(val_set, batch_size=batch_size,shuffle=True)

    transforms = torch.nn.Sequential(
            T.RandomHorizontalFlip(p=1),
        )

    # Transfer Learning
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    model.classifier = DeepLabHead(2048, 2)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # runing machine
    model.to(device)

    train_loss = []
    valid_loss = []

    min_valid_loss = float('inf')

    # fit
    for epoch in range(num_epochs):
        running_loss = 0
        for i, train_data in enumerate(trainloader, 0):
            inputs, labels = train_data
            inputs = normalize(inputs)
            labels = labels.squeeze()

            inputs, labels = inputs.to(device), labels.to(device)  

            # data augmentation
            p = torch.rand(1)
            if p >= 0.5:
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

        print(f'epoch {epoch+1}, training loss = {running_loss/(i+1)}')
        train_loss.append(running_loss/(i+1))

        running_loss = 0
        for i, val_data in enumerate(validloader):
            inputs, labels = val_data
            inputs = normalize(inputs)
            labels = labels.squeeze()

            inputs, labels = inputs.to(device), labels.to(device)  

            # forward + backward + optimize
            outputs = model(inputs)['out']
            loss = criterion(outputs, labels) 
            running_loss += loss.item()  
            
        # Save the best model
        if running_loss/(i+1) < min_valid_loss:
            print(f'epoch {epoch+1}, validation loss = {running_loss/(i+1)}, lowest validation loss = True, save model')  
            torch.save(model.state_dict(), f'{model_name}.pt')    
            min_valid_loss = running_loss/(i+1)
        else:
            print(f'epoch {epoch+1}, validation loss = {running_loss/(i+1)}, lowest validation loss = False, do not save model')
      
        valid_loss.append(running_loss/(i+1))    

    print(f'Model has been saved to {model_name}.pt\n')    

    return model, train_loss, valid_loss

if __name__ == '__main__':

    oxpet_train = Oxpet_Dataset(
        os.path.join("datasets-oxpet-rewritten", "train","images.h5"),
        os.path.join("datasets-oxpet-rewritten", "train","binary.h5"), 
        os.path.join("datasets-oxpet-rewritten", "train","bboxes.h5"),
        os.path.join("datasets-oxpet-rewritten", "train","masks.h5"), 
        require_binary=False,
        require_bbox=False,
        require_masks=True)
    oxpet_valid = Oxpet_Dataset(
        os.path.join("datasets-oxpet-rewritten", "val","images.h5"),
        os.path.join("datasets-oxpet-rewritten", "val","binary.h5"), 
        os.path.join("datasets-oxpet-rewritten", "val","bboxes.h5"),
        os.path.join("datasets-oxpet-rewritten", "val","masks.h5"), 
        require_binary=False,
        require_bbox=False,
        require_masks=True
    )
    oxpet_test = Oxpet_Dataset(
        os.path.join("datasets-oxpet-rewritten", "test","images.h5"),
        os.path.join("datasets-oxpet-rewritten", "test","binary.h5"), 
        os.path.join("datasets-oxpet-rewritten", "test","bboxes.h5"),
        os.path.join("datasets-oxpet-rewritten", "test","masks.h5"), 
        require_binary=False,
        require_bbox=False,
        require_masks=True
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_epochs = 10
    batch_size = 8
   
    model, train_loss, valid_loss = train(oxpet_train, oxpet_valid, batch_size, num_epochs, device, 'COCO_transferred')

    testloader = DataLoader(oxpet_test, batch_size=batch_size,shuffle=True)

    precision, recall, accuracy, F_1, IOU = 0, 0, 0, 0, 0
    for i, data in enumerate(testloader):
        images,targets = data
        images = normalize(images)
        images,targets = images.to(device),targets.to(device)
        p, r, a, f, iou = utils.Evaluation_mask(model, images, targets.squeeze())
        precision += p
        recall += r
        accuracy += a
        F_1 += f
        IOU += iou

    print(f'accuracy={accuracy/(i+1)}')
    print(f'recall={recall/(i+1)}')
    print(f'precision={precision/(i+1)}')
    print(f'f1={F_1/(i+1)}')
    print(f'IOU={IOU/(i+1)}\n')

    stat = [float(accuracy/(i+1)), float(recall/(i+1)), float(precision/(i+1)), float(F_1/(i+1)), float(IOU/(i+1))]

    stat_name = 'COCO'
    stat_file_name = 'COCO_stats.csv'

    # Output the stats
    with open(stat_file_name, 'w') as f:

        write = csv.writer(f)
        write.writerow(train_loss)
        write.writerow(valid_loss)
        write.writerow(stat)
    
    print(f'{stat_name} statistics has been saved to {stat_file_name}')