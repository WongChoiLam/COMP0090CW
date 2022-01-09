import os
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as T
import utils
from torch.utils.data import DataLoader
from Oxpet_Dataset import Oxpet_Dataset
from ISIC2018_Dataset import ISIC2018_Dataset
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import csv

def train(trainset, val_set, batch_size, num_epochs, device, num_classes, model_name, pretrained_model):

    trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
    if pretrained_model:
        validloader = DataLoader(val_set, batch_size=batch_size,shuffle=True)

    transforms = torch.nn.Sequential(
            T.RandomHorizontalFlip(p=1),
        )

    # initialize model
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)

    # Transfer Learning
    if pretrained_model:

        # Load Pretrained model
        model.load_state_dict(torch.load(pretrained_model))
        model.eval()

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad=False
        
        # Add new classifier
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

    if pretrained_model:
        print('Start transfer learning...')
    else:
        print('Start pretraining model...')

    # fit
    for epoch in range(num_epochs):
        running_loss = 0
        for i, train_data in enumerate(trainloader, 0):
            inputs, labels = train_data
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

        if pretrained_model:
            running_loss = 0
            for i, val_data in enumerate(validloader):
                inputs, labels = val_data
                labels = labels.squeeze()

                inputs, labels = inputs.to(device), labels.to(device)  

                # forward + backward + optimize
                outputs = model(inputs)['out']
                loss = criterion(outputs, labels) 
                running_loss += loss.item()  

            print(f'epoch {epoch+1}, val loss = {running_loss/(i+1)}')          
            valid_loss.append(running_loss/(i+1))

    #Save Model
    torch.save(model.state_dict(), f'{model_name}.pt')

    if pretrained_model:
        print(f'Transfer Learned model has been saved to {model_name}.pt\n')
    else:
        print(f'Pretrained model has been saved to {model_name}.pt\n')

    return model, train_loss, valid_loss

if __name__ == '__main__':

    ISIC_dataset = ISIC2018_Dataset(os.path.join('ISIC-rewritten', 'ISIC2018_Task1-2_Training_Input.h5'), os.path.join('ISIC-rewritten', 'ISIC2018_Task1_Training_GroundTruth.h5'))
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

    # Train the model with ISIC
    train(ISIC_dataset, None, batch_size, num_epochs, device, 2, 'ISIC_pretrained', None)
    
    # Do transfer Learning with oxpet
    ISIC_transferred, train_loss, valid_loss = train(oxpet_train, oxpet_valid, batch_size, num_epochs, device, 2, 'ISIC_trainsferred', 'ISIC_pretrained.pt')
    
    # Test the results
    testloader = DataLoader(oxpet_test, batch_size=batch_size,shuffle=True)

    precision, recall, accuracy, F_1, IOU = 0, 0, 0, 0, 0
    for i, data in enumerate(testloader):
        images,targets = data
        images,targets = images.to(device),targets.to(device)
        p, r, a, f, iou = utils.Evaluation_mask(ISIC_transferred, images, targets.squeeze())
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

    # Output the stats
    with open('ISIC_stats.csv', 'w') as f:

        write = csv.writer(f)
        write.writerow(train_loss)
        write.writerow(valid_loss)
        write.writerow(stat)
    
    print('ISIC statistics has been saved to ISIC_stats.csv')