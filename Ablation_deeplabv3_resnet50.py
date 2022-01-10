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

def K_Fold_split(data_len, k):
    """Split the data into k-fold data, return the according training and validation indices

    Args:
        data (Tensor): Data to be split
        k (scalar): number of k-fold required

    Returns:
        list: The indices of training and validation indices
    """    
    indices = torch.randperm(data_len)
    ratio = data_len//k
    indices_list = []

    for i in range(1, k + 1):
        test_ind = indices[(i-1)*ratio:i*ratio]
        train_ind = torch.cat([indices[:(i-1)*ratio], indices[i*ratio:]])
        indices_list.append((train_ind, test_ind))
        
    return indices_list

def train(trainloader, validloader, num_epochs, device, freeze, save, model_name):

    transforms = torch.nn.Sequential(
            T.RandomHorizontalFlip(p=1),
        )

    # Transfer Learning
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    if freeze:
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

    if save:
        min_valid_loss = float('inf')

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

        running_loss = 0
        for i, val_data in enumerate(validloader):
            inputs, labels = val_data
            labels = labels.squeeze()

            inputs, labels = inputs.to(device), labels.to(device)  

            # forward + backward + optimize
            outputs = model(inputs)['out']
            loss = criterion(outputs, labels) 
            running_loss += loss.item()  

        if save:
            
            # Save the best model
            if running_loss/(i+1) < min_valid_loss:
                print(f'epoch {epoch+1}, validation loss = {running_loss/(i+1)}, lowest validation loss = True, save model')  
                torch.save(model.state_dict(), f'{model_name}.pt')    
                min_valid_loss = running_loss/(i+1)
            else:
                print(f'epoch {epoch+1}, validation loss = {running_loss/(i+1)}, lowest validation loss = False, do not save model')
        else:
            print(f'epoch {epoch+1}, val loss = {running_loss/(i+1)}')          
        valid_loss.append(running_loss/(i+1))    

    if save:
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

    # Ablation Study
    train_losses_freeze = []
    valid_losses_freeze = []
    train_losses_unfreeze = []
    valid_losses_unfreeze = []
    for kfold, (train_idxs, valid_idxs) in enumerate(K_Fold_split(oxpet_train.__len__(), 3)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idxs)
        trainloader = torch.utils.data.DataLoader(oxpet_train, batch_size=batch_size, sampler=train_subsampler)
        validloader = torch.utils.data.DataLoader(oxpet_train, batch_size=batch_size, sampler=valid_subsampler)

        # Experiment with freezed version
        print(f'Kfold {kfold+1}, freezed version')
        model, train_loss, valid_loss = train(trainloader, validloader, num_epochs, device, True, False, None)
        train_losses_freeze.append(train_loss)
        valid_losses_freeze.append(valid_loss)
        print('\n')
        # Experiment with unfreezed version
        print(f'Kfold {kfold+1}, unfreezed version')
        model, train_loss, valid_loss = train(trainloader, validloader, num_epochs, device, False, False, None)
        train_losses_unfreeze.append(train_loss)
        valid_losses_unfreeze.append(valid_loss)
        print('\n')

    trainloader = DataLoader(oxpet_test, batch_size=batch_size,shuffle=True)
    validloader = DataLoader(oxpet_test, batch_size=batch_size,shuffle=True)
    testloader = DataLoader(oxpet_test, batch_size=batch_size,shuffle=True)

    # Use Whole training set to train and see if the trend agrees with cross validation
    model_unfreeze, train_loss, valid_loss = train(trainloader, validloader, num_epochs, device, False, True, 'COCO_transferred_unfreeze')

    # Load the trained freezed version
    model_freeze = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model_freeze.classifier = DeepLabHead(2048, 2)
    model_freeze.load_state_dict(torch.load('COCO_transferred.pt'))
    model_freeze.eval()
    model_freeze.to(device)

    precision_unfreeze, recall_unfreeze, accuracy_unfreeze, F_1_unfreeze, IOU_unfreeze = 0, 0, 0, 0, 0
    for i, data in enumerate(testloader):
        images,targets = data
        images,targets = images.to(device),targets.to(device)
        p, r, a, f, iou = utils.Evaluation_mask(model_unfreeze, images, targets.squeeze())
        precision_unfreeze += p
        recall_unfreeze += r
        accuracy_unfreeze += a
        F_1_unfreeze += f
        IOU_unfreeze += iou

    precision_freeze, recall_freeze, accuracy_freeze, F_1_freeze, IOU_freeze = 0, 0, 0, 0, 0
    for i, data in enumerate(testloader):
        images,targets = data
        images,targets = images.to(device),targets.to(device)
        p, r, a, f, iou = utils.Evaluation_mask(model_freeze, images, targets.squeeze())
        precision_freeze += p
        recall_freeze += r
        accuracy_freeze += a
        F_1_freeze += f
        IOU_freeze += iou

    print(f'unfreezed accuracy={accuracy_unfreeze/(i+1)}, freezed accuracy={accuracy_freeze/(i+1)}')
    print(f'unfreezed recall={recall_unfreeze/(i+1)}, freezed recall={recall_freeze/(i+1)}')
    print(f'unfreezed precision={precision_unfreeze/(i+1)}, freezed precision={precision_freeze/(i+1)}')
    print(f'unfreezed F_1={F_1_unfreeze/(i+1)}, freezed F_1={F_1_freeze/(i+1)}')
    print(f'unfreezed IOU={IOU_unfreeze/(i+1)}, freezed IOU={IOU_freeze/(i+1)}')

    stat_unfreeze = [float(accuracy_unfreeze/(i+1)), float(recall_unfreeze/(i+1)), float(precision_unfreeze/(i+1)), float(F_1_unfreeze/(i+1)), float(IOU_unfreeze/(i+1))]
    stat_freeze = [float(accuracy_freeze/(i+1)), float(recall_freeze/(i+1)), float(precision_freeze/(i+1)), float(F_1_freeze/(i+1)), float(IOU_freeze/(i+1))]

    stat_name = 'Ablation'
    stat_file_name = 'Ablation_stats.csv'

    # Output the stats
    with open(stat_file_name, 'w') as f:

        write = csv.writer(f)
        write.writerows(train_losses_freeze)
        write.writerows(valid_losses_freeze)
        write.writerows(train_losses_unfreeze)
        write.writerows(valid_losses_unfreeze)
        write.writerow(train_loss)
        write.writerow(valid_loss)
        write.writerow(stat_unfreeze)
        write.writerow(stat_freeze)
    
    print(f'{stat_name} statistics has been saved to {stat_file_name}')