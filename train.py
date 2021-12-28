from UNet import UNet
from Oxpet_Dataset import Oxpet_Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import os

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 20
    trainset = Oxpet_Dataset(os.path.join("datasets-oxpet", "train","images.h5"),os.path.join("datasets-oxpet", "train","binary.h5"),os.path.join("datasets-oxpet", "train","bboxes.h5"),os.path.join("datasets-oxpet", "train","masks.h5"), require_binary=False, require_bbox=False)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle= True,num_workers=4)
    validset = Oxpet_Dataset(os.path.join("datasets-oxpet", "val","images.h5"),os.path.join("datasets-oxpet", "val","binary.h5"),os.path.join("datasets-oxpet", "val","bboxes.h5"),os.path.join("datasets-oxpet", "val","masks.h5"), require_binary=False, require_bbox=False)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle= True,num_workers=4)

    net = UNet(num_classes=2)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Training done.')

    # save trained model
    torch.save(net.state_dict(), 'saved_model.pt')
    print('Model saved.')