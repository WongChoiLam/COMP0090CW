from UNet import UNet
from Oxpet_Dataset import Oxpet_Dataset
from torch.utils.data import DataLoader
import utils
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 20
    trainset = Oxpet_Dataset(os.path.join("datasets-oxpet-rewritten", "train","images.h5"),os.path.join("datasets-oxpet-rewritten", "train","binary.h5"),os.path.join("datasets-oxpet-rewritten", "train","bboxes.h5"),os.path.join("datasets-oxpet-rewritten", "train","masks.h5"),False,False)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle= True,num_workers=0)
    # validset = Oxpet_Dataset(os.path.join("datasets-oxpet", "val","images.h5"),os.path.join("datasets-oxpet", "val","binary.h5"),os.path.join("datasets-oxpet", "val","bboxes.h5"),os.path.join("datasets-oxpet", "val","masks.h5"), require_binary=False, require_bbox=False)
    # validloader = DataLoader(validset, batch_size=batch_size, shuffle= True,num_workers=4)

    net = UNet(1).to(device)

    criterion = utils.DiceLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_loss = []

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))

            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            # if i % 20 == 19:
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 20))
            #     running_loss = 0.0
        train_loss.append(loss.item()/i)
    
    plt.plot(train_loss)
    plt.savefig('train_loss')

    print('Training done.')

    # save trained model
    torch.save(net.state_dict(), 'saved_model.pt')
    print('Model saved.')