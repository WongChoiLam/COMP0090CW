import torch
import torchvision
import torch.nn as nn
from Oxpet_Dataset import Oxpet_Dataset
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
def Evaluation_binary(model, training_data, targets):
    prediction = model(training_data)
    confusion_matrix = torch.zeros(2,2)
    for t,p in zip(targets,prediction):
        if t == p:
            confusion_matrix[t.long(),p.long()] += 1
    precision = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    recall = confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[1,0])
    Accuracy = (confusion_matrix[0,0]+confusion_matrix[1,1])/targets.view(-1).size(0)
    F_1 = 2 * confusion_matrix[1,1]/(2 * confusion_matrix[1,1] + confusion_matrix[0,1]+confusion_matrix[1,0])
    return precision,recall,Accuracy,F_1

def Evaluation_bboxes(model, training_data, targets):
    IoU = []
    prediction = model(training_data)
    bboxes_hat = prediction[0]['boxes']
    for b_hat,t in zip(bboxes_hat,targets):   
        xy_max = torch.min(b_hat[2:], t[2:])
        xy_min = torch.max(b_hat[:2], t[:2])
        inter = torch.clamp(xy_max-xy_min, min=0)
        inter_area = inter[0]*inter[1]
        area_pre = (b_hat[2]-b_hat[0])*(b_hat[3]-b_hat[1])
        area_tar = (t[2]-t[0])*(t[3]-t[1])
        union = area_pre + area_tar- inter_area
        IoU.append(inter_area/union) 
    return IoU

def Evaluation_mask(model,training_data, targets):
    prediction = model(training_data)
    masks_hat = prediction['out'].softmax(1)
    masks_hat = torch.softmax(masks_hat,dim=1)       
    masks_hat = getBinaryTensor(masks_hat)
    confusion_matrix = torch.zeros(2,2)
    area_pre = masks_hat.sum()
    area_tar = targets.sum()
    inter_area  = 0
    for tar,mask in zip(targets,masks_hat):
        for row1,row2 in zip(tar.squeeze(),mask.squeeze()):
            for t,m in zip(row1,row2):
                if t == m:  
                    inter_area += 1
                    confusion_matrix[t.long(),m.long()] += 1
    union = area_pre + area_tar- inter_area
    IoU = inter_area/union
    precision = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    recall = confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[1,0])
    Accuracy = (confusion_matrix[0,0]+confusion_matrix[1,1])/targets.view(-1).size(0)
    F_1 = 2 * confusion_matrix[1,1]/(2 * confusion_matrix[1,1] + confusion_matrix[0,1]+confusion_matrix[1,0])
    return precision,recall,Accuracy,F_1,IoU

def getBinaryTensor(input, boundary = 0.5):
    one = torch.ones_like(input)
    zero = torch.zeros_like(input)
    return torch.where(input > boundary, one, zero)

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset = Oxpet_Dataset(
        os.path.join("datasets-oxpet-rewritten", "train","images.h5"),
        os.path.join("datasets-oxpet-rewritten", "train","binary.h5"), 
        os.path.join("datasets-oxpet-rewritten", "train","bboxes.h5"),
        os.path.join("datasets-oxpet-rewritten", "train","masks.h5"),
        require_binary=False,
        require_bbox=False,
        require_masks=True)
    
    data_loader = DataLoader(dataset,batch_size=4,shuffle=True)
    images,targets = next(iter(data_loader))
    
    images,targets = images.to(device),targets.to(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model = model.to(device)
    model.eval()
    
    res = model(images)['out']
    print(res.shape)
    res = res.argmax(1)
    print(res.shape)
    
    
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(res[2].byte().cpu().numpy()).resize((256, 256))
    r.putpalette(colors)

    plt.imshow(r)
    plt.show()

    
    