import torch
import torchvision
from Oxpet_Dataset_RAM import Oxpet_Dataset
import os
from torch.utils.data import DataLoader

def Evaluation_binary(model, training_data, targets):
    prediction = model(training_data)
    confusion_matrix = torch.zeros(2,2)
    for t,p in zip(targets,prediction):
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
    masks_hat = prediction[0]['mask']
    confusion_matrix = torch.zeros(2,2)
    for row1,row2 in zip(targets,masks_hat):
        for t,p in zip(row1,row2):
            confusion_matrix[t.long(),p.long()] += 1
    precision = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    recall = confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[1,0])
    Accuracy = (confusion_matrix[0,0]+confusion_matrix[1,1])/targets.view(-1).size(0)
    F_1 = 2 * confusion_matrix[1,1]/(2 * confusion_matrix[1,1] + confusion_matrix[0,1]+confusion_matrix[1,0])
    return precision,recall,Accuracy,F_1



if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset = Oxpet_Dataset(
        os.path.join("datasets-oxpet", "train","images.h5"),
        os.path.join("datasets-oxpet", "train","binary.h5"), 
        os.path.join("datasets-oxpet", "train","bboxes.h5"),
        os.path.join("datasets-oxpet", "train","masks.h5"), 
        require_binary=False,
        require_bbox=True,
        require_masks=False)
    
    images,targets = [x[0].numpy() for x in dataset],[x[1].numpy() for x in dataset]

    images,targets =  torch.tensor(images), torch.tensor(targets)

    images,targets = images[:10],targets[:10]
    images,targets = images.to(device),targets.to(device)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.eval()
    
    boxes_res = Evaluation_bboxes(model, images, targets)
    mask_res = Evaluation_mask(model,images, targets)
    # print(boxes_res)
    