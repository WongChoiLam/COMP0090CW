import matplotlib.pyplot as plt
import torch
import torchvision
from Oxpet_Dataset import Oxpet_Dataset
import os
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def show_images_Base_Ablation_COCO(models, origin_images,targets,batch_size):

    outputs_Baseline = models['Baseline'](origin_images)['out']
    outputs_Baseline = torch.argmax(outputs_Baseline,dim=1)
    outputs_Abaltion = models['Abaltion'](origin_images)['out']
    outputs_Abaltion = torch.argmax(outputs_Abaltion,dim=1)
    outputs_COCO = models['COCO'](origin_images)['out']
    outputs_COCO = torch.argmax(outputs_COCO,dim=1)
    
    plt.figure(figsize=(20,10))
    for i in range(8):
        plt.subplot(batch_size,5,5*i+1)
        plt.imshow(origin_images[i].permute(1,2,0))
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('Origin')
        plt.subplot(batch_size,5,5*i+2)
        plt.imshow(outputs_Baseline[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('BaseLine')
        plt.subplot(batch_size,5,5*i+3)
        plt.imshow(outputs_Abaltion[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('Abaltion')
        plt.subplot(batch_size,5,5*i+4)
        plt.imshow(outputs_COCO[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('COCO')
        plt.subplot(batch_size,5,5*i+5)
        plt.imshow(targets[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('target')
    plt.show()

def show_images_Base_OEQ(models, origin_images,targets,batch_size):
    # needs to change
    outputs_Baseline = models['Baseline'](origin_images)['out']
    outputs_Baseline = torch.argmax(outputs_Baseline,dim=1)
    outputs_ISIC2018 = models['ISIC2018'](origin_images)['out']
    outputs_ISIC2018 = torch.argmax(outputs_ISIC2018,dim=1)
    outputs_Cityscapes = models['Cityscapes'](origin_images)['out']
    outputs_Cityscapes = torch.argmax(outputs_Cityscapes,dim=1)
    outputs_MAS3K = models['MAS3K'](origin_images)['out']
    outputs_MAS3K = torch.argmax(outputs_MAS3K,dim=1)
    outputs_VOC2012 = models['VOC2012'](origin_images)['out']
    outputs_VOC2012 = torch.argmax(outputs_VOC2012,dim=1)
    
    plt.figure(figsize=(20,10))
    for i in range(8):
        plt.subplot(batch_size,7,7*i+1)
        plt.imshow(origin_images[i].permute(1,2,0))
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('Origin')
        plt.subplot(batch_size,5,5*i+2)
        plt.imshow(outputs_Baseline[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('BaseLine')
        plt.subplot(batch_size,5,5*i+3)
        plt.imshow(outputs_ISIC2018[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('ISIC2018')
        plt.subplot(batch_size,5,5*i+4)
        plt.imshow(outputs_Cityscapes[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('Cityscapes')
        plt.subplot(batch_size,7,7*i+5)
        plt.imshow(outputs_MAS3K[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('MAS3K')
        plt.subplot(batch_size,7,7*i+6)
        plt.imshow(outputs_VOC2012[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('VOC2012')
        plt.subplot(batch_size,7,7*i+7)
        plt.imshow(targets[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('target')
    plt.show()


def load_models_Base_Ablation_COCO(path):
    models = {}
    model_base = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    model_ablation = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model_ablation.classifier = DeepLabHead(2048, 2)
    model_COCO  = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model_COCO.classifier = DeepLabHead(2048, 2)
    
    model_base.load_state_dict(torch.load(path['Baseline']))
    model_ablation.load_state_dict(torch.load(path['Abaltion']))
    model_COCO.load_state_dict(torch.load(path['COCO']))
    
    model_base.eval()
    model_ablation.eval()
    model_COCO.eval()
    
    models['Baseline'] = model_base
    models['Abaltion'] = model_ablation
    models['COCO'] = model_COCO
    return models

def load_models_Base_OEQ(path):
    models = []
    #needs to implement
    
    return models 

oxpet_test = Oxpet_Dataset(
    os.path.join("datasets-oxpet-rewritten", "test","images.h5"),
    os.path.join("datasets-oxpet-rewritten", "test","binary.h5"), 
    os.path.join("datasets-oxpet-rewritten", "test","bboxes.h5"),
    os.path.join("datasets-oxpet-rewritten", "test","masks.h5"), 
    require_binary=False,
    require_bbox=False,
    require_masks=True
)
path_base_ablation_COCO = {'Baseline': 'BaseLine.pt', 'COCO': 'COCO_transferred.pt','Abaltion': 'COCO_transferred_unfreeze.pt'}
# needs to change
path_base_OEQ= {'Baseline': 'BaseLine.pt', 'COCO': 'COCO_transferred.pt','Abaltion': 'COCO_transferred_unfreeze.pt'}

testloader = DataLoader(oxpet_test, batch_size=8,shuffle=True)

images,targets = next(iter(testloader))
models_Base_Ablation_COCO = load_models_Base_Ablation_COCO(path_base_ablation_COCO)
show_images_Base_Ablation_COCO(models_Base_Ablation_COCO, images,targets,8)


models_OEQ = load_models_Base_OEQ(path_base_OEQ)
show_images_Base_OEQ(models_OEQ, images,targets,8)






    