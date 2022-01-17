import matplotlib.pyplot as plt
import torch
import torchvision
from Oxpet_Dataset import Oxpet_Dataset
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def show(models, origin_images,targets,batch_size):

    outputs_Baseline = models['Baseline'](origin_images)['out']
    outputs_Baseline = torch.argmax(outputs_Baseline,dim=1)
    outputs_ISIC2018 = models['ISIC'](origin_images)['out']
    outputs_ISIC2018 = torch.argmax(outputs_ISIC2018,dim=1)
    outputs_Cityscapes = models['City'](origin_images)['out']
    outputs_Cityscapes = torch.argmax(outputs_Cityscapes,dim=1)
    outputs_MAS3K = models['MAS3K'](origin_images)['out']
    outputs_MAS3K = torch.argmax(outputs_MAS3K,dim=1)
    outputs_VOC2012 = models['VOC'](origin_images)['out']
    outputs_VOC2012 = torch.argmax(outputs_VOC2012,dim=1)
    outputs_Abaltion = models['Ablation'](origin_images)['out']
    outputs_Abaltion = torch.argmax(outputs_Abaltion,dim=1)
    outputs_COCO = models['COCO'](origin_images)['out']
    outputs_COCO = torch.argmax(outputs_COCO,dim=1)
    
    plt.figure(figsize=(20,10))

    for i in range(batch_size):
        plt.subplot(batch_size,9,9*i+1)
        plt.imshow(origin_images[i].permute(1,2,0))
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('Origin', fontsize=20)
        plt.subplot(batch_size,9,9*i+2)
        plt.imshow(outputs_Baseline[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('Baseline', fontsize=20)
        plt.subplot(batch_size,9,9*i+3)
        plt.imshow(outputs_COCO[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('COCO', fontsize=20)
        plt.subplot(batch_size,9,9*i+4)
        plt.imshow(outputs_Abaltion[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('COCO Ablation', fontsize=20)
        plt.subplot(batch_size,9,9*i+5)
        plt.imshow(outputs_ISIC2018[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('ISIC2018', fontsize=20)
        plt.subplot(batch_size,9,9*i+6)
        plt.imshow(outputs_Cityscapes[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('Cityscapes', fontsize=20)
        plt.subplot(batch_size,9,9*i+7)
        plt.imshow(outputs_MAS3K[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('MAS3K', fontsize=20)
        plt.subplot(batch_size,9,9*i+8)
        plt.imshow(outputs_VOC2012[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('VOC2012', fontsize=20)
        plt.subplot(batch_size,9,9*i+9)
        plt.imshow(targets[i].reshape(256,256),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.title('Target', fontsize=20)
    plt.savefig('Visualisation.png')

def load_models(path):
    models = {}
    model_base = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    model_ISIC = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    model_MAS3K = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    model_City = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    model_VOC = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    model_ablation = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model_ablation.classifier = DeepLabHead(2048, 2)
    model_COCO  = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model_COCO.classifier = DeepLabHead(2048, 2)
    model_Ablation  = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model_Ablation.classifier = DeepLabHead(2048, 2)

    model_base.load_state_dict(torch.load(path['Baseline']))
    model_COCO.load_state_dict(torch.load(path['COCO']))
    model_Ablation.load_state_dict(torch.load(path['Ablation']))
    model_ISIC.load_state_dict(torch.load(path['ISIC']))
    model_MAS3K.load_state_dict(torch.load(path['MAS3K']))
    model_City.load_state_dict(torch.load(path['Cityscapes']))
    model_VOC.load_state_dict(torch.load(path['VOC']))

    model_base.eval()
    model_COCO.eval()
    model_Ablation.eval()
    model_ISIC.eval()
    model_MAS3K.eval()
    model_City.eval()
    model_VOC.eval()
    
    models['Baseline'] = model_base
    models['COCO'] = model_COCO
    models['Ablation'] = model_Ablation
    models['ISIC'] = model_ISIC
    models['MAS3K'] = model_MAS3K
    models['City'] = model_City
    models['VOC'] = model_VOC
    return models 

if __name__ == '__main__':
    import os
    
    oxpet_test = Oxpet_Dataset(
        os.path.join("datasets-oxpet-rewritten", "test","images.h5"),
        os.path.join("datasets-oxpet-rewritten", "test","binary.h5"), 
        os.path.join("datasets-oxpet-rewritten", "test","bboxes.h5"),
        os.path.join("datasets-oxpet-rewritten", "test","masks.h5"), 
        require_binary=False,
        require_bbox=False,
        require_masks=True
    )
    
    path = {'Baseline': 'BaseLine.pt', 'COCO': 'COCO_transferred.pt', 'Ablation': 'COCO_transferred_unfreeze.pt', 'ISIC': 'ISIC_transferred.pt','Cityscapes':'cityscapes_transferred.pt','VOC':'VOC_transferred.pt','MAS3K':'MAS3K_transferred.pt'}

    testloader = DataLoader(oxpet_test, batch_size=5,shuffle=True)

    images,targets = next(iter(testloader))
    models = load_models(path)
    show(models, images, targets, 5)