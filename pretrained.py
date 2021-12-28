from Oxpet_Dataset import Oxpet_Dataset

import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image

import torchvision

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs, bbox=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        if bbox != None and i == 0:
            for j in range(0, min(1, bbox.shape[0])):
                x1, y1, x2, y2 = bbox[j]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                axs[0, i].add_patch(rect)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        

training_data = Oxpet_Dataset(
    os.path.join("datasets-oxpet", "train","images.h5"),
    os.path.join("datasets-oxpet", "train","binary.h5"), 
    os.path.join("datasets-oxpet", "train","bboxes.h5"),
    os.path.join("datasets-oxpet", "train","masks.h5"), 
    require_binary=False,
    require_bbox=False,
    require_masks=False)
# print(training_data.__getitem__(0)[1].shape)
ox_dataloader = DataLoader(training_data, batch_size=1, shuffle= True,num_workers=4)

# see https://pytorch.org/vision/main/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.maskrcnn_resnet50_fpn
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

while True:
    imgs, = next(iter(ox_dataloader))
    print(imgs.shape)    
    imgs = list(img/255 for img in imgs)

    output = model(imgs)
    print(output[0]['labels'])
    show([imgs[0], output[0]['masks'][0]], output[0]['boxes'])
    # show(output[0]['masks'][0])
    plt.show()