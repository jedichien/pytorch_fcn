import os 
import numpy as np
import re
import random
import time
import scipy.misc
import torch
from glob import glob
from imageio import imread
from PIL import Image

mean=[0.485, 0.456, 0.460] # R, G, B
std =[0.229, 0.224, 0.225] # R, G, B


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_shape, isTrain=True, n_class=2):
        self.data_dir = data_dir
        self.image_shape = image_shape
        self.isTrain = isTrain
        self.class_map = np.array([[255, 0, 0], [255, 0, 255]], dtype=np.float32)
        if self.isTrain:
            self.image_paths = glob(os.path.join(self.data_dir, 'training', 'image_2', '*.png'))
            self.label_paths = {
                re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                for path in glob(os.path.join(self.data_dir, 'training', 'gt_image_2', '*_road_*.png'))
            }
        else:
            self.image_paths = glob(os.path.join(self.data_dir, 'testing', 'image_2', '*.png'))
        
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # raw
        img_path = self.image_paths[idx]
        image = np.array(Image.fromarray(imread(img_path)).resize(self.image_shape[0:2][::-1]))
        image = np.transpose(image, (2, 0, 1))/255.0
        image[0] = (image[0]-mean[0]) / std[0]
        image[1] = (image[1]-mean[1]) / std[1]
        image[2] = (image[2]-mean[2]) / std[2]
        image = torch.from_numpy(image.copy()).float()
        if self.isTrain:
            # label
            label_path = self.label_paths[os.path.basename(img_path)]
            label_img = np.array(Image.fromarray(imread(label_path)).resize(self.image_shape[0:2][::-1]))
            label = []
            # one-hot-like labels by class
            for (index, classmap) in enumerate(self.class_map):
                label_cls = np.all(label_img==classmap, axis=2).astype(np.uint8)
                label.append(label_cls)
            label = np.array(label, dtype=np.uint8)
            # to tensor
            label = torch.from_numpy(label.copy()).float()
            sample = {
                'X': image,
                'Y': label
            }
        else:
            sample = {
                'X': image
            }
        return sample
    

def denormal(tensor):
    dtensor = tensor.clone()
    dtensor[:, 0, ...] = dtensor[:, 0, :, :].mul_(std[0]).add_(mean[0])
    dtensor[:, 1, ...] = dtensor[:, 1, :, :].mul_(std[1]).add_(mean[1])
    dtensor[:, 2, ...] = dtensor[:, 2, :, :].mul_(std[2]).add_(mean[2])
    return dtensor

def toRGB(tensor, dtype=np.uint8):
    dtensor = tensor.clone()
    dtensor = (dtensor.data.cpu().numpy().transpose(0, 2, 3, 1)*255).astype(dtype)
    return dtensor
