import os
import numpy as np
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import WeightedRandomSampler
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

classes = ['cracked', 'fadelamp', 'foggy']

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))

#--------------------------------------------------------------------------------------
def get_transform(data_mean, data_std, img_size, dataset_type):
    if dataset_type == "train":
        transform = A.Compose([
                    A.Flip(p = 0.2),
                    A.Resize(height = img_size[0], width = img_size[1], always_apply=True),
                    A.GaussNoise(var_limit=(60.0,200.0), p =0.4),
                    
                    #---------tweak augmentation
                    A.OneOf([A.RandomBrightnessContrast(brightness_limit =[-0.1, 0.1], contrast_limit =[-0.3, 0.3], p = 0.4),
                                A.ChannelShuffle(p = 0.4)
                            ], p=0.6),
                    
                    
                    A.OneOf([
                            A.ShiftScaleRotate(shift_limit=0.13, rotate_limit=15, interpolation = 1, p = 0.3),
                            A.Affine(shear=(-8, 8), rotate=(-8, 8), p=0.3)
                            ], p = 0.6),
                       
                    A.OneOf([
                            A.ImageCompression(quality_lower=30, quality_upper=40, p = 0.3),
                            A.Downscale(scale_min = 0.35, scale_max = 0.6, p = 0.3),
                            ], p =0.6),
                    #-----------------tweak augmentation
                    
                    A.Normalize(mean = data_mean, std = data_std, always_apply = True),
                    ToTensorV2(always_apply = True)
                    ])
                    
    elif dataset_type == "val":
        transform = A.Compose([
                        A.Resize(height = img_size[0], width = img_size[1],always_apply=True),
                        A.Normalize(mean = data_mean, std = data_std, always_apply = True),
                        ToTensorV2(always_apply = True)
                        ])
        
    return transform

#------------------------------------------------------------------------------------------

def get_dataloader_old(data_dir, batch_size, data_mean, data_std, img_size, dataset_type, weights_sampler =False):
    my_transforms  = get_transform(data_mean, data_std, img_size, dataset_type = dataset_type)
    
    dataset = datasets.ImageFolder(root = data_dir, transform = Transforms(transforms = my_transforms))
    
    if weights_sampler:
        class_weights=  []
        for class_dir in os.listdir(data_dir):
            if class_dir.endswith("Store"): continue
            total_images = len(os.listdir(f"{data_dir}/{class_dir}"))
            class_weights.append(1 / total_images)
            
        #inititate empty data weights
        sample_weights = [0] * len(dataset)
        
        #apply the class_weight to each data sample
        for idx, (data, label) in enumerate(dataset):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight
            
        sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement = True)
        
        loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler)
        
        return loader
        
    # No weighted sampler for val set
    loader = DataLoader(dataset, batch_size = batch_size)
    return loader

#--------------------------------------------------------------------------------------

class LampDataset(Dataset):
    def __init__(self, data, albu_transform = None, transform = None):
        '''
        data: [[imgname, imgpath, ['damage1', 'damage2'], 1, 0 , 1],[], [] ... ]
        '''
        self.data = data
        self.transform = transform
        self.albu_transform = albu_transform
        self.transform = transform
        
    def __len__(self):
        ''' Returns the len of the dataset
        '''
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx][1] #index of imgpath in array in 1
        
        #Read image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Get Label
        label = self.data[idx][-3:].astype('float32') #index of the corresponding classes are from -3:
        
        #apply tranforms
        if self.albu_transform:
            image = self.albu_transform(image = image)['image']
        if self.transform:
            image = self.transform(Image.fromarray(np.uint8(image)))
            
    
        label = torch.tensor(label, dtype = torch.float32)
            
        return (image, label)
    
    
def get_dataloader(data_dir, data, batch_size, data_mean, data_std, img_size, dataset_type, weights_sampler =False):
    
    transform = get_transform(data_mean, data_std, img_size, dataset_type = dataset_type)
    
    dataset = LampDataset(data, albu_transform=transform)
    
    if weights_sampler:
        class_weights=  []
        for class_dir in classes:
            total_images = len(os.listdir(f"{data_dir}/{class_dir}"))
            class_weights.append(1 / total_images)
            
        #inititate empty data weights
        sample_weights = [0] * len(dataset)
        
        #apply the class_weight to each data sample. 
        for idx, (data, label) in enumerate(dataset):
            if label.sum().item() > 1: #More than one labels
                if label[1] == 1: # if fadelamp is one of them
                    class_weight = class_weights[1]
                    sample_weights[idx] = class_weight
                else: #Choose randomly between cracked and foggy
                    class_weight = class_weights[random.choice([0,2])]
                    sample_weights[idx] = class_weight
            else: # Only one label for the image. choose the corresponding weight
                class_weight = class_weights[label.tolist().index(1)] #get index where 1 is present
                sample_weights[idx] = class_weight
                        
        sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement = True)
        
        loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler)
        
        return loader
    
    # No weighted sampler for val set
    loader = DataLoader(dataset, batch_size = batch_size)
    return loader
    
    
    