import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import WeightedRandomSampler
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader


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
                    
                    #---------tweak augmentation
                    A.OneOf([
                            A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=15, interpolation = 1, p = 0.2),
                            A.RandomBrightnessContrast(brightness_limit =[-0.2, 0.2], contrast_limit =[-0.3, 0.3], p = 0.2)
                            ], p = 0.3),
                                
                    A.OneOf([
                            A.ImageCompression(quality_lower=70, quality_upper=80, p = 0.2),
                            A.Affine(shear=(-8, 8), rotate=(-8, 8), p=0.2)
                            ], p =0.3),
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

def get_dataloader(data_dir, batch_size, data_mean, data_std, img_size, dataset_type, weights_sampler =False):
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
        
        return loader, dataset
        
    # No weighted sampler for val set
    loader = DataLoader(dataset, batch_size = batch_size)
    return loader, dataset