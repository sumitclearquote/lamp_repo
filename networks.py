from torchvision import transforms as t
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features , num_classes)
        
    def forward(self, x):
        output = self.resnet(x)
        return output
    
    
class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features , num_classes)
        
    def forward(self, x):
        output = self.resnet(x)
        return output
    
class EfficientNetB4(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(EfficientNetB4, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b4', pretrained=pretrained)
        num_features = self.efficientnet.classifier.fc.in_features
        self.efficientnet.classifier.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        output = self.efficientnet(x)
        return output






            
        
        
    
    
    