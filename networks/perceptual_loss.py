import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as f

# Feature reconstruction loss similar to https://arxiv.org/pdf/1603.08155.pdf
# Using the first 5 layers of resnet18

class ShortenedResnet(nn.Module):
    def __init__(self, requires_grad = True, depth = 2):
        super().__init__()
        self.classifier_list = list(resnet18(pretrained=True).children())[:depth]
   
        for classifier, i in zip(self.classifier_list, range(len(self.classifier_list))):
            self.add_module("cl_"+str(i), classifier)
            for param in classifier.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):
        previous = x
        for layer in self.classifier_list:
            previous = layer.forward(previous)
            
        return previous
    
class FeatureReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_net = ShortenedResnet(False)
        self.prediction_net = ShortenedResnet()
        self.loss = MSELoss()
        
    def forward(self, prediction, target, down_image):
        up_image = f.interpolate(down_image, (prediction.shape[-2], prediction.shape[-1]), mode="bilinear", antialias=True)
        target_image = up_image + .5 * target
        prediction_image = up_image + .5 * prediction
        target_out = self.target_net.forward(target_image)
        prediction_out = self.forward(prediction_image)
        
        target_shape = target_out.shape
        divisor = target_shape[-1] * target_shape[-2] * target_shape[-3]
        
        loss = self.loss.forward(prediction_out, target_out) / divisor
        return loss
        
        
class StyleReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_net = ShortenedResnet(False)
        self.prediction_net = ShortenedResnet()
        self.loss = L1Loss()
        self.direct_loss = L1Loss()
        
    def forward(self, prediction, target, down_image):
        up_image = f.interpolate(down_image, (prediction.shape[-2], prediction.shape[-1]), mode="bilinear", antialias=True)
        target_image = up_image + .5 * target
        prediction_image = up_image + .5 * prediction
        target_out = self.target_net.forward(target_image)
        prediction_out = self.prediction_net.forward(prediction_image)
        
        target_shape = target_out.shape
        divisor = 1# target_shape[-1] * target_shape[-2] * target_shape[-3]
        
        loss = self.loss.forward(prediction_out, target_out) / divisor
        l1 = self.direct_loss.forward(prediction, target)
        return loss + .01 * l1
        
        
