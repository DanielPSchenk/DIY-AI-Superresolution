import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import ResNet34_Weights, resnet34


class ResBlock(nn.Module):
    def __init__(self, channels) -> None:
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, padding="same")
        )
        
    def forward(self, x):
        output = self.network.forward(x)
        y = output + x
        return y
    

class CNN_Backbone(nn.Module):
    def __init__(self, hp) -> None:
        super().__init__()
        
        self.num_blocks = hp["num_blocks"]
        dimension = hp["backbone_dimension"]
        self.transform_dimension = nn.Sequential(
            nn.Conv2d(3, dimension, 1, padding = "same"),
            nn.LeakyReLU()
        )
        self.blocks = [ResBlock(dimension) for i in range(self.num_blocks)]
        
    def forward(self, x):
        output = self.transform_dimension.forward(x)
        for block in self.blocks:
            output = block.forward(output)
            
        return output


class UpscalerResNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
               
        weights = ResNet34_Weights.DEFAULT
        self.resnet_preprocess = weights.transforms()
        
        classifier_list = list(resnet34(pretrained=True).children())
        
        self.pretrained_classifier_layers = [torch.nn.Sequential(*(classifier_list[:-i])) for i in range(len(classifier_list) - 1, 0, -1)]
        
        self.out_dims = [layer.forward(torch.zeros(1, 3, 256, 256)).shape for layer in self.pretrained_classifier_layers]
        
        print(self.out_dims)