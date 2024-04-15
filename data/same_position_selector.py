import torchvision.transforms.v2 as transforms
import random
import numpy as np
import torch.nn.functional as f
import torch

class SamePartSelector(transforms.Transform):
    def __init__(self, minimum_size = 1024, output_size=1024, device="cuda", multiplier=2):
        self.minimum_size = minimum_size
        self.output_size = output_size
        self.device = device
        self.multiplier = multiplier
    
    def __call__(self, img):
        smaller_dimension = img.shape[1]
        if img.shape[2] < smaller_dimension:
            smaller_dimension = img.shape[2]
            
        #print(self.minimum_size)
        size = smaller_dimension
            
        #print(img.shape)
        #print(size)
        start_x = 0
        start_y = 0
        crop = img[:, start_x: start_x + size, start_y:start_y + size]
        rescaled_crop = f.interpolate(crop.unsqueeze(0), (self.output_size, self.output_size), mode="bilinear", antialias=True)
        del crop
        
        down_image = f.interpolate(rescaled_crop, scale_factor=(.5, .5), mode="bilinear", antialias=True)
        
        target = (rescaled_crop - f.interpolate(down_image, scale_factor=(2, 2), mode="bilinear")) * self.multiplier
        
        
        return (down_image.squeeze(0), target.squeeze(0))