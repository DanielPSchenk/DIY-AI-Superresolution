import torchvision.transforms.v2 as transforms
import random
import numpy as np
import torch.nn.functional as f
import torch

class RandomPartSelector(transforms.Transform):
    def __init__(self, minimum_size = 1024, output_size=1024, device="cuda"):
        self.minimum_size = minimum_size
        self.output_size = output_size
        self.device = device
    
    def __call__(self, img):
        smaller_dimension = img.shape[1]
        if img.shape[2] < smaller_dimension:
            smaller_dimension = img.shape[2]
            
        #print(self.minimum_size)
        size = self.minimum_size
        if smaller_dimension > self.minimum_size:
            size = random.randint(self.minimum_size, smaller_dimension)
        else:
            size = smaller_dimension
            
        #print(img.shape)
        #print(size)
        start_x = random.randint(0, img.shape[1] - size)
        start_y = random.randint(0, img.shape[2] - size)
        crop = img[:, start_x: start_x + size, start_y:start_y + size]
        rescaled_crop = f.interpolate(crop.unsqueeze(0), (self.output_size, self.output_size), mode="bilinear", antialias=True).squeeze(0)
        del crop
        return rescaled_crop.to(torch.float16)