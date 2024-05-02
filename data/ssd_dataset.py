import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

class SSDData(data.Dataset):
    
    def __init__(self, image_folders, transform, start = 0, size=100) -> None:
        super().__init__()
        files = []
        
        for image_folder in image_folders:
            files = files + [os.path.join(image_folder, file) for file in os.listdir(image_folder)]
        #self.image_folder = image_folder
        self.to_tensor = transforms.ToTensor()
        self.files = [file for file in files if file.endswith("jpg") or file.endswith("png") and self.to_tensor(Image.open(file).convert("RGB")).max() <= 1]
        print(len(self.files))
        self.files = [self.files[i] for i in range(start, start + size)]
        
        
        
        
            
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
            
    def __getitem__(self, key):
        image = self.to_tensor(Image.open(self.files[key]).convert("RGB"))
        
        if(self.transform):
            image = self.transform(image)        
        return image
    