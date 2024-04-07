import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

class SSDData(data.Dataset):
    
    def __init__(self, image_folder, transform, start = 0, size=100) -> None:
        super().__init__()
        files = os.listdir(image_folder)
        self.image_folder = image_folder
        self.files = [file for file in files if file.endswith("jpg") or file.endswith("png")]
        print(len(self.files))
        self.files = [self.files[i] for i in range(start, start + size)]
        self.tensor_images = []
        
        self.to_tensor = transforms.ToTensor()
        
            
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
            
    def __getitem__(self, key):
        image = self.to_tensor(Image.open(os.path.join(self.image_folder, self.files[key])).convert("RGB"))
        
        if(self.transform):
            image = self.transform(image)        
        return image
    