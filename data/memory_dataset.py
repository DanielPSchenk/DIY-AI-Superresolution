import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

class MemoryData(data.Dataset):
    
    def __init__(self, image_folder, transform=None, start=0, size=100) -> None:
        super().__init__()
        files = os.listdir(image_folder)
        self.files = [file for file in files if file.endswith("jpg") or file.endswith("png")]
        self.tensor_images = []
        
        to_tensor = transforms.ToTensor()
        
        for file in [self.files[i] for i in range(start, start + size)]:
            image = Image.open(os.path.join(image_folder, file)).convert("RGB")
            self.tensor_images.append(to_tensor(image))
        self.transform = transform
        
    def __len__(self):
        return len(self.tensor_images)
            
    def __getitem__(self, key):
        image = self.tensor_images[key]
        if(self.transform):
            image = self.transform(image)        
        return image
    