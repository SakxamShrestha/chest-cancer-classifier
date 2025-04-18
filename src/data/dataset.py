import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import yaml

class ChestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Load parameters
        with open("configs/params.yaml") as f:
            self.params = yaml.safe_load(f)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Assuming directory structure .../class_name/image.jpg
        label = 1 if 'cancer' in os.path.dirname(img_path).lower() else 0
        
        return image, label