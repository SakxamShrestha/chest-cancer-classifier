import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import yaml
from pathlib import Path

class ChestDataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None):
        """
        Args:
            base_dir (str): Base directory of processed data
            split (str): One of 'train', 'test', or 'valid'
            transform: Optional transform to be applied
        """
        self.base_dir = Path(base_dir) / split
        self.transform = transform
        
        # Load parameters to get class names
        with open("configs/params.yaml") as f:
            params = yaml.safe_load(f)
        self.class_names = params['class_names']
        
        # Get all image paths and labels
        self.data = []
        
        # Add images from each class
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.base_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.data.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label