import os
import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image

def preprocess_data():
    # Load parameters
    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)
    
    raw_data_dir = Path(params['paths']['raw_data'])
    processed_data_dir = Path(params['paths']['processed_data'])
    
    # Create processed data directories
    for split in ['train', 'val', 'test']:
        for class_name in ['cancer', 'normal']:
            (processed_data_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name in ['cancer', 'normal']:
        images = list((raw_data_dir / class_name).glob('*.jpg'))
        
        # Split data
        train_imgs, temp_imgs = train_test_split(
            images, 
            train_size=params['data']['train_split'],
            random_state=42
        )
        
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            train_size=params['data']['val_split']/(params['data']['val_split'] + params['data']['test_split']),
            random_state=42
        )
        
        # Process and save images
        for img_path, split in zip(
            [*train_imgs, *val_imgs, *test_imgs],
            ['train']*len(train_imgs) + ['val']*len(val_imgs) + ['test']*len(test_imgs)
        ):
            process_image(img_path, processed_data_dir / split / class_name, params)

def process_image(img_path, output_dir, params):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((params['data']['image_size'], params['data']['image_size']))
    img.save(output_dir / img_path.name)

if __name__ == "__main__":
    preprocess_data()