import os
import shutil
from pathlib import Path
import yaml
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data():
    # Load parameters
    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)
    
    raw_data_dir = Path(params['paths']['raw_data']) / 'data'
    processed_data_dir = Path(params['paths']['processed_data'])
    
    # Class mapping
    class_mapping = {
        'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'adenocarcinoma',
        'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'large_cell_carcinoma',
        'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'squamous_cell_carcinoma',
        'normal': 'normal'
    }
    
    # Process each split (train, test, valid)
    for split in ['train', 'test', 'valid']:
        logger.info(f"Processing {split} split...")
        split_dir = raw_data_dir / split
        
        # Process each class
        for orig_class_name, new_class_name in class_mapping.items():
            source_dir = split_dir / orig_class_name
            target_dir = processed_data_dir / split / new_class_name
            
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Process images
            if source_dir.exists():
                process_class_directory(
                    source_dir,
                    target_dir,
                    params['data']['image_size']
                )
                logger.info(f"Processed {orig_class_name} -> {new_class_name}")
            else:
                logger.warning(f"Directory not found: {source_dir}")

def process_class_directory(source_dir, target_dir, image_size):
    """Process all images in a class directory."""
    for img_path in source_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            target_path = target_dir / img_path.name
            process_image(img_path, target_path, image_size)

def process_image(input_path, output_path, image_size):
    """Process a single image."""
    try:
        with Image.open(input_path) as img:
            img = img.convert('RGB')
            img = img.resize((image_size, image_size))
            img.save(output_path)
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    preprocess_data()