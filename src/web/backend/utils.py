# src/web/backend/utils.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def transform_image(image: Image.Image) -> torch.Tensor:
    """
    Transform an input image for model prediction.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        torch.Tensor: Transformed image tensor
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image)
    except Exception as e:
        logger.error(f"Error transforming image: {e}")
        raise

def get_image_size(image: Image.Image) -> Tuple[int, int]:
    """
    Get the size of an input image.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        tuple: (width, height) of the image
    """
    return image.size

def validate_image(image: Image.Image) -> bool:
    """
    Validate if an image meets the requirements.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    min_size = 64  # Minimum dimension size
    width, height = get_image_size(image)
    
    if width < min_size or height < min_size:
        return False
    
    return True