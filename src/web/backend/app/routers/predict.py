from fastapi import APIRouter, UploadFile, File
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.models.model import ChestCancerClassifier
import io
import yaml
from pathlib import Path
import os

router = APIRouter()

# Get absolute paths
current_file = Path(__file__).resolve()
# Fix: Navigate up to ml-final-project (one more level up from src)
project_root = current_file.parent.parent.parent.parent.parent.parent  # Six levels up from predict.py
config_path = project_root / "configs" / "params.yaml"
model_path = project_root / "src" / "models" / "best_model.pth"

print(f"Current file: {current_file}")
print(f"Project root: {project_root}")
print(f"Looking for config at: {config_path}")
print(f"Model path: {model_path}")
print(f"Config exists: {config_path.exists()}")
print(f"Current working directory: {os.getcwd()}")

# Load config
try:
    with open(config_path) as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Available files in configs directory: {list((project_root / 'configs').glob('*'))}")
    raise

# Load model
# In predict.py, where we load the model
model = ChestCancerClassifier()
checkpoint = torch.load(str(model_path), map_location=torch.device('cpu'))
if 'model_state_dict' in checkpoint:
    # If it's a training checkpoint
    state_dict = checkpoint['model_state_dict']
else:
    # If it's just the model state dict
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = model.class_names[predicted.item()]
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        class_probs = {
            class_name: float(prob)
            for class_name, prob in zip(model.class_names, probabilities)
        }
    
    return {
        "prediction": predicted_class,
        "probabilities": class_probs
    }