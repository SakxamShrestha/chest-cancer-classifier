# src/web/backend/model.py
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import io
import logging

logger = logging.getLogger(__name__)

class ChestCancerClassifier(nn.Module):
    def __init__(self, num_classes=4, architecture="efficientnet_b0", pretrained=True):
        super().__init__()
        self.model = timm.create_model(architecture, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class PredictionPipeline:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        self.class_names = [
            "adenocarcinoma",
            "large.cell.carcinoma",
            "squamous.cell.carcinoma",
            "normal"
        ]
    
    def _load_model(self, model_path):
        try:
            model = ChestCancerClassifier(num_classes=len(self.class_names))
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_bytes):
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Transform image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # Get probabilities for all classes
            class_probabilities = {
                self.class_names[i]: float(prob) * 100
                for i, prob in enumerate(probabilities[0])
            }
            
            return {
                "predicted_class": self.class_names[predicted_class],
                "confidence": float(probabilities[0][predicted_class]) * 100,
                "class_probabilities": class_probabilities
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise