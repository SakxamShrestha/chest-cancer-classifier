import torch
import torch.nn as nn
import timm  # We'll use timm's EfficientNet implementation
import yaml
from pathlib import Path

class ChestCancerClassifier(nn.Module):
    def __init__(self):
        super(ChestCancerClassifier, self).__init__()
        
        # Get absolute path to config file
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        config_path = project_root / "configs" / "params.yaml"
        
        # Load parameters
        with open(config_path) as f:
            params = yaml.safe_load(f)
        
        # Use EfficientNet from timm
        self.model = timm.create_model('efficientnet_b0', pretrained=params['model']['pretrained'])
        
        # Modify classifier head for multi-class
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=params['model']['dropout_rate']),
            nn.Linear(num_features, params['model']['num_classes'])
        )
        
        # Store class names
        self.class_names = params['class_names']
    
    def forward(self, x):
        return self.model(x)
    
    def predict_class(self, x):
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
            return [self.class_names[idx] for idx in predicted]

    @property
    def num_classes(self):
        return len(self.class_names)