import torch
import torch.nn as nn
import torchvision.models as models
import yaml

class ChestCancerClassifier(nn.Module):
    def __init__(self):
        super(ChestCancerClassifier, self).__init__()
        
        # Load parameters
        with open("configs/params.yaml") as f:
            params = yaml.safe_load(f)
        
        # Use EfficientNet as backbone
        self.model = models.efficientnet_b0(pretrained=params['model']['pretrained'])
        
        # Modify classifier head for multi-class
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=params['model']['dropout_rate']),
            nn.Linear(num_features, params['model']['num_classes'])
        )
        
        self.class_names = params['class_names']
    
    def forward(self, x):
        return self.model(x)
    
    def predict_class(self, x):
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
            return [self.class_names[idx] for idx in predicted]
        