import torch
import torch.nn as nn
import torchvision.models as models
import yaml

class ChestCancerClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ChestCancerClassifier, self).__init__()
        
        # Load model parameters
        with open("configs/params.yaml") as f:
            params = yaml.safe_load(f)
        
        # Use EfficientNet as backbone
        self.model = models.efficientnet_b0(pretrained=params['model']['pretrained'])
        
        # Modify classifier head
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=params['model']['dropout_rate']),
            nn.Linear(num_features, params['model']['num_classes'])
        )
    
    def forward(self, x):
        return self.model(x)
