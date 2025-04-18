import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import json
from pathlib import Path
from torchvision import transforms
from src.models.model import ChestCancerClassifier
from src.data.dataset import ChestDataset
from src.utils.metrics import calculate_metrics

def train():
    # Load parameters
    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((params['data']['image_size'], params['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((params['data']['image_size'], params['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ChestDataset(
        Path(params['paths']['processed_data']) / 'train',
        transform=train_transform
    )
    val_dataset = ChestDataset(
        Path(params['paths']['processed_data']) / 'val',
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['data']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['data']['batch_size']
    )
    
    # Initialize model
    model = ChestCancerClassifier(
        num_classes=params['model']['num_classes'],
        pretrained=params['model']['pretrained']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['training']['learning_rate']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=params['training']['scheduler_patience'],
        factor=params['training']['scheduler_factor']
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(params['training']['epochs']):
        model.train()
        train_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Path(params['paths']['model_dir']) / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= params['training']['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"Epoch [{epoch+1}/{params['training']['epochs']}]")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Metrics: {metrics}")
    
    # Save final metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(targets, predictions)
    return val_loss/len(dataloader), metrics

if __name__ == "__main__":
    train()