import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import json
from pathlib import Path
from torchvision import transforms
import logging
from src.models.model import ChestCancerClassifier
from src.data.dataset import ChestDataset
from src.utils.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # Load parameters
    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((params['data']['image_size'], params['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        params['paths']['processed_data'],
        split='train',
        transform=train_transform
    )
    val_dataset = ChestDataset(
        params['paths']['processed_data'],
        split='valid',
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['data']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['data']['batch_size'],
        num_workers=4
    )
    
    # Initialize model
    model = ChestCancerClassifier().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['training']['learning_rate']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=params['training']['scheduler_patience'],
        factor=params['training']['scheduler_factor']
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(params['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{params['training']['epochs']}] "
                          f"Batch [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")
        
        # Validation phase
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        # Calculate training metrics
        train_metrics = calculate_metrics(train_targets, train_predictions)
        
        # Logging
        logger.info(f"Epoch [{epoch+1}/{params['training']['epochs']}]")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
        logger.info(f"Train Metrics: {train_metrics}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics: {val_metrics}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, Path(params['paths']['model_dir']) / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= params['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Save final metrics
    metrics = {
        'final_val_loss': float(val_loss),
        'final_val_metrics': val_metrics,
        'best_val_loss': float(best_val_loss)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

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