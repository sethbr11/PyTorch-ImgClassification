# Train the model
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from config import get_config, get_device
from model import build_transformer, build_cnn
from dataset import get_dataloaders
from train_validation import validate

import warnings

def get_model(config):
    model = build_transformer(config)
    return model

def train_model(config, validate_with_images=False, use_cnn=False, dynamic_epochs=False):
    # Initialize model, loss, optimizer, and scheduler
    device = get_device()
    if use_cnn:
        model = build_cnn(config).to(device)
    else:
        model = get_model(config).to(device)

    train_loader, val_loader, test_loader = get_dataloaders(config)
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config['lr_start_val'], betas=(0.9, 0.95), weight_decay=config['weight_decay'])
    
    # Use a scheduler to adjust the learning rate during training. Warmup for the first few epochs, then cosine annealing
    warmup_epochs = config['warmup_epochs']
    scheduler = lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[
            lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs), 
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'] - warmup_epochs, eta_min=config['min_lr'])
        ], 
        milestones=[warmup_epochs]
    )

    # Training loop
    loss_history_train = [] # Keep track of training loss values to check for overfitting
    loss_history_val = [] # Keep track of validation loss values to check for plateaued loss
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        loss_history_train.append(avg_train_loss)
        
        # Step the scheduler
        scheduler.step()
        
        # Print epoch loss and learning rate
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Run validation
        val_loss = validate(model, val_loader, device, epoch+1, validate_with_images)
        
        # Save the loss for early stopping
        loss_history_val.append(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if use_cnn:
                torch.save(model.state_dict(), 'best_cnn_model.pth')
            else:
                torch.save(model.state_dict(), 'best_vit_model.pth')
        
        # Check for overfitting:
        if len(loss_history_train) > 1 and len(loss_history_val) > 1:
            # Check if validation loss is increasing while training loss is still decreasing
            if loss_history_val[-1] > loss_history_val[-2] * 1.05 and loss_history_train[-1] < loss_history_train[-2]:
                print(f"Potential overfitting detected at epoch {epoch+1}. Validation loss increased significantly while training loss is still decreasing.")

        # Check for early stopping
        if dynamic_epochs and epoch >= config['patience']:
            recent_losses = loss_history_train[-config['patience']:]
            if np.std(recent_losses) < config['loss_threshold']:
                print(f"Stopping early at epoch {epoch+1} due to low loss standard deviation.")
                break

    # Load the best model saved during training
    if use_cnn:
        model.load_state_dict(torch.load('best_cnn_model.pth'))
    else:
        model.load_state_dict(torch.load('best_vit_model.pth'))

    # Evaluate the best model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy of the best model: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description='Train a Transformer model.')
    parser.add_argument('--valimg', action='store_true', help='Show images used in validation')
    parser.add_argument('--cnn', action='store_true', help='Use CNN instead of Transformer')
    parser.add_argument('--dynamic_epochs', action='store_true', help='Use dynamic epochs')
    args = parser.parse_args()
    
    # Feedback to the user based on the arguments
    if args.valimg:
        print("> Validation images will be displayed after each epoch.")
    if args.cnn:
        print("> Using CNN instead of ViT model.")
    if args.dynamic_epochs:
        print("> Dynamic epoch adjustment is enabled.")
    
    config = get_config()
    train_model(config, args.valimg, args.cnn, args.dynamic_epochs)