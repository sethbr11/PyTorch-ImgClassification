# Train the model
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from config import get_config, get_device
from model import build_transformer
from dataset import get_dataloaders
from train_validation import validate

import warnings

def get_model(config):
    model = build_transformer(config)
    return model

def train_model(config, validate_with_images=False):
    # Initialize model, loss, and optimizer
    device = get_device()
    model = get_model(config).to(device)
    train_loader, test_loader = get_dataloaders(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Training loop
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
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Run validation
        validate(model, test_loader, device, epoch+1, validate_with_images)

    # Evaluate model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description='Train a Transformer model.')
    parser.add_argument('--valimg', action='store_true', help='Show images used in validation')
    args = parser.parse_args()
    
    # Display validation images if the flag is set
    if args.valimg:
        print("Validation images will be displayed after each epoch.")
    
    config = get_config()
    train_model(config, args.valimg)