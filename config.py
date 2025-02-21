# Configuration file for the model
import torch

def get_config():
    return {
        'img_size': 32,  # CIFAR-10 images are 32x32
        'patch_size': 4,  # Each patch will be 4x4 pixels
        'embed_dim': 128,  # Size of the embedding vector
        'num_heads': 8,   # Multi-head attention heads
        'num_layers': 12,  # Transformer encoder layers
        'num_classes': 10,  # CIFAR-10 has 10 classes
        'batch_size': 128, # Number of images in each batch
        'epochs': 20, # An epoch is a full pass through the dataset
        'lr': 1e-4, # Learning rate
        'weight_decay': 1e-4, # Weight decay
    }
    
def get_class_names():
    return {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
