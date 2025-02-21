# Configuration file for the model
import torch

def get_config():
    # Hyperparameters that will always stay the same for this dataset
    fixed_params = {
        'img_size': 32,  # CIFAR-10 images are 32x32
        'num_classes': 10,  # CIFAR-10 has 10 classes
    }
    
    # Hyperparameters that can be tuned
    tunable_params = {
        'patch_size': 4,  # Each patch will be 4x4 pixels
        'embed_dim': 64,  # Size of the embedding vector
        'num_heads': 8,   # Multi-head attention heads
        'num_layers': 6,  # Transformer encoder layers
        'batch_size': 128, # Number of images in each batch. Adjust based on GPU memory
        'epochs': 100, # Max epochs to run. An epoch is a full pass through the dataset
        'warmup_epochs': 5, # Number of warmup epochs (using LinearLR scheduler)
        'patience': 15, # Minimum epochs to run before stopping early
        'loss_threshold': 1e-3, # Threshold for early stopping (difference in loss)
        'lr_start_val': 1e-3, # Initial learning rate
        'min_lr': 1e-6, # Minimum learning rate
        'weight_decay': 0.05, # Weight decay for AdamW optimizer
        'dropout_rate': 0.1, # Dropout rate
    }
    
    # Combine both dictionaries and return
    return {**fixed_params, **tunable_params}
    
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
