import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

def get_dataloaders(config):
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Randomly crop & resize
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Slightly increased rotation range
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),  # Convert image to tensor before normalization
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Cutout regularization
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Only normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)

    # Split the training data for validation (90% for training, 10% for validation)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    # num_workers is set to 4 to speed up data loading
    # pin_memory is set to True to speed up data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader