# /utils/data_loading.py
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.pairedimage_dataset import PairedImageDataset

def get_transforms(config):
    """Generate torchvision transforms based on config."""
    img_height = config['training']['img_height']
    img_width = config['training']['img_width']

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    return transform

def get_dataloaders(config):
    """Create DataLoader for training, validation, and testing datasets."""
    transform = get_transforms(config)

    # Assuming the same transform is applied to both 'before' and 'after' images
    train_dataset = PairedImageDataset(config['directories']['data']['train'], before_transform=transform, after_transform=transform)
    val_dataset = PairedImageDataset(config['directories']['data']['val'], before_transform=transform, after_transform=transform)
    test_dataset = PairedImageDataset(config['directories']['data']['test'], before_transform=transform, after_transform=transform)

    # Create DataLoaders using batch size from config
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=config['training']['shuffle'], pin_memory=config['training']['pin_memory'], num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=config['training']['shuffle'], pin_memory=config['training']['pin_memory'], num_workers=config['training']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=config['training']['shuffle'], pin_memory=config['training']['pin_memory'], num_workers=config['training']['num_workers'])

    return train_loader, val_loader, test_loader
