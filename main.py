# main.py
from utils.config_manager import load_config
from utils.data_loading import get_dataloaders
from utils.training import train_one_epoch, validate
from utils.testing import test
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.optim_scheduler import setup_optimizer_scheduler
from utils.plotting import plot_metrics
from model_parts.unet import UNet
import torch
import torch.nn as nn
import os

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = UNet().to(device)

    optimizer, scheduler = setup_optimizer_scheduler(model, config)
    l1_criterion = nn.L1Loss().to(device)
    mse_criterion = nn.MSELoss().to(device)
    
    model_path = os.path.join(config['model']['path'] + config['model']['file_name'])
    if os.path.exists(model_path):
        load_checkpoint(model, model_path, device)
        print(f"Model loaded from {model_path}.")
    else:
        print("No checkpoint found. Training model from scratch.")

    # Initialize best metrics and early stopping counters
    best_val_loss = float('inf')
    best_val_psnr = 0
    epochs_since_improvement = 0 
    train_losses, val_losses, train_psnrs, val_psnrs = [], [], [], []
    
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        train_loss, train_psnr = train_one_epoch(model, device, train_loader, optimizer, l1_criterion, mse_criterion, config['training'])
        val_loss, val_psnr = validate(model, device, val_loader, l1_criterion, mse_criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_psnrs.append(train_psnr)
        val_psnrs.append(val_psnr)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} summary: Train Loss: {train_loss}, Train PSNR: {train_psnr}, Val Loss: {val_loss}, Val PSNR: {val_psnr}")

        # Check for improvement in val_loss for checkpointing and early stopping
        if val_loss < best_val_loss or val_psnr > best_val_psnr:
            best_val_loss = min(val_loss, best_val_loss)
            best_val_psnr = max(val_psnr, best_val_psnr)
            epochs_since_improvement = 0
            save_checkpoint(model.state_dict(), model_path)
            print(f"Improved model checkpoint saved to {model_path}.")
        else:
            epochs_since_improvement += 1
        
        if epochs_since_improvement >= config['training']['early_stopping_patience']:
            print("Early stopping triggered.")
            break
        
    test_loss, test_psnr = test(model, device, test_loader, l1_criterion, mse_criterion)
    plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs, config)
    # Print statements for test_loss and test_psnr
    print(f"Test Loss: {test_loss}")
    print(f"Test PSNR: {test_psnr}")
    print(f"Final Test Loss: {test_loss}, Final Test PSNR: {test_psnr}.")

if __name__ == "__main__":
    main()
