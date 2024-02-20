# /utils/training.py
import torch
from utils.metrics import psnr

def train_one_epoch(model, device, train_loader, optimizer, l1_criterion, mse_criterion, config):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    
    # Use get() method with a default value to avoid KeyError
    accumulation_steps = config.get('accumulation_steps', 1)
    
    for i, (before_imgs, after_imgs) in enumerate(train_loader):
        before_imgs, after_imgs = before_imgs.to(device), after_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(before_imgs)
        l1_loss = l1_criterion(outputs, after_imgs)
        mse_loss = mse_criterion(outputs, after_imgs)
        loss = l1_loss + mse_loss

        loss.backward()
        if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        batch_psnr = psnr(outputs, after_imgs, max_pixel=1.0)
        running_psnr += batch_psnr

    avg_loss = running_loss / len(train_loader)
    avg_psnr = running_psnr / len(train_loader)
    return avg_loss, avg_psnr


def validate(model, device, val_loader, l1_criterion, mse_criterion):
    model.eval()
    val_running_loss = 0.0
    val_running_psnr = 0.0

    with torch.no_grad():
        for before_imgs, after_imgs in val_loader:
            before_imgs, after_imgs = before_imgs.to(device), after_imgs.to(device)
            outputs = model(before_imgs)
            l1_loss = l1_criterion(outputs, after_imgs)
            mse_loss = mse_criterion(outputs, after_imgs)
            loss = l1_loss + mse_loss

            val_running_loss += loss.item()
            val_running_psnr += psnr(outputs, after_imgs)

    avg_loss = val_running_loss / len(val_loader)
    avg_psnr = val_running_psnr / len(val_loader)
    return avg_loss, avg_psnr
