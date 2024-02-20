# /utils/plotting.py
import matplotlib.pyplot as plt
from threading import Timer
import os


def plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs, config):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_psnrs, label="Training PSNR")
    plt.plot(val_psnrs, label="Validation PSNR")
    plt.title("Training and Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.grid(True)

    # Determine the base save path from config
    model_dir = config['model']['path']
    base_filename = "metrics_figure"
    file_extension = ".png"
    save_path = os.path.join(model_dir, base_filename + file_extension)
    
    # Check if the file exists and append a suffix if it does
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(model_dir, f"{base_filename}_{counter}{file_extension}")
        counter += 1

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics figure saved to {save_path}")