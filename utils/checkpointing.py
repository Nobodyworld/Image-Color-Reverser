# /utils/checkpointing.py
import torch
import os

def save_checkpoint(state_dict, filename='other_model.pth'):

    # Ensure directory exists (extract directory from filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state_dict, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, filename, device):

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded from {filename}")
