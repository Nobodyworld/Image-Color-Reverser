# ./tests/test_count_param.py
import torch
import json
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

# Assuming the script is executed from the project root directory
config_path = os.path.join(project_root, 'config', 'config.json')
config = load_config(config_path)

from model_parts.unet import UNet

# Initialize your model
model = UNet()

# Assuming model_checkpoint_path is relative to the project root
model_checkpoint_path = os.path.join(project_root, config['model']['path'], config['model']['file_name'])

state_dict = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {total_params}")
