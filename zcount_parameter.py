from model import UNet
import torch

model = UNet()
model.load_state_dict(torch.load('best_psnr_denocoder_pytorch.pth'))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {total_params}")