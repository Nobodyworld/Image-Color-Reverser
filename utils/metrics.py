# /utils/metrics.py
import torch

def psnr(pred, target, max_pixel=1.0, eps=1e-10, reduction='mean'):

    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr_val = 20 * torch.log10(max_pixel / torch.sqrt(mse + eps))

    if reduction == 'mean':
        return psnr_val.mean().cpu().item()
    elif reduction == 'none':
        return psnr_val.cpu().numpy()
    else:
        raise ValueError("Invalid reduction mode. Supported modes are 'mean' and 'none'.")