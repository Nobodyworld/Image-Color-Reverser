# /utils/transforms.py
import torchvision
from utils.transforms import RandomColorJitterWithRandomFactors

def get_transforms(config):
    transforms_list = [torchvision.transforms.Resize((config['training']['img_height'], config['training']['img_width']))]

    if config['augmentation']['color_jitter']['enabled']:
        jitter_params = config['augmentation']['color_jitter']
        transforms_list.append(RandomColorJitterWithRandomFactors(
            brightness=jitter_params['brightness'],
            contrast=jitter_params['contrast'],
            saturation=jitter_params['saturation'],
            hue=jitter_params['hue'],
            p=jitter_params['p']
        ))

    transforms_list.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms_list)
