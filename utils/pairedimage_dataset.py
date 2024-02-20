# /utils/pairedimage_dataset.py
from PIL import Image
import os
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    def __init__(self, directory, before_transform=None, after_transform=None):
        self.directory = directory
        self.before_transform = before_transform
        self.after_transform = after_transform
        self.image_pairs = [file for file in os.listdir(directory) if '_before' in file]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        before_image_name = self.image_pairs[idx]
        after_image_name = before_image_name.replace('_before', '_after')
        
        before_image_path = os.path.join(self.directory, before_image_name)
        after_image_path = os.path.join(self.directory, after_image_name)

        before_image = Image.open(before_image_path).convert("RGB")
        after_image = Image.open(after_image_path).convert("RGB")

        if self.before_transform:
            before_image = self.before_transform(before_image)
        if self.after_transform:
            after_image = self.after_transform(after_image)

        return before_image, after_image