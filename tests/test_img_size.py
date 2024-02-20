# ./tests/test_img_size.py
import os
import json
from PIL import Image

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def find_images_with_different_size(root_dirs, width, height):
    for root_dir in root_dirs:
        print(f"Checking images in: {root_dir}")
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            if img.size != (width, height):
                                print(f"Found mismatched size in {file_path}: {img.size}")
                    except IOError:
                        print(f"Could not open {file_path}")

if __name__ == "__main__":
    config_path = os.path.join(os.getcwd(), 'config', 'config.json')
    config = load_config(config_path)

    # Extracting image dimensions from config
    img_width = config['training']['img_width']
    img_height = config['training']['img_height']

    # Collecting all data directories to check
    data_directories = [config['directories']['data'][key] for key in ['train', 'test', 'val', 'play_data', 'stage_data']]
    full_data_directories = [os.path.join(os.getcwd(), dir_path) for dir_path in data_directories]

    find_images_with_different_size(full_data_directories, img_width, img_height)
