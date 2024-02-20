import os
from PIL import Image
import json

def load_config(config_path='config/config.json'):
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        raise Exception("config.json file not found.")
    except json.JSONDecodeError:
        raise Exception("Failed to decode config.json.")

img_height = load_config()['training']['img_height']
img_width = load_config()['training']['img_width']

# Get the current script directory
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
stage_data = os.path.join(project_root, "stage_data")

def orient_to_portrait(image):
    if image.width > image.height:
        image = image.rotate(90, expand=True)
    return image

def resize_image(image, target_size):
    return image.resize(target_size, Image.LANCZOS)

# Process images in the directory
for filename in os.listdir(stage_data):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            image_path = os.path.join(stage_data, filename)
            with Image.open(image_path) as image:
                image = orient_to_portrait(image)
                if image.size != (img_width, img_height):
                    image_resized = resize_image(image, (img_width, img_height))
                    image_resized.save(image_path, quality=100)
                    print(f'Resized {filename}.')
                else:
                    print(f'Skipped {filename} (same dimensions).')
        except Exception as e:
            print(f'Error processing {filename}: {e}')
    else:
        print(f'Ignored {filename} (not an image file).')
