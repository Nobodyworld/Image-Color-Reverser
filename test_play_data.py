import json
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_parts.unet import UNet

try:
    with open('config/config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("Error: config.json file not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: Failed to decode config.json.")
    exit(1)

play_data = os.path.join(config['directories']['data']['play_data'])
play_data_output = os.path.join(config['directories']['data']['play_data_output'])
model_path = os.path.join(config['model']['path'] + config['model']['file_name'])

print("Loading model from:", model_path)

img_height = config['training']['img_height']
img_width = config['training']['img_width']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet()
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file {model_path} not found.")
    exit(1)
except RuntimeError as e:
    print(f"Error loading the model: {e}")
    exit(1)

model = model.to(device)
model.eval()

before_transform = transforms.Compose([
    transforms.Resize((img_height, img_width), antialias=True),            
    transforms.ToTensor(),
])

def process_single_image(input_img_path, output_img_path, before_transform):
    try:
        input_img = Image.open(input_img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Input image {input_img_path} not found.")
        return
    except IOError:
        print(f"Error: Failed to open {input_img_path}.")
        return

    input_tensor = before_transform(input_img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_batch = model(input_batch)
    
    output_tensor = output_batch.squeeze(0).cpu()
    output_img = transforms.ToPILImage()(output_tensor)

    try:
        output_img.save(output_img_path)
    except IOError:
        print(f"Error: Failed to save {output_img_path}.")

def process_all_images(play_data, play_data_output, before_transform):
    if not os.path.isdir(play_data):
        print(f"Error: Input directory {play_data} does not exist.")
        return

    os.makedirs(play_data_output, exist_ok=True)

    for file_name in os.listdir(play_data):
        input_file_path = os.path.join(play_data, file_name)
        
        if os.path.isfile(input_file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_file_path = os.path.join(play_data_output, file_name).replace('_before', '_after')
            process_single_image(input_file_path, output_file_path, before_transform)

process_all_images(play_data, play_data_output, before_transform)