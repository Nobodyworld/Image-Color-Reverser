import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import UNet

img_height = 384
img_width = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = UNet()
model.load_state_dict(torch.load("best_denocoder_pytorch.pth"))
model = model.to(device)
model.eval()

# Define transforms for the input image (noisy)
noisy_transform = transforms.Compose([
    transforms.Resize((img_height, img_width), antialias=None),            
    transforms.ToTensor(),
    #transforms.RandomHorizontalFlip(p=0.1),
    #transforms.RandomVerticalFlip(p=0.1),
    #RandomColorJitterWithRandomFactors(p=0.2),
    # Add any other noisy-specific transformations
])

clean_transform = transforms.Compose([
    transforms.Resize((img_height, img_width), antialias=None),   
    transforms.ToTensor(),   
    # Add any other clean-specific transformations
])


def process_single_image(input_img_path, output_img_path, noisy_transform, clean_transform):
    # Load the input image
    input_img = Image.open(input_img_path).convert('RGB')
    
    # Apply the noisy_transform
    input_tensor = noisy_transform(input_img)
    
    # Add a batch dimension
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Run the model
    with torch.no_grad():
        output_batch = model(input_batch)
    
    # Remove the batch dimension and convert the tensor back to an image
    output_tensor = output_batch.squeeze(0).cpu()
    output_img = transforms.ToPILImage()(output_tensor)
    
    # Save the output image
    output_img.save(output_img_path)

def process_all_images(input_dir, output_dir, noisy_transform, clean_transform):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file_name)
        
        # Check if the file is an image
        if os.path.isfile(input_file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_file_path = os.path.join(output_dir, file_name)
            process_single_image(input_file_path, output_file_path, noisy_transform, clean_transform)

# Usage example
input_dir = './test_these'
output_dir = './test_these/output'
process_all_images(input_dir, output_dir, noisy_transform, clean_transform)
