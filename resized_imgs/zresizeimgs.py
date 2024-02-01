import os
from PIL import Image

# Set image dimensions
img_height = 384
img_width = 256

# Define data directory
input_dir = './resized_imgs'

# Check if directory exists
if not os.path.exists(input_dir):
    print(f"Directory '{input_dir}' does not exist. Please check the path.")
else:
    # Resize images in the same directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                image_path = os.path.join(input_dir, filename)
                with Image.open(image_path) as image:
                    width, height = image.size
                    if width != img_width or height != img_height:
                        image_resized = image.resize((img_width, img_height), Image.LANCZOS)
                        image_resized.save(image_path, quality=100)
                        print(f'Resized {filename}.')
                    else:
                        print(f'Skipped {filename} (same dimensions).')
            except IOError:
                print(f'Failed to open or process {filename}.')
        else:
            print(f'Ignored {filename} (not an image file).')

