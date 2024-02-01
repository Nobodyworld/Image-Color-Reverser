import os
import glob
from PIL import Image

# Set the directory path
directory_path = r'C:\Users\Nobod\Documents\GitHub\Public-Unet-Image-Color-Reverser-or-Processor\resized_imgs'

# List all the files ending with '_before.jpg' in the directory
files_to_convert = glob.glob(os.path.join(directory_path, '*_before.jpg'))

for file_path in files_to_convert:
    try:
        # Open the image file
        img = Image.open(file_path)
        # Convert the image to grayscale
        img_gray = img.convert('L')
        # Save the grayscale image back to the same file
        img_gray.save(file_path)
        print(f'Converted {file_path} to grayscale.')
    except Exception as e:
        print(f'Error converting {file_path}: {e}')
