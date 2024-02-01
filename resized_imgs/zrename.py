import os

# Define data directory
input_dir = './resized_imgs'

# Initialize counter for new filenames
counter = 1

# Rename each image in the directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
        # Construct new filename
        new_filename = f'wikks_{str(counter).zfill(2)}_after{os.path.splitext(filename)[1]}'
        # Rename the file
        os.rename(os.path.join(input_dir, filename), os.path.join(input_dir, new_filename))
        # Increment counter
        counter += 1
        print(f'Renamed {filename} to {new_filename}.')
    else:
        print(f'Ignored {filename} (not an image file).')
