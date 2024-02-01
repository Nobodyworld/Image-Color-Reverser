import os

# Define data directory
input_dir = 'test_these\output'

# Rename each image with '_before' in its filename
for filename in os.listdir(input_dir):
    if filename.lower().endswith('_before.jpg') or filename.lower().endswith('_before.jpeg') or filename.lower().endswith('_before.png'):
        # Construct new filename
        new_filename = filename.replace('_before', '_after')
        # Rename the file
        os.rename(os.path.join(input_dir, filename), os.path.join(input_dir, new_filename))
        print(f'Renamed {filename} to {new_filename}.')
    else:
        print(f'Ignored {filename} (not an image file with "_before" in its filename).')