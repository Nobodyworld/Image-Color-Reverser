import os
import shutil
import random

def move_pair(dest_folder, img_before, img_after):
    """
    Moves a pair of images (before and after) to a specified destination folder.
    """
    # Corrected source folder path to match the actual structure
    source_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "stage_data")
    
    # Move the before image
    shutil.move(os.path.join(source_folder, img_before), os.path.join(dest_folder, img_before))
    # Move the after image
    shutil.move(os.path.join(source_folder, img_after), os.path.join(dest_folder, img_after))


def main():
    # Get the current script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)

    stage_data = os.path.join(project_root, "stage_data")
    train_dir = os.path.join(project_root, "train")
    val_dir = os.path.join(project_root, "val")
    test_dir = os.path.join(project_root, "test")

    # Ensure destination directories exist
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    images = [img for img in os.listdir(stage_data) if img.endswith("_before.jpg")]
    random.shuffle(images)

    num_images = len(images)
    num_train = int(num_images * 0.80) # Percent of dataset allocated to `/train`
    num_val = int(num_images * 0.12) # Percent of dataset allocated to `/val`
    # The remaining percent is allocated to `/test`. In this case, (.08) or eight percent.

    for i, img_before in enumerate(images):
        img_after = img_before.replace("_before.jpg", "_after.jpg")

        if i < num_train:
            move_pair(train_dir, img_before, img_after)
        elif i < num_train + num_val:
            move_pair(val_dir, img_before, img_after)
        else:
            move_pair(test_dir, img_before, img_after)

if __name__ == "__main__":
    main()
