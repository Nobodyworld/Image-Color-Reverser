import os
import shutil

def move_image_to_dest(src_folder, stage_data, img_name):
    shutil.move(os.path.join(src_folder, img_name), os.path.join(stage_data, img_name))

def main():
    # Get the current script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)

    stage_data = os.path.join(project_root, "stage_data")
    train_dir = os.path.join(project_root, "train")
    val_dir = os.path.join(project_root, "val")
    test_dir = os.path.join(project_root, "test")

    # Ensure the destination folder exists
    if not os.path.exists(stage_data):
        os.makedirs(stage_data)

    # Function to process each directory
    def process_directory(data_dir):
        for img_name in os.listdir(data_dir):
            if img_name.endswith("_after.jpg") or img_name.endswith("_before.jpg"):
                move_image_to_dest(data_dir, stage_data, img_name)

    # Process each directory: train, val, test
    process_directory(train_dir)
    process_directory(val_dir)
    process_directory(test_dir)

if __name__ == "__main__":
    main()
