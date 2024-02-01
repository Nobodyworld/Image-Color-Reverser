import os
import shutil

def move_pair_back(src_clean, src_noisy, dest_folder, img_before, img_after):
    shutil.move(os.path.join(src_noisy, img_before), os.path.join(dest_folder, img_before))
    shutil.move(os.path.join(src_clean, img_after), os.path.join(dest_folder, img_after))

def main():
    dest_folder = "resized_imgs"
    train_clean = "train/clean"
    train_noisy = "train/noisy"
    val_clean = "val/clean"
    val_noisy = "val/noisy"
    test_clean = "test/clean"
    test_noisy = "test/noisy"

    # Define a function to process each set
    def process_set(clean_folder, noisy_folder):
        for img_after in os.listdir(clean_folder):
            if img_after.endswith("_after.jpg"):
                img_before = img_after.replace("_after.jpg", "_before.jpg")
                move_pair_back(clean_folder, noisy_folder, dest_folder, img_before, img_after)

    # Process each set: train, val, test
    process_set(train_clean, train_noisy)
    process_set(val_clean, val_noisy)
    process_set(test_clean, test_noisy)

if __name__ == "__main__":
    main()
