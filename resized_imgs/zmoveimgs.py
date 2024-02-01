import os
import shutil
import random

def move_pair(src_folder, dest_clean, dest_noisy, img_before, img_after):
    shutil.move(os.path.join(src_folder, img_before), os.path.join(dest_noisy, img_before))
    shutil.move(os.path.join(src_folder, img_after), os.path.join(dest_clean, img_after))

def main():
    src_folder = "resized_imgs"
    train_clean = "train/clean"
    train_noisy = "train/noisy"
    val_clean = "val/clean"
    val_noisy = "val/noisy"
    test_clean = "test/clean"
    test_noisy = "test/noisy"

    images = [img for img in os.listdir(src_folder) if img.endswith("_before.jpg")]
    random.shuffle(images)

    num_images = len(images)
    num_train = int(num_images * 0.8)
    num_val = int(num_images * 0.12)

    for i, img_before in enumerate(images):
        img_after = img_before.replace("_before.jpg", "_after.jpg")

        if i < num_train:
            move_pair(src_folder, train_clean, train_noisy, img_before, img_after)
        elif i < num_train + num_val:
            move_pair(src_folder, val_clean, val_noisy, img_before, img_after)
        else:
            move_pair(src_folder, test_clean, test_noisy, img_before, img_after)

if __name__ == "__main__":
    main()
