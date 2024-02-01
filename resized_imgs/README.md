# Image Processing Scripts
This directory contains a collection of Python scripts designed to assist in the management and preparation of image datasets for a larger project. The primary function of these scripts is to consolidate 'before' and 'after' images into the `resized_imgs` folder for resizing, editing, and additions. After you can redistribute them into `test`, `train`, and `validate` folders for further use.

## Overview of Scripts
1. **`zmoveimgs.py`**: This is the primary script used for MOVE OUT images into training, validation, and test datasets. It segregates the images from the `resized_imgs` folder into `train`, `val`, and `test` folders, each having `clean` and `noisy` subfolders.

2. **`zgetimgs.py`**: This script is designed for moving images BACK INTO the resized_imgs folder from the train, val, and test directories. It effectively reverses the actions of zmoveimgs.py. Each image pair, consisting of a 'before' image from the noisy subfolders and an 'after' image from the clean subfolders, is relocated back to the resized_imgs folder. This is particularly useful when you need to consolidate your dataset back into a single directory, either for additional preprocessing steps or for archiving.

3. **`zresizeimgs.py`**: Use this script to resize images in the `resized_imgs` folder. It should be run before `zmove.py` if you have added new jpeg images to the resized images directory to ensure all images are of uniform size.

4. **`zname.py` and `zrename.py`**: If you add new images to the data set you can consider using these. The name scripts are used for renaming images to match the standard convention fo the project. They should be used cautiously to avoid disrupting the naming convention of your dataset. It's recommended to make backups before using these scripts.


## Instructions
### Setup
Ensure that all images are initially placed in the `resized_imgs` directory. The directory structure for the `train`, `val`, and `test` folders will be created automatically by the scripts.

### Using `zresizeimgs.py`
Run this script to resize all images in the `resized_imgs` folder. This is an essential step to standardize image sizes across your dataset, and will throw an error if you during training if you have an image in which the aspect ratio is mismatched with the array being used for matrix multiplication. 

Simple Example: If we have a 2:3 and a 3:4 image, in terms of aspect ratio the dot products would mismatch as follows.....
...
2x3:               3x4:
x-x-Error          x-x-x
x-x-Error          x-x-x
x-x-Error          x-x-x
Error-Error-Error  x-x-x
...
Note: Though this a simple example, a nugget to take away from this is that if we have the same aspect ratio across two different images we can normalize before the training step of 'model.py' or maintain consideration as to the intrinsic meaning of what we could accomplish with different size images and relative aspect ratios. (I have not been creative or logical enough to come up with a significant use case, but I am sure a very powerful one exists.)

### Using `zmove.py`
This script is the main tool for distributing images into appropriate datasets. It randomly assigns images to the `train`, `val`, and `test` folders based on specified proportions.

1. **Running the Script**: Simply execute `zmove.py`. The script will automatically shuffle and distribute the images.

2. **Folder Structure**:
   - `train/clean`
   - `train/noisy`
   - `val/clean`
   - `val/noisy`
   - `test/clean`
   - `test/noisy`

### Using `zname.py` and `zrename.py`
These scripts are used for renaming image files. They must be used with caution to maintain the integrity of your dataset.

1. **Backup First**: Always create a backup of your images before running these scripts.

2. **Testing on a Subset**: Try these scripts on a small set of images first to ensure they work as expected.

3. **Running the Scripts**: Use `zname.py` for initial naming and `zrename.py` for any subsequent renaming needs.

## Caution
- **Backup**: Always backup your images before using `zname.py` and `zrename.py`.
- **Testing**: Test the scripts on a small subset of your images first.
- **Order of Execution**: Run `zresizeimgs.py` before `zmove.py` to ensure all images are properly sized.

## Additional Notes
This folder is part of a larger project focused on image processing. The scripts are tailored for a specific workflow involving the preparation of image datasets. For any modifications or custom use-cases, please review and adjust the scripts accordingly.

---