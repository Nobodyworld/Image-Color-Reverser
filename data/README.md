# ImageMLDataset
## All You Need To Know
This repository is intended for quick synchronization and setup of a machine learning image dataset for **before** and **after** scenarios. It provides scripts for resizing images and organizing them into training, validation, and testing sets, simplifying the preparation process for your ML projects.

### Manual Copy (For Infrequent Updates)
For those instances where updates to the dataset are infrequent, manually copying the `/data` directory from the ImageMLDataset repository into your project's root is the simplest approach. This method is straightforward but does not maintain a link to the original repository for easy updates.

**Caution** 
- Resyncing will cause all of your images to be moved back into `/data/stage_data`. Do this sparingly once you have established a trained model on this data.

### Git Submodule Method
To keep a link to the original repository, allowing for easy updates, you can add the ImageMLDataset as a submodule to your current Git repository:
```git
git submodule add https://github.com/Nobodyworld/ImageMLDataset.git Data
```
This command clones the ImageMLDataset repository into a directory named `Data` within your current repository.


**Running the Script:**
- Ensure to place your `config.json` in the `.config/` directory at the root of your project, then execute `stage_resize.py` to resize the images.
- Ensure you have reviewed the `requirements.txt`

**Implementing Safe Dimensions in Data Preparation**
- When resizing images using `stage_resize.py`, refer to the `odd_pixel_count.png` by layers to view the idea of an appropriate starting dimension. 

### Using `stage_resize.py`
Resize all images in `./data/stage_data/` according to the parameters set in the expected `.config/config.json`. Here is a condensed example schema for the configuration file:
```json
{
  "training": {
    "img_height": 1920,
    "img_width": 1280
  }
}
```

### Using `stage_move_imgs.py`
Organize your images for machine learning purposes with `stage_move_imgs.py`, located in `/data/data_designer/`. This script moves images from `./data/stage_data/` to appropriately named directories for different dataset splits: `/test`, `/train`, and `/val`, creating them if necessary.
```python
num_train = int(num_images * 0.80)  # 80% of dataset allocated to `/train`
num_val = int(num_images * 0.12)    # 12% of dataset allocated to `/val`
# The remaining 8% is allocated to `/test`
```
**Note**: Adjust the percentages in the script as necessary to fit the needs of your project.

---
### Ensuring Compatible Image Dimensions for Neural Network Layers

When preparing image datasets for convolutional neural networks, especially those employing pooling layers that halve image dimensions, it's essential to start with dimensions that remain integers throughout the network's depth. This avoids tensor size mismatches that can occur when an image's width or height is not divisible by 2^n, where n is the number of times the image is halved by pooling layers.

#### The Issue with Irrational Dimensions

In PyTorch and other deep learning frameworks, certain operations like max pooling perform integer division on image dimensions. When an operation attempts to halve an odd number, the result is a non-integer which gets floored, potentially leading to downstream tensor size mismatches. This is particularly problematic in deep networks where pooling operations occur multiple times, compounding the issue.

### Optimizing Image Data for ML Models

Working with image datasets for machine learning, especially in architectures like U-Net in PyTorch, requires a nuanced understanding of how different aspects of image data—like file size and dimensions—impact computational resources. Here, we delve into optimizing your workflow for both efficiency and effectiveness.

#### Understanding File Sizes vs. Image Dimensions

The consumption of GPU memory during training or inference is more directly influenced by the dimensions of an image (width x height), rather than the file size on disk. This distinction is crucial for managing resources effectively:

1. **Loading Images**: Loading images from disk into RAM depends on file size, which might influence loading times but not GPU memory usage. It's the image dimensions that play a critical role when images are loaded into GPU for processing.
   
2. **Transforming Images**: Pre-processing steps often involve resizing and normalizing images to a consistent size for the neural network, converting compressed file formats into raw pixel tensors.

3. **GPU Memory Utilization**: The dimensions of these tensors, corresponding to the image dimensions post-transformation, directly determine the amount of GPU memory used. For example, a tensor representing a 960x640 pixel image consumes GPU memory based on its dimensions, regardless of the original image's file size.

4. **Batch Processing Implications**: Training typically processes images in batches, amplifying the memory requirement proportional to the batch size.

#### Efficiently Handling High-Quality Images

For those incorporating high-quality images in the `/data` directory, leveraging dynamic resizing can significantly optimize GPU memory usage. PyTorch's `torchvision.transforms` offers a practical approach to adjust image sizes during dataset loading, balancing quality and computational efficiency.

Example of dynamic resizing:

```python
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Define transformations
transform = transforms.Compose([
    transforms.Resize((desired_height, desired_width)),  # Adjust to your model's needs
    transforms.ToTensor(),
    # Consider additional transformations as needed
])

# Apply transformations to the dataset
dataset = ImageFolder(root='./data/train', transform=transform)
```

This method ensures that while high-quality images are stored on disk, their resized versions are what's loaded into GPU memory, thus optimizing resource use without compromising on the quality available for inference or further transformations.

### Next Steps: Leveraging `/play_data` for Inference

The `/play_data` directory is designed to facilitate easy inference with "image input" based neural networks, serving as a sandbox for testing and demonstrations. The subdirectory `/play_data/output` is specifically intended for storing images post-inference or transformation, showcasing the practical applications of your trained models.

#### Conclusion

This guide complements the initial setup and usage instructions, empowering you to manage image datasets more effectively for machine learning projects. By understanding and applying these principles, you can ensure efficient use of computational resources, enabling the training of more complex models or the processing of larger datasets without undue strain on your hardware.