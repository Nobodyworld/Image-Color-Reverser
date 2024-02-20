# Public Unet Image Color Reverser or Processor
An attempt to train a Pytroch model to trun black and white images into color and the reverse of this with loss conversions reflectively recordered to acheive the desired outcome and its opposite..

This repository hosts the De-Noisy Image Project, leveraging a U-Net architecture to denoise images. It's structured to accommodate varying levels of compute power across three project sizes: normal, small, and extra-small. 

### Project Overview

Utilizing a U-Net architecture, this project aims to denoise images effectively. The project is divided into three scales:
- **Normal**: For standard compute resources.
- **Small**: Reduced requirements for less powerful machines.
- **Extra-Small (xtra-small)**: Minimized compute needs.

### Training Configuration

To adapt to different compute capabilities, the image sizes, number of layers, and several key training parameters can be adjusted. Below is an example configuration:

```json
{
  "training": {
    "epochs": 36,
    "batch_size": 2,
    "accumulation_steps": 1,
    "num_workers": 4,
    "pin_memory": true,
    "shuffle": true,
    "early_stopping": true,
    "early_stopping_patience": 8,
    "step_decrease_interval": 8,
    "img_height": 1920,
    "img_width": 1280
  }
}
```
## Implementation Notes

### Safe Dimensions
- **Resizing**: Utilize `stage_resize.py` for image resizing. Consider the `odd_pixel_count.png` to determine appropriate starting dimensions, ensuring compatibility with the network's pooling layers.

### Batching
- The batch size reflects how many images are processed together through the network. A smaller batch size does not limit the total number of images seen per epoch but indicates that images are fed in smaller groups. For example, with 20 images and a batch size of 2, there would be 10 batches per epoch.

### Pooling Layers
- With image dimensions of 640x960, up to 7 pooling layers can be applied. This number is derived from halving each dimension until reaching a minimal size that remains a whole number. Adjust the number of pooling layers based on your specific task and network architecture.

### Tensor Mismatch and Ceil Mode
- **Ceil Mode**: Opting for `ceil` instead of `floor` in pooling layers can influence the output shape, potentially leading to tensor mismatches. This option must be carefully managed to ensure compatibility between layers.

### Directory Structure

1. **`./data/`**: Contains subdirectories for different stages of data processing and storage.
2. **`./model_parts/` & `./config/`**: Holds model parameters and configuration settings.
3. **Scripts**: Including `main.py` for training and architecture maintenance, and `test_play_data.py` for inference testing.

### Running the Project

- **Training**: Execute `main.py` with images in the `train`, `val`, and `test` folders.
- **Inference**: Place images in `./data/play_data/` and run `test_play_data.py` for denoised outputs.

### Important Considerations

- Ensure compatibility with your hardware, especially GPU capabilities, by running `./tests/test_gpu.py`.
- Backup data before executing scripts that modify or move files to prevent data loss or corruption.
- Adjust parameters in the model or configuration files to tailor the project to your dataset and requirements.

### Model Architecture

- **Encoder**: Convolutional layers with downsampling for feature extraction.
- **Middle Section**: Bridges encoder and decoder, processing feature maps further.
- **Decoder**: Upsamples feature maps, reduces channels, and uses skip connections for feature concatenation.
- **Output**: Final layer mapping feature maps to the target output channels.

### Modifications & Key Features

- Designed for flexibility across different computing environments.
- Supports adjustments in image dimensions and model layers to match computational resources.
- Includes detailed documentation on the model's layers and operational flow.