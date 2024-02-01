# Public Unet Image Color Reverser or Processor
An attempt to train a Pytroch model to trun black and white images into color and the reverse of this with loss conversions reflectively recordered to acheive the desired outcome and its opposite..

# Summary
The main components are the `resized_imgs` folder for storing and preprocessing images before training, `test_these` for extra inference, and the `model.py` script for training and architecture. The `model.py` will run without an NVIDIA GPU but you should consider reducing the image sizes, the number of layers, or these varaibles mentioned further below. (See lines 190-200 of `model.py`)

This build has slight variations from the original, and small "De-Noisy Image Project."

  """
    # Set batch size, image dimensions
    batch_size = 64
    img_height = 384
    img_width = 256
    epochs = 120
    accumulation_steps = 4
  """

Note: As you can see above we have reduced the image height and width significantly from the large repo for this project to account for less GPU or CPU compute power. Do not forget to run the `ztest_gpu.py` file to verify you have a connected cuda device.

Batching Note: Having a batch size of two does not mean that my model only sees two images per 'epoch' but rather that my model is fed two images at a time, of all images in the dataset, until it finishes, which completes one epoch. So if I have 20 images I will then have 10 batches per epoch.

For image dimensions of 640 in width and 960 in height, you can apply a maximum of 7 pooling layers. This number is calculated based on how many times each dimension can be divided by 2 (halved) until it reaches a minimum size while remaining a whole number. However, in practice, you may not need or want to use the maximum number of pooling layers, as each pooling layer reduces the spatial resolution of your feature maps. The actual number to use would depend on the specifics of your task and the architecture of your neural network.

"ceil_mode plici(bool) – when True, will use ceil instead of floor to compute the output shape." https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

![image](https://github.com/Nobodyworld/De-Noisy-Image-Project-Small/assets/127373451/ae4f539f-41d8-4ecf-9d2a-f82dfb5b0682)

Tensor mismatch
torch.Size([6, 112, 14, 10])
torch.Size([6, 112, 15, 10])

## Directory Structure

1. **`resized_imgs`**: This folder is used to store images prior to training a model so that you can keep all 'before' and 'after' images in a consolidated location and perform edits should you choose. (I highly recommend not cropping or resizing otherwise you risk skewing the before and after pairs.) After images in this directory are manipulated use 'zmoveimgs.py' to move them proprtionally to `train`, `test`, and `val` folders using the customizable logic below.    
  
  """  
    num_images = len(images)
    num_train = int(num_images * 0.80)
    num_val = int(num_images * 0.12)

  """

Note: If not images are not in train or val an else statement adds them to '/test' within the 'zmoveimgs.py' script. So .08, or 8 percent of the images were allocated to '/test'.

Similarly, if you wanted to get the images back int the '/resized_imgs' folder all you need to do is run the 'zgetimgs.py' script.

Please refer to the 'resized_imgs\readme.txt' file for more info.

2. **`test_these`**: Place your images for additional inference here. Images should be named with a `_before` suffix (e.g., `image_before.jpg`). These images can be processed using the `zrename_test_before.py` script.

3. **`test_these/output`**: This directory will contain the output of the inference process. Images from the `test_these` folder are processed and their denoised versions are saved here.


## Key Scripts

1. **`model.py`**: This script is used for both training the U-Net model and maintaining the architecture referenced during inference.

### Training the Model
Run `model.py` to train the U-Net model. The script will use images from the `train`, `val`, and `test` folders.


2. **`test_these.py`**: Script for manual inference after a model has been trainined.

### Running Manual Inference After Training
1. Place your images for inference in the `test_these` folder, or use the images already there. Ensure additions are named with a `_before` suffix by using the `zrename_test_before.py` script. (Ex: Image1_before.jpg)
2. Run `test_these.py`. It will process the images and save the denoised outputs to the `test_these/output` folder.


## Note
- Ensure that the Python environment has all the necessary libraries installed (PyTorch, torchvision, PIL, etc.).
- Adjust model parameters in `model.py` as needed to suit your dataset and training requirements.
- Always backup your data before running scripts that modify or move files.

---

## Model.py Layers Explained

### Encoder Section
The encoder part of the U-Net architecture consists of several convolutional layers. Each `enc_conv` layer includes two convolutional layers with batch normalization and ReLU activation. The `pool` layers are used for downsampling the feature maps, reducing their dimensions by half.

- `enc_conv1` to `enc_conv8`: These are sequential blocks, each containing two convolutional layers. Each convolution is followed by batch normalization and ReLU activation. These blocks progressively increase the number of channels while capturing complex features from the input images.
- `pool1` to `pool7`: MaxPooling layers used for downsampling. They reduce the spatial dimensions of the feature maps by half, which helps the network to learn increasingly abstract features.
- `res_enc1` to `res_enc7`: These are residual connections in each encoding stage. They help in alleviating the vanishing gradient problem and enable the training of deeper networks.

### Middle Section
The middle part is a bridge between the encoder and decoder sections. It further processes the feature maps from the encoder.

- `mid_conv1` to `mid_conv9`: These are convolutional blocks similar to the encoder, designed to further process the feature maps. The number of channels remains constant throughout these layers. This section can be simplified or expanded based on the complexity required.

### Decoder Section
The decoder part of the U-Net upsamples the feature maps and reduces the number of channels. It also concatenates the feature maps from the encoder using skip connections.

- `dec_conv8` to `dec_conv1`: Each `dec_conv` layer consists of two convolutional layers with batch normalization and ReLU activation. The number of channels is progressively reduced.
- `up7` to `up1`: These are upsampling layers that increase the spatial dimensions of the feature maps. They use bilinear interpolation for upsampling.
- The `torch.cat` operations in the decoder concatenate the upsampled features with the corresponding features from the encoder. This is a crucial part of U-Net, allowing the network to use both high-level and low-level features for reconstruction.

### Output Section
The final output is generated through a convolutional layer that maps the feature maps to the desired number of output channels (e.g., 3 for RGB images).

- `out_conv`: This layer uses a 1x1 convolution to reduce the number of output channels to match the number of channels in the target images. It is followed by a Sigmoid activation function to ensure the output values are in a valid range (e.g., [0, 1] for normalized images).

### Forward Function
This function defines the forward pass of the network. It sequentially applies all the layers and functions defined in the `__init__` method. The feature maps are processed through the encoder, middle, and decoder sections, and the final output is produced. The forward function ensures that the skip connections are correctly utilized by concatenating the encoder features with the corresponding decoder features.

This architecture is a standard U-Net model used for tasks like image segmentation and denoising. The use of residual connections and skip connections typically helps in training deeper models more effectively.


##Example Script Outputs:
/test_these.py
PLEASE REFER to the test_these folder for inference.

/ztest_gpu.py
GPU: NVIDIA GeForce RTX 3060
tensor([5., 7., 9.], device='cuda:0')

/zcount_parameter.py
Number of trainable parameters: 1587411

/model.py" (example taken from Small Model.)

