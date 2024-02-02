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

/model.py
No pre-trained model found. Training from scratch.
Epoch [1/120], Loss: 0.1296
Validation Loss: 0.5545
Epoch [2/120], Loss: 0.1268
Validation Loss: 0.5521
Epoch [3/120], Loss: 0.1201
Validation Loss: 0.5297
Epoch [4/120], Loss: 0.1210
Validation Loss: 0.5027
Epoch [5/120], Loss: 0.1140
Validation Loss: 0.4896
Epoch [6/120], Loss: 0.1127
Validation Loss: 0.4862
Epoch [7/120], Loss: 0.1111
Validation Loss: 0.4871
Epoch [8/120], Loss: 0.1103
Validation Loss: 0.4828
Epoch [9/120], Loss: 0.1067
Validation Loss: 0.4779
Epoch [10/120], Loss: 0.1060
Validation Loss: 0.4717
Epoch [11/120], Loss: 0.1031
Validation Loss: 0.4647
Epoch [12/120], Loss: 0.1035
Validation Loss: 0.4582
Epoch [13/120], Loss: 0.1014
Validation Loss: 0.4524
Epoch [14/120], Loss: 0.0999
Validation Loss: 0.4471
Epoch [15/120], Loss: 0.1006
Validation Loss: 0.4423
Epoch [16/120], Loss: 0.0977
Validation Loss: 0.4383
Epoch [17/120], Loss: 0.0942
Validation Loss: 0.4344
Epoch [18/120], Loss: 0.1005
Validation Loss: 0.4322
Epoch [19/120], Loss: 0.0947
Validation Loss: 0.4294
Epoch [20/120], Loss: 0.0938
Validation Loss: 0.4265
Epoch [21/120], Loss: 0.0942
Validation Loss: 0.4242
Epoch [22/120], Loss: 0.0983
Validation Loss: 0.4216
Epoch [23/120], Loss: 0.0960
Validation Loss: 0.4193
Epoch [24/120], Loss: 0.0951
Validation Loss: 0.4168
Epoch [25/120], Loss: 0.0935
Validation Loss: 0.4145
Epoch [26/120], Loss: 0.0904
Validation Loss: 0.4121
Epoch [27/120], Loss: 0.0930
Validation Loss: 0.4104
Epoch [28/120], Loss: 0.0952
Validation Loss: 0.4095
Epoch [29/120], Loss: 0.0916
Validation Loss: 0.4080
Epoch [30/120], Loss: 0.0876
Validation Loss: 0.4061
Epoch [31/120], Loss: 0.0928
Validation Loss: 0.4046
Epoch [32/120], Loss: 0.0903
Validation Loss: 0.4030
Epoch [33/120], Loss: 0.0879
Validation Loss: 0.4016
Epoch [34/120], Loss: 0.0876
Validation Loss: 0.4006
Epoch [35/120], Loss: 0.0874
Validation Loss: 0.3995
Epoch [36/120], Loss: 0.0901
Validation Loss: 0.3991
Epoch [37/120], Loss: 0.0896
Validation Loss: 0.3979
Epoch [38/120], Loss: 0.0846
Validation Loss: 0.3967
Epoch [39/120], Loss: 0.0868
Validation Loss: 0.3955
Epoch [40/120], Loss: 0.0856
Validation Loss: 0.3940
Epoch [41/120], Loss: 0.0890
Validation Loss: 0.3932
Epoch [42/120], Loss: 0.0867
Validation Loss: 0.3917
Epoch [43/120], Loss: 0.0883
Validation Loss: 0.3904
Epoch [44/120], Loss: 0.0888
Validation Loss: 0.3896
Epoch [45/120], Loss: 0.0879
Validation Loss: 0.3890
Epoch [46/120], Loss: 0.0869
Validation Loss: 0.3880
Epoch [47/120], Loss: 0.0842
Validation Loss: 0.3862
Epoch [48/120], Loss: 0.0841
Validation Loss: 0.3848
Epoch [49/120], Loss: 0.0868
Validation Loss: 0.3840
Epoch [50/120], Loss: 0.0830
Validation Loss: 0.3825
Epoch [51/120], Loss: 0.0836
Validation Loss: 0.3820
Epoch [52/120], Loss: 0.0825
Validation Loss: 0.3812
Epoch [53/120], Loss: 0.0822
Validation Loss: 0.3803
Epoch [54/120], Loss: 0.0891
Validation Loss: 0.3793
Epoch [55/120], Loss: 0.0872
Validation Loss: 0.3783
Epoch [56/120], Loss: 0.0869
Validation Loss: 0.3772
Epoch [57/120], Loss: 0.0822
Validation Loss: 0.3762
Epoch [58/120], Loss: 0.0854
Validation Loss: 0.3750
Epoch [59/120], Loss: 0.0855
Validation Loss: 0.3737
Epoch [60/120], Loss: 0.0866
Validation Loss: 0.3738
Epoch [61/120], Loss: 0.0821
Validation Loss: 0.3725
Epoch [62/120], Loss: 0.0837
Validation Loss: 0.3716
Epoch [63/120], Loss: 0.0863
Validation Loss: 0.3711
Epoch [64/120], Loss: 0.0794
Validation Loss: 0.3703
Epoch [65/120], Loss: 0.0834
Validation Loss: 0.3702
Epoch [66/120], Loss: 0.0827
Validation Loss: 0.3696
Epoch [67/120], Loss: 0.0820
Validation Loss: 0.3684
Epoch [68/120], Loss: 0.0812
Validation Loss: 0.3672
Epoch [69/120], Loss: 0.0840
Validation Loss: 0.3662
Epoch [70/120], Loss: 0.0809
Validation Loss: 0.3654
Epoch [71/120], Loss: 0.0773
Validation Loss: 0.3643
Epoch [72/120], Loss: 0.0827
Validation Loss: 0.3637
Epoch [73/120], Loss: 0.0799
Validation Loss: 0.3631
Epoch [74/120], Loss: 0.0837
Validation Loss: 0.3627
Epoch [75/120], Loss: 0.0790
Validation Loss: 0.3613
Epoch [76/120], Loss: 0.0778
Validation Loss: 0.3607
Epoch [77/120], Loss: 0.0804
Validation Loss: 0.3605
Epoch [78/120], Loss: 0.0772
Validation Loss: 0.3596
Epoch [79/120], Loss: 0.0788
Validation Loss: 0.3589
Epoch [80/120], Loss: 0.0794
Validation Loss: 0.3578
Epoch [81/120], Loss: 0.0812
Validation Loss: 0.3576
Epoch [82/120], Loss: 0.0786
Validation Loss: 0.3567
Epoch [83/120], Loss: 0.0813
Validation Loss: 0.3558
Epoch [84/120], Loss: 0.0777
Validation Loss: 0.3545
Epoch [85/120], Loss: 0.0801
Validation Loss: 0.3544
Epoch [86/120], Loss: 0.0774
Validation Loss: 0.3536
Epoch [87/120], Loss: 0.0768
Validation Loss: 0.3526
Epoch [88/120], Loss: 0.0785
Validation Loss: 0.3523
Epoch [89/120], Loss: 0.0807
Validation Loss: 0.3523
Epoch [90/120], Loss: 0.0800
Validation Loss: 0.3520
Epoch [91/120], Loss: 0.0788
Validation Loss: 0.3511
Epoch [92/120], Loss: 0.0762
Validation Loss: 0.3504
Epoch [93/120], Loss: 0.0787
Validation Loss: 0.3499
Epoch [94/120], Loss: 0.0816
Validation Loss: 0.3496
Epoch [95/120], Loss: 0.0749
Validation Loss: 0.3485
Epoch [96/120], Loss: 0.0788
Validation Loss: 0.3480
Epoch [97/120], Loss: 0.0769
Validation Loss: 0.3473
Epoch [98/120], Loss: 0.0735
Validation Loss: 0.3466
Epoch [99/120], Loss: 0.0724
Validation Loss: 0.3457
Epoch [100/120], Loss: 0.0793
Validation Loss: 0.3455
Epoch [101/120], Loss: 0.0796
Validation Loss: 0.3458
Epoch [102/120], Loss: 0.0761
Validation Loss: 0.3448
Epoch [103/120], Loss: 0.0732
Validation Loss: 0.3442
Epoch [104/120], Loss: 0.0778
Validation Loss: 0.3436
Epoch [105/120], Loss: 0.0763
Validation Loss: 0.3431
Epoch [106/120], Loss: 0.0739
Validation Loss: 0.3422
Epoch [107/120], Loss: 0.0792
Validation Loss: 0.3423
Epoch [108/120], Loss: 0.0753
Validation Loss: 0.3412
Epoch [109/120], Loss: 0.0750
Validation Loss: 0.3403
Epoch [110/120], Loss: 0.0798
Validation Loss: 0.3406
Epoch [111/120], Loss: 0.0755
Validation Loss: 0.3399
Epoch [112/120], Loss: 0.0787
Validation Loss: 0.3392
Epoch [113/120], Loss: 0.0765
Validation Loss: 0.3383
Epoch [114/120], Loss: 0.0720
Validation Loss: 0.3379
Epoch [115/120], Loss: 0.0755
Validation Loss: 0.3374
Epoch [116/120], Loss: 0.0766
Validation Loss: 0.3370
Epoch [117/120], Loss: 0.0763
Validation Loss: 0.3363
Epoch [118/120], Loss: 0.0744
Validation Loss: 0.3363
Epoch [119/120], Loss: 0.0762
Validation Loss: 0.3361
Epoch [120/120], Loss: 0.0754
Validation Loss: 0.3358
Test Loss: 0.3018

/model.py
Pre-trained model loaded.
Epoch [1/120], Loss: 0.0754
Validation Loss: 0.3351
Epoch [2/120], Loss: 0.0762
Validation Loss: 0.3317
Epoch [3/120], Loss: 0.0728
Validation Loss: 0.3288
Epoch [4/120], Loss: 0.0768
Validation Loss: 0.3282
Epoch [5/120], Loss: 0.0724
Validation Loss: 0.3269
Epoch [6/120], Loss: 0.0728
Validation Loss: 0.3256
Epoch [7/120], Loss: 0.0733
Validation Loss: 0.3249
Epoch [8/120], Loss: 0.0740
Validation Loss: 0.3244
Epoch [9/120], Loss: 0.0725
Validation Loss: 0.3249
Epoch [10/120], Loss: 0.0728
Validation Loss: 0.3249
Epoch [11/120], Loss: 0.0714
Validation Loss: 0.3240
Epoch [12/120], Loss: 0.0729
Validation Loss: 0.3234
Epoch [13/120], Loss: 0.0720
Validation Loss: 0.3230
Epoch [14/120], Loss: 0.0715
Validation Loss: 0.3218
Epoch [15/120], Loss: 0.0730
Validation Loss: 0.3191
Epoch [16/120], Loss: 0.0710
Validation Loss: 0.3169
Epoch [17/120], Loss: 0.0684
Validation Loss: 0.3152
Epoch [18/120], Loss: 0.0744
Validation Loss: 0.3162
Epoch [19/120], Loss: 0.0697
Validation Loss: 0.3168
Epoch [20/120], Loss: 0.0693
Validation Loss: 0.3160
Epoch [21/120], Loss: 0.0699
Validation Loss: 0.3156
Epoch [22/120], Loss: 0.0739
Validation Loss: 0.3143
Epoch [23/120], Loss: 0.0719
Validation Loss: 0.3135
Epoch [24/120], Loss: 0.0714
Validation Loss: 0.3131
Epoch [25/120], Loss: 0.0702
Validation Loss: 0.3124
Epoch [26/120], Loss: 0.0678
Validation Loss: 0.3110
Epoch [27/120], Loss: 0.0700
Validation Loss: 0.3099
Epoch [28/120], Loss: 0.0723
Validation Loss: 0.3105
Epoch [29/120], Loss: 0.0693
Validation Loss: 0.3105
Epoch [30/120], Loss: 0.0659
Validation Loss: 0.3093
Epoch [31/120], Loss: 0.0706
Validation Loss: 0.3091
Epoch [32/120], Loss: 0.0685
Validation Loss: 0.3088
Epoch [33/120], Loss: 0.0665
Validation Loss: 0.3074
Epoch [34/120], Loss: 0.0665
Validation Loss: 0.3058
Epoch [35/120], Loss: 0.0662
Validation Loss: 0.3043
Epoch [36/120], Loss: 0.0688
Validation Loss: 0.3044
Epoch [37/120], Loss: 0.0685
Validation Loss: 0.3047
Epoch [38/120], Loss: 0.0639
Validation Loss: 0.3045
Epoch [39/120], Loss: 0.0662
Validation Loss: 0.3046
Epoch [40/120], Loss: 0.0650
Validation Loss: 0.3034
Epoch [41/120], Loss: 0.0683
Validation Loss: 0.3026
Epoch [42/120], Loss: 0.0660
Validation Loss: 0.3002
Epoch [43/120], Loss: 0.0677
Validation Loss: 0.2988
Epoch [44/120], Loss: 0.0682
Validation Loss: 0.2990
Epoch [45/120], Loss: 0.0676
Validation Loss: 0.2998
Epoch [46/120], Loss: 0.0667
Validation Loss: 0.3000
Epoch [47/120], Loss: 0.0642
Validation Loss: 0.2983
Epoch [48/120], Loss: 0.0641
Validation Loss: 0.2969
Epoch [49/120], Loss: 0.0666
Validation Loss: 0.2966
Epoch [50/120], Loss: 0.0632
Validation Loss: 0.2957
Epoch [51/120], Loss: 0.0636
Validation Loss: 0.2955
Epoch [52/120], Loss: 0.0629
Validation Loss: 0.2953
Epoch [53/120], Loss: 0.0624
Validation Loss: 0.2954
Epoch [54/120], Loss: 0.0691
Validation Loss: 0.2954
Epoch [55/120], Loss: 0.0671
Validation Loss: 0.2955
Epoch [56/120], Loss: 0.0670
Validation Loss: 0.2951
Epoch [57/120], Loss: 0.0628
Validation Loss: 0.2935
Epoch [58/120], Loss: 0.0657
Validation Loss: 0.2914
Epoch [59/120], Loss: 0.0660
Validation Loss: 0.2896
Epoch [60/120], Loss: 0.0670
Validation Loss: 0.2896
Epoch [61/120], Loss: 0.0628
Validation Loss: 0.2891
Epoch [62/120], Loss: 0.0644
Validation Loss: 0.2892
Epoch [63/120], Loss: 0.0669
Validation Loss: 0.2889
Epoch [64/120], Loss: 0.0605
Validation Loss: 0.2877
Epoch [65/120], Loss: 0.0643
Validation Loss: 0.2875
Epoch [66/120], Loss: 0.0636
Validation Loss: 0.2873
Epoch [67/120], Loss: 0.0628
Validation Loss: 0.2867
Epoch [68/120], Loss: 0.0621
Validation Loss: 0.2865
Epoch [69/120], Loss: 0.0650
Validation Loss: 0.2867
Epoch [70/120], Loss: 0.0622
Validation Loss: 0.2869
Epoch [71/120], Loss: 0.0586
Validation Loss: 0.2858
Epoch [72/120], Loss: 0.0637
Validation Loss: 0.2854
Epoch [73/120], Loss: 0.0614
Validation Loss: 0.2844
Epoch [74/120], Loss: 0.0649
Validation Loss: 0.2841
Epoch [75/120], Loss: 0.0603
Validation Loss: 0.2829
Epoch [76/120], Loss: 0.0593
Validation Loss: 0.2822
Epoch [77/120], Loss: 0.0618
Validation Loss: 0.2820
Epoch [78/120], Loss: 0.0589
Validation Loss: 0.2812
Epoch [79/120], Loss: 0.0604
Validation Loss: 0.2811
Epoch [80/120], Loss: 0.0609
Validation Loss: 0.2808
Epoch [81/120], Loss: 0.0626
Validation Loss: 0.2811
Epoch [82/120], Loss: 0.0604
Validation Loss: 0.2809
Epoch [83/120], Loss: 0.0627
Validation Loss: 0.2805
Epoch [84/120], Loss: 0.0596
Validation Loss: 0.2788
Epoch [85/120], Loss: 0.0618
Validation Loss: 0.2782
Epoch [86/120], Loss: 0.0592
Validation Loss: 0.2776
Epoch [87/120], Loss: 0.0589
Validation Loss: 0.2773
Epoch [88/120], Loss: 0.0603
Validation Loss: 0.2775
Epoch [89/120], Loss: 0.0623
Validation Loss: 0.2782
Epoch [90/120], Loss: 0.0618
Validation Loss: 0.2786
Epoch [91/120], Loss: 0.0606
Validation Loss: 0.2778
Epoch [92/120], Loss: 0.0583
Validation Loss: 0.2765
Epoch [93/120], Loss: 0.0607
Validation Loss: 0.2757
Epoch [94/120], Loss: 0.0633
Validation Loss: 0.2756
Epoch [95/120], Loss: 0.0571
Validation Loss: 0.2747
Epoch [96/120], Loss: 0.0607
Validation Loss: 0.2742
Epoch [97/120], Loss: 0.0593
Validation Loss: 0.2739
Epoch [98/120], Loss: 0.0560
Validation Loss: 0.2733
Epoch [99/120], Loss: 0.0549
Validation Loss: 0.2727
Epoch [100/120], Loss: 0.0615
Validation Loss: 0.2729
Epoch [101/120], Loss: 0.0617
Validation Loss: 0.2732
Epoch [102/120], Loss: 0.0584
Validation Loss: 0.2721
Epoch [103/120], Loss: 0.0557
Validation Loss: 0.2710
Epoch [104/120], Loss: 0.0600
Validation Loss: 0.2706
Epoch [105/120], Loss: 0.0587
Validation Loss: 0.2703
Epoch [106/120], Loss: 0.0564
Validation Loss: 0.2697
Epoch [107/120], Loss: 0.0615
Validation Loss: 0.2703
Epoch [108/120], Loss: 0.0578
Validation Loss: 0.2696
Epoch [109/120], Loss: 0.0575
Validation Loss: 0.2691
Epoch [110/120], Loss: 0.0622
Validation Loss: 0.2701
Epoch [111/120], Loss: 0.0580
Validation Loss: 0.2695
Epoch [112/120], Loss: 0.0611
Validation Loss: 0.2686
Epoch [113/120], Loss: 0.0589
Validation Loss: 0.2674
Epoch [114/120], Loss: 0.0546
Validation Loss: 0.2667
Epoch [115/120], Loss: 0.0580
Validation Loss: 0.2661
Epoch [116/120], Loss: 0.0589
Validation Loss: 0.2658
Epoch [117/120], Loss: 0.0588
Validation Loss: 0.2654
Epoch [118/120], Loss: 0.0571
Validation Loss: 0.2658
Epoch [119/120], Loss: 0.0587
Validation Loss: 0.2658
Epoch [120/120], Loss: 0.0580
Validation Loss: 0.2658
Test Loss: 0.2342

