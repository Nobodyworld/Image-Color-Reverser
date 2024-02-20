# ./model_parts/encoder.py
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Define all encoder components here (e.g., enc_conv1, pool1, res_enc1, etc.)
        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.res_enc1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(16), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(16))
        self.enc_conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.res_enc2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32))
        self.enc_conv3 = nn.Sequential(nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.res_enc3 = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(48), nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(48))
        self.enc_conv4 = nn.Sequential(nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.res_enc4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64))
        self.enc_conv5 = nn.Sequential(nn.Conv2d(64, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True), nn.Conv2d(80, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
        self.pool5 = nn.MaxPool2d(2, 2)
        self.res_enc5 = nn.Sequential(nn.Conv2d(80, 80, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(80), nn.Conv2d(80, 80, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(80))
        self.enc_conv6 = nn.Sequential(nn.Conv2d(80, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.pool6 = nn.MaxPool2d(2, 2)
        self.res_enc6 = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(96), nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(96))
        self.enc_conv7 = nn.Sequential(nn.Conv2d(96, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True), nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))
        self.pool7 = nn.MaxPool2d(2, 2)
        self.res_enc7 = nn.Sequential(nn.Conv2d(112, 112, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(112), nn.Conv2d(112, 112, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(112))
        self.enc_conv8 = nn.Sequential(nn.Conv2d(112, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, x):
        # Implement the forward pass for the encoder
        x1 = self.enc_conv1(x)
        x2 = self.pool1(x1)
        x2 = x2 + self.res_enc1(x2)
        x3 = self.enc_conv2(x2)
        x4 = self.pool2(x3)
        x4 = x4 + self.res_enc2(x4)
        x5 = self.enc_conv3(x4)
        x6 = self.pool3(x5)
        x6 = x6 + self.res_enc3(x6)
        x7 = self.enc_conv4(x6)
        x8 = self.pool4(x7)
        x8 = x8 + self.res_enc4(x8)
        x9 = self.enc_conv5(x8)
        x10 = self.pool5(x9)
        x10 = x10 + self.res_enc5(x10)
        x11 = self.enc_conv6(x10)
        x12 = self.pool6(x11)
        x12 = x12 + self.res_enc6(x12)
        x13 = self.enc_conv7(x12)
        x14 = self.pool7(x13)
        x15 = self.enc_conv8(x14)
        
        # Return the outputs of all encoder stages
        return x1, x3, x5, x7, x9, x11, x13, x15
