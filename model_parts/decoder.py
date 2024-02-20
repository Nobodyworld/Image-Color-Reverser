# ./model_parts/decoder.py
import torch
import torch.nn as nn
 
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Define all decoder components here (e.g., dec_conv8, up7, dec_conv7, etc.)
        self.dec_conv8 = nn.Sequential(nn.Conv2d(128, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True), nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv7 = nn.Sequential(nn.Conv2d(224, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv6 = nn.Sequential(nn.Conv2d(192, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True), nn.Conv2d(80, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv5 = nn.Sequential(nn.Conv2d(160, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = nn.Sequential(nn.Conv2d(128, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Modify the last decoder layer to have 32 output channels
        self.dec_conv1 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        # Keep the output layer's input channels at 32
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()  # or nn.Sigmoid()
        )
  
    def forward(self, x15, x13, x11, x9, x7, x5, x3, x1):
        # Implement the forward pass for the decoder using the outputs from the encoder
        x16 = self.dec_conv8(x15)
        x17 = self.up7(x16)
        x18 = self.dec_conv7(torch.cat([x17, x13], dim=1))
        x19 = self.up6(x18)
        x20 = self.dec_conv6(torch.cat([x19, x11], dim=1))
        x21 = self.up5(x20)
        x22 = self.dec_conv5(torch.cat([x21, x9], dim=1))
        x23 = self.up4(x22)
        x24 = self.dec_conv4(torch.cat([x23, x7], dim=1))
        x25 = self.up3(x24)
        x26 = self.dec_conv3(torch.cat([x25, x5], dim=1))
        x27 = self.up2(x26)
        x28 = self.dec_conv2(torch.cat([x27, x3], dim=1))
        x29 = self.up1(x28)
        x30 = self.dec_conv1(torch.cat([x29, x1], dim=1))

        # Output
        x31 = self.out_conv(x30)
        return x31