import os
import logging

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def deconv3x3(in_planes, out_planes, stride=1):
    "3x3 deconvolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=True)


class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = conv3x3(in_planes=1, out_planes=64, stride=2)
        self.conv2 = conv3x3(in_planes=64, out_planes=64, stride=2)
        self.conv3 = conv3x3(in_planes=64, out_planes=64, stride=2)
        self.conv4 = conv3x3(in_planes=64, out_planes=64, stride=1)
        self.conv5 = conv3x3(in_planes=64, out_planes=128, stride=2)
        self.conv6 = conv3x3(in_planes=128, out_planes=128, stride=1)
        self.conv7 = conv3x3(in_planes=128, out_planes=256, stride=2)
        self.conv8 = conv3x3(in_planes=256, out_planes=128, stride=1)
        self.conv9 = conv3x3(in_planes=128, out_planes=64, stride=1)
        self.conv10 = nn.Conv2d(64, 500, kernel_size=8, stride=stride,
                                padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.lrelu(x)
        x = self.conv5(x)
        x = self.lrelu(x)
        x = self.conv6(x)
        x = self.lrelu(x)
        x = self.conv7(x)
        x = self.lrelu(x)
        x = self.conv8(x)
        x = self.lrelu(x)
        x = self.conv9(x)
        x = self.lrelu(x)
        x = self.conv10(x)

        return x


class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.deconv1 = nn.ConvTranspose2d(500, 64, kernel_size=8, stride=1,
                                          padding=1, bias=True)
        self.deconv2 = deconv3x3(in_planes=64, out_planes=64, stride=1)
        self.deconv3 = deconv3x3(in_planes=64, out_planes=64, stride=1)
        self.deconv4 = deconv3x3(in_planes=64, out_planes=256, stride=1)
        self.deconv5 = deconv3x3(in_planes=256, out_planes=128, stride=2)
        self.deconv6 = deconv3x3(in_planes=128, out_planes=128, stride=1)
        self.deconv7 = deconv3x3(in_planes=128, out_planes=64, stride=2)
        self.deconv8 = deconv3x3(in_planes=64, out_planes=64, stride=1)
        self.deconv9 = deconv3x3(in_planes=64, out_planes=64, stride=2)
        self.deconv10 = deconv3x3(in_planes=64, out_planes=1, stride=2)
        self.deconv11 = deconv3x3(in_planes=64, out_planes=1, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.lrelu(x)
        x = self.deconv2(x)
        x = self.lrelu(x)
        x = self.deconv3(x)
        x = self.lrelu(x)
        x = self.deconv4(x)
        x = self.lrelu(x)
        x = self.deconv5(x)
        x = self.lrelu(x)
        x = self.deconv6(x)
        x = self.lrelu(x)
        x = self.deconv7(x)
        x = self.lrelu(x)
        x = self.deconv8(x)
        x = self.lrelu(x)
        x = self.deconv9(x)
        x = self.lrelu(x)
        x = self.deconv10(x)
        x = self.lrelu(x)
        x = self.deconv11(x)
        x = self.sigmoid(x)

        return x
