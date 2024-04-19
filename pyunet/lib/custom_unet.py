import torch
import torch.nn as nn
import attention_conv_2d as AttentionConv2d
from depthwise_seperable_conv import DepthwiseSeperableConv

class CustomDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(CustomDoubleConv, self).__init__()

        self.conv = nn.Sequential(
            AttentionConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeperableConv(out_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)