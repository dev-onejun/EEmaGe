"""
This is the model presented in the work: 
    S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah, 
    Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features,  
    IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909
"""

import torch
import torch.nn as nn

import math


class EEGFeaturesExtractor(nn.Module):
    def __init__(
        self,
        in_channels=1,
        temp_channels=10,
        out_channels=50,
        input_width=440,
        input_height=128,
        temporal_kernel=(1, 33),
        temporal_stride=(1, 2),
        temporal_dilation_list=[(1, 1), (1, 2), (1, 4), (1, 8), (1, 16)],
        num_temporal_layers=5,
        num_spatial_layers=4,
        spatial_stride=(2, 1),
        num_residual_blocks=4,
        down_kernel=3,
        down_stride=2,
    ):
        super().__init__()

        self.temporal_block = TemporalBlock(
            in_channels,
            temp_channels,
            num_temporal_layers,
            temporal_kernel,
            temporal_stride,
            temporal_dilation_list,
        )

        self.spatial_block = SpatialBlock(
            temp_channels * num_temporal_layers,
            out_channels,
            num_spatial_layers,
            spatial_stride,
            input_height,
        )

        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(
                        out_channels * num_spatial_layers,
                        out_channels * num_spatial_layers,
                    ),
                    ConvLayer2D(
                        out_channels * num_spatial_layers,
                        out_channels * num_spatial_layers,
                        down_kernel,
                        down_stride,
                        0,
                        1,
                    ),
                )
                for i in range(num_residual_blocks)
            ]
        )

        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers, out_channels, down_kernel, 1, 0, 1
        )

    def forward(self, x):
        out = self.temporal_block(x)

        out = self.spatial_block(out)

        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)

        out = self.final_conv(out)

        out = out.view(-1)

        return out


class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
        )
        self.add_module("drop", nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        kernel_size,
        stride,
        dilation_list,
    ):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (
                n_layers - len(dilation_list)
            )

        padding = []
        # Compute padding for each temporal layer to have a fixed size output
        # Output size is controlled by striding to be 1 / 'striding' of the original size
        for dilation in dilation_list:
            filter_size = kernel_size[1] * dilation[1] - 1
            temp_pad = math.floor((filter_size - 1) / 2) - 1 * (dilation[1] // 2 - 1)
            padding.append((0, temp_pad))

        self.layers = nn.ModuleList(
            [
                ConvLayer2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding[i],
                    dilation_list[i],
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)
        return out


class SpatialBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_spatial_layers, stride, input_height
    ):
        super().__init__()

        kernel_list = []
        for i in range(num_spatial_layers):
            kernel_list.append(((input_height // (i + 1)), 1))

        padding = []
        for kernel in kernel_list:
            temp_pad = math.floor((kernel[0] - 1) / 2)
            padding.append((temp_pad, 0))

        self.layers = nn.ModuleList(
            [
                ConvLayer2D(
                    in_channels, out_channels, kernel_list[i], stride, padding[i], 1
                )
                for i in range(num_spatial_layers)
            ]
        )

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)

        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
