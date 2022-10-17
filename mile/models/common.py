from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RouteEncode(nn.Module):
    def __init__(self, out_channels, backbone='resnet18'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=[4])
        self.out_channels = out_channels
        feature_info = self.backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        self.fc = nn.Linear(feature_info[-1]['num_chs'], out_channels)

    def forward(self, route):
        x = self.backbone(route)[0]
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.fc(x)


class GRUCellLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size, reset_bias=1.0):
        super().__init__()
        self.reset_bias = reset_bias

        self.update_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.update_norm = nn.LayerNorm(hidden_size)

        self.reset_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.reset_norm = nn.LayerNorm(hidden_size)

        self.proposal_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.proposal_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, state):
        update = self.update_layer(torch.cat([inputs, state], -1))
        update = torch.sigmoid(self.update_norm(update))

        reset = self.reset_layer(torch.cat([inputs, state], -1))
        reset = torch.sigmoid(self.reset_norm(reset) + self.reset_bias)

        h_n = self.proposal_layer(torch.cat([inputs, reset * state], -1))
        h_n = torch.tanh(self.proposal_norm(h_n))
        output = (1 - update) * h_n + update * state
        return output


class Policy(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(True),
            nn.Linear(in_channels // 2, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, feature_info, out_channels):
        super().__init__()
        n_upsample_skip_convs = len(feature_info) - 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_info[-1]['num_chs'], out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.upsample_skip_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(feature_info[-i]['num_chs'], out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
            for i in range(2, n_upsample_skip_convs+2)
        )

        self.out_channels = out_channels

    def forward(self, xs: List[Tensor]) -> Tensor:
        x = self.conv1(xs[-1])

        for i, conv in enumerate(self.upsample_skip_convs):
            size = xs[-(i+2)].shape[-2:]
            x = conv(xs[-(i+2)]) + F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ConvInstanceNorm(in_channels, out_channels, latent_n_channels)
        self.conv2 = ConvInstanceNorm(out_channels, out_channels, latent_n_channels)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv1(x, w)
        return self.conv2(x, w)


class ConvInstanceNorm(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels):
        super().__init__()
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.adaptive_norm = AdaptiveInstanceNorm(latent_n_channels, out_channels)

    def forward(self, x, w):
        x = self.conv_act(x)
        return self.adaptive_norm(x, w)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, latent_n_channels, out_channels, epsilon=1e-8):
        super().__init__()
        self.out_channels = out_channels
        self.epsilon = epsilon

        self.latent_affine = nn.Linear(latent_n_channels, 2 * out_channels)

    def forward(self, x, style):
        # Instance norm
        mean = x.mean(dim=(-1, -2), keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x**2, dim=(-1, -2), keepdim=True) + self.epsilon)
        x = x / std

        # Normalising with the style vector
        style = self.latent_affine(style).unsqueeze(-1).unsqueeze(-1)
        scale, bias = torch.split(style, split_size_or_sections=self.out_channels, dim=1)
        out = scale * x + bias
        return out


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = {
            f'bev_segmentation_{self.downsample_factor}': self.segmentation_head(x),
            f'bev_instance_offset_{self.downsample_factor}': self.instance_offset_head(x),
            f'bev_instance_center_{self.downsample_factor}': self.instance_center_head(x),
        }
        return output


class RGBHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.rgb_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = {
            f'rgb_{self.downsample_factor}': self.rgb_head(x),
        }
        return output


class BevDecoder(nn.Module):
    def __init__(self, latent_n_channels, semantic_n_channels, constant_size=(3, 3), is_segmentation=True):
        super().__init__()
        n_channels = 512

        self.constant_tensor = nn.Parameter(torch.randn((n_channels, *constant_size), dtype=torch.float32))

        # Input 512 x 3 x 3
        self.first_norm = AdaptiveInstanceNorm(latent_n_channels, out_channels=n_channels)
        self.first_conv = ConvInstanceNorm(n_channels, n_channels, latent_n_channels)
        # 512 x 3 x 3

        self.middle_conv = nn.ModuleList(
            [DecoderBlock(n_channels, n_channels, latent_n_channels, upsample=True) for _ in range(3)]
        )

        head_module = SegmentationHead if is_segmentation else RGBHead
        # 512 x 24 x 24
        self.conv1 = DecoderBlock(n_channels, 256, latent_n_channels, upsample=True)
        self.head_4 = head_module(256, semantic_n_channels, downsample_factor=4)
        # 256 x 48 x 48

        self.conv2 = DecoderBlock(256, 128, latent_n_channels, upsample=True)
        self.head_2 = head_module(128, semantic_n_channels, downsample_factor=2)
        # 128 x 96 x 96

        self.conv3 = DecoderBlock(128, 64, latent_n_channels, upsample=True)
        self.head_1 = head_module(64, semantic_n_channels, downsample_factor=1)
        # 64 x 192 x 192

    def forward(self, w: Tensor) -> Tensor:
        b = w.shape[0]
        x = self.constant_tensor.unsqueeze(0).repeat([b, 1, 1, 1])

        x = self.first_norm(x, w)
        x = self.first_conv(x, w)

        for module in self.middle_conv:
            x = module(x, w)

        x = self.conv1(x, w)
        output_4 = self.head_4(x)
        x = self.conv2(x, w)
        output_2 = self.head_2(x)
        x = self.conv3(x, w)
        output_1 = self.head_1(x)

        output = {**output_4, **output_2, **output_1}
        return output
