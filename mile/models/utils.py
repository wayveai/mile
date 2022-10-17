import torch
import torch.nn as nn


class Concat(nn.Module):
    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=-1)


class PixelNorm(nn.Module):
    def forward(self, x, epsilon=1e-8):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdims=True) + epsilon)

