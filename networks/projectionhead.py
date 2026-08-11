import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead1(nn.Module):
    def __init__(self, input_channels=2048, num_class=128):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Linear(input_channels, input_channels)
        self.f2 = nn.Linear(input_channels, num_class)

    def forward(self, x):
        x = self.gap(x)
        y = self.f1(x.squeeze())
        y = self.f2(y)

        return y
    
class ProjectionHead2(nn.Module):
    def __init__(self, input_channels=192, num_class=128):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)  # 线性变换
        self.conv2 = nn.Conv2d(input_channels, num_class, kernel_size=1)  # 映射到 num_class 维度

    def forward(self, x):
        x = self.conv1(x)  # (B, C, H, W) → (B, C, H, W)
        y = self.conv2(x)  # (B, C, H, W) → (B, num_class, H, W)

        return y