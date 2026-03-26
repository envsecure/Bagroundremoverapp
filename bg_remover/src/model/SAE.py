import torch
import torch.nn as nn


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // ratio, bias=False)
        self.fc2 = nn.Linear(channels // ratio, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        # Global Average Pooling → (B, C)
        se = x.mean(dim=[2, 3])
        # FC → ReLU → FC → Sigmoid
        se = torch.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        # Reshape to (B, C, 1, 1) and scale
        se = se.view(b, c, 1, 1)
        return x * se