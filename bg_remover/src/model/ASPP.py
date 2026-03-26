import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # Branch 1: Global Average Pooling → 1x1 conv
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_pool = nn.BatchNorm2d(out_channels)

        # Branch 2: 1x1 convolution
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

        # Branch 3: 3x3 dilated conv, rate=6
        self.conv3x3_r6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn3x3_r6 = nn.BatchNorm2d(out_channels)

        # Branch 4: 3x3 dilated conv, rate=12
        self.conv3x3_r12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3x3_r12 = nn.BatchNorm2d(out_channels)

        # Branch 5: 3x3 dilated conv, rate=18
        self.conv3x3_r18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn3x3_r18 = nn.BatchNorm2d(out_channels)

        # Final 1x1 conv after concatenation (5 * out_channels → out_channels)
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # Branch 1
        y1 = self.pool(x)
        y1 = F.relu(self.bn_pool(self.conv_pool(y1)))
        y1 = F.interpolate(y1, size=(h, w), mode="bilinear", align_corners=False)

        # Branch 2
        y2 = F.relu(self.bn1x1(self.conv1x1(x)))

        # Branch 3
        y3 = F.relu(self.bn3x3_r6(self.conv3x3_r6(x)))

        # Branch 4
        y4 = F.relu(self.bn3x3_r12(self.conv3x3_r12(x)))

        # Branch 5
        y5 = F.relu(self.bn3x3_r18(self.conv3x3_r18(x)))

        # Concatenate + final conv
        y = torch.cat([y1, y2, y3, y4, y5], dim=1)
        y = F.relu(self.bn_out(self.conv_out(y)))
        return y