import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from .ASPP import ASPP
from .SAE import SqueezeAndExcite


class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Encoder (ResNet50 pretrained on ImageNet) ──────────────────────
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Low-level features: output of layer1 (equivalent to conv2_block2_out)
        self.encoder_low = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # → 256 channels, stride 4
        )
        # High-level features: layer2 → layer3 → layer4 (equivalent to conv4_block6_out)
        self.encoder_high = nn.Sequential(
            resnet.layer2,  # → 512 channels, stride 8
            resnet.layer3,  # → 1024 channels, stride 16
        )

        # ── ASPP on high-level features ────────────────────────────────────
        self.aspp = ASPP(in_channels=1024, out_channels=256)

        # ── Low-level feature projection (256 → 48) ───────────────────────
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeAndExcite(256)

        # ── Final classifier ──────────────────────────────────────────────
        self.final_conv = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # Encoder
        low = self.encoder_low(x)        # stride 4
        high = self.encoder_high(low)     # stride 16

        # ASPP
        x_a = self.aspp(high)
        x_a = F.interpolate(x_a, size=low.shape[2:], mode="bilinear", align_corners=False)  # 4× upsample

        # Low-level projection
        x_b = self.low_proj(low)

        # Concatenate + decode
        x = torch.cat([x_a, x_b], dim=1)
        x = self.decoder_conv(x)
        x = self.se(x)

        # 4× upsample to original resolution
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x


def deeplabv3_plus(shape):
    """
    Factory function — keeps the same call signature as the old TF version.
    `shape` is (H, W, C), but PyTorch model doesn't need it at construction time.
    """
    return DeepLabV3Plus()
