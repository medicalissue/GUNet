"""
Simple UNet for 10-channel Gaussian prediction.

Input: RGB image (3ch)
Output: 10-channel Gaussian parameters per pixel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .gaussian_utils import parse_gaussian_params_11ch


class DoubleConv(nn.Module):
    """(ReflectionPad -> Conv -> BN -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionPool2d(nn.Module):
    """Attention-based 2x2 pooling - learns which pixels to attend to."""
    def __init__(self, in_ch):
        super().__init__()
        self.attn_conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to 2x2 blocks
        x = x.view(B, C, H // 2, 2, W // 2, 2)  # (B, C, H/2, 2, W/2, 2)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, C, H/2, W/2, 2, 2)
        x = x.reshape(B, C, H // 2, W // 2, 4)  # (B, C, H/2, W/2, 4)

        # Compute attention weights for 2x2 blocks
        # Use mean pooled features to compute attention
        x_for_attn = x.mean(dim=1, keepdim=True)  # (B, 1, H/2, W/2, 4)
        attn = F.softmax(x_for_attn, dim=-1)  # (B, 1, H/2, W/2, 4)

        # Apply attention
        out = (x * attn).sum(dim=-1)  # (B, C, H/2, W/2)
        return out


class Down(nn.Module):
    """Downsample -> DoubleConv with configurable pooling."""
    def __init__(self, in_ch, out_ch, pool_type: str = "max"):
        super().__init__()
        self.pool_type = pool_type

        if pool_type == "max":
            self.pool = nn.MaxPool2d(2)
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(2)
        elif pool_type == "attn":
            self.pool = AttentionPool2d(in_ch)
        elif pool_type == "strided":
            # Strided conv does pooling + channel change together
            self.pool = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. Use 'max', 'avg', 'attn', or 'strided'")

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upsample -> Concat skip -> DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if sizes don't match (use reflect/mirror padding)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2], mode='reflect')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GaussianSplitNet(nn.Module):
    """
    Simple UNet for 11-channel Gaussian prediction.

    Architecture:
        3 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 11

    Output: 11 channels per pixel
        - delta_mu (2): position offset
        - scale (2): Gaussian size
        - rotation (1): angle
        - color (3): RGB
        - opacity (1): alpha
        - importance (1): for top-k selection
        - depth (1): for occlusion ordering

    Pool types:
        - 'max': MaxPool2d (default) - preserves strong features/edges
        - 'avg': AvgPool2d - smoother, reduces noise
        - 'attn': AttentionPool2d - learned adaptive pooling
        - 'strided': Strided convolution - learnable downsampling
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 4,  # Unused, kept for API compatibility
        pool_type: str = "max",
    ):
        super().__init__()
        _ = num_levels
        self.pool_type = pool_type

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)      # 3 -> 64
        self.down1 = Down(base_channels, base_channels * 2, pool_type)    # 64 -> 128
        self.down2 = Down(base_channels * 2, base_channels * 4, pool_type)  # 128 -> 256
        self.down3 = Down(base_channels * 4, base_channels * 8, pool_type)  # 256 -> 512

        # Decoder
        self.up1 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)  # 512+256 -> 256
        self.up2 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)  # 256+128 -> 128
        self.up3 = Up(base_channels * 2 + base_channels, base_channels)          # 128+64 -> 64

        # Output head: 64 -> 11 (with depth)
        self.outc = nn.Conv2d(base_channels, 11, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)

        # Decoder
        x = self.up1(x4, x3)  # (B, 256, H/4, W/4)
        x = self.up2(x, x2)   # (B, 128, H/2, W/2)
        x = self.up3(x, x1)   # (B, 64, H, W)

        # Output
        out = self.outc(x)    # (B, 11, H, W)

        # Parse to Gaussian parameters (with depth)
        gaussians = parse_gaussian_params_11ch(out)

        return {
            'gaussians': gaussians,
            'multi_scale_gaussians': [gaussians],
            'importance_maps': None,
        }

    def get_num_gaussians(self, H: int, W: int) -> int:
        return H * W

    def get_gaussian_info(self, H: int, W: int) -> str:
        return f"GaussianSplitNet (pool={self.pool_type}): {H}x{W} = {H*W:,} Gaussians"
