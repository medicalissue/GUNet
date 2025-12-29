"""
Multi-Scale Gaussian UNet.

Architecture:
- Encoder: Multiple levels with downsampling
- Decoder: Each level outputs 9-channel Gaussian parameters
- Skip connections from encoder to decoder

Output at each level l:
- Resolution: (H/2^l, W/2^l)
- Channels: 9 (delta_mu, scale, rotation, color, opacity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from .gaussian_utils import parse_gaussian_params, upsample_gaussians, merge_multi_scale_gaussians


class ConvBlock(nn.Module):
    """Double convolution block with ReflectionPad, BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block: Upsample + Concat skip + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class GaussianHead(nn.Module):
    """
    Output head for Gaussian parameters (9 channels).

    Uses a deeper network with 3x3 conv for better spatial context,
    followed by 1x1 conv for final parameter prediction.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        hidden_channels = max(in_channels // 2, 32)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 9, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiScaleGaussianUNet(nn.Module):
    """
    Multi-Scale Gaussian UNet.

    Produces Gaussians at multiple resolutions and merges them
    into a single set for rendering.

    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        base_channels: Base number of channels (doubled at each level)
        num_levels: Number of pyramid levels for encoder/decoder
        start_level: First level to output Gaussians (0=full res, 1=half, etc.)
                     Levels < start_level will not output Gaussians.

    Example with num_levels=4, start_level=1, image_size=256:
        - Level 0 (256x256): NO Gaussians (skipped)
        - Level 1 (128x128): 16,384 Gaussians
        - Level 2 (64x64): 4,096 Gaussians
        - Level 3 (32x32): 1,024 Gaussians
        Total: 21,504 Gaussians
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 5,
        start_level: int = 0,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.start_level = start_level

        # Channel progression: 64 -> 128 -> 256 -> 512 -> 512
        channels = [base_channels * min(2**i, 8) for i in range(num_levels)]

        # Encoder
        self.enc_first = ConvBlock(in_channels, channels[0])
        self.encoders = nn.ModuleList([
            DownBlock(channels[i], channels[i + 1])
            for i in range(num_levels - 1)
        ])

        # Decoder
        self.decoders = nn.ModuleList([
            UpBlock(channels[i + 1] + channels[i], channels[i])
            for i in range(num_levels - 2, -1, -1)
        ])

        # Gaussian output heads only for levels >= start_level
        self.gaussian_heads = nn.ModuleDict()
        for i in range(num_levels):
            if i >= start_level:
                self.gaussian_heads[str(i)] = GaussianHead(channels[i])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Dictionary containing:
            - 'gaussians': Merged Gaussian parameters for rendering
            - 'multi_scale_gaussians': List of Gaussians at each level
        """
        B, C, H, W = x.shape

        # Encoder forward
        enc_features = []

        # Level 0 (full resolution)
        f = self.enc_first(x)
        enc_features.append(f)

        # Levels 1 to L-1 (downsampling)
        for encoder in self.encoders:
            f = encoder(f)
            enc_features.append(f)

        # Decoder forward with Gaussian outputs
        multi_scale_raw = []
        dec_features = [enc_features[-1]]  # Start from bottleneck

        # Bottleneck Gaussian output (if >= start_level)
        bottleneck_level = self.num_levels - 1
        if bottleneck_level >= self.start_level:
            raw_params = self.gaussian_heads[str(bottleneck_level)](enc_features[-1])
            multi_scale_raw.append((raw_params, bottleneck_level))

        # Decode and produce Gaussians at each level
        for i, decoder in enumerate(self.decoders):
            level_idx = self.num_levels - 2 - i
            skip = enc_features[level_idx]
            f = decoder(dec_features[-1], skip)
            dec_features.append(f)

            # Gaussian output for this level (if >= start_level)
            if level_idx >= self.start_level:
                raw_params = self.gaussian_heads[str(level_idx)](f)
                multi_scale_raw.append((raw_params, level_idx))

        # Parse and upsample Gaussians from all levels
        multi_scale_gaussians = []
        for raw_params, level in multi_scale_raw:
            parsed = parse_gaussian_params(raw_params)
            upsampled = upsample_gaussians(parsed, level, H, W)
            multi_scale_gaussians.append(upsampled)

        # Merge all levels
        merged_gaussians = merge_multi_scale_gaussians(multi_scale_gaussians)

        return {
            'gaussians': merged_gaussians,
            'multi_scale_gaussians': multi_scale_gaussians,
        }

    def get_num_gaussians(self, H: int, W: int) -> int:
        """Calculate total number of Gaussians for given image size."""
        total = 0
        for l in range(self.start_level, self.num_levels):
            h = H // (2 ** l)
            w = W // (2 ** l)
            total += h * w
        return total

    def get_gaussian_info(self, H: int, W: int) -> str:
        """Get detailed info about Gaussian distribution."""
        lines = []
        total = 0
        for l in range(self.num_levels):
            h = H // (2 ** l)
            w = W // (2 ** l)
            count = h * w
            if l >= self.start_level:
                lines.append(f"  Level {l}: {h}x{w} = {count:,} Gaussians")
                total += count
            else:
                lines.append(f"  Level {l}: {h}x{w} = SKIPPED")
        lines.append(f"  Total: {total:,} Gaussians")
        return "\n".join(lines)
