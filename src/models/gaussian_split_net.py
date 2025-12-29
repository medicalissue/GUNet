"""
Gaussian UNet with Importance-based Selection.

Input: Image → Initial Gaussians (direct pixel-to-Gaussian conversion)
Process: Standard UNet with feature channels (64→128→256→...)
Output: 10-channel Gaussians at full resolution

10 output channels:
- delta_mu (2): offset from pixel center
- scale (2): Gaussian scale
- rotation (1): rotation angle
- color (3): RGB
- opacity (1): alpha
- importance (1): for top-k selection (learned, not same as opacity!)

Key insight: importance ≠ opacity
- Opacity: "How transparent is this Gaussian?"
- Importance: "Should this Gaussian be rendered at all?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .gaussian_utils import parse_gaussian_params_10ch


def image_to_initial_gaussians(image: torch.Tensor) -> torch.Tensor:
    """
    Convert image pixels to initial Gaussian parameters (no learning).

    Each pixel becomes a Gaussian with:
    - delta_mu = 0 (centered at pixel)
    - scale = 1 (pixel-sized)
    - rotation = 0
    - color = pixel color
    - opacity = 1
    - importance = 0.5

    Args:
        image: (B, 3, H, W) RGB image in [0, 1]

    Returns:
        raw_params: (B, 10, H, W) raw parameters for parsing
    """
    B, _, H, W = image.shape
    device = image.device

    # Create raw parameter tensor
    # These values, when parsed, give the desired initial Gaussians
    raw = torch.zeros(B, 10, H, W, device=device, dtype=image.dtype)

    # delta_mu: tanh(0) * 0.5 = 0 → centered at pixel
    raw[:, 0:2] = 0.0

    # scale: softplus(0.54) ≈ 1.0 → pixel-sized
    raw[:, 2:4] = 0.54

    # rotation: 0 → no rotation
    raw[:, 4:5] = 0.0

    # color: logit(pixel_color) so sigmoid gives back pixel color
    # Clamp to avoid inf
    color_clamped = image.clamp(0.001, 0.999)
    raw[:, 5:8] = torch.log(color_clamped / (1 - color_clamped))

    # opacity: sigmoid(6) ≈ 0.997 ≈ 1.0 → fully opaque
    raw[:, 8:9] = 6.0

    # importance: sigmoid(0) = 0.5 → neutral importance
    raw[:, 9:10] = 0.0

    return raw


class ConvBlock(nn.Module):
    """Standard convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: pool → conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class GaussianSplitNet(nn.Module):
    """
    UNet for Gaussian prediction with importance-based selection.

    Architecture:
    1. Image → Initial Gaussians (direct conversion, no learning)
    2. Standard UNet with feature channels (10→64→128→256→...)
    3. Output: 10 channels at full resolution
    4. Top-k selection based on importance

    This allows:
    - Large Gaussians with low opacity but high importance to survive
    - Network learns refinements from pixel-sized initial Gaussians
    """

    def __init__(
        self,
        in_channels: int = 3,  # Unused, kept for API compatibility
        base_channels: int = 64,
        num_levels: int = 4,
    ):
        super().__init__()
        _ = in_channels  # API compatibility
        self.num_levels = num_levels
        self.base_channels = base_channels

        # Channel progression: 10 → 64 → 128 → 256 → 512 (capped at 512)
        channels = [10]  # Input is 10-channel initial Gaussians
        for i in range(num_levels):
            channels.append(base_channels * min(2**i, 8))

        # Encoder
        self.enc_first = ConvBlock(channels[0], channels[1])
        self.encoders = nn.ModuleList([
            DownBlock(channels[i + 1], channels[i + 2])
            for i in range(num_levels - 1)
        ])

        # Bottleneck
        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        # Decoder (reverse order, with skip connections)
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 1):
            # in_channels = current + skip, out_channels = target
            dec_in = channels[num_levels - i] + channels[num_levels - i]
            dec_out = channels[num_levels - 1 - i]
            self.decoders.append(UpBlock(dec_in, dec_out))

        # Final output head: features → 10 channels
        self.output_head = nn.Conv2d(channels[1], 10, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Dictionary with 'gaussians' containing parsed parameters
        """
        B, C, H, W = x.shape

        # Image → Initial Gaussians (no learning)
        g = image_to_initial_gaussians(x)

        # Encoder
        enc_features = []
        g = self.enc_first(g)
        enc_features.append(g)

        for encoder in self.encoders:
            g = encoder(g)
            enc_features.append(g)

        # Bottleneck
        g = self.bottleneck(g)

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(enc_features) - 1 - i
            skip = enc_features[skip_idx]
            g = decoder(g, skip)

        # Output head → 10 channels
        g = self.output_head(g)

        # Parse Gaussian parameters
        gaussians = parse_gaussian_params_10ch(g)

        return {
            'gaussians': gaussians,
            'multi_scale_gaussians': [gaussians],  # For compatibility
            'importance_maps': None,
        }

    def get_num_gaussians(self, H: int, W: int) -> int:
        """Total Gaussians = H * W (single scale)"""
        return H * W

    def get_gaussian_info(self, H: int, W: int) -> str:
        """Get Gaussian distribution info."""
        count = H * W
        return f"Gaussian UNet (single scale):\n  Level 0: {H}x{W} = {count:,} Gaussians\n  Total: {count:,} Gaussians"
