"""
Simple UNet for 11-channel Gaussian prediction.

Input: RGB image (3ch)
Output: 11-channel Gaussian parameters per pixel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .gaussian_utils import parse_gaussian_params_11ch


def get_padding_layer(pad_type: str, padding: int):
    """
    Get padding layer based on type.

    Args:
        pad_type: 'zero', 'reflect', 'replicate', 'circular'
        padding: padding size

    Returns:
        nn.Module or None (for zero padding, use conv's built-in padding)
    """
    if pad_type == "zero":
        return None  # Use conv's built-in padding
    elif pad_type == "reflect":
        return nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        return nn.ReplicationPad2d(padding)
    elif pad_type == "circular":
        return nn.CircularPad2d(padding)
    else:
        raise ValueError(f"Unknown pad_type: {pad_type}. Use 'zero', 'reflect', 'replicate', or 'circular'")


def get_pad_mode(pad_type: str) -> str:
    """Get F.pad mode string from pad_type."""
    if pad_type == "zero":
        return "constant"
    elif pad_type == "reflect":
        return "reflect"
    elif pad_type == "replicate":
        return "replicate"
    elif pad_type == "circular":
        return "circular"
    else:
        return "reflect"  # default


class DoubleConv(nn.Module):
    """(Pad -> Conv -> BN -> ReLU) x 2 with configurable padding."""
    def __init__(self, in_ch, out_ch, pad_type: str = "reflect"):
        super().__init__()
        self.pad_type = pad_type

        if pad_type == "zero":
            # Use conv's built-in padding for zero padding
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            # Use explicit padding layer
            pad_layer = get_padding_layer(pad_type, 1)
            self.net = nn.Sequential(
                pad_layer,
                nn.Conv2d(in_ch, out_ch, 3, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                get_padding_layer(pad_type, 1),
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
    """Downsample -> DoubleConv with configurable pooling and padding."""
    def __init__(self, in_ch, out_ch, pool_type: str = "max", pad_type: str = "reflect"):
        super().__init__()
        self.pool_type = pool_type
        self.pad_type = pad_type

        if pool_type == "max":
            self.pool = nn.MaxPool2d(2)
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(2)
        elif pool_type == "attn":
            self.pool = AttentionPool2d(in_ch)
        elif pool_type == "strided":
            # Strided conv with configurable padding
            if pad_type == "zero":
                self.pool = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                )
            else:
                pad_layer = get_padding_layer(pad_type, 1)
                self.pool = nn.Sequential(
                    pad_layer,
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                )
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. Use 'max', 'avg', 'attn', or 'strided'")

        self.conv = DoubleConv(in_ch, out_ch, pad_type)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upsample -> Concat skip -> DoubleConv with configurable padding."""
    def __init__(self, in_ch, out_ch, pad_type: str = "reflect"):
        super().__init__()
        self.pad_type = pad_type
        self.pad_mode = get_pad_mode(pad_type)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, pad_type)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if sizes don't match
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2], mode=self.pad_mode)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CrossAttentionSelector(nn.Module):
    """
    Cross-attention based Gaussian selector.

    k learnable queries attend to HxW candidate Gaussians,
    outputting k selected/aggregated Gaussians.

    This allows global reasoning about which Gaussians to keep,
    naturally handling redundancy by letting queries compete.
    """

    def __init__(
        self,
        num_queries: int = 1024,
        input_dim: int = 11,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable query embeddings
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # Project input (11ch Gaussian params) to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            ) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection: hidden_dim -> 10 (no importance needed)
        self.output_proj = nn.Linear(hidden_dim, 10)

    def forward(self, candidates: torch.Tensor, H: int, W: int) -> Dict[str, torch.Tensor]:
        """
        Args:
            candidates: (B, 11, H, W) - per-pixel Gaussian predictions
            H, W: image dimensions for position encoding

        Returns:
            gaussians dict with (B, k, *) tensors
        """
        B = candidates.shape[0]
        device = candidates.device

        # Flatten spatial dims: (B, 11, H, W) -> (B, HW, 11)
        candidates_flat = candidates.flatten(2).permute(0, 2, 1)  # (B, HW, 11)

        # Project to hidden dim
        kv = self.input_proj(candidates_flat)  # (B, HW, hidden_dim)

        # Expand queries for batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, k, hidden_dim)

        # Cross-attention layers
        for attn, norm, ffn, ffn_norm in zip(self.layers, self.norms, self.ffns, self.ffn_norms):
            # Cross-attention: queries attend to candidates
            attn_out, _ = attn(q, kv, kv)
            q = norm(q + attn_out)

            # FFN
            q = ffn_norm(q + ffn(q))

        # Output projection -> 10 params per Gaussian
        output = self.output_proj(q)  # (B, k, 10)

        # Parse output to Gaussian parameters
        return self._parse_output(output, H, W)

    def _parse_output(self, output: torch.Tensor, H: int, W: int) -> Dict[str, torch.Tensor]:
        """Parse cross-attention output to Gaussian dict."""
        B, k, _ = output.shape

        # Position: directly predict absolute position (sigmoid * image_size)
        means = torch.sigmoid(output[..., 0:2]) * torch.tensor([W-1, H-1], device=output.device)

        # Scale: softplus + minimum
        scales = F.softplus(output[..., 2:4]) + 1.0

        # Rotation: unbounded
        rotations = output[..., 4:5]

        # Color: sigmoid
        colors = torch.sigmoid(output[..., 5:8])

        # Opacity: sigmoid
        opacities = torch.sigmoid(output[..., 8:9])

        # Depth: softplus (positive)
        depths = F.softplus(output[..., 9:10])

        return {
            'means': means,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'opacities': opacities,
            'depths': depths,
        }


class GaussianSplitNet(nn.Module):
    """
    UNet for Gaussian prediction with configurable output mode.

    Architecture:
        in -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> output
        (in = 3 for RGB, 11 for identity Gaussians)

    Output modes:
        - 'conv': Per-pixel 11-channel prediction (H*W Gaussians, needs top-k)
        - 'cross_attn': Cross-attention selection (k Gaussians, no top-k needed)

    Output (conv mode): 11 channels per pixel
        - delta_mu (2): position offset
        - scale (2): Gaussian size
        - rotation (1): angle
        - color (3): RGB
        - opacity (1): alpha
        - importance (1): for top-k selection
        - depth (1): for occlusion ordering

    Output (cross_attn mode): k Gaussians via learned queries
        - No importance needed, queries naturally select/aggregate

    Identity input mode (use_identity_input=True):
        - Input: 11ch identity Gaussians instead of 3ch RGB
        - Each pixel starts as a valid Gaussian (pos=center, scale=1, color=RGB, etc.)
        - Network learns to REFINE rather than predict from scratch
        - Enables residual learning: output ≈ identity + delta

    Pool types:
        - 'max': MaxPool2d (default) - preserves strong features/edges
        - 'avg': AvgPool2d - smoother, reduces noise
        - 'attn': AttentionPool2d - learned adaptive pooling
        - 'strided': Strided convolution - learnable downsampling

    Pad types:
        - 'zero': Zero padding - may cause border artifacts
        - 'reflect': Reflection padding (default) - mirrors at boundary
        - 'replicate': Replication padding - repeats edge pixels
        - 'circular': Circular padding - wraps around (for tiled textures)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 4,  # Unused, kept for API compatibility
        pool_type: str = "max",
        pad_type: str = "reflect",
        output_mode: str = "conv",  # 'conv' or 'cross_attn'
        num_gaussians: int = 1024,  # Only for cross_attn mode
        use_identity_input: bool = False,  # Use 11ch identity Gaussians as input
    ):
        super().__init__()
        _ = num_levels
        self.pool_type = pool_type
        self.pad_type = pad_type
        self.output_mode = output_mode
        self.num_gaussians = num_gaussians
        self.use_identity_input = use_identity_input

        # Determine actual input channels
        actual_in_channels = 11 if use_identity_input else in_channels

        # Encoder
        self.inc = DoubleConv(actual_in_channels, base_channels, pad_type)  # 3 or 11 -> 64
        self.down1 = Down(base_channels, base_channels * 2, pool_type, pad_type)    # 64 -> 128
        self.down2 = Down(base_channels * 2, base_channels * 4, pool_type, pad_type)  # 128 -> 256
        self.down3 = Down(base_channels * 4, base_channels * 8, pool_type, pad_type)  # 256 -> 512

        # Decoder
        self.up1 = Up(base_channels * 8 + base_channels * 4, base_channels * 4, pad_type)  # 512+256 -> 256
        self.up2 = Up(base_channels * 4 + base_channels * 2, base_channels * 2, pad_type)  # 256+128 -> 128
        self.up3 = Up(base_channels * 2 + base_channels, base_channels, pad_type)          # 128+64 -> 64

        # Output head: 64 -> 11 (with depth)
        self.outc = nn.Conv2d(base_channels, 11, kernel_size=1)

        # Cross-attention selector (for cross_attn mode)
        if output_mode == "cross_attn":
            self.selector = CrossAttentionSelector(
                num_queries=num_gaussians,
                input_dim=11,
                hidden_dim=256,
                num_heads=8,
                num_layers=2,
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape

        # Encoder
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)

        # Decoder
        feat = self.up1(x4, x3)  # (B, 256, H/4, W/4)
        feat = self.up2(feat, x2)   # (B, 128, H/2, W/2)
        feat = self.up3(feat, x1)   # (B, 64, H, W)

        # Per-pixel Gaussian prediction
        candidates = self.outc(feat)    # (B, 11, H, W)

        if self.output_mode == "conv":
            # Use all per-pixel predictions (need top-k later)
            gaussians = parse_gaussian_params_11ch(candidates)
        else:  # cross_attn
            # Cross-attention selects k Gaussians from HxW candidates
            gaussians = self.selector(candidates, H, W)

        return {
            'gaussians': gaussians,
            'multi_scale_gaussians': [gaussians],
            'importance_maps': None,
        }

    def get_num_gaussians(self, H: int, W: int) -> int:
        if self.output_mode == "cross_attn":
            return self.num_gaussians
        return H * W

    def get_gaussian_info(self, H: int, W: int) -> str:
        input_type = "identity(11ch)" if self.use_identity_input else "RGB(3ch)"
        if self.output_mode == "cross_attn":
            return f"GaussianSplitNet (cross_attn, {input_type}): {self.num_gaussians:,} Gaussians from {H}x{W} candidates"
        return f"GaussianSplitNet (pool={self.pool_type}, pad={self.pad_type}, {input_type}): {H}x{W} = {H*W:,} Gaussians"
