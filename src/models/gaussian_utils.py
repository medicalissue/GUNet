"""
Gaussian utilities for Multi-Scale Gaussian UNet.

Each Gaussian has 9 parameters:
- delta_mu: (2) offset from pixel center, in range [-0.5, 0.5]
- scale: (2) Gaussian scale (in log space from network, exp() applied)
- rotation: (1) rotation angle in radians
- color: (3) RGB color
- opacity: (1) opacity value (sigmoid applied to get [0,1])
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict


def create_pixel_coords(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Create pixel coordinate grid.

    Returns:
        coords: (H, W, 2) tensor of (x, y) coordinates
    """
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
    return coords


def parse_gaussian_params(raw_params: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Parse raw 9-channel output into Gaussian parameters.

    Args:
        raw_params: (B, 9, H, W) raw network output

    Returns:
        Dictionary with parsed parameters:
        - delta_mu: (B, H, W, 2) offset in [-0.5, 0.5]
        - scale: (B, H, W, 2) positive scale values
        - rotation: (B, H, W, 1) rotation angle in radians
        - color: (B, H, W, 3) RGB in [0, 1]
        - opacity: (B, H, W, 1) opacity in [0, 1]
    """
    B, C, H, W = raw_params.shape
    assert C == 9, f"Expected 9 channels, got {C}"

    # Rearrange to (B, H, W, 9)
    params = raw_params.permute(0, 2, 3, 1)

    # Parse each parameter
    delta_mu = torch.tanh(params[..., 0:2]) * 0.5  # [-0.5, 0.5]
    scale = torch.exp(params[..., 2:4])  # positive, log-space output
    rotation = params[..., 4:5]  # radians, unbounded
    # Color: direct output with clamp for full dynamic range
    color = params[..., 5:8].clamp(0, 1)  # [0, 1]
    # Opacity: direct output with clamp (allows exact 0/1 for sparsity)
    opacity = params[..., 8:9].clamp(0, 1)  # [0, 1]

    return {
        'delta_mu': delta_mu,
        'scale': scale,
        'rotation': rotation,
        'color': color,
        'opacity': opacity,
    }


def upsample_gaussians(
    params: Dict[str, torch.Tensor],
    level: int,
    target_H: int,
    target_W: int,
) -> Dict[str, torch.Tensor]:
    """
    Upsample Gaussians from level l to full resolution.

    For level l:
    - mean position: μ = 2^l * (pixel_coord + delta_mu)
    - scale: s = 2^l * s_local
    - rotation, color, opacity: unchanged

    Args:
        params: Dictionary with Gaussian parameters at level l
        level: The pyramid level (0 = full resolution)
        target_H: Target height (full resolution)
        target_W: Target width (full resolution)

    Returns:
        Dictionary with Gaussians in full resolution coordinates
    """
    B, H, W, _ = params['delta_mu'].shape
    device = params['delta_mu'].device
    scale_factor = 2 ** level

    # Create local pixel coordinates for this level
    local_coords = create_pixel_coords(H, W, device)  # (H, W, 2)

    # Compute full-resolution mean positions
    # μ = 2^l * (local_coord + delta_mu)
    local_positions = local_coords.unsqueeze(0) + params['delta_mu']  # (B, H, W, 2)
    means = scale_factor * local_positions

    # Scale the Gaussian scales
    scales = scale_factor * params['scale']

    return {
        'means': means,  # (B, H, W, 2) in full resolution coordinates
        'scales': scales,  # (B, H, W, 2)
        'rotations': params['rotation'],  # (B, H, W, 1)
        'colors': params['color'],  # (B, H, W, 3)
        'opacities': params['opacity'],  # (B, H, W, 1)
        'level': level,
    }


def flatten_gaussians(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Flatten spatial dimensions of Gaussians.

    Args:
        params: Dictionary with (B, H, W, C) tensors

    Returns:
        Dictionary with (B, N, C) tensors where N = H * W
    """
    B = params['means'].shape[0]

    return {
        'means': params['means'].reshape(B, -1, 2),
        'scales': params['scales'].reshape(B, -1, 2),
        'rotations': params['rotations'].reshape(B, -1, 1),
        'colors': params['colors'].reshape(B, -1, 3),
        'opacities': params['opacities'].reshape(B, -1, 1),
    }


def merge_multi_scale_gaussians(
    multi_scale_params: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Merge Gaussians from all pyramid levels into a single set.

    Args:
        multi_scale_params: List of Gaussian parameter dicts from each level

    Returns:
        Merged dictionary with all Gaussians concatenated
        Each tensor has shape (B, N_total, C) where N_total = sum of all levels
    """
    flattened = [flatten_gaussians(p) for p in multi_scale_params]

    merged = {
        'means': torch.cat([f['means'] for f in flattened], dim=1),
        'scales': torch.cat([f['scales'] for f in flattened], dim=1),
        'rotations': torch.cat([f['rotations'] for f in flattened], dim=1),
        'colors': torch.cat([f['colors'] for f in flattened], dim=1),
        'opacities': torch.cat([f['opacities'] for f in flattened], dim=1),
    }

    return merged


def compute_total_gaussians(H: int, W: int, num_levels: int) -> int:
    """
    Compute total number of Gaussians across all pyramid levels.

    N = sum_{l=0}^{L-1} (H/2^l * W/2^l) ≈ 4/3 * H * W
    """
    total = 0
    for l in range(num_levels):
        h = H // (2 ** l)
        w = W // (2 ** l)
        total += h * w
    return total
