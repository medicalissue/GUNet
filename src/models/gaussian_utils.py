"""
Gaussian utilities for Multi-Scale Gaussian UNet.

Each Gaussian has 10 parameters:
- delta_mu: (2) offset from pixel center, in range [-0.5, 0.5]
- scale: (2) Gaussian scale (softplus applied for stability)
- rotation: (1) rotation angle in radians
- color: (3) RGB color
- opacity: (1) opacity value (sigmoid applied to get [0,1])
- importance: (1) importance score for top-k selection (separate from opacity!)

Key insight: importance ≠ opacity
- Opacity: "How transparent is this Gaussian?"
- Importance: "Should this Gaussian be rendered at all?"

This allows large, low-opacity Gaussians to survive top-k selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    scale = F.softplus(params[..., 2:4])  # positive, no explosion (unlike exp)
    rotation = params[..., 4:5]  # radians, unbounded

    # Color: STE (Straight-Through Estimator)
    # Forward: clamp to [0,1], Backward: gradient passes through unchanged
    color_raw = params[..., 5:8]
    # color = color_raw + (color_raw.clamp(0, 1) - color_raw).detach()
    color = torch.sigmoid(color_raw)
    
    # Opacity: sigmoid for smooth [0,1] mapping
    opacity = torch.sigmoid(params[..., 8:9])

    return {
        'delta_mu': delta_mu,
        'scale': scale,
        'rotation': rotation,
        'color': color,
        'opacity': opacity,
    }


def parse_gaussian_params_10ch(raw_params: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Parse raw 10-channel output into Gaussian parameters.

    10 channels = 9 Gaussian params + 1 importance score.

    Args:
        raw_params: (B, 10, H, W) raw network output

    Returns:
        Dictionary with parsed parameters:
        - means: (B, N, 2) absolute positions in image coordinates
        - scales: (B, N, 2) positive scale values
        - rotations: (B, N, 1) rotation angle in radians
        - colors: (B, N, 3) RGB in [0, 1]
        - opacities: (B, N, 1) opacity in [0, 1]
        - importance: (B, N, 1) importance score in [0, 1]
    """
    B, C, H, W = raw_params.shape
    assert C == 10, f"Expected 10 channels, got {C}"
    device = raw_params.device

    # Rearrange to (B, H, W, 10)
    params = raw_params.permute(0, 2, 3, 1)

    # Parse each parameter
    delta_mu = torch.tanh(params[..., 0:2]) * 0.5  # [-0.5, 0.5]
    scale = F.softplus(params[..., 2:4]) + 1.0  # minimum scale of 1.0
    rotation = params[..., 4:5]  # radians, unbounded
    color = torch.sigmoid(params[..., 5:8])  # [0, 1]
    opacity = torch.sigmoid(params[..., 8:9])  # [0, 1]
    importance = torch.sigmoid(params[..., 9:10])  # [0, 1]

    # Create pixel coordinate grid
    coords = create_pixel_coords(H, W, device)  # (H, W, 2)

    # Compute absolute mean positions
    # means = pixel_coord + delta_mu
    means = coords.unsqueeze(0) + delta_mu  # (B, H, W, 2)

    # Flatten spatial dimensions: (B, H, W, C) -> (B, N, C)
    N = H * W

    return {
        'means': means.reshape(B, N, 2),
        'scales': scale.reshape(B, N, 2),
        'rotations': rotation.reshape(B, N, 1),
        'colors': color.reshape(B, N, 3),
        'opacities': opacity.reshape(B, N, 1),
        'importance': importance.reshape(B, N, 1),
    }


def parse_gaussian_params_11ch(raw_params: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Parse raw 11-channel output into Gaussian parameters with depth.

    11 channels = 9 Gaussian params + 1 importance + 1 depth.

    Args:
        raw_params: (B, 11, H, W) raw network output

    Returns:
        Dictionary with parsed parameters:
        - means: (B, N, 2) absolute positions in image coordinates
        - scales: (B, N, 2) positive scale values
        - rotations: (B, N, 1) rotation angle in radians
        - colors: (B, N, 3) RGB in [0, 1]
        - opacities: (B, N, 1) opacity in [0, 1]
        - importance: (B, N, 1) importance score in [0, 1]
        - depths: (B, N, 1) depth values (positive, for front-to-back ordering)
    """
    B, C, H, W = raw_params.shape
    assert C == 11, f"Expected 11 channels, got {C}"
    device = raw_params.device

    # Rearrange to (B, H, W, 11)
    params = raw_params.permute(0, 2, 3, 1)

    # Parse each parameter
    delta_mu = torch.tanh(params[..., 0:2]) * 0.5  # [-0.5, 0.5]
    scale = F.softplus(params[..., 2:4]) + 1.0  # minimum scale of 1.0
    rotation = params[..., 4:5]  # radians, unbounded
    color = torch.sigmoid(params[..., 5:8])  # [0, 1]
    opacity = torch.sigmoid(params[..., 8:9])  # [0, 1]
    importance = torch.sigmoid(params[..., 9:10])  # [0, 1]
    # Depth: softplus ensures positive, +0.1 to avoid z=0
    depth = F.softplus(params[..., 10:11])  # positive depth

    # Create pixel coordinate grid
    coords = create_pixel_coords(H, W, device)  # (H, W, 2)

    # Compute absolute mean positions
    # means = pixel_coord + delta_mu
    means = coords.unsqueeze(0) + delta_mu  # (B, H, W, 2)

    # Flatten spatial dimensions: (B, H, W, C) -> (B, N, C)
    N = H * W

    return {
        'means': means.reshape(B, N, 2),
        'scales': scale.reshape(B, N, 2),
        'rotations': rotation.reshape(B, N, 1),
        'colors': color.reshape(B, N, 3),
        'opacities': opacity.reshape(B, N, 1),
        'importance': importance.reshape(B, N, 1),
        'depths': depth.reshape(B, N, 1),
    }


def apply_topk_importance(
    gaussians: Dict[str, torch.Tensor],
    k: int,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Apply top-k selection based on IMPORTANCE with STE for gradient flow.

    Forward: exact hard top-k mask
    Backward: gradients flow through soft gate (importance-based)

    Args:
        gaussians: Dictionary with Gaussian parameters (B, N, C)
        k: Number of top Gaussians to keep
        temperature: Sharpness of soft gate (higher = sharper gradient)

    Returns:
        Dictionary with masked opacities
    """
    importance = gaussians['importance']  # (B, N, 1)
    opacities = gaussians['opacities']  # (B, N, 1)
    B, N, _ = importance.shape

    k = min(k, N)

    # Hard mask from top-k (non-differentiable)
    importance_flat = importance.squeeze(-1)  # (B, N)
    _, topk_indices = torch.topk(importance_flat, k, dim=1)  # (B, k)
    hard_mask = torch.zeros_like(importance_flat)  # (B, N)
    hard_mask.scatter_(1, topk_indices, 1.0)
    hard_mask = hard_mask.unsqueeze(-1)  # (B, N, 1)

    # Soft gate from importance (differentiable)
    # temperature > 1: sharper (importance^temp approaches 0 or 1)
    # temperature < 1: softer
    soft_gate = importance ** temperature  # (B, N, 1)

    # STE: forward=hard, backward=soft
    # mask = hard + (soft - soft.detach())
    # Forward: hard (exact top-k)
    # Backward: d(mask)/d(importance) = d(soft)/d(importance)
    mask = hard_mask + (soft_gate - soft_gate.detach())

    # Apply mask to opacities
    masked_opacities = opacities * mask

    result = {
        'means': gaussians['means'],
        'scales': gaussians['scales'],
        'rotations': gaussians['rotations'],
        'colors': gaussians['colors'],
        'opacities': masked_opacities,
        'importance': gaussians['importance'],
    }

    # Pass through depths if present
    if 'depths' in gaussians:
        result['depths'] = gaussians['depths']

    return result


def apply_topk_torchsort(
    gaussians: Dict[str, torch.Tensor],
    k: int,
    regularization_strength: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Apply top-k selection using torchsort's differentiable ranking.

    Unlike STE which only passes gradients through importance values,
    torchsort makes the RANKING itself differentiable. This means:
    - Gradients flow through "which elements are in top-k"
    - Elements near the k-th boundary get meaningful gradients

    Args:
        gaussians: Dictionary with Gaussian parameters (B, N, C)
        k: Number of top Gaussians to keep
        regularization_strength: Controls softness of ranking
            - Lower = harder (more like true ranking)
            - Higher = softer (smoother gradients)

    Returns:
        Dictionary with masked opacities
    """
    from torchsort import soft_rank

    importance = gaussians['importance']  # (B, N, 1)
    opacities = gaussians['opacities']  # (B, N, 1)
    B, N, _ = importance.shape

    k = min(k, N)

    # Get soft ranks (lower rank = higher importance)
    # Negate importance so high importance -> low rank
    importance_flat = importance.squeeze(-1)  # (B, N)
    ranks = soft_rank(
        -importance_flat,
        regularization='l2',
        regularization_strength=regularization_strength,
    )  # (B, N), ranks in [1, N]

    # Soft mask: rank <= k -> 1, rank > k -> 0
    # Use sigmoid with sharpness factor for smooth boundary
    sharpness = 10.0 / regularization_strength  # sharper when reg is lower
    soft_mask = torch.sigmoid((k + 0.5 - ranks) * sharpness)  # (B, N)
    soft_mask = soft_mask.unsqueeze(-1)  # (B, N, 1)

    # For forward pass, use hard mask (exact top-k)
    hard_mask = (ranks <= k).float().unsqueeze(-1)  # (B, N, 1)

    # STE: forward=hard, backward=soft
    mask = hard_mask + (soft_mask - soft_mask.detach())

    # Apply mask to opacities
    masked_opacities = opacities * mask

    result = {
        'means': gaussians['means'],
        'scales': gaussians['scales'],
        'rotations': gaussians['rotations'],
        'colors': gaussians['colors'],
        'opacities': masked_opacities,
        'importance': gaussians['importance'],
    }

    # Pass through depths if present
    if 'depths' in gaussians:
        result['depths'] = gaussians['depths']

    return result


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


def make_identity_gaussians(image: torch.Tensor) -> torch.Tensor:
    """
    Create identity Gaussian input from RGB image.

    Instead of feeding raw RGB (3ch) to the network, we create 11-channel
    "identity Gaussians" where each pixel is initialized as a valid Gaussian.

    Identity Gaussian for pixel (x, y) with color RGB:
    - delta_mu = 0 (position = pixel center)
    - scale = 1 (unit scale)
    - rotation = 0 (no rotation)
    - color = RGB (from image)
    - opacity = 1 (fully opaque)
    - importance = 1 (fully important)
    - depth = 1 (unit depth)

    The network learns to REFINE these rather than predict from scratch.
    This is residual learning - output ≈ identity + delta.

    Args:
        image: (B, 3, H, W) RGB image in [0, 1]

    Returns:
        identity_gaussians: (B, 11, H, W) identity Gaussian parameters
            - ch 0-1: delta_mu = 0
            - ch 2-3: scale_raw = 0 (so softplus(0)+1 = 1)
            - ch 4: rotation = 0
            - ch 5-7: color_raw = logit(RGB) (so sigmoid(logit(RGB)) = RGB)
            - ch 8: opacity_raw = 5 (so sigmoid(5) ≈ 0.993)
            - ch 9: importance_raw = 5 (so sigmoid(5) ≈ 0.993)
            - ch 10: depth_raw = 0.54 (so softplus(0.54) ≈ 1)
    """
    B, C, H, W = image.shape
    assert C == 3, f"Expected 3-channel RGB image, got {C} channels"
    device = image.device
    dtype = image.dtype

    # Initialize 11-channel identity
    identity = torch.zeros(B, 11, H, W, device=device, dtype=dtype)

    # Channels 0-1: delta_mu = 0 (already zero)
    # Channels 2-3: scale_raw = 0 (softplus(0) + 1 = 1)
    # Channel 4: rotation = 0 (already zero)

    # Channels 5-7: color_raw = logit(RGB)
    # logit(x) = log(x / (1-x)) = log(x) - log(1-x)
    # Clamp to avoid log(0) or log(inf)
    eps = 1e-6
    rgb_clamped = image.clamp(eps, 1 - eps)
    color_raw = torch.log(rgb_clamped) - torch.log(1 - rgb_clamped)
    identity[:, 5:8] = color_raw

    # Channel 8: opacity_raw = 5 (sigmoid(5) ≈ 0.993)
    identity[:, 8] = 5.0

    # Channel 9: importance_raw = 5 (sigmoid(5) ≈ 0.993)
    identity[:, 9] = 5.0

    # Channel 10: depth_raw ≈ 0.54 (softplus(0.54) ≈ 1)
    # softplus(x) = log(1 + exp(x)), inverse: x = log(exp(y) - 1)
    # For y=1: x = log(exp(1) - 1) ≈ log(1.718) ≈ 0.54
    identity[:, 10] = 0.54

    return identity


def apply_topk_opacity(
    gaussians: Dict[str, torch.Tensor],
    k: int,
) -> Dict[str, torch.Tensor]:
    """
    Apply top-k selection based on opacity with Straight-Through Estimator.

    Only the top-k Gaussians by opacity will contribute to rendering (forward),
    but gradients flow through all Gaussians (backward) via STE.

    Args:
        gaussians: Dictionary with Gaussian parameters (B, N, C)
        k: Number of top Gaussians to keep

    Returns:
        Dictionary with masked opacities (top-k only in forward, all in backward)
    """
    opacities = gaussians['opacities']  # (B, N, 1)
    B, N, _ = opacities.shape

    # Clamp k to valid range
    k = min(k, N)

    # Get top-k indices based on opacity
    opacity_flat = opacities.squeeze(-1)  # (B, N)
    _, topk_indices = torch.topk(opacity_flat, k, dim=1)  # (B, k)

    # Create binary mask for top-k
    mask = torch.zeros_like(opacity_flat)  # (B, N)
    mask.scatter_(1, topk_indices, 1.0)
    mask = mask.unsqueeze(-1)  # (B, N, 1)

    # STE: forward uses masked opacity, backward passes gradient through all
    # output = soft + (hard - soft).detach()
    # Forward: soft + (hard - soft) = hard = opacities * mask
    # Backward: d(soft)/d(params) = d(opacities)/d(params)
    masked_opacities = opacities + (opacities * mask - opacities).detach()

    # Return new gaussians dict with masked opacities
    return {
        'means': gaussians['means'],
        'scales': gaussians['scales'],
        'rotations': gaussians['rotations'],
        'colors': gaussians['colors'],
        'opacities': masked_opacities,
    }
