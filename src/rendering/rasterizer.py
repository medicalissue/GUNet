"""
2D Gaussian Rasterizer using gsplat.

2D Gaussians are rendered as 3D Gaussians with depth-aware ordering.
Supports learned depth for proper occlusion (front Gaussians block back ones).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from gsplat import rasterization


def render_gaussians_2d(
    gaussians: Dict[str, torch.Tensor],
    H: int,
    W: int,
    background: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Render 2D Gaussians using gsplat with depth-aware ordering.

    Args:
        gaussians: Dictionary with:
            - means: (B, N, 2) pixel coordinates
            - scales: (B, N, 2) pixel scales
            - rotations: (B, N, 1) rotation angles (radians)
            - colors: (B, N, 3) RGB [0,1]
            - opacities: (B, N, 1) opacity [0,1]
            - depths: (B, N, 1) optional depth values (if not present, all z=1)
        H, W: Output dimensions
        background: Optional background color

    Returns:
        (B, 3, H, W) rendered image
    """
    means = gaussians['means']
    scales = gaussians['scales']
    rotations = gaussians['rotations']
    colors = gaussians['colors']
    opacities = gaussians['opacities']

    B, N, _ = means.shape
    device = means.device
    dtype = means.dtype

    # 2D -> 3D: use learned depth if available, else z=1
    if 'depths' in gaussians:
        depths = gaussians['depths']  # (B, N, 1)
        means_3d = torch.cat([means, depths], dim=-1)
    else:
        means_3d = torch.cat([means, torch.ones(B, N, 1, device=device, dtype=dtype)], dim=-1)

    scales_3d = torch.cat([scales, torch.full((B, N, 1), 0.001, device=device, dtype=dtype)], dim=-1)

    # Rotation angle -> quaternion (z-axis rotation)
    half = rotations / 2
    quats = torch.cat([
        torch.cos(half),           # w
        torch.zeros_like(half),    # x
        torch.zeros_like(half),    # y
        torch.sin(half),           # z
    ], dim=-1)

    # Camera: identity view, identity intrinsic (orthographic-like)
    viewmat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    K = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous()

    # Render each batch
    images = []
    for b in range(B):
        render_colors, _, _ = rasterization(
            means=means_3d[b],
            quats=quats[b],
            scales=scales_3d[b],
            opacities=opacities[b, :, 0],
            colors=colors[b],
            viewmats=viewmat[b:b+1],
            Ks=K[b:b+1],
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=100.0,
        )
        images.append(render_colors.squeeze(0))

    # (B, H, W, 3) -> (B, 3, H, W)
    image = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    return image.clamp(0, 1)


class GaussianRenderer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gaussians: Dict[str, torch.Tensor], H: int, W: int,
                background: Optional[torch.Tensor] = None) -> torch.Tensor:
        return render_gaussians_2d(gaussians, H, W, background)
