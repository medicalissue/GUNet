"""
Training script for Multi-Scale Gaussian UNet.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --data_dir ./data --image_size 256 --batch_size 8
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MultiScaleGaussianUNet, GaussianSplitNet
from src.models.gaussian_utils import apply_topk_opacity, apply_topk_importance
from src.rendering import render_gaussians_2d


def cast_gaussians_to_float32(gaussians: dict) -> dict:
    """Cast all Gaussian tensors to float32 for rendering (gsplat requires float32)."""
    return {k: v.float() if torch.is_tensor(v) else v for k, v in gaussians.items()}


from src.losses import ReconstructionLoss
from src.data import ImageDataset
from torch.utils.data import DataLoader


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between pred and target images."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_bpp(n_gaussians: int, H: int, W: int, bits_per_param: int = 32) -> float:
    """
    Compute Bits Per Pixel for Gaussian representation.

    Each Gaussian has 9 parameters (delta_mu:2, scale:2, rotation:1, color:3, opacity:1).
    BPP = (N_gaussians * 9 * bits_per_param) / (H * W)
    """
    total_bits = n_gaussians * 9 * bits_per_param
    return total_bits / (H * W)


def compute_effective_bpp(gaussians: dict, H: int, W: int, bits_per_param: int = 32) -> float:
    """
    Compute effective BPP weighted by opacity.

    Only counts Gaussians with opacity > 0.5 as "active".
    """
    opacities = gaussians['opacities']  # (B, N, 1)
    n_effective = (opacities > 0.5).float().sum().item() / opacities.shape[0]
    total_bits = n_effective * 9 * bits_per_param
    return total_bits / (H * W)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Scale Gaussian UNet')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to training images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')

    # Model
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'split'],
                        help='Model architecture: unet (UNet decoder) or split (Gaussian split decoder)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--num_levels', type=int, default=5,
                        help='Number of pyramid levels')
    parser.add_argument('--start_level', type=int, default=0,
                        help='First level to output Gaussians (0=full res, 1=half, etc.) - only for unet')

    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=0.8,
                        help='L1 loss weight')
    parser.add_argument('--ssim_weight', type=float, default=0.2,
                        help='SSIM loss weight')
    parser.add_argument('--lpips_weight', type=float, default=0.0,
                        help='LPIPS perceptual loss weight (0 to disable)')
    parser.add_argument('--count_weight', type=float, default=0.0,
                        help='Gaussian count loss weight (0 to disable)')
    parser.add_argument('--target_count', type=int, default=None,
                        help='Target number of effective Gaussians')
    parser.add_argument('--scale_weight', type=float, default=0.0,
                        help='Scale loss weight - penalizes small scales (0 to disable)')
    parser.add_argument('--sparsity_weight', type=float, default=0.0,
                        help='Sparsity loss weight - encourages binary opacity (0 to disable)')

    # Top-k selection
    parser.add_argument('--use_topk', action='store_true',
                        help='Use top-k importance selection for rendering')
    parser.add_argument('--topk_count', type=int, default=1024,
                        help='Number of top-k Gaussians to render')
    parser.add_argument('--topk_temperature', type=float, default=1.0,
                        help='Soft gate sharpness for STE (>1: sharper, <1: softer)')

    # Rendering
    parser.add_argument('--camera_model', type=str, default='ortho',
                        help='Camera model: "pinhole" (perspective) or "ortho" (orthographic)')

    # Mixed precision
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision (AMP) for faster training')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Visualization
    parser.add_argument('--vis_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--vis_every', type=int, default=100,
                        help='Visualize every N batches')

    # Config file (overrides command line args)
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config YAML file')

    # Misc
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override args with config
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Ensure correct types for numeric values
    args.lr = float(args.lr)
    args.l1_weight = float(args.l1_weight)
    args.ssim_weight = float(args.ssim_weight)
    args.lpips_weight = float(args.lpips_weight)
    args.count_weight = float(args.count_weight)
    args.scale_weight = float(args.scale_weight)
    args.sparsity_weight = float(args.sparsity_weight)
    if args.target_count is not None:
        args.target_count = int(args.target_count)

    # Handle use_topk from config (can be True/False in yaml)
    if hasattr(args, 'use_topk') and args.use_topk is not None:
        args.use_topk = bool(args.use_topk)
    else:
        args.use_topk = False
    args.topk_count = int(args.topk_count)
    if hasattr(args, 'topk_temperature'):
        args.topk_temperature = float(args.topk_temperature)
    else:
        args.topk_temperature = 1.0

    # Handle use_amp from config
    if hasattr(args, 'use_amp') and args.use_amp is not None:
        args.use_amp = bool(args.use_amp)
    else:
        args.use_amp = False

    # Print loaded config
    print(f"=== Config ===")
    print(f"data_dir: {args.data_dir}")
    print(f"image_size: {args.image_size}")
    print(f"batch_size: {args.batch_size}")
    print(f"model_type: {args.model_type}")
    print(f"num_levels: {args.num_levels}")
    if args.model_type == 'unet':
        print(f"start_level: {args.start_level}")
    print(f"use_topk: {args.use_topk}")
    if args.use_topk:
        print(f"topk_count: {args.topk_count}")
        print(f"topk_temperature: {args.topk_temperature}")
    print(f"use_amp: {args.use_amp}")
    print(f"camera_model: {args.camera_model}")
    print(f"vis_every: {args.vis_every}")
    print(f"==============")

    return args


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f'Saved checkpoint to {save_path}')


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    return start_epoch


@torch.no_grad()
def visualize_reconstruction(
    model: nn.Module,
    images: torch.Tensor,
    save_path: str,
    epoch: int,
    batch_idx: int,
):
    """
    Visualize reconstruction results.

    Creates a figure with:
    - Original image
    - Reconstructed image
    - Per-level Gaussian visualizations
    - Difference map
    - Importance maps (for GaussianSplitNet)
    """
    model.eval()

    B, C, H, W = images.shape
    device = images.device

    # Forward pass
    output = model(images)
    gaussians = output['gaussians']
    multi_scale = output['multi_scale_gaussians']

    # Cast to float32 and render full reconstruction
    gaussians = cast_gaussians_to_float32(gaussians)
    rendered = render_gaussians_2d(gaussians, H, W)

    # Compute metrics
    psnr = compute_psnr(rendered, images)
    n_gaussians = gaussians['means'].shape[1]
    bpp = compute_bpp(n_gaussians, H, W)
    bpp_eff = compute_effective_bpp(gaussians, H, W)

    # Take first image in batch
    orig = images[0].cpu().permute(1, 2, 0).numpy()
    recon = rendered[0].cpu().permute(1, 2, 0).numpy()
    diff = np.abs(orig - recon)

    # Create figure - add extra row for encoder/decoder comparison if available
    num_levels = len(multi_scale)
    has_enc_dec = 'encoder_gaussians' in output and output['encoder_gaussians'] is not None
    num_rows = 3 if has_enc_dec else 2
    fig, axes = plt.subplots(num_rows, num_levels + 2, figsize=(4 * (num_levels + 2), 4 * num_rows))

    # Row 1: Original, Reconstructed, Per-level contributions
    axes[0, 0].imshow(orig.clip(0, 1))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(recon.clip(0, 1))
    axes[0, 1].set_title(f'Reconstructed\nPSNR={psnr:.2f}dB\nBPP={bpp:.2f} (eff={bpp_eff:.2f})')
    axes[0, 1].axis('off')

    # Render each level separately
    for i, level_gaussians in enumerate(multi_scale):
        # Flatten this level's gaussians and cast to float32
        level_dict = {
            'means': level_gaussians['means'][:1].reshape(1, -1, 2).float(),
            'scales': level_gaussians['scales'][:1].reshape(1, -1, 2).float(),
            'rotations': level_gaussians['rotations'][:1].reshape(1, -1, 1).float(),
            'colors': level_gaussians['colors'][:1].reshape(1, -1, 3).float(),
            'opacities': level_gaussians['opacities'][:1].reshape(1, -1, 1).float(),
        }
        level_rendered = render_gaussians_2d(level_dict, H, W)
        level_img = level_rendered[0].cpu().permute(1, 2, 0).numpy()

        level_idx = level_gaussians.get('level', i)
        axes[0, i + 2].imshow(level_img.clip(0, 1))
        axes[0, i + 2].set_title(f'Level {level_idx}\n({level_dict["means"].shape[1]} G)')
        axes[0, i + 2].axis('off')

    # Row 2: Difference map, Gaussian positions, Scale distributions
    axes[1, 0].imshow(diff.clip(0, 1))
    axes[1, 0].set_title('Difference (|orig - recon|)')
    axes[1, 0].axis('off')

    # Gaussian positions scatter plot
    axes[1, 1].set_xlim(0, W)
    axes[1, 1].set_ylim(H, 0)  # Flip y-axis
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_title('Gaussian Centers (all levels)')

    colors_list = plt.cm.viridis(np.linspace(0, 1, num_levels))
    for i, level_gaussians in enumerate(multi_scale):
        means = level_gaussians['means'][0].cpu().numpy()
        level_idx = level_gaussians.get('level', i)
        axes[1, 1].scatter(
            means[:, 0], means[:, 1],
            s=1, alpha=0.5,
            c=[colors_list[i]],
            label=f'L{level_idx}'
        )
    axes[1, 1].legend(markerscale=5, loc='upper right')

    # Per-level scale distributions
    for i, level_gaussians in enumerate(multi_scale):
        scales = level_gaussians['scales'][0].cpu().numpy()
        mean_scale = scales.mean()

        axes[1, i + 2].hist(scales.flatten(), bins=50, alpha=0.7)
        axes[1, i + 2].set_title(f'L{level_gaussians.get("level", i)} Scales\nmean={mean_scale:.2f}')
        axes[1, i + 2].set_xlabel('Scale (pixels)')

    # Row 3: Encoder vs Decoder comparison (for symmetric network)
    encoder_gaussians = output.get('encoder_gaussians', None)
    decoder_gaussians = output.get('decoder_gaussians', None)

    if encoder_gaussians and decoder_gaussians:
        # Render encoder-only reconstruction
        from src.models.gaussian_utils import merge_multi_scale_gaussians
        enc_merged = merge_multi_scale_gaussians(encoder_gaussians)
        enc_merged = cast_gaussians_to_float32(enc_merged)
        enc_rendered = render_gaussians_2d(enc_merged, H, W)
        enc_img = enc_rendered[0].cpu().permute(1, 2, 0).numpy()

        axes[2, 0].imshow(enc_img.clip(0, 1))
        axes[2, 0].set_title(f'Encoder Only\n({enc_merged["means"].shape[1]} G)')
        axes[2, 0].axis('off')

        # Render decoder-only reconstruction
        dec_merged = merge_multi_scale_gaussians(decoder_gaussians)
        dec_merged = cast_gaussians_to_float32(dec_merged)
        dec_rendered = render_gaussians_2d(dec_merged, H, W)
        dec_img = dec_rendered[0].cpu().permute(1, 2, 0).numpy()

        axes[2, 1].imshow(dec_img.clip(0, 1))
        axes[2, 1].set_title(f'Decoder Only\n({dec_merged["means"].shape[1]} G)')
        axes[2, 1].axis('off')

        # Opacity distributions for encoder and decoder
        enc_opacities = enc_merged['opacities'][0].cpu().numpy().squeeze()
        dec_opacities = dec_merged['opacities'][0].cpu().numpy().squeeze()

        axes[2, 2].hist(enc_opacities, bins=50, alpha=0.7, color='blue', label='Encoder')
        axes[2, 2].hist(dec_opacities, bins=50, alpha=0.7, color='orange', label='Decoder')
        axes[2, 2].legend()
        axes[2, 2].set_title('Opacity Distributions')
        axes[2, 2].set_xlabel('Opacity')

        # Fill remaining cells with per-level encoder Gaussians
        for i in range(min(len(encoder_gaussians), num_levels - 1)):
            if i + 3 < num_levels + 2:
                level_g = encoder_gaussians[i]
                level_dict = {
                    'means': level_g['means'][:1].reshape(1, -1, 2).float(),
                    'scales': level_g['scales'][:1].reshape(1, -1, 2).float(),
                    'rotations': level_g['rotations'][:1].reshape(1, -1, 1).float(),
                    'colors': level_g['colors'][:1].reshape(1, -1, 3).float(),
                    'opacities': level_g['opacities'][:1].reshape(1, -1, 1).float(),
                }
                level_rendered = render_gaussians_2d(level_dict, H, W)
                level_img = level_rendered[0].cpu().permute(1, 2, 0).numpy()
                axes[2, i + 3].imshow(level_img.clip(0, 1))
                axes[2, i + 3].set_title(f'Enc L{level_g.get("level", i)}')
                axes[2, i + 3].axis('off')

    plt.suptitle(f'Epoch {epoch}, Batch {batch_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    model.train()


@torch.no_grad()
def visualize_gaussians_detail(
    model: nn.Module,
    images: torch.Tensor,
    save_path: str,
):
    """
    Detailed visualization of Gaussian parameters.
    Includes: importance map, scale spatial distribution, opacity, colors, etc.
    """
    model.eval()

    B, C, H, W = images.shape

    output = model(images)
    gaussians = output['gaussians']
    gaussians = cast_gaussians_to_float32(gaussians)

    # Get parameters
    means = gaussians['means'][0].cpu().numpy()
    scales = gaussians['scales'][0].cpu().numpy()
    opacities = gaussians['opacities'][0].cpu().numpy().squeeze()
    colors = gaussians['colors'][0].cpu().numpy()

    # Check if importance is available (GaussianSplitNet)
    has_importance = 'importance' in gaussians and gaussians['importance'] is not None
    if has_importance:
        importance = gaussians['importance'][0].cpu().numpy().squeeze()

    # Check if depth is available
    has_depth = 'depths' in gaussians and gaussians['depths'] is not None
    if has_depth:
        depths = gaussians['depths'][0].cpu().numpy().squeeze()

    # Get rotation for ellipse drawing
    rotations = gaussians['rotations'][0].cpu().numpy().squeeze()

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))

    # Row 0: Original, Rendered, Difference
    orig = images[0].cpu().permute(1, 2, 0).numpy()
    rendered = render_gaussians_2d(gaussians, H, W)
    recon = rendered[0].cpu().permute(1, 2, 0).numpy()
    diff = np.abs(orig - recon)

    axes[0, 0].imshow(orig.clip(0, 1))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(recon.clip(0, 1))
    axes[0, 1].set_title('Rendered')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(diff.clip(0, 1))
    axes[0, 2].set_title('Difference')
    axes[0, 2].axis('off')

    # Row 1: Importance map, Scale map, Opacity map (as spatial heatmaps)
    # Importance map (reshape to HxW)
    if has_importance:
        importance_map = importance.reshape(H, W)
        im = axes[1, 0].imshow(importance_map, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Importance Map\nmean={importance.mean():.3f}, max={importance.max():.3f}')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    else:
        axes[1, 0].text(0.5, 0.5, 'No importance\n(not GaussianSplitNet)',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Importance Map')
        axes[1, 0].axis('off')

    # Scale map (mean of scale_x and scale_y, reshape to HxW)
    mean_scales = scales.mean(axis=1)
    scale_map = mean_scales.reshape(H, W)
    im = axes[1, 1].imshow(scale_map, cmap='viridis')
    axes[1, 1].set_title(f'Scale Map (mean)\nmean={mean_scales.mean():.2f}, max={mean_scales.max():.2f}')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # Depth map (if available) or Opacity map
    if has_depth:
        depth_map = depths.reshape(H, W)
        im = axes[1, 2].imshow(depth_map, cmap='plasma')
        axes[1, 2].set_title(f'Depth Map\nmean={depths.mean():.2f}, range=[{depths.min():.2f}, {depths.max():.2f}]')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    else:
        opacity_map = opacities.reshape(H, W)
        im = axes[1, 2].imshow(opacity_map, cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Opacity Map\nmean={opacities.mean():.3f}')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    # Row 2: Histograms - Scale distribution, Opacity distribution, Importance distribution
    # Scale histogram (2D: scale_x vs scale_y)
    axes[2, 0].hist2d(scales[:, 0], scales[:, 1], bins=50, cmap='Blues')
    axes[2, 0].set_xlabel('Scale X')
    axes[2, 0].set_ylabel('Scale Y')
    axes[2, 0].set_title(f'Scale Distribution\nmean_x={scales[:,0].mean():.2f}, mean_y={scales[:,1].mean():.2f}')
    axes[2, 0].set_aspect('equal')

    # Opacity histogram
    axes[2, 1].hist(opacities, bins=50, alpha=0.7, color='gray')
    axes[2, 1].axvline(x=0.5, color='r', linestyle='--', label='threshold=0.5')
    n_active = (opacities > 0.5).sum()
    axes[2, 1].set_xlabel('Opacity')
    axes[2, 1].set_title(f'Opacity Distribution\n{n_active}/{len(opacities)} active (>{0.5})')
    axes[2, 1].legend()

    # Importance/Depth histogram
    if has_depth:
        # Show depth distribution
        axes[2, 2].hist(depths, bins=50, alpha=0.7, color='purple')
        axes[2, 2].set_xlabel('Depth')
        axes[2, 2].set_title(f'Depth Distribution\nmean={depths.mean():.2f}, std={depths.std():.2f}')
    elif has_importance:
        axes[2, 2].hist(importance, bins=50, alpha=0.7, color='orange')
        axes[2, 2].axvline(x=0.5, color='r', linestyle='--', label='threshold=0.5')
        n_important = (importance > 0.5).sum()
        axes[2, 2].set_xlabel('Importance')
        axes[2, 2].set_title(f'Importance Distribution\n{n_important}/{len(importance)} important (>{0.5})')
        axes[2, 2].legend()
    else:
        # Color distribution (RGB channels)
        for i, (ch, name) in enumerate(zip([0, 1, 2], ['R', 'G', 'B'])):
            axes[2, 2].hist(colors[:, ch], bins=50, alpha=0.5, label=name)
        axes[2, 2].legend()
        axes[2, 2].set_title('Color Distribution')

    # Row 3: Gaussian ellipse boundaries
    # Select top-k Gaussians for visualization (avoid clutter)
    n_ellipses = min(512, len(means))
    if has_importance:
        top_indices = np.argsort(importance)[-n_ellipses:]
    else:
        top_indices = np.argsort(opacities)[-n_ellipses:]

    # Helper function to draw ellipses
    def draw_ellipses(ax, indices, alpha=0.5, color_by='depth'):
        ellipses = []
        ellipse_colors = []
        for idx in indices:
            cx, cy = means[idx]
            sx, sy = scales[idx]
            angle_deg = np.degrees(rotations[idx])
            # Ellipse width/height = 2 * scale (1-sigma boundary)
            ellipse = Ellipse(
                xy=(cx, cy),
                width=2 * sx,
                height=2 * sy,
                angle=angle_deg,
            )
            ellipses.append(ellipse)
            # Color by depth (if available), importance, or opacity
            if color_by == 'depth' and has_depth:
                # Normalize depth for colormap
                d_min, d_max = depths.min(), depths.max()
                ellipse_colors.append((depths[idx] - d_min) / (d_max - d_min + 1e-6))
            elif color_by == 'importance' and has_importance:
                ellipse_colors.append(importance[idx])
            else:
                ellipse_colors.append(opacities[idx])

        # Use plasma for depth, hot for importance/opacity
        cmap = plt.cm.plasma if (color_by == 'depth' and has_depth) else plt.cm.hot
        collection = PatchCollection(
            ellipses,
            facecolor='none',
            edgecolor=cmap(ellipse_colors),
            linewidth=0.5,
            alpha=alpha,
        )
        ax.add_collection(collection)

    # Row 3, Col 0: Rendered + all top-k ellipses
    axes[3, 0].imshow(recon.clip(0, 1))
    draw_ellipses(axes[3, 0], top_indices, alpha=0.7)
    axes[3, 0].set_xlim(0, W)
    axes[3, 0].set_ylim(H, 0)
    axes[3, 0].set_title(f'Rendered + Top-{n_ellipses} Ellipses')
    axes[3, 0].axis('off')

    # Row 3, Col 1: Original + all top-k ellipses
    axes[3, 1].imshow(orig.clip(0, 1))
    draw_ellipses(axes[3, 1], top_indices, alpha=0.7)
    axes[3, 1].set_xlim(0, W)
    axes[3, 1].set_ylim(H, 0)
    axes[3, 1].set_title(f'Original + Top-{n_ellipses} Ellipses')
    axes[3, 1].axis('off')

    # Row 3, Col 2: Zoomed view of center region
    zoom_size = H // 4
    cx, cy = W // 2, H // 2
    x1, x2 = cx - zoom_size, cx + zoom_size
    y1, y2 = cy - zoom_size, cy + zoom_size

    # Filter ellipses in zoom region
    in_region = (means[:, 0] >= x1) & (means[:, 0] <= x2) & \
                (means[:, 1] >= y1) & (means[:, 1] <= y2)
    region_indices = np.where(in_region)[0]
    if has_importance:
        region_indices = region_indices[np.argsort(importance[region_indices])[-min(256, len(region_indices)):]]
    else:
        region_indices = region_indices[np.argsort(opacities[region_indices])[-min(256, len(region_indices)):]]

    axes[3, 2].imshow(recon[y1:y2, x1:x2].clip(0, 1), extent=[x1, x2, y2, y1])
    draw_ellipses(axes[3, 2], region_indices, alpha=0.8)
    axes[3, 2].set_xlim(x1, x2)
    axes[3, 2].set_ylim(y2, y1)
    axes[3, 2].set_title(f'Zoomed Center ({len(region_indices)} ellipses)')
    axes[3, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    model.train()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    vis_dir: Path,
    vis_every: int = 100,
    use_topk: bool = False,
    topk_count: int = 1024,
    topk_temperature: float = 1.0,
    use_amp: bool = False,
    scaler: GradScaler = None,
    camera_model: str = "ortho",
) -> dict:
    """Train for one epoch with visualization."""
    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_count = 0.0
    total_n_eff = 0.0
    total_avg_scale = 0.0
    total_psnr = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        B, C, H, W = images.shape

        # Forward pass
        optimizer.zero_grad()

        # Use autocast for mixed precision (only for model forward)
        with autocast('cuda', enabled=use_amp):
            output = model(images)
            gaussians = output['gaussians']

            # Apply top-k selection if enabled
            if use_topk:
                # Use importance-based selection if available (GaussianSplitNet)
                if 'importance' in gaussians:
                    gaussians = apply_topk_importance(gaussians, topk_count, topk_temperature)
                else:
                    gaussians = apply_topk_opacity(gaussians, topk_count)

        # Render Gaussians OUTSIDE autocast (gsplat requires float32)
        gaussians_f32 = cast_gaussians_to_float32(gaussians)
        rendered = render_gaussians_2d(gaussians_f32, H, W, camera_model=camera_model)

        # Compute loss (pass gaussians for count loss)
        losses = criterion(rendered, images, gaussians_f32)
        loss = losses['total']

        # Compute PSNR
        with torch.no_grad():
            psnr = compute_psnr(rendered, images)

        # Backward pass with scaler for mixed precision
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_l1 += losses['l1'].item()
        total_ssim += losses['ssim'].item()
        total_count += losses['count'].item()
        total_n_eff += losses['n_effective'].item()
        total_avg_scale += losses['avg_scale'].item()
        total_psnr += psnr
        num_batches += 1

        # Update progress bar
        postfix = {
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr:.2f}',
            'l1': f'{losses["l1"].item():.4f}',
            'ssim': f'{losses["ssim"].item():.4f}',
        }
        if criterion.lpips_weight > 0:
            postfix['lpips'] = f'{losses["lpips"].item():.4f}'
        if criterion.count_weight > 0:
            postfix['n_eff'] = f'{losses["n_effective"].item():.0f}'
        if criterion.scale_weight > 0:
            postfix['avg_s'] = f'{losses["avg_scale"].item():.2f}'
        pbar.set_postfix(postfix)

        # Visualization
        if batch_idx % vis_every == 0:
            vis_path = vis_dir / f'epoch{epoch:04d}_batch{batch_idx:06d}.png'
            visualize_reconstruction(model, images, str(vis_path), epoch, batch_idx)

            # Detailed visualization every 10x vis_every
            if batch_idx % (vis_every) == 0:
                detail_path = vis_dir / f'detail_epoch{epoch:04d}_batch{batch_idx:06d}.png'
                visualize_gaussians_detail(model, images, str(detail_path))

    return {
        'loss': total_loss / num_batches,
        'l1': total_l1 / num_batches,
        'ssim': total_ssim / num_batches,
        'count': total_count / num_batches,
        'n_effective': total_n_eff / num_batches,
        'avg_scale': total_avg_scale / num_batches,
        'psnr': total_psnr / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    camera_model: str = "ortho",
) -> dict:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        B, C, H, W = images.shape

        output = model(images)
        gaussians = output['gaussians']
        gaussians = cast_gaussians_to_float32(gaussians)
        rendered = render_gaussians_2d(gaussians, H, W, camera_model=camera_model)

        losses = criterion(rendered, images)

        total_loss += losses['total'].item()
        total_l1 += losses['l1'].item()
        total_ssim += losses['ssim'].item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'l1': total_l1 / num_batches,
        'ssim': total_ssim / num_batches,
    }


def plot_training_curves(history: dict, save_path: str):
    """Plot training loss curves."""
    # Check which optional metrics are being tracked
    has_n_eff = 'n_effective' in history and any(v > 0 for v in history['n_effective'])
    has_avg_scale = 'avg_scale' in history and any(v > 0 for v in history['avg_scale'])
    has_psnr = 'psnr' in history and len(history['psnr']) > 0
    n_plots = 3 + int(has_n_eff) + int(has_avg_scale) + int(has_psnr)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    epochs = range(len(history['loss']))

    axes[0].plot(epochs, history['loss'], 'b-', label='Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True)

    axes[1].plot(epochs, history['l1'], 'g-', label='L1 Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('L1 Loss')
    axes[1].set_title('L1 Loss')
    axes[1].grid(True)

    axes[2].plot(epochs, history['ssim'], 'r-', label='SSIM Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM Loss')
    axes[2].set_title('SSIM Loss (1 - SSIM)')
    axes[2].grid(True)

    plot_idx = 3
    if has_psnr:
        axes[plot_idx].plot(epochs, history['psnr'], 'orange', label='PSNR')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('PSNR (dB)')
        axes[plot_idx].set_title('PSNR')
        axes[plot_idx].grid(True)
        plot_idx += 1

    if has_n_eff:
        axes[plot_idx].plot(epochs, history['n_effective'], 'm-', label='N Effective')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Count')
        axes[plot_idx].set_title('Effective Gaussians')
        axes[plot_idx].grid(True)
        plot_idx += 1

    if has_avg_scale:
        axes[plot_idx].plot(epochs, history['avg_scale'], 'c-', label='Avg Scale')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Scale')
        axes[plot_idx].set_title('Average Scale')
        axes[plot_idx].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Dataset and DataLoader
    dataset = ImageDataset(args.data_dir, image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    if args.model_type == 'split':
        model = GaussianSplitNet(
            in_channels=3,
            base_channels=args.base_channels,
            num_levels=args.num_levels,
        ).to(device)
    else:  # 'unet'
        model = MultiScaleGaussianUNet(
            in_channels=3,
            base_channels=args.base_channels,
            num_levels=args.num_levels,
            start_level=args.start_level,
        ).to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {num_params:,}')

    print(f'\n=== Gaussian Distribution ===')
    print(model.get_gaussian_info(args.image_size, args.image_size))
    print(f'==============================\n')

    # Loss
    criterion = ReconstructionLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        lpips_weight=args.lpips_weight,
        count_weight=args.count_weight,
        target_count=args.target_count,
        scale_weight=args.scale_weight,
        sparsity_weight=args.sparsity_weight,
    )
    if args.lpips_weight > 0:
        print(f'LPIPS loss enabled: weight={args.lpips_weight}')
    if args.count_weight > 0:
        print(f'Count loss enabled: weight={args.count_weight}, target={args.target_count}')
    if args.scale_weight > 0:
        print(f'Scale loss enabled: weight={args.scale_weight}')
    if args.sparsity_weight > 0:
        print(f'Sparsity loss enabled: weight={args.sparsity_weight}')
    if args.use_topk:
        print(f'Top-k selection enabled: k={args.topk_count}')
    if args.use_amp:
        print(f'Mixed precision (AMP) enabled')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision scaler
    scaler = GradScaler('cuda', enabled=args.use_amp)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)

    # Training history
    history = {'loss': [], 'l1': [], 'ssim': [], 'n_effective': [], 'avg_scale': [], 'psnr': []}

    # Training loop
    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_one_epoch(
            model, dataloader, criterion, optimizer, device, epoch,
            vis_dir=vis_dir, vis_every=args.vis_every,
            use_topk=args.use_topk, topk_count=args.topk_count,
            topk_temperature=args.topk_temperature,
            use_amp=args.use_amp, scaler=scaler,
            camera_model=args.camera_model,
        )

        # Update scheduler
        scheduler.step()

        # Record history
        history['loss'].append(train_metrics['loss'])
        history['l1'].append(train_metrics['l1'])
        history['ssim'].append(train_metrics['ssim'])
        history['n_effective'].append(train_metrics['n_effective'])
        history['avg_scale'].append(train_metrics['avg_scale'])
        history['psnr'].append(train_metrics['psnr'])

        # Print metrics
        msg = (f'Epoch {epoch}: loss={train_metrics["loss"]:.4f}, '
               f'psnr={train_metrics["psnr"]:.2f}dB, '
               f'l1={train_metrics["l1"]:.4f}, ssim={train_metrics["ssim"]:.4f}')
        if args.count_weight > 0:
            msg += f', n_eff={train_metrics["n_effective"]:.0f}'
        if args.scale_weight > 0:
            msg += f', avg_s={train_metrics["avg_scale"]:.2f}'
        print(msg)

        # Plot training curves
        plot_training_curves(history, str(vis_dir / 'training_curves.png'))

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = save_dir / f'checkpoint_epoch{epoch:04d}.pt'
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], save_path)

        # Save best model
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            save_path = save_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], save_path)

    # Save final model
    save_path = save_dir / 'final_model.pt'
    save_checkpoint(model, optimizer, args.epochs - 1, train_metrics['loss'], save_path)
    print('Training complete!')


if __name__ == '__main__':
    main()
