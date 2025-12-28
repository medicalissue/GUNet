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
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MultiScaleGaussianUNet
from src.rendering import render_gaussians_2d
from src.losses import ReconstructionLoss
from src.data import ImageDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Scale Gaussian UNet')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to training images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')

    # Model
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--num_levels', type=int, default=5,
                        help='Number of pyramid levels')
    parser.add_argument('--start_level', type=int, default=0,
                        help='First level to output Gaussians (0=full res, 1=half, etc.)')

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
    if args.target_count is not None:
        args.target_count = int(args.target_count)

    # Print loaded config
    print(f"=== Config ===")
    print(f"data_dir: {args.data_dir}")
    print(f"image_size: {args.image_size}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_levels: {args.num_levels}")
    print(f"start_level: {args.start_level}")
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
    """
    model.eval()

    B, C, H, W = images.shape
    device = images.device

    # Forward pass
    output = model(images)
    gaussians = output['gaussians']
    multi_scale = output['multi_scale_gaussians']

    # Render full reconstruction
    rendered = render_gaussians_2d(gaussians, H, W)

    # Take first image in batch
    orig = images[0].cpu().permute(1, 2, 0).numpy()
    recon = rendered[0].cpu().permute(1, 2, 0).numpy()
    diff = np.abs(orig - recon)

    # Create figure
    num_levels = len(multi_scale)
    fig, axes = plt.subplots(2, num_levels + 2, figsize=(4 * (num_levels + 2), 8))

    # Row 1: Original, Reconstructed, Difference, Per-level contributions
    axes[0, 0].imshow(orig.clip(0, 1))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(recon.clip(0, 1))
    axes[0, 1].set_title(f'Reconstructed\nL1={np.mean(diff):.4f}')
    axes[0, 1].axis('off')

    # Render each level separately
    for i, level_gaussians in enumerate(multi_scale):
        # Flatten this level's gaussians
        level_dict = {
            'means': level_gaussians['means'][:1].reshape(1, -1, 2),
            'scales': level_gaussians['scales'][:1].reshape(1, -1, 2),
            'rotations': level_gaussians['rotations'][:1].reshape(1, -1, 1),
            'colors': level_gaussians['colors'][:1].reshape(1, -1, 3),
            'opacities': level_gaussians['opacities'][:1].reshape(1, -1, 1),
        }
        level_rendered = render_gaussians_2d(level_dict, H, W)
        level_img = level_rendered[0].cpu().permute(1, 2, 0).numpy()

        level_idx = level_gaussians.get('level', i)
        axes[0, i + 2].imshow(level_img.clip(0, 1))
        axes[0, i + 2].set_title(f'Level {level_idx}\n({level_dict["means"].shape[1]} G)')
        axes[0, i + 2].axis('off')

    # Row 2: Difference map, Gaussian positions visualization
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
    """
    model.eval()

    B, C, H, W = images.shape

    output = model(images)
    gaussians = output['gaussians']

    # Get parameters
    means = gaussians['means'][0].cpu().numpy()
    scales = gaussians['scales'][0].cpu().numpy()
    opacities = gaussians['opacities'][0].cpu().numpy().squeeze()
    colors = gaussians['colors'][0].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Mean positions colored by opacity
    sc = axes[0, 0].scatter(means[:, 0], means[:, 1], c=opacities, s=1, cmap='hot', alpha=0.5)
    axes[0, 0].set_xlim(0, W)
    axes[0, 0].set_ylim(H, 0)
    axes[0, 0].set_title('Gaussian Positions (color=opacity)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0, 0])

    # Scale distribution
    axes[0, 1].hist2d(scales[:, 0], scales[:, 1], bins=50, cmap='Blues')
    axes[0, 1].set_xlabel('Scale X')
    axes[0, 1].set_ylabel('Scale Y')
    axes[0, 1].set_title('Scale Distribution')

    # Opacity distribution
    axes[0, 2].hist(opacities, bins=50, alpha=0.7)
    axes[0, 2].set_xlabel('Opacity')
    axes[0, 2].set_title(f'Opacity Distribution\nmean={opacities.mean():.3f}')

    # Color distribution (RGB channels)
    for i, (c, name) in enumerate(zip([0, 1, 2], ['R', 'G', 'B'])):
        axes[1, 0].hist(colors[:, c], bins=50, alpha=0.5, label=name)
    axes[1, 0].legend()
    axes[1, 0].set_title('Color Distribution')

    # Mean scale vs position
    mean_scales = scales.mean(axis=1)
    sc = axes[1, 1].scatter(means[:, 0], means[:, 1], c=mean_scales, s=1, cmap='viridis', alpha=0.5)
    axes[1, 1].set_xlim(0, W)
    axes[1, 1].set_ylim(H, 0)
    axes[1, 1].set_title('Positions (color=scale)')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1, 1])

    # Rendered vs original
    rendered = render_gaussians_2d(gaussians, H, W)
    axes[1, 2].imshow(rendered[0].cpu().permute(1, 2, 0).numpy().clip(0, 1))
    axes[1, 2].set_title('Rendered')
    axes[1, 2].axis('off')

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
) -> dict:
    """Train for one epoch with visualization."""
    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_count = 0.0
    total_n_eff = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        B, C, H, W = images.shape

        # Forward pass
        optimizer.zero_grad()

        output = model(images)
        gaussians = output['gaussians']

        # Render Gaussians
        rendered = render_gaussians_2d(gaussians, H, W)

        # Compute loss (pass gaussians for count loss)
        losses = criterion(rendered, images, gaussians)
        loss = losses['total']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_l1 += losses['l1'].item()
        total_ssim += losses['ssim'].item()
        total_count += losses['count'].item()
        total_n_eff += losses['n_effective'].item()
        num_batches += 1

        # Update progress bar
        postfix = {
            'loss': f'{loss.item():.4f}',
            'l1': f'{losses["l1"].item():.4f}',
            'ssim': f'{losses["ssim"].item():.4f}',
        }
        if criterion.lpips_weight > 0:
            postfix['lpips'] = f'{losses["lpips"].item():.4f}'
        if criterion.count_weight > 0:
            postfix['n_eff'] = f'{losses["n_effective"].item():.0f}'
        pbar.set_postfix(postfix)

        # Visualization
        if batch_idx % vis_every == 0:
            vis_path = vis_dir / f'epoch{epoch:04d}_batch{batch_idx:06d}.png'
            visualize_reconstruction(model, images, str(vis_path), epoch, batch_idx)

            # Detailed visualization every 10x vis_every
            if batch_idx % (vis_every * 10) == 0:
                detail_path = vis_dir / f'detail_epoch{epoch:04d}_batch{batch_idx:06d}.png'
                visualize_gaussians_detail(model, images, str(detail_path))

    return {
        'loss': total_loss / num_batches,
        'l1': total_l1 / num_batches,
        'ssim': total_ssim / num_batches,
        'count': total_count / num_batches,
        'n_effective': total_n_eff / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
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
        rendered = render_gaussians_2d(gaussians, H, W)

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
    # Check if n_effective tracking is enabled
    has_n_eff = 'n_effective' in history and any(v > 0 for v in history['n_effective'])
    n_plots = 4 if has_n_eff else 3

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

    if has_n_eff:
        axes[3].plot(epochs, history['n_effective'], 'm-', label='N Effective')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Count')
        axes[3].set_title('Effective Gaussians')
        axes[3].grid(True)

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
    )
    if args.lpips_weight > 0:
        print(f'LPIPS loss enabled: weight={args.lpips_weight}')
    if args.count_weight > 0:
        print(f'Count loss enabled: weight={args.count_weight}, target={args.target_count}')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)

    # Training history
    history = {'loss': [], 'l1': [], 'ssim': [], 'n_effective': []}

    # Training loop
    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_one_epoch(
            model, dataloader, criterion, optimizer, device, epoch,
            vis_dir=vis_dir, vis_every=args.vis_every
        )

        # Update scheduler
        scheduler.step()

        # Record history
        history['loss'].append(train_metrics['loss'])
        history['l1'].append(train_metrics['l1'])
        history['ssim'].append(train_metrics['ssim'])
        history['n_effective'].append(train_metrics['n_effective'])

        # Print metrics
        msg = (f'Epoch {epoch}: loss={train_metrics["loss"]:.4f}, '
               f'l1={train_metrics["l1"]:.4f}, ssim={train_metrics["ssim"]:.4f}')
        if args.count_weight > 0:
            msg += f', n_eff={train_metrics["n_effective"]:.0f}'
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
