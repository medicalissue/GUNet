"""
Reconstruction losses for 2D Gaussian Splatting.

Includes:
- L1 Loss: Pixel-wise absolute difference
- SSIM Loss: Structural similarity
- Combined weighted loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

try:
    from pytorch_msssim import ssim, ms_ssim
    PYTORCH_MSSSIM_AVAILABLE = True
except ImportError:
    PYTORCH_MSSSIM_AVAILABLE = False


def gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a Gaussian window for SSIM computation."""
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)


def ssim_loss_pytorch(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Compute SSIM loss (1 - SSIM) using pure PyTorch.

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian window

    Returns:
        SSIM loss (scalar)
    """
    B, C, H, W = pred.shape
    device = pred.device

    # Create Gaussian window
    window = gaussian_window(window_size, sigma, device)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)
    window = window.expand(C, 1, -1, -1)  # (C, 1, window_size, window_size)

    # Padding
    pad = window_size // 2

    # Compute means
    mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
    mu_target = F.conv2d(target, window, padding=pad, groups=C)

    # Compute variances and covariance
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=C) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_pred_target

    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula
    ssim_num = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    ssim_den = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    ssim_map = ssim_num / ssim_den

    # Average SSIM
    ssim_val = ssim_map.mean()

    # Return loss (1 - SSIM)
    return 1 - ssim_val


class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss with L1 and SSIM components.

    Loss = λ_l1 * L1 + λ_ssim * (1 - SSIM)

    Args:
        l1_weight: Weight for L1 loss (default: 0.8)
        ssim_weight: Weight for SSIM loss (default: 0.2)
        use_pytorch_msssim: Use pytorch_msssim library if available
    """

    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        use_pytorch_msssim: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.use_pytorch_msssim = use_pytorch_msssim and PYTORCH_MSSSIM_AVAILABLE

        if use_pytorch_msssim and not PYTORCH_MSSSIM_AVAILABLE:
            print("Warning: pytorch_msssim not available, using pure PyTorch SSIM")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]

        Returns:
            Dictionary with:
            - 'total': Combined loss
            - 'l1': L1 loss component
            - 'ssim': SSIM loss component
        """
        # L1 loss
        l1_loss = F.l1_loss(pred, target)

        # SSIM loss
        if self.use_pytorch_msssim:
            ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
            ssim_loss = 1 - ssim_val
        else:
            ssim_loss = ssim_loss_pytorch(pred, target)

        # Combined loss
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss

        return {
            'total': total_loss,
            'l1': l1_loss,
            'ssim': ssim_loss,
        }


class PerceptualLoss(nn.Module):
    """
    Optional perceptual loss using VGG features.

    Computes L1 distance in VGG feature space.
    """

    def __init__(self, layers: tuple = ('relu1_2', 'relu2_2', 'relu3_3')):
        super().__init__()
        self.layers = layers

        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features

        # Layer indices for VGG16
        layer_indices = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
        }

        # Build feature extractor
        max_idx = max(layer_indices[l] for l in layers)
        self.features = nn.Sequential(*list(vgg.children())[:max_idx + 1])
        self.layer_indices = [layer_indices[l] for l in layers]

        # Freeze weights
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]

        Returns:
            Perceptual loss (scalar)
        """
        # Normalize
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        # Extract features
        loss = 0.0
        x_pred = pred_norm
        x_target = target_norm

        for i, layer in enumerate(self.features):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if i in self.layer_indices:
                loss = loss + F.l1_loss(x_pred, x_target)

        return loss / len(self.layer_indices)
