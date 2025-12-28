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

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


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
    Combined reconstruction loss with L1, SSIM, LPIPS, and Gaussian count components.

    Loss = λ_l1 * L1 + λ_ssim * (1 - SSIM) + λ_lpips * LPIPS + λ_count * CountLoss

    Count loss penalizes the effective number of Gaussians (sum of opacities).
    If target_count is set, only penalizes when N_effective > target_count.

    Args:
        l1_weight: Weight for L1 loss (default: 0.8)
        ssim_weight: Weight for SSIM loss (default: 0.2)
        lpips_weight: Weight for LPIPS loss (default: 0.0, disabled)
        count_weight: Weight for count loss (default: 0.0, disabled)
        target_count: Target number of effective Gaussians (default: None, penalize all)
        use_pytorch_msssim: Use pytorch_msssim library if available
    """

    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        lpips_weight: float = 0.0,
        count_weight: float = 0.0,
        target_count: Optional[int] = None,
        use_pytorch_msssim: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.count_weight = count_weight
        self.target_count = target_count
        self.use_pytorch_msssim = use_pytorch_msssim and PYTORCH_MSSSIM_AVAILABLE

        if use_pytorch_msssim and not PYTORCH_MSSSIM_AVAILABLE:
            print("Warning: pytorch_msssim not available, using pure PyTorch SSIM")

        # Initialize LPIPS if enabled
        self.lpips_fn = None
        if lpips_weight > 0:
            if LPIPS_AVAILABLE:
                self.lpips_fn = lpips.LPIPS(net='vgg').eval()
                # Freeze LPIPS weights
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            else:
                print("Warning: lpips not available, install with 'pip install lpips'")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gaussians: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]
            gaussians: Optional dict with 'opacities' tensor for count loss

        Returns:
            Dictionary with:
            - 'total': Combined loss
            - 'l1': L1 loss component
            - 'ssim': SSIM loss component
            - 'lpips': LPIPS loss component (if enabled)
            - 'count': Count loss component (if enabled)
            - 'n_effective': Effective number of Gaussians
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

        # LPIPS loss (if enabled)
        lpips_loss = torch.tensor(0.0, device=pred.device)
        if self.lpips_weight > 0 and self.lpips_fn is not None:
            # LPIPS expects [-1, 1] range
            pred_lpips = pred * 2 - 1
            target_lpips = target * 2 - 1
            # Move LPIPS model to same device if needed
            if next(self.lpips_fn.parameters()).device != pred.device:
                self.lpips_fn = self.lpips_fn.to(pred.device)
            lpips_loss = self.lpips_fn(pred_lpips, target_lpips).mean()
            total_loss = total_loss + self.lpips_weight * lpips_loss

        # Count loss (if enabled)
        count_loss = torch.tensor(0.0, device=pred.device)
        n_effective = torch.tensor(0.0, device=pred.device)

        if self.count_weight > 0 and gaussians is not None:
            opacities = gaussians['opacities']  # (B, N, 1)
            n_effective = opacities.sum() / opacities.shape[0]  # per-image average

            if self.target_count is not None:
                # Penalize only when exceeding target
                count_loss = F.relu(n_effective - self.target_count)
            else:
                # Penalize total count (normalized by total slots)
                n_total = opacities.shape[1]
                count_loss = n_effective / n_total

            total_loss = total_loss + self.count_weight * count_loss

        return {
            'total': total_loss,
            'l1': l1_loss,
            'ssim': ssim_loss,
            'lpips': lpips_loss,
            'count': count_loss,
            'n_effective': n_effective,
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
