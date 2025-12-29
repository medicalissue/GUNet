"""
Reconstruction losses for 2D Gaussian Splatting.

Includes:
- L1 Loss: Pixel-wise absolute difference
- SSIM Loss: Structural similarity
- Importance Loss: Guide importance maps to match image complexity
- Combined weighted loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

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


def compute_edge_map(image: torch.Tensor) -> torch.Tensor:
    """
    Compute edge magnitude map from image using Sobel filters.

    High values indicate complex regions (edges, textures).
    Low values indicate simple regions (flat areas).

    Args:
        image: (B, 3, H, W) input image in [0, 1]

    Returns:
        edge_map: (B, 1, H, W) edge magnitude in [0, 1]
    """
    B, C, H, W = image.shape
    device = image.device

    # Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # Compute gradients
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    # Edge magnitude
    edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # Normalize to [0, 1] per-image
    edge_min = edge_mag.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    edge_max = edge_mag.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    edge_map = (edge_mag - edge_min) / (edge_max - edge_min + 1e-8)

    return edge_map


def compute_importance_loss(
    importance_maps: List[torch.Tensor],
    target_image: torch.Tensor,
) -> torch.Tensor:
    """
    Compute importance guidance loss.

    Encourages importance maps to be high where image has edges/complexity.

    Args:
        importance_maps: List of (B, 1, H_l, W_l) importance maps at each level
        target_image: (B, 3, H, W) target image

    Returns:
        importance_loss: Scalar loss
    """
    B, C, H, W = target_image.shape
    device = target_image.device

    # Compute edge map at full resolution
    edge_map = compute_edge_map(target_image)  # (B, 1, H, W)

    total_loss = 0.0
    for imp in importance_maps:
        # Downsample edge map to match importance map resolution
        imp_h, imp_w = imp.shape[2:]
        edge_downsampled = F.interpolate(edge_map, size=(imp_h, imp_w),
                                         mode='bilinear', align_corners=True)

        # MSE loss between importance and edge map
        # This encourages: high edges → high importance
        total_loss = total_loss + F.mse_loss(imp, edge_downsampled)

    return total_loss / len(importance_maps)


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
    Combined reconstruction loss with L1, SSIM, LPIPS, count, scale, and sparsity components.

    Loss = λ_l1 * L1 + λ_ssim * (1 - SSIM) + λ_lpips * LPIPS + λ_count * CountLoss
           + λ_scale * ScaleLoss + λ_sparsity * SparsityLoss

    Count loss penalizes the effective number of Gaussians (sum of opacities).
    Scale loss penalizes small Gaussians - smaller scale = larger penalty.
    Sparsity loss encourages binary opacity (0 or 1) to prevent "cheating".

    Args:
        l1_weight: Weight for L1 loss (default: 0.8)
        ssim_weight: Weight for SSIM loss (default: 0.2)
        lpips_weight: Weight for LPIPS loss (default: 0.0, disabled)
        count_weight: Weight for count loss (default: 0.0, disabled)
        target_count: Target number of effective Gaussians (default: None, penalize all)
        scale_weight: Weight for scale loss (default: 0.0, disabled)
        sparsity_weight: Weight for sparsity loss (default: 0.0, disabled)
        use_pytorch_msssim: Use pytorch_msssim library if available
    """

    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        lpips_weight: float = 0.0,
        count_weight: float = 0.0,
        target_count: Optional[int] = None,
        scale_weight: float = 0.0,
        sparsity_weight: float = 0.0,
        use_pytorch_msssim: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.count_weight = count_weight
        self.target_count = target_count
        self.scale_weight = scale_weight
        self.sparsity_weight = sparsity_weight
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

        # Scale loss (if enabled) - inverse penalty for small scales
        scale_loss = torch.tensor(0.0, device=pred.device)
        avg_scale = torch.tensor(0.0, device=pred.device)

        if self.scale_weight > 0 and gaussians is not None:
            scales = gaussians['scales']  # (B, N, 2)
            # Use mean of x,y scales per Gaussian
            scale_mean = scales.mean(dim=-1)  # (B, N)

            # Inverse penalty: smaller scale = larger penalty
            # 1 / (scale + 1) gives:
            #   - scale=0: penalty = 1.0
            #   - scale=1: penalty = 0.5
            #   - scale=3: penalty = 0.25
            #   - scale=9: penalty = 0.1
            #   - scale→∞: penalty → 0
            scale_loss = (1.0 / (scale_mean + 1.0)).mean()
            avg_scale = scale_mean.mean()

            total_loss = total_loss + self.scale_weight * scale_loss

        # Sparsity loss (if enabled) - encourage binary opacity (0 or 1)
        sparsity_loss = torch.tensor(0.0, device=pred.device)

        if self.sparsity_weight > 0 and gaussians is not None:
            opacities = gaussians['opacities']  # (B, N, 1)
            # min(opacity, 1-opacity) is maximized at 0.5, minimized at 0 or 1
            # This encourages opacity to be either 0 or 1, not in between
            sparsity_loss = torch.min(opacities, 1 - opacities).mean()
            total_loss = total_loss + self.sparsity_weight * sparsity_loss

        return {
            'total': total_loss,
            'l1': l1_loss,
            'ssim': ssim_loss,
            'lpips': lpips_loss,
            'count': count_loss,
            'scale': scale_loss,
            'sparsity': sparsity_loss,
            'n_effective': n_effective,
            'avg_scale': avg_scale,
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
