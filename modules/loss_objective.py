import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from dreamsim import dreamsim
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

class LossObjective(ABC):
    """
    Abstract base class for loss objectives used in camera pose optimization.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.setup_models()
    
    @abstractmethod
    def setup_models(self):
        """Setup any models/networks needed for the loss computation."""
        pass
    
    @abstractmethod
    def compute_loss(self, rendered_image: torch.Tensor, target_image: torch.Tensor, 
                    rendered_mask: torch.Tensor, target_mask: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        """
        Compute the loss between rendered and target images.
        
        Args:
            rendered_image: [C, H, W] - Rendered image
            target_image: [C, H, W] - Target image
            rendered_mask: [H, W] - Rendered mask
            target_mask: [H, W] - Target mask
            
        Returns:
            Loss value as a scalar tensor
        """
        pass
    
    def get_name(self) -> str:
        """Return the name of this loss objective."""
        return self.__class__.__name__
    
    def get_description(self) -> str:
        """Return a description of this loss objective."""
        return "Base loss objective"

class DreamSimBoundaryLoss(LossObjective):
    """
    Original loss objective combining DreamSim perceptual loss with boundary consistency.
    """
    
    def setup_models(self):
        self.dreamsim_model, _ = dreamsim(pretrained=True, device=self.device)
        self.boundary_weight = 0.1
        
    def compute_loss(self, rendered_image: torch.Tensor, target_image: torch.Tensor, 
                    rendered_mask: torch.Tensor, target_mask: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        
        # Apply mask to both images
        masked_rendered = rendered_image * rendered_mask
        masked_target = target_image * target_mask
        
        # Resize for DreamSim (expects [N, C, H, W])
        masked_rendered_resized = F.interpolate(
            masked_rendered.unsqueeze(0), (224, 224), mode='bicubic'
        )
        masked_target_resized = F.interpolate(
            masked_target.unsqueeze(0), (224, 224), mode='bicubic'
        )
        
        # DreamSim loss
        dreamsim_loss = self.dreamsim_model(masked_rendered_resized, masked_target_resized)
        
        # Boundary consistency loss
        boundary_loss = F.mse_loss(rendered_mask, target_mask, reduction='none').mean()
        boundary_loss = boundary_loss * (target_mask.numel() / target_mask.sum()) * self.boundary_weight
        
        return dreamsim_loss + boundary_loss
    
    def get_description(self) -> str:
        return "DreamSim perceptual loss + boundary consistency (original)"

class PerceptualLoss(LossObjective):
    """
    LPIPS perceptual loss with boundary consistency.
    """
    
    def setup_models(self):
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        self.boundary_weight = 0.1
        
    def compute_loss(self, rendered_image: torch.Tensor, target_image: torch.Tensor, 
                    rendered_mask: torch.Tensor, target_mask: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        
        # Apply mask to both images
        masked_rendered = rendered_image * rendered_mask
        masked_target = target_image * target_mask
        
        # LPIPS expects [N, C, H, W] in range [-1, 1]
        lpips_loss = self.lpips_model(
            masked_rendered.unsqueeze(0) * 2 - 1,
            masked_target.unsqueeze(0) * 2 - 1
        )
        
        # Boundary consistency loss
        boundary_loss = F.mse_loss(rendered_mask, target_mask, reduction='none').mean()
        boundary_loss = boundary_loss * (target_mask.numel() / target_mask.sum()) * self.boundary_weight
        
        return lpips_loss + boundary_loss
    
    def get_description(self) -> str:
        return "LPIPS perceptual loss + boundary consistency"

class L1L2Loss(LossObjective):
    """
    Combined L1 and L2 loss with boundary consistency.
    """
    
    def setup_models(self):
        self.l1_weight = 0.5
        self.l2_weight = 0.5
        self.boundary_weight = 0.1
        
    def compute_loss(self, rendered_image: torch.Tensor, target_image: torch.Tensor, 
                    rendered_mask: torch.Tensor, target_mask: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        
        # Apply mask to both images
        masked_rendered = rendered_image * rendered_mask
        masked_target = target_image * target_mask
        
        # L1 loss
        l1_loss = F.l1_loss(masked_rendered, masked_target)
        
        # L2 loss
        l2_loss = F.mse_loss(masked_rendered, masked_target)
        
        # Boundary consistency loss
        boundary_loss = F.mse_loss(rendered_mask, target_mask, reduction='none').mean()
        boundary_loss = boundary_loss * (target_mask.numel() / target_mask.sum()) * self.boundary_weight
        
        return self.l1_weight * l1_loss + self.l2_weight * l2_loss + boundary_loss
    
    def get_description(self) -> str:
        return f"L1 ({self.l1_weight}) + L2 ({self.l2_weight}) + boundary consistency"

class SSIMLoss(LossObjective):
    """
    Structural Similarity Index (SSIM) loss with boundary consistency.
    """
    
    def setup_models(self):
        self.boundary_weight = 0.1
        
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """
        Compute SSIM between two images.
        """
        # Convert to grayscale if needed
        if img1.shape[0] == 3:
            img1 = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
            img2 = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # Add batch dimension
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        
        # Constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create 2D Gaussian kernel
        sigma = 1.5
        # Create 1D Gaussian kernel
        kernel_1d = torch.exp(-torch.linspace(-(window_size // 2), window_size // 2, window_size) ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D Gaussian kernel by outer product
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Add batch and channel dimensions: [out_channels, in_channels, height, width]
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(img1.device)
        
        # Compute local means
        mu1 = F.conv2d(img1, kernel, padding=window_size // 2)
        mu2 = F.conv2d(img2, kernel, padding=window_size // 2)
        
        # Compute local variances and covariance
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2) - mu1_mu2
        
        # Compute SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
        
        return ssim_map.mean()
        
    def compute_loss(self, rendered_image: torch.Tensor, target_image: torch.Tensor, 
                    rendered_mask: torch.Tensor, target_mask: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        
        # Apply mask to both images
        masked_rendered = rendered_image * rendered_mask
        masked_target = target_image * target_mask
        
        # SSIM loss (1 - SSIM for minimization)
        ssim_value = self.compute_ssim(masked_rendered, masked_target)
        ssim_loss = 1 - ssim_value
        
        # Boundary consistency loss
        boundary_loss = F.mse_loss(rendered_mask, target_mask, reduction='none').mean()
        boundary_loss = boundary_loss * (target_mask.numel() / target_mask.sum()) * self.boundary_weight
        
        return ssim_loss + boundary_loss
    
    def get_description(self) -> str:
        return "SSIM loss + boundary consistency"

class HybridLoss(LossObjective):
    """
    Hybrid loss combining multiple loss functions.
    """
    
    def setup_models(self):
        self.dreamsim_model, _ = dreamsim(pretrained=True, device=self.device)
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        
        # Loss weights
        self.dreamsim_weight = 0.4
        self.lpips_weight = 0.3
        self.l1_weight = 0.2
        self.boundary_weight = 0.1
        
    def compute_loss(self, rendered_image: torch.Tensor, target_image: torch.Tensor, 
                    rendered_mask: torch.Tensor, target_mask: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        
        # Apply mask to both images
        masked_rendered = rendered_image * rendered_mask
        masked_target = target_image * target_mask
        
        # DreamSim loss
        masked_rendered_224 = F.interpolate(masked_rendered.unsqueeze(0), (224, 224), mode='bicubic')
        masked_target_224 = F.interpolate(masked_target.unsqueeze(0), (224, 224), mode='bicubic')
        dreamsim_loss = self.dreamsim_model(masked_rendered_224, masked_target_224)
        
        # LPIPS loss
        lpips_loss = self.lpips_model(
            masked_rendered.unsqueeze(0) * 2 - 1,
            masked_target.unsqueeze(0) * 2 - 1
        )
        
        # L1 loss
        l1_loss = F.l1_loss(masked_rendered, masked_target)
        
        # Boundary consistency loss
        boundary_loss = F.mse_loss(rendered_mask, target_mask, reduction='none').mean()
        boundary_loss = boundary_loss * (target_mask.numel() / target_mask.sum()) * self.boundary_weight
        
        total_loss = (self.dreamsim_weight * dreamsim_loss + 
                     self.lpips_weight * lpips_loss + 
                     self.l1_weight * l1_loss + 
                     boundary_loss)
        
        return total_loss
    
    def get_description(self) -> str:
        return f"Hybrid: DreamSim({self.dreamsim_weight}) + LPIPS({self.lpips_weight}) + L1({self.l1_weight}) + boundary"

class MaskOnlyLoss(LossObjective):
    """
    Simple mask-only loss for testing geometric alignment.
    """
    
    def setup_models(self):
        pass
        
    def compute_loss(self, rendered_image: torch.Tensor, target_image: torch.Tensor, 
                    rendered_mask: torch.Tensor, target_mask: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        
        # Only use mask consistency loss
        mask_loss = F.mse_loss(rendered_mask, target_mask, reduction='none').mean()
        mask_loss = mask_loss * (target_mask.numel() / target_mask.sum())
        
        return mask_loss
    
    def get_description(self) -> str:
        return "Mask-only loss for geometric alignment"

# Factory function to create loss objectives
def create_loss_objective(loss_type: str, device: str = 'cuda', **kwargs) -> LossObjective:
    """
    Factory function to create different loss objectives.
    
    Args:
        loss_type: Type of loss objective ('dreamsim', 'lpips', 'l1l2', 'ssim', 'hybrid', 'mask_only')
        device: Device to run on
        **kwargs: Additional arguments for specific loss types
        
    Returns:
        LossObjective instance
    """
    loss_objectives = {
        'dreamsim': DreamSimBoundaryLoss,
        'lpips': PerceptualLoss,
        'l1l2': L1L2Loss,
        'ssim': SSIMLoss,
        'hybrid': HybridLoss,
        'mask_only': MaskOnlyLoss,
    }
    
    if loss_type not in loss_objectives:
        raise ValueError(f"Unknown loss type: {loss_type}. Available types: {list(loss_objectives.keys())}")
    
    return loss_objectives[loss_type](device=device)

# Utility function to list all available loss objectives
def list_loss_objectives():
    """List all available loss objectives with descriptions."""
    objectives = [
        ('dreamsim', 'DreamSim perceptual loss + boundary consistency (original)'),
        ('lpips', 'LPIPS perceptual loss + boundary consistency'),
        ('l1l2', 'Combined L1 and L2 loss + boundary consistency'),
        ('ssim', 'SSIM loss + boundary consistency'),
        ('hybrid', 'Hybrid loss combining multiple objectives'),
        ('mask_only', 'Mask-only loss for geometric alignment'),
    ]
    
    print("Available Loss Objectives:")
    print("=" * 50)
    for name, desc in objectives:
        print(f"🎯 {name:12} - {desc}")
    print("=" * 50)
