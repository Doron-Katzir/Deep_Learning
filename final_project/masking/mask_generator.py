"""
Mask Generation Interface and Base Classes

This module defines the plug-and-play interface for different masking strategies
in Masked Autoencoder training. All masking strategies must:

1. Preserve exact mask ratio
2. Be reproducible given a seed
3. Work batch-wise on GPU
4. Return binary masks (1 = keep, 0 = remove)

Masking Strategies:
- RandomMask: Baseline random masking (MAE paper)
- GridMask: Deterministic periodic patterns
- SaliencyMask: Fast heuristic based on edge detection (no pretrained models)
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class MaskGenerator(ABC):
    """
    Abstract base class for mask generation strategies.

    All mask generators must implement the `generate` method which returns
    a binary mask tensor indicating which patches to keep (1) or mask (0).

    The mask ratio must be preserved exactly for fair comparison across strategies.
    """

    def __init__(self, mask_ratio: float = 0.75):
        """
        Args:
            mask_ratio: Fraction of patches to mask (0.0 to 1.0)
        """
        assert 0.0 <= mask_ratio < 1.0, f"mask_ratio must be in [0, 1), got {mask_ratio}"
        self.mask_ratio = mask_ratio

    @abstractmethod
    def generate(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
        images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate binary mask for a batch of images.

        Args:
            batch_size: Number of images in batch
            num_patches: Total number of patches per image (H/P * W/P)
            device: Device to create mask on
            images: Optional input images (B, C, H, W) for content-aware masking

        Returns:
            Binary mask of shape (B, num_patches):
                1 = keep (visible)
                0 = mask (hidden)

        Note: The number of masked patches should be exactly:
              num_masked = round(num_patches * mask_ratio)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mask_ratio={self.mask_ratio})"


def create_mask_generator(mask_type: str, mask_ratio: float, **kwargs) -> MaskGenerator:
    """
    Factory function to create mask generators.

    Args:
        mask_type: Type of masking ('random', 'grid', 'saliency')
        mask_ratio: Fraction of patches to mask
        **kwargs: Additional arguments for specific mask generators

    Returns:
        MaskGenerator instance

    Example:
        >>> mask_gen = create_mask_generator('random', 0.75)
        >>> mask = mask_gen.generate(32, 16, torch.device('cuda'))
    """
    mask_type = mask_type.lower()

    if mask_type == 'random':
        from .random_mask import RandomMaskGenerator
        return RandomMaskGenerator(mask_ratio)
    elif mask_type == 'grid':
        from .grid_mask import GridMaskGenerator
        return GridMaskGenerator(mask_ratio, **kwargs)
    elif mask_type == 'saliency':
        from .saliency_mask import SaliencyMaskGenerator
        return SaliencyMaskGenerator(mask_ratio, **kwargs)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}. "
                        f"Supported types: random, grid, saliency")
