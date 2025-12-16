"""
Random Masking Strategy

This is the baseline masking strategy from the MAE paper (He et al., CVPR 2022).
Patches are masked uniformly at random without any structure.

Key properties:
- Content-agnostic: doesn't look at image content
- Uniform distribution: each patch has equal probability of being masked
- Reproducible: given the same random seed, produces same masks
"""

from typing import Optional
import torch
from .mask_generator import MaskGenerator


class RandomMaskGenerator(MaskGenerator):
    """
    Random masking baseline from MAE paper.

    Generates masks by shuffling patch indices and selecting the first N patches
    to keep, where N = num_patches * (1 - mask_ratio).

    This is the standard approach that forces the model to rely on context
    rather than local information.
    """

    def __init__(self, mask_ratio: float = 0.75):
        """
        Args:
            mask_ratio: Fraction of patches to mask (default: 0.75 from MAE paper)
        """
        super().__init__(mask_ratio)

    def generate(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
        images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate random binary masks.

        Implementation follows MAE paper: use random noise + argsort for shuffling,
        then create binary mask based on number of patches to keep.

        Args:
            batch_size: Number of images in batch
            num_patches: Total number of patches per image
            device: Device to create mask on
            images: Not used for random masking (kept for interface consistency)

        Returns:
            Binary mask (B, num_patches): 1 = keep, 0 = mask
        """
        # Calculate number of patches to keep (visible)
        num_keep = int(num_patches * (1 - self.mask_ratio))

        # Generate random noise for shuffling
        # Using random noise + argsort is more efficient than torch.randperm for batches
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)  # Shuffle indices

        # Create binary mask: first num_keep are visible (1), rest are masked (0)
        mask = torch.zeros(batch_size, num_patches, device=device)
        mask[:, :num_keep] = 1

        # Unshuffle mask to match original patch order
        # This is important so the mask corresponds to spatial patch positions
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def __repr__(self) -> str:
        return f"RandomMaskGenerator(mask_ratio={self.mask_ratio})"
