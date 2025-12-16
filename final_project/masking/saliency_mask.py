"""
Saliency-Guided Masking Strategy

This masking strategy uses fast, lightweight heuristics to identify salient patches
and preferentially mask them. This tests whether forcing the model to reconstruct
more informative patches leads to better representations.

IMPORTANT: No pretrained models allowed (per project constraints).

Approach:
- Use Sobel edge detection to compute patch saliency scores
- Mask patches with HIGH saliency (harder reconstruction task)
- Fast and GPU-compatible

Research motivation:
- Salient patches contain more information (edges, textures)
- Forcing reconstruction of salient patches may learn better features
- Alternative: mask low-saliency patches (easier task)
- Hypothesis: harder task → better representations
"""

from typing import Optional
import torch
import torch.nn.functional as F
import math
from .mask_generator import MaskGenerator


class SaliencyMaskGenerator(MaskGenerator):
    """
    Saliency-guided masking using Sobel edge detection.

    Computes saliency score per patch based on edge magnitude, then masks
    patches with high saliency to create a harder reconstruction task.

    No learned models or pretrained networks - purely heuristic-based.
    """

    def __init__(self, mask_ratio: float = 0.75, mode: str = 'high'):
        """
        Args:
            mask_ratio: Fraction of patches to mask
            mode: 'high' = mask high-saliency patches (harder)
                  'low' = mask low-saliency patches (easier)
        """
        super().__init__(mask_ratio)
        assert mode in ['high', 'low'], f"mode must be 'high' or 'low', got {mode}"
        self.mode = mode

        # Sobel kernels for edge detection
        # Horizontal edges (Gx)
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3) / 8.0

        # Vertical edges (Gy)
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3) / 8.0

    def compute_saliency(self, images: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        Compute saliency score for each patch using Sobel edge magnitude.

        Args:
            images: (B, C, H, W) input images
            patch_size: Size of each patch

        Returns:
            (B, num_patches) saliency scores
        """
        B, C, H, W = images.shape
        device = images.device

        # Move Sobel kernels to same device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # Convert to grayscale (average across channels)
        if C == 3:
            gray = images.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            gray = images

        # Apply Sobel filters
        # Expand kernels for all input channels
        sobel_x = sobel_x.repeat(1, 1, 1, 1)
        sobel_y = sobel_y.repeat(1, 1, 1, 1)

        grad_x = F.conv2d(gray, sobel_x, padding=1)  # (B, 1, H, W)
        grad_y = F.conv2d(gray, sobel_y, padding=1)  # (B, 1, H, W)

        # Compute gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # (B, 1, H, W)

        # Average pooling to get patch-level saliency
        # Each patch is patch_size x patch_size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        # Reshape to patches
        grad_mag = grad_mag.squeeze(1)  # (B, H, W)
        grad_mag = grad_mag.reshape(B, num_patches_h, patch_size, num_patches_w, patch_size)
        grad_mag = grad_mag.permute(0, 1, 3, 2, 4)  # (B, num_patches_h, num_patches_w, patch_size, patch_size)

        # Average over each patch
        patch_saliency = grad_mag.mean(dim=(-2, -1))  # (B, num_patches_h, num_patches_w)
        patch_saliency = patch_saliency.reshape(B, -1)  # (B, num_patches)

        return patch_saliency

    def generate(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
        images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate saliency-guided binary masks.

        Args:
            batch_size: Number of images in batch
            num_patches: Total number of patches per image
            device: Device to create mask on
            images: (B, C, H, W) input images - REQUIRED for saliency masking

        Returns:
            Binary mask (B, num_patches): 1 = keep, 0 = mask
        """
        if images is None:
            raise ValueError("SaliencyMaskGenerator requires input images")

        B, C, H, W = images.shape
        assert B == batch_size, f"Batch size mismatch: {B} vs {batch_size}"

        # Infer patch size
        patch_size = H // int(math.sqrt(num_patches))
        assert patch_size * int(math.sqrt(num_patches)) == H, \
            "Image size must be divisible by patch grid size"

        # Compute per-patch saliency scores
        saliency = self.compute_saliency(images, patch_size)  # (B, num_patches)

        # Rank patches by saliency
        if self.mode == 'high':
            # Mask high-saliency patches (harder task)
            # Sort descending: high saliency first
            _, indices = torch.sort(saliency, dim=1, descending=True)
        else:
            # Mask low-saliency patches (easier task)
            # Sort ascending: low saliency first
            _, indices = torch.sort(saliency, dim=1, descending=False)

        # Determine number of patches to keep vs mask
        num_mask = int(num_patches * self.mask_ratio)
        num_keep = num_patches - num_mask

        # Create mask: patches to keep are 1, patches to mask are 0
        mask = torch.zeros(batch_size, num_patches, device=device)

        # Assign 1 to kept patches (based on saliency ranking)
        # If mode='high', we keep LOW saliency patches (mask high)
        # If mode='low', we keep HIGH saliency patches (mask low)
        for b in range(batch_size):
            # Keep the LAST num_keep patches in sorted order
            keep_indices = indices[b, num_mask:]
            mask[b, keep_indices] = 1

        return mask

    def __repr__(self) -> str:
        return f"SaliencyMaskGenerator(mask_ratio={self.mask_ratio}, mode={self.mode})"
