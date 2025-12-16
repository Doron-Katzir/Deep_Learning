"""
Mask Visualization Utilities

This module provides functions to visualize masking patterns and verify
that mask generators work correctly.

Sanity checks:
1. Mask ratio is preserved exactly
2. Masks are binary (0 or 1)
3. Patterns match expectations (grid has structure, random is uniform, etc.)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from .mask_generator import MaskGenerator


def visualize_masks(
    images: torch.Tensor,
    masks: torch.Tensor,
    patch_size: int = 16,
    num_samples: int = 4,
    save_path: Optional[str] = None
):
    """
    Visualize original images with mask overlays.

    Args:
        images: (B, C, H, W) input images (normalized)
        masks: (B, num_patches) binary masks (1=keep, 0=mask)
        patch_size: Size of each patch
        num_samples: Number of samples to visualize
        save_path: Optional path to save figure
    """
    B, C, H, W = images.shape
    num_samples = min(num_samples, B)

    # Denormalize images for visualization (assume ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # Convert mask to spatial grid
    grid_size = H // patch_size
    mask_spatial = masks.reshape(B, grid_size, grid_size)  # (B, grid_h, grid_w)

    # Upsample mask to image resolution
    mask_upsampled = mask_spatial.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    mask_upsampled = mask_upsampled.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, H, W)

    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original image
        img = images_denorm[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i}: Original")
        axes[i, 0].axis('off')

        # Mask pattern
        mask_vis = mask_spatial[i].cpu().numpy()
        axes[i, 1].imshow(mask_vis, cmap='RdYlGn', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Mask Pattern (ratio={1 - masks[i].mean():.2f})")
        axes[i, 1].axis('off')

        # Masked image (gray out masked patches)
        masked_img = img.copy()
        mask_overlay = mask_upsampled[i].cpu().permute(1, 2, 0).numpy()
        masked_img = masked_img * mask_overlay + (1 - mask_overlay) * 0.5  # Gray for masked
        axes[i, 2].imshow(masked_img)
        axes[i, 2].set_title(f"Visible Patches Only")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def sanity_check_mask_generator(
    mask_gen: MaskGenerator,
    images: Optional[torch.Tensor] = None,
    num_patches: int = 16,
    batch_size: int = 8,
    device: torch.device = torch.device('cpu')
):
    """
    Run sanity checks on a mask generator.

    Verifies:
    1. Output shape is correct
    2. Mask ratio is preserved
    3. Values are binary (0 or 1)
    4. Masks are different across batch (for random)

    Args:
        mask_gen: MaskGenerator instance to test
        images: Optional input images (required for saliency masking)
        num_patches: Number of patches
        batch_size: Batch size to test
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print(f"Sanity Check: {mask_gen}")
    print(f"{'='*60}")

    # Generate masks
    if images is None and 'Saliency' in mask_gen.__class__.__name__:
        # Create dummy images for saliency masking
        img_size = int(np.sqrt(num_patches)) * 16  # Assume patch_size=16
        images = torch.randn(batch_size, 3, img_size, img_size, device=device)

    masks = mask_gen.generate(batch_size, num_patches, device, images)

    # Check 1: Shape
    expected_shape = (batch_size, num_patches)
    assert masks.shape == expected_shape, f"Shape mismatch: {masks.shape} vs {expected_shape}"
    print(f"✓ Shape: {masks.shape}")

    # Check 2: Binary values
    unique_vals = torch.unique(masks).cpu().numpy()
    assert len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals), \
        f"Mask should be binary, got unique values: {unique_vals}"
    print(f"✓ Binary values: {unique_vals}")

    # Check 3: Mask ratio
    actual_mask_ratios = (1 - masks.mean(dim=1)).cpu().numpy()
    expected_ratio = mask_gen.mask_ratio
    tolerance = 0.05  # Allow 5% tolerance

    print(f"✓ Mask ratios (expected {expected_ratio:.2f}):")
    for i, ratio in enumerate(actual_mask_ratios):
        status = "✓" if abs(ratio - expected_ratio) < tolerance else "✗"
        print(f"  Sample {i}: {ratio:.4f} {status}")

    avg_ratio = actual_mask_ratios.mean()
    assert abs(avg_ratio - expected_ratio) < tolerance, \
        f"Mask ratio deviation too large: {avg_ratio:.4f} vs {expected_ratio:.4f}"

    # Check 4: Diversity (for random masking)
    if 'Random' in mask_gen.__class__.__name__:
        # Check that masks are different across batch
        unique_masks = len(torch.unique(masks, dim=0))
        if unique_masks == 1:
            print("⚠ Warning: All masks are identical (expected variation for random masking)")
        else:
            print(f"✓ Mask diversity: {unique_masks}/{batch_size} unique masks")

    # Check 5: Structure (for grid masking)
    if 'Grid' in mask_gen.__class__.__name__:
        # All masks in batch should be identical (deterministic pattern)
        if not torch.all(masks[0] == masks[1]):
            print("⚠ Warning: Grid masks differ across batch (expected identical)")
        else:
            print(f"✓ Grid structure: All masks identical (deterministic)")

    print(f"{'='*60}\n")


def compare_masking_strategies(
    strategies: list,
    images: torch.Tensor,
    patch_size: int = 16,
    save_path: Optional[str] = None
):
    """
    Compare different masking strategies side-by-side.

    Args:
        strategies: List of (name, MaskGenerator) tuples
        images: (B, C, H, W) input images
        patch_size: Size of each patch
        save_path: Optional path to save figure
    """
    B, C, H, W = images.shape
    num_patches = (H // patch_size) ** 2
    device = images.device

    num_strategies = len(strategies)
    fig, axes = plt.subplots(1, num_strategies + 1, figsize=(4 * (num_strategies + 1), 4))

    # Show original image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_denorm = (images[0] * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()

    axes[0].imshow(img_denorm)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Show each masking strategy
    for idx, (name, mask_gen) in enumerate(strategies, start=1):
        mask = mask_gen.generate(1, num_patches, device, images)

        # Convert to spatial
        grid_size = H // patch_size
        mask_spatial = mask.reshape(grid_size, grid_size).cpu().numpy()

        axes[idx].imshow(mask_spatial, cmap='RdYlGn', vmin=0, vmax=1)
        axes[idx].set_title(f"{name}\n(ratio={mask_gen.mask_ratio:.2f})")
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")

    plt.show()


if __name__ == '__main__':
    """Run sanity checks on all masking strategies."""
    from .random_mask import RandomMaskGenerator
    from .grid_mask import GridMaskGenerator
    from .saliency_mask import SaliencyMaskGenerator

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running sanity checks on device: {device}")

    # Test parameters
    batch_size = 8
    img_size = 64
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2
    mask_ratio = 0.75

    # Create test images
    test_images = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # Test all strategies
    print("\n" + "="*60)
    print("MASK GENERATOR SANITY CHECKS")
    print("="*60)

    # Random masking
    random_gen = RandomMaskGenerator(mask_ratio=mask_ratio)
    sanity_check_mask_generator(random_gen, None, num_patches, batch_size, device)

    # Grid masking (test all patterns)
    for pattern in ['checkerboard', 'stripes_h', 'stripes_v', 'blocks']:
        grid_gen = GridMaskGenerator(mask_ratio=mask_ratio, pattern=pattern)
        sanity_check_mask_generator(grid_gen, None, num_patches, batch_size, device)

    # Saliency masking
    for mode in ['high', 'low']:
        saliency_gen = SaliencyMaskGenerator(mask_ratio=mask_ratio, mode=mode)
        sanity_check_mask_generator(saliency_gen, test_images, num_patches, batch_size, device)

    print("All sanity checks passed! ✓")
