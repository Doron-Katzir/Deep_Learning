"""
Grid-based Structured Masking Strategy

This masking strategy creates deterministic periodic patterns where visible patches
form a regular grid. This tests whether structured spatial patterns help or hurt
representation learning.

Research motivation:
- Random masking may be too uniform
- Grid patterns provide consistent spatial structure
- Easier to predict missing patches (might be easier or harder to learn from)
- Tests inductive bias of spatial coherence
"""

from typing import Optional
import torch
import math
from .mask_generator import MaskGenerator


class GridMaskGenerator(MaskGenerator):
    """
    Grid-based masking with periodic visible patches.

    Creates masks where visible patches form a regular grid pattern.
    For example, with mask_ratio=0.75 (25% visible), we might keep
    every 2nd patch in a checkerboard pattern.

    Implementation ensures exact mask ratio while maintaining grid structure.
    """

    def __init__(self, mask_ratio: float = 0.75, pattern: str = 'checkerboard'):
        """
        Args:
            mask_ratio: Fraction of patches to mask
            pattern: Grid pattern type ('checkerboard', 'stripes_h', 'stripes_v', 'blocks')
        """
        super().__init__(mask_ratio)
        self.pattern = pattern

        # Validate pattern
        valid_patterns = ['checkerboard', 'stripes_h', 'stripes_v', 'blocks']
        if pattern not in valid_patterns:
            raise ValueError(f"Pattern must be one of {valid_patterns}, got {pattern}")

    def generate(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
        images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate grid-based binary masks.

        Args:
            batch_size: Number of images in batch
            num_patches: Total number of patches per image
            device: Device to create mask on
            images: Not used for grid masking

        Returns:
            Binary mask (B, num_patches): 1 = keep, 0 = mask
        """
        # Calculate grid size (assume square)
        grid_size = int(math.sqrt(num_patches))
        assert grid_size * grid_size == num_patches, \
            f"Grid masking requires square patch grid, got {num_patches} patches"

        # Calculate stride to achieve desired mask ratio
        # stride = how many patches to skip between visible patches
        num_keep = int(num_patches * (1 - self.mask_ratio))
        stride = max(1, int(math.sqrt(num_patches / num_keep)))

        # Generate base mask pattern for one image
        if self.pattern == 'checkerboard':
            mask = self._checkerboard_pattern(grid_size, stride)
        elif self.pattern == 'stripes_h':
            mask = self._horizontal_stripes_pattern(grid_size, stride)
        elif self.pattern == 'stripes_v':
            mask = self._vertical_stripes_pattern(grid_size, stride)
        elif self.pattern == 'blocks':
            mask = self._blocks_pattern(grid_size, stride)

        mask = mask.flatten()  # (num_patches,)

        # Adjust to match exact mask ratio
        # Since stride is approximate, we may need to adjust
        current_keep = mask.sum().item()
        target_keep = num_keep

        if current_keep > target_keep:
            # Too many visible patches: randomly remove some
            visible_indices = torch.where(mask == 1)[0]
            num_remove = int(current_keep - target_keep)
            remove_indices = visible_indices[torch.randperm(len(visible_indices))[:num_remove]]
            mask[remove_indices] = 0
        elif current_keep < target_keep:
            # Too few visible patches: randomly add some
            masked_indices = torch.where(mask == 0)[0]
            num_add = int(target_keep - current_keep)
            add_indices = masked_indices[torch.randperm(len(masked_indices))[:num_add]]
            mask[add_indices] = 1

        # Repeat for batch
        mask = mask.unsqueeze(0).repeat(batch_size, 1).to(device)

        return mask

    def _checkerboard_pattern(self, grid_size: int, stride: int) -> torch.Tensor:
        """Create checkerboard pattern (alternating visible patches)."""
        mask = torch.zeros(grid_size, grid_size)
        for i in range(0, grid_size, stride):
            for j in range(0, grid_size, stride):
                # Checkerboard: alternate based on parity
                if (i // stride + j // stride) % 2 == 0:
                    mask[i, j] = 1
        return mask

    def _horizontal_stripes_pattern(self, grid_size: int, stride: int) -> torch.Tensor:
        """Create horizontal stripe pattern."""
        mask = torch.zeros(grid_size, grid_size)
        for i in range(0, grid_size, stride):
            mask[i, :] = 1
        return mask

    def _vertical_stripes_pattern(self, grid_size: int, stride: int) -> torch.Tensor:
        """Create vertical stripe pattern."""
        mask = torch.zeros(grid_size, grid_size)
        for j in range(0, grid_size, stride):
            mask[:, j] = 1
        return mask

    def _blocks_pattern(self, grid_size: int, stride: int) -> torch.Tensor:
        """Create block pattern (visible blocks of patches)."""
        mask = torch.zeros(grid_size, grid_size)
        block_size = max(1, stride // 2)

        for i in range(0, grid_size, stride):
            for j in range(0, grid_size, stride):
                # Create small visible blocks
                i_end = min(i + block_size, grid_size)
                j_end = min(j + block_size, grid_size)
                mask[i:i_end, j:j_end] = 1

        return mask

    def __repr__(self) -> str:
        return f"GridMaskGenerator(mask_ratio={self.mask_ratio}, pattern={self.pattern})"
