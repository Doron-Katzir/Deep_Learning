"""
Generate Combined Reconstruction Visualization

Creates a single figure comparing reconstructions across all mask ratios.

Usage:
    python -m evaluation.generate_reconstructions_combined
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from models.mae_vit import mae_vit_tiny
from datasets.tinyimagenet import TinyImageNet


def denormalize(images: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize images for visualization."""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
    return images * std + mean


def unpatchify(x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Convert patches back to images.

    Args:
        x: (B, num_patches, patch_size**2 * 3)
        patch_size: Size of each patch

    Returns:
        images: (B, 3, H, W)
    """
    B, num_patches, _ = x.shape
    h = w = int(num_patches ** 0.5)

    x = x.reshape(B, h, w, patch_size, patch_size, 3)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.reshape(B, 3, h * patch_size, w * patch_size)

    return x


def generate_masked_image(original: torch.Tensor, mask: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """Apply mask to original image for visualization."""
    B, C, H, W = original.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    mask_spatial = mask.reshape(B, num_patches_h, num_patches_w)
    mask_img = mask_spatial.unsqueeze(1).float()
    mask_img = F.interpolate(mask_img, size=(H, W), mode='nearest')
    masked = original * mask_img

    return masked


@torch.no_grad()
def generate_reconstructions_for_checkpoint(
    checkpoint_path: Path,
    images: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    patch_size: int = 16
):
    """Generate reconstructions for a single checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = mae_vit_tiny(img_size=64, patch_size=patch_size, norm_pix_loss=True)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()

    images = images.to(device)
    mask = mask.to(device)

    loss, pred, mask_out = model(images, mask)
    reconstruction = unpatchify(pred, patch_size)

    return reconstruction


def visualize_all_reconstructions(
    original: torch.Tensor,
    masked_dict: dict,
    reconstruction_dict: dict,
    mask_ratios: list,
    output_path: Path,
    num_images: int = 8
):
    """
    Create combined visualization showing all mask ratios.

    Layout: Original | Masked_0.5 | Recon_0.5 | Masked_0.75 | Recon_0.75 | Masked_0.9 | Recon_0.9

    Args:
        original: (B, 3, H, W) original images
        masked_dict: Dict mapping ratio -> masked images
        reconstruction_dict: Dict mapping ratio -> reconstructions
        mask_ratios: List of mask ratios
        output_path: Where to save figure
        num_images: Number of images to show
    """
    num_images = min(num_images, original.shape[0])

    # Denormalize
    original = denormalize(original[:num_images]).cpu()
    original = torch.clamp(original, 0, 1)

    for ratio in mask_ratios:
        masked_dict[ratio] = torch.clamp(denormalize(masked_dict[ratio][:num_images]).cpu(), 0, 1)
        reconstruction_dict[ratio] = torch.clamp(denormalize(reconstruction_dict[ratio][:num_images]).cpu(), 0, 1)

    # Create figure: 1 (original) + 2 * num_ratios (masked + recon per ratio)
    num_cols = 1 + 2 * len(mask_ratios)

    fig = plt.figure(figsize=(num_cols * 2, num_images * 2))
    gs = GridSpec(num_images, num_cols, figure=fig, hspace=0.02, wspace=0.02)

    for row in range(num_images):
        col_idx = 0

        # Original
        ax = fig.add_subplot(gs[row, col_idx])
        img = original[row].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
        if row == 0:
            ax.set_title('Original', fontsize=11, fontweight='bold', pad=10)
        col_idx += 1

        # For each mask ratio: show masked and reconstruction
        for ratio in mask_ratios:
            # Masked
            ax = fig.add_subplot(gs[row, col_idx])
            img = masked_dict[ratio][row].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
            if row == 0:
                ax.set_title(f'Masked\n{ratio:.0%}', fontsize=11, fontweight='bold', pad=10)
            col_idx += 1

            # Reconstruction
            ax = fig.add_subplot(gs[row, col_idx])
            img = reconstruction_dict[ratio][row].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
            if row == 0:
                ax.set_title(f'Reconstruction\n{ratio:.0%}', fontsize=11, fontweight='bold', pad=10)
            col_idx += 1

    plt.suptitle('MAE Reconstructions Across Mask Ratios',
                 fontsize=14, fontweight='bold', y=0.995)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined reconstruction visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate Combined Reconstruction Visualization')
    parser.add_argument('--num_images', type=int, default=8, help='Number of images to visualize')
    parser.add_argument('--batch_idx', type=int, default=0, help='Which batch to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/analysis/reconstructions_combined.png',
                        help='Output path')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading TinyImageNet validation set...")
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TinyImageNet(root='./data/tiny-imagenet-200', split='val', transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.num_images,
        shuffle=False,
        num_workers=0
    )

    # Get fixed batch
    for i, (images, _) in enumerate(loader):
        if i == args.batch_idx:
            break

    print(f"Loaded batch of {images.shape[0]} images")

    # Parameters
    patch_size = 16
    img_size = 64
    num_patches = (img_size // patch_size) ** 2
    mask_ratios = [0.5, 0.75, 0.9]

    # Storage
    masked_dict = {}
    reconstruction_dict = {}

    print("\nGenerating reconstructions for all mask ratios...")
    print("=" * 70)

    for mask_ratio in mask_ratios:
        print(f"\nProcessing mask_ratio={mask_ratio}")

        # Generate mask (same pattern for all ratios for fair comparison)
        B = images.shape[0]
        num_masked = int(num_patches * mask_ratio)

        torch.manual_seed(args.seed)  # Same mask pattern
        noise = torch.rand(B, num_patches)
        ids_shuffle = torch.argsort(noise, dim=1)

        mask = torch.ones(B, num_patches)
        mask[:, :num_patches - num_masked] = 0
        mask = torch.gather(mask, dim=1, index=ids_shuffle)

        # Generate masked input
        masked_input = generate_masked_image(images, 1 - mask, patch_size=patch_size)
        masked_dict[mask_ratio] = masked_input

        # Load checkpoint and generate reconstruction
        checkpoint_path = Path(f'checkpoints/random_mask_{mask_ratio}/checkpoint_best.pth')

        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue

        reconstruction = generate_reconstructions_for_checkpoint(
            checkpoint_path=checkpoint_path,
            images=images,
            mask=mask,
            device=device,
            patch_size=patch_size
        )

        reconstruction_dict[mask_ratio] = reconstruction
        print(f"  ✓ Completed")

    # Create combined visualization
    print("\n" + "=" * 70)
    print("Creating combined visualization...")

    output_path = Path(args.output)
    visualize_all_reconstructions(
        original=images,
        masked_dict=masked_dict,
        reconstruction_dict=reconstruction_dict,
        mask_ratios=mask_ratios,
        output_path=output_path,
        num_images=args.num_images
    )

    print("\n" + "=" * 70)
    print("✓ Combined reconstruction visualization completed!")
    print("=" * 70)
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
