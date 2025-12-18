"""
Generate Qualitative Reconstructions

Loads trained MAE checkpoints and generates visual reconstructions
to show what the model learned during pretraining.

Usage:
    python -m evaluation.generate_reconstructions
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
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, 3, h, patch_size, w, patch_size)
    x = x.reshape(B, 3, h * patch_size, w * patch_size)

    return x


def visualize_reconstruction(
    original: torch.Tensor,
    masked: torch.Tensor,
    reconstruction: torch.Tensor,
    mask: torch.Tensor,
    mask_ratio: float,
    output_path: Path,
    num_images: int = 8
):
    """
    Create visualization grid showing original, masked, and reconstructed images.

    Args:
        original: (B, 3, H, W) original images
        masked: (B, 3, H, W) masked images
        reconstruction: (B, 3, H, W) reconstructed images
        mask: (B, num_patches) binary mask
        mask_ratio: Mask ratio used
        output_path: Where to save the figure
        num_images: Number of images to show
    """
    num_images = min(num_images, original.shape[0])

    # Move to CPU and denormalize
    original = denormalize(original[:num_images]).cpu()
    masked = denormalize(masked[:num_images]).cpu()
    reconstruction = denormalize(reconstruction[:num_images]).cpu()

    # Clip to valid range
    original = torch.clamp(original, 0, 1)
    masked = torch.clamp(masked, 0, 1)
    reconstruction = torch.clamp(reconstruction, 0, 1)

    # Create figure
    fig = plt.figure(figsize=(15, num_images * 1.5))
    gs = GridSpec(num_images, 3, figure=fig, hspace=0.05, wspace=0.05)

    for i in range(num_images):
        # Original
        ax = fig.add_subplot(gs[i, 0])
        img = original[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
        if i == 0:
            ax.set_title('Original', fontsize=12, fontweight='bold')

        # Masked
        ax = fig.add_subplot(gs[i, 1])
        img = masked[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
        if i == 0:
            ax.set_title(f'Masked ({mask_ratio:.0%})', fontsize=12, fontweight='bold')

        # Reconstruction
        ax = fig.add_subplot(gs[i, 2])
        img = reconstruction[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
        if i == 0:
            ax.set_title('Reconstruction', fontsize=12, fontweight='bold')

    plt.suptitle(f'MAE Reconstructions (Mask Ratio: {mask_ratio:.0%})',
                 fontsize=14, fontweight='bold', y=0.995)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstruction visualization to {output_path}")
    plt.close()


def generate_masked_image(original: torch.Tensor, mask: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Apply mask to original image for visualization.

    Args:
        original: (B, 3, H, W)
        mask: (B, num_patches) - 1 = keep, 0 = remove
        patch_size: Size of each patch

    Returns:
        masked_image: (B, 3, H, W) with masked patches set to gray
    """
    B, C, H, W = original.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Reshape mask to spatial grid
    mask_spatial = mask.reshape(B, num_patches_h, num_patches_w)

    # Upsample mask to image size
    mask_img = mask_spatial.unsqueeze(1).float()  # (B, 1, h, w)
    mask_img = F.interpolate(mask_img, size=(H, W), mode='nearest')  # (B, 1, H, W)

    # Apply mask (set masked patches to gray = 0 in normalized space)
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
    """
    Generate reconstructions for a single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        images: (B, 3, H, W) input images
        mask: (B, num_patches) binary mask (1 = masked, 0 = visible)
        device: Device to use
        patch_size: Patch size

    Returns:
        reconstruction: (B, 3, H, W) reconstructed images
    """
    # Load model
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = mae_vit_tiny(img_size=64, patch_size=patch_size, norm_pix_loss=True)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()

    images = images.to(device)
    mask = mask.to(device)

    # Forward pass
    loss, pred, mask_out = model(images, mask)

    # Unpatchify prediction
    reconstruction = unpatchify(pred, patch_size)

    return reconstruction


def main():
    parser = argparse.ArgumentParser(description='Generate Reconstruction Visualizations')
    parser.add_argument('--num_images', type=int, default=8, help='Number of images to visualize')
    parser.add_argument('--batch_idx', type=int, default=0, help='Which batch to use from validation set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for masking')
    parser.add_argument('--output_dir', type=str, default='results/analysis', help='Output directory')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load TinyImageNet validation data
    print("Loading TinyImageNet validation set...")
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TinyImageNet(root='./data', split='val', transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.num_images,
        shuffle=False,
        num_workers=0
    )

    # Get a fixed batch of images
    for i, (images, _) in enumerate(loader):
        if i == args.batch_idx:
            break

    print(f"Loaded batch of {images.shape[0]} images")

    # Generate a fixed random mask for all ratios
    # This ensures fair comparison across mask ratios
    patch_size = 16
    img_size = 64
    num_patches = (img_size // patch_size) ** 2

    # Process each mask ratio
    mask_ratios = [0.5, 0.75, 0.9]

    for mask_ratio in mask_ratios:
        print(f"\n{'='*60}")
        print(f"Generating reconstructions for mask_ratio={mask_ratio}")
        print(f"{'='*60}")

        # Generate mask (1 = masked, 0 = visible)
        B = images.shape[0]
        num_masked = int(num_patches * mask_ratio)

        # Create random mask
        torch.manual_seed(args.seed)  # Same mask pattern for all ratios
        noise = torch.rand(B, num_patches)
        ids_shuffle = torch.argsort(noise, dim=1)

        mask = torch.ones(B, num_patches)
        mask[:, :num_patches - num_masked] = 0  # 0 = keep
        mask = torch.gather(mask, dim=1, index=ids_shuffle)  # Unshuffle

        # Generate masked input for visualization
        masked_input = generate_masked_image(images, 1 - mask, patch_size=patch_size)

        # Load checkpoint and generate reconstruction
        checkpoint_path = Path(f'checkpoints/random_mask_{mask_ratio}/checkpoint_best.pth')

        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            print("Skipping this mask ratio...")
            continue

        reconstruction = generate_reconstructions_for_checkpoint(
            checkpoint_path=checkpoint_path,
            images=images,
            mask=mask,
            device=device,
            patch_size=patch_size
        )

        # Create visualization
        output_path = Path(args.output_dir) / f'reconstructions_{mask_ratio}.png'
        visualize_reconstruction(
            original=images,
            masked=masked_input,
            reconstruction=reconstruction,
            mask=mask,
            mask_ratio=mask_ratio,
            output_path=output_path,
            num_images=args.num_images
        )

    print(f"\n{'='*60}")
    print("All reconstructions generated successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
