"""
Attention Map Visualization

Visualize self-attention patterns in Vision Transformer layers.
Shows which patches the model attends to, helping understand learned representations.

Usage:
    python -m evaluation.visualize_attention --checkpoint checkpoint.pth --image_path image.jpg --output attention.png
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from models.mae_vit import mae_vit_tiny


def get_attention_maps(model: nn.Module, image: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
    """
    Extract attention maps from a specific transformer layer.

    Args:
        model: MAE ViT model
        image: (1, C, H, W) input image
        layer_idx: Layer index (-1 for last layer)

    Returns:
        (num_heads, num_patches, num_patches) attention maps
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)

    # Register hook to capture attention weights
    attention_maps = []

    def hook_fn(module, input, output):
        # In the Attention module, we need to capture attention weights
        # This requires modifying the forward pass slightly
        pass

    # Alternative: Extract attention manually
    # We'll do a modified forward pass to capture attention

    # Patch embedding
    x = model.patch_embed(image)  # (1, num_patches, embed_dim)
    x = x + model.pos_embed

    # Process through blocks
    for i, block in enumerate(model.blocks):
        if i == layer_idx or (layer_idx == -1 and i == len(model.blocks) - 1):
            # Capture attention for this layer
            B, N, C = x.shape

            # Compute attention (replicate attention module logic)
            qkv = block.attn.qkv(block.norm1(x))
            qkv = qkv.reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Attention weights
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)

            attention_maps.append(attn[0].detach().cpu())  # (num_heads, N, N)

        # Continue forward pass
        x = block(x)

    if len(attention_maps) == 0:
        raise ValueError(f"Could not extract attention from layer {layer_idx}")

    return attention_maps[0]  # (num_heads, num_patches, num_patches)


def visualize_attention(
    image: torch.Tensor,
    attention_maps: torch.Tensor,
    output_path: Path = None,
    num_heads_to_show: int = 6
):
    """
    Visualize attention maps.

    Args:
        image: (C, H, W) input image (denormalized)
        attention_maps: (num_heads, num_patches, num_patches) attention weights
        output_path: Path to save figure
        num_heads_to_show: Number of attention heads to visualize
    """
    num_heads, num_patches, _ = attention_maps.shape
    grid_size = int(np.sqrt(num_patches))

    # Select which heads to show
    heads_to_show = min(num_heads_to_show, num_heads)

    # Average attention to each patch (from all patches)
    # Shape: (num_heads, num_patches)
    avg_attention = attention_maps.mean(dim=1)

    # Reshape to spatial grid
    # Shape: (num_heads, grid_size, grid_size)
    spatial_attention = avg_attention.reshape(num_heads, grid_size, grid_size)

    # Create visualization
    fig, axes = plt.subplots(2, (heads_to_show + 2) // 2, figsize=(15, 6))
    axes = axes.flatten()

    # Show original image first
    img_np = image.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    for idx in range(heads_to_show):
        axes[idx].imshow(img_np, alpha=0.5)
        attn_map = spatial_attention[idx].numpy()

        # Upsample attention map to image size
        from scipy.ndimage import zoom
        zoom_factor = img_np.shape[0] // grid_size
        attn_map_upsampled = zoom(attn_map, zoom_factor, order=1)

        # Overlay attention
        im = axes[idx].imshow(attn_map_upsampled, cmap='jet', alpha=0.5)
        axes[idx].set_title(f'Head {idx + 1}', fontsize=10)
        axes[idx].axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)

    plt.suptitle('Self-Attention Maps (Average Attention per Patch)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Attention Maps')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--layer', type=int, default=-1, help='Layer index (-1 for last)')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads to show')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = mae_vit_tiny(img_size=64, patch_size=16)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    print(f"Loading image: {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')

    # Transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize
    ])

    image_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)

    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image_tensor[0] * std + mean

    # Extract attention maps
    print(f"Extracting attention maps from layer {args.layer}...")
    with torch.no_grad():
        attention_maps = get_attention_maps(model, image_tensor, layer_idx=args.layer)

    # Visualize
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visualize_attention(image_denorm, attention_maps, output_path, num_heads_to_show=args.num_heads)

    print("Attention visualization completed!")


if __name__ == '__main__':
    main()
