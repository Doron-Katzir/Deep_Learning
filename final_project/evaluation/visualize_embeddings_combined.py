"""
Combined t-SNE Visualization for Multiple Mask Ratios

Generates side-by-side t-SNE plots comparing embedding spaces
learned by MAE models with different mask ratios.

Usage:
    python -m evaluation.visualize_embeddings_combined
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from models.mae_vit import mae_vit_tiny


@torch.no_grad()
def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple:
    """
    Extract features from encoder.

    Args:
        model: MAE encoder
        dataloader: Data loader
        device: Device

    Returns:
        features: (N, embed_dim) numpy array
        labels: (N,) numpy array
    """
    model.eval()

    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc='Extracting features'):
        images = images.to(device)

        # Get encoder features (mean pool across patches)
        features = model.encode(images)  # (B, num_patches, embed_dim)
        features = features.mean(dim=1)  # (B, embed_dim)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def visualize_embeddings_combined(
    features_dict: dict,
    labels: np.ndarray,
    mask_ratios: list,
    output_path: Path,
    method: str = 'tsne',
    num_classes: int = 100
):
    """
    Create combined t-SNE visualization for multiple mask ratios.

    Args:
        features_dict: Dict mapping mask_ratio -> features array
        labels: (N,) class labels
        mask_ratios: List of mask ratios
        output_path: Output file path
        method: 'tsne' or 'pca'
        num_classes: Number of classes for coloring
    """
    print(f"Performing {method.upper()} dimensionality reduction for all models...")

    # Reduce dimensionality for each model
    embeddings_dict = {}

    for ratio in mask_ratios:
        print(f"  Processing mask_ratio={ratio}")
        features = features_dict[ratio]

        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            embeddings_2d = reducer.fit_transform(features)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(features)
        else:
            raise ValueError(f"Unknown method: {method}")

        embeddings_dict[ratio] = embeddings_2d

    # Create combined visualization
    print("Creating combined visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ratio in enumerate(mask_ratios):
        ax = axes[idx]
        embeddings_2d = embeddings_dict[ratio]

        # Create scatter plot with class colors
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab20',
            s=1,
            alpha=0.6,
            rasterized=True
        )

        ax.set_title(f'Mask Ratio {ratio:.0%}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(f'{method.upper()} 1', fontsize=11)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=11)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'{method.upper()} Visualization of MAE Representations',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Combined Embedding Visualization')
    parser.add_argument('--mask_ratios', nargs='+', type=float, default=[0.5, 0.75, 0.9],
                        help='Mask ratios to visualize')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='Dimensionality reduction method')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='results/analysis/tsne_combined.png',
                        help='Output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")

    # Load CIFAR-100
    print("Loading CIFAR-100...")
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Subsample if needed
    if args.num_samples < len(dataset):
        indices = np.random.permutation(len(dataset))[:args.num_samples]
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=250,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"Using {len(dataset)} samples\n")

    # Extract features for all mask ratios
    features_dict = {}

    for ratio in args.mask_ratios:
        print(f"{'='*70}")
        print(f"Processing mask_ratio={ratio}")
        print(f"{'='*70}")

        checkpoint_path = Path(f'checkpoints/random_mask_{ratio}/checkpoint_best.pth')

        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            print("Skipping this mask ratio...")
            continue

        # Load model
        print(f"Loading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model = mae_vit_tiny(img_size=64, patch_size=16)
        model.load_state_dict(checkpoint['model'], strict=False)
        model = model.to(device)
        model.eval()

        # Extract features
        features, labels = extract_features(model, dataloader, device)
        features_dict[ratio] = features

        print(f"Extracted features: {features.shape}\n")

    if not features_dict:
        print("ERROR: No features extracted for any mask ratio!")
        return

    # Create combined visualization
    print(f"{'='*70}")
    print("Creating combined visualization")
    print(f"{'='*70}\n")

    visualize_embeddings_combined(
        features_dict=features_dict,
        labels=labels,
        mask_ratios=args.mask_ratios,
        output_path=Path(args.output),
        method=args.method,
        num_classes=100
    )

    print(f"\n{'='*70}")
    print("✓ Combined visualization completed!")
    print(f"{'='*70}")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
