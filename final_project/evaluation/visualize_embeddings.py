"""
Embedding Visualization with t-SNE and PCA

This script visualizes learned representations using dimensionality reduction.
Helps understand how well different masking strategies separate classes.

Usage:
    python -m evaluation.visualize_embeddings --checkpoint checkpoint.pth --method tsne --output embeddings_tsne.png
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

from models.mae_vit import mae_vit_tiny


@torch.no_grad()
def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple:
    """
    Extract features and labels from model.

    Args:
        model: MAE encoder
        dataloader: DataLoader
        device: Device

    Returns:
        features: (n, d) feature array
        labels: (n,) label array
    """
    model.eval()
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc='Extracting features'):
        images = images.to(device)

        # Extract features (average over patches)
        features = model.encode(images)  # (B, num_patches, embed_dim)
        features = features.mean(dim=1)  # (B, embed_dim)

        all_features.append(features.cpu())
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    return features, labels


def visualize_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    output_path: Path = None,
    title: str = None,
    num_classes: int = 100
):
    """
    Visualize embeddings using dimensionality reduction.

    Args:
        features: (n, d) feature matrix
        labels: (n,) label array
        method: 'tsne' or 'pca'
        output_path: Path to save figure
        title: Plot title
        num_classes: Number of classes (for color palette)
    """
    print(f"Performing {method.upper()} dimensionality reduction...")

    if method == 'tsne':
        # t-SNE: nonlinear, preserves local structure
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = reducer.fit_transform(features)
    elif method == 'pca':
        # PCA: linear, preserves global structure
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(features)
        explained_var = reducer.explained_variance_ratio_
        print(f"Explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create visualization
    plt.figure(figsize=(12, 10))

    # Plot with class colors
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Use a colormap
    if n_labels <= 20:
        # Use distinct colors for few classes
        palette = sns.color_palette('tab20', n_labels)
    else:
        # Use continuous colormap for many classes
        palette = sns.color_palette('hsv', n_labels)

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[palette[idx]],
            label=f'Class {label}' if n_labels <= 20 else None,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )

    plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)

    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f'{method.upper()} Visualization of Learned Representations', fontsize=14, fontweight='bold')

    # Add legend only for few classes
    if n_labels <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Embeddings')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'], help='Dimensionality reduction method')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device (support MPS for Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = mae_vit_tiny(img_size=64, patch_size=16)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()

    # Create dataloader
    print("Loading CIFAR-100...")
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Subsample dataset
    if args.num_samples < len(dataset):
        indices = np.random.permutation(len(dataset))[:args.num_samples]
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Extract features
    features, labels = extract_features(model, dataloader, device)
    print(f"Extracted features: {features.shape}")

    # Visualize
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visualize_embeddings(
        features,
        labels,
        method=args.method,
        output_path=output_path,
        title=f'{args.method.upper()} Visualization - {Path(args.checkpoint).stem}'
    )

    print("Visualization completed!")


if __name__ == '__main__':
    main()
