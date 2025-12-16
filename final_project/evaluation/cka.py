"""
Centered Kernel Alignment (CKA) Analysis

CKA measures similarity between learned representations across different layers
and models. It's invariant to orthogonal transformations and isotropic scaling.

Reference:
    Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019

Usage:
    python -m evaluation.cka --checkpoints checkpoint1.pth checkpoint2.pth --output cka_comparison.png
"""

import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models.mae_vit import mae_vit_tiny


def centering(K: np.ndarray) -> np.ndarray:
    """
    Center kernel matrix K.

    Args:
        K: (n, n) kernel matrix

    Returns:
        Centered kernel matrix
    """
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH


def rbf_kernel(X: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix.

    Args:
        X: (n, d) data matrix
        sigma: Kernel bandwidth (if None, use median heuristic)

    Returns:
        (n, n) kernel matrix
    """
    # Compute pairwise squared distances
    X = X.reshape(X.shape[0], -1)  # Flatten features
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T

    if sigma is None:
        # Median heuristic for bandwidth
        mdist = np.median(KX[KX != 0])
        sigma = np.sqrt(mdist)

    KX *= -0.5 / (sigma ** 2)
    KX = np.exp(KX)

    return KX


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """
    Compute linear kernel matrix.

    Args:
        X: (n, d) data matrix

    Returns:
        (n, n) kernel matrix
    """
    X = X.reshape(X.shape[0], -1)
    return np.dot(X, X.T)


def cka_score(X: np.ndarray, Y: np.ndarray, kernel: str = 'linear') -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two feature matrices.

    CKA(X, Y) = ||HKXH · HKYH||_F / (||HKXH||_F · ||HKYH||_F)

    Args:
        X: (n, d1) feature matrix from model 1
        Y: (n, d2) feature matrix from model 2
        kernel: 'linear' or 'rbf'

    Returns:
        CKA similarity score in [0, 1]
    """
    # Compute kernel matrices
    if kernel == 'linear':
        K_X = linear_kernel(X)
        K_Y = linear_kernel(Y)
    elif kernel == 'rbf':
        K_X = rbf_kernel(X)
        K_Y = rbf_kernel(Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Center kernel matrices
    K_X = centering(K_X)
    K_Y = centering(K_Y)

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic = np.sum(K_X * K_Y)

    # Normalize
    norm_X = np.linalg.norm(K_X, ord='fro')
    norm_Y = np.linalg.norm(K_Y, ord='fro')

    if norm_X == 0 or norm_Y == 0:
        return 0.0

    cka = hsic / (norm_X * norm_Y)

    return cka


class FeatureExtractor:
    """Extract layer-wise features from MAE encoder."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        """
        Args:
            model: MAE encoder model
            layer_names: List of layer names to extract features from
        """
        self.model = model
        self.layer_names = layer_names
        self.features = {name: [] for name in layer_names}
        self.hooks = []

        # Register forward hooks
        for name in layer_names:
            layer = self._get_layer(name)
            hook = layer.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)

    def _get_layer(self, name: str):
        """Get layer by name."""
        if name == 'patch_embed':
            return self.model.patch_embed
        elif name.startswith('block_'):
            idx = int(name.split('_')[1])
            return self.model.blocks[idx]
        elif name == 'norm':
            return self.model.norm
        else:
            raise ValueError(f"Unknown layer name: {name}")

    def _hook_fn(self, name: str):
        """Create hook function for layer."""
        def hook(module, input, output):
            self.features[name].append(output.detach().cpu())
        return hook

    def extract_features(self, dataloader: DataLoader, max_batches: int = None) -> Dict[str, np.ndarray]:
        """
        Extract features from all layers.

        Args:
            dataloader: DataLoader for images
            max_batches: Maximum number of batches to process

        Returns:
            Dictionary mapping layer names to feature arrays (n, d)
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc='Extracting features')):
                if max_batches and batch_idx >= max_batches:
                    break

                images = images.to(device)
                _ = self.model.encode(images)  # Triggers hooks

        # Convert to numpy arrays
        features_np = {}
        for name, feat_list in self.features.items():
            # Concatenate batches
            features = torch.cat(feat_list, dim=0)  # (n, num_patches, embed_dim)
            # Average over patches
            features = features.mean(dim=1)  # (n, embed_dim)
            features_np[name] = features.numpy()

        # Remove hooks
        for hook in self.hooks:
            hook.remove()

        return features_np


def compute_cka_matrix(checkpoints: List[Path], dataloader: DataLoader, kernel: str = 'linear') -> np.ndarray:
    """
    Compute pairwise CKA matrix between multiple checkpoints.

    Args:
        checkpoints: List of checkpoint paths
        dataloader: DataLoader for images
        kernel: Kernel type for CKA

    Returns:
        (n_models, n_models) CKA similarity matrix
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_models = len(checkpoints)

    # Extract features from all models
    print("Extracting features from all models...")
    all_features = []

    for ckpt_path in checkpoints:
        print(f"\nLoading checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Load model
        model = mae_vit_tiny(img_size=64, patch_size=16)
        model.load_state_dict(checkpoint['model'], strict=False)
        model = model.to(device)
        model.eval()

        # Extract features (use final layer representation)
        layer_names = ['norm']  # Final encoder layer
        extractor = FeatureExtractor(model, layer_names)
        features = extractor.extract_features(dataloader, max_batches=50)  # Limit for speed
        all_features.append(features['norm'])

    # Compute CKA matrix
    print(f"\nComputing CKA matrix with {kernel} kernel...")
    cka_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                cka_matrix[i, j] = 1.0
            elif i < j:
                cka = cka_score(all_features[i], all_features[j], kernel=kernel)
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka  # Symmetric

    return cka_matrix


def plot_cka_matrix(cka_matrix: np.ndarray, labels: List[str], output_path: Path):
    """Plot CKA similarity matrix as heatmap."""
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cka_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'label': 'CKA Similarity'}
    )

    plt.title('Centered Kernel Alignment (CKA) Similarity Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"CKA matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CKA Analysis')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='Checkpoint paths')
    parser.add_argument('--labels', nargs='+', default=None, help='Model labels for plot')
    parser.add_argument('--output', type=str, default='cka_matrix.png', help='Output path')
    parser.add_argument('--kernel', type=str, default='linear', choices=['linear', 'rbf'], help='Kernel type')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()

    checkpoints = [Path(p) for p in args.checkpoints]

    # Generate labels if not provided
    if args.labels:
        labels = args.labels
    else:
        labels = [f"Model {i+1}" for i in range(len(checkpoints))]

    assert len(labels) == len(checkpoints), "Number of labels must match number of checkpoints"

    # Create dataloader
    print("Loading CIFAR-100 for feature extraction...")
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Compute CKA matrix
    cka_matrix = compute_cka_matrix(checkpoints, dataloader, kernel=args.kernel)

    # Print matrix
    print("\n" + "="*60)
    print("CKA Similarity Matrix")
    print("="*60)
    for i, label_i in enumerate(labels):
        row_str = f"{label_i:15s}"
        for j in range(len(labels)):
            row_str += f"  {cka_matrix[i, j]:.3f}"
        print(row_str)
    print("="*60)

    # Plot matrix
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_cka_matrix(cka_matrix, labels, output_path)

    print("\nCKA analysis completed!")


if __name__ == '__main__':
    main()
