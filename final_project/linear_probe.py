"""
Linear Probing Evaluation Script

This script evaluates pretrained MAE encoders by training a linear classifier
on frozen representations. This tests the quality of learned representations
without fine-tuning the encoder.

Protocol:
1. Load pretrained encoder weights
2. Freeze all encoder parameters
3. Add linear classification head
4. Train only the head on CIFAR-100
5. Report top-1 accuracy

Usage:
    python linear_probe.py --checkpoint checkpoints/checkpoint_best.pth --data_fraction 1.0
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models.mae_vit import mae_vit_tiny


class LinearClassifier(nn.Module):
    """Linear classification head for frozen encoder."""

    def __init__(self, encoder: nn.Module, num_classes: int = 100, pool_type: str = 'mean'):
        """
        Args:
            encoder: Pretrained MAE encoder (will be frozen)
            num_classes: Number of classes for classification
            pool_type: How to pool patch features ('mean', 'max', 'cls')
        """
        super().__init__()
        self.encoder = encoder
        self.pool_type = pool_type

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Get encoder output dimension
        embed_dim = encoder.pos_embed.shape[-1]

        # Linear classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_classes) logits
        """
        # Encode (no masking for evaluation)
        with torch.no_grad():
            features = self.encoder.encode(x)  # (B, num_patches, embed_dim)

        # Pool patch features to single vector
        if self.pool_type == 'mean':
            pooled = features.mean(dim=1)  # (B, embed_dim)
        elif self.pool_type == 'max':
            pooled = features.max(dim=1)[0]
        elif self.pool_type == 'first':
            pooled = features[:, 0]  # Use first patch token
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Classification
        logits = self.head(pooled)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description='Linear Probing Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of training data to use (for label efficiency)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--pool_type', type=str, default='mean', choices=['mean', 'max', 'first'], help='Pooling strategy')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cifar100_loaders(data_fraction: float = 1.0, batch_size: int = 256, img_size: int = 64):
    """
    Create CIFAR-100 dataloaders.

    Args:
        data_fraction: Fraction of training data to use (for label efficiency)
        batch_size: Batch size
        img_size: Image size (CIFAR-100 is 32x32, we resize to match TinyImageNet)

    Returns:
        train_loader, val_loader
    """
    # CIFAR-100 mean/std
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )

    # Training augmentation
    train_transform = transforms.Compose([
        transforms.Resize(img_size),  # Resize to match pretraining resolution
        transforms.RandomCrop(img_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])

    # Load datasets
    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=val_transform
    )

    # Subsample training data if needed (for label efficiency)
    if data_fraction < 1.0:
        num_train = len(train_dataset)
        num_samples = int(num_train * data_fraction)
        indices = np.random.permutation(num_train)[:num_samples]
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {num_samples}/{num_train} training samples ({data_fraction*100:.1f}%)")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Evaluating')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'acc': 100. * correct / total})

    accuracy = 100. * correct / total
    return accuracy


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pretrained encoder
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create encoder
    encoder = mae_vit_tiny(img_size=64, patch_size=16)
    encoder.load_state_dict(checkpoint['model'], strict=False)  # Load encoder weights
    encoder = encoder.to(device)
    encoder.eval()

    # Create linear classifier
    print(f"Creating linear classifier with {args.pool_type} pooling")
    model = LinearClassifier(encoder, num_classes=100, pool_type=args.pool_type)
    model = model.to(device)

    # Count trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.2f}M (linear head only)")

    # Create dataloaders
    print("Loading CIFAR-100...")
    train_loader, val_loader = get_cifar100_loaders(
        data_fraction=args.data_fraction,
        batch_size=args.batch_size,
        img_size=64
    )

    # Create optimizer (only for head)
    optimizer = torch.optim.SGD(
        model.head.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting linear probing for {args.epochs} epochs...")
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Evaluate
        val_acc = evaluate(model, val_loader, device)
        print(f"Val Acc: {val_acc:.2f}%")

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': best_acc,
                'args': args
            }, output_dir / 'linear_probe_best.pth')

    print(f"\nLinear probing completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'linear_probe_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data fraction: {args.data_fraction}\n")
        f.write(f"Pool type: {args.pool_type}\n")
        f.write(f"Best validation accuracy: {best_acc:.2f}%\n")

    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
