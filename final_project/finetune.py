"""
Fine-tuning Evaluation Script

This script fine-tunes pretrained MAE encoders end-to-end on CIFAR-100.
Unlike linear probing, this updates all encoder weights during training.

Protocol:
1. Load pretrained encoder weights
2. Add classification head
3. Fine-tune entire model on CIFAR-100
4. Report top-1 accuracy

Usage:
    python finetune.py --checkpoint checkpoints/checkpoint_best.pth --data_fraction 1.0
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


class FineTuneClassifier(nn.Module):
    """Classification model with fine-tunable encoder."""

    def __init__(self, encoder: nn.Module, num_classes: int = 100, pool_type: str = 'mean'):
        """
        Args:
            encoder: Pretrained MAE encoder (will be fine-tuned)
            num_classes: Number of classes for classification
            pool_type: How to pool patch features ('mean', 'max', 'first')
        """
        super().__init__()
        self.encoder = encoder
        self.pool_type = pool_type

        # Get encoder output dimension
        embed_dim = encoder.pos_embed.shape[-1]

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_classes) logits
        """
        # Encode (gradients flow through encoder)
        features = self.encoder.encode(x)  # (B, num_patches, embed_dim)

        # Pool patch features
        if self.pool_type == 'mean':
            pooled = features.mean(dim=1)
        elif self.pool_type == 'max':
            pooled = features.max(dim=1)[0]
        elif self.pool_type == 'first':
            pooled = features[:, 0]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Classification
        logits = self.head(pooled)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tuning Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of training data to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--pool_type', type=str, default='mean', choices=['mean', 'max', 'first'], help='Pooling strategy')
    parser.add_argument('--layer_decay', type=float, default=0.65, help='Layer-wise LR decay')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cifar100_loaders(data_fraction: float = 1.0, batch_size: int = 128, img_size: int = 64):
    """Create CIFAR-100 dataloaders."""
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )

    # Training augmentation (stronger than linear probing)
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)

    # Subsample if needed
    if data_fraction < 1.0:
        num_train = len(train_dataset)
        num_samples = int(num_train * data_fraction)
        indices = np.random.permutation(num_train)[:num_samples]
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {num_samples}/{num_train} training samples ({data_fraction*100:.1f}%)")

    # num_workers=0 and pin_memory=False for MPS compatibility
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader


def get_parameter_groups(model: FineTuneClassifier, lr: float, weight_decay: float, layer_decay: float):
    """
    Create parameter groups with layer-wise learning rate decay.

    Lower layers get smaller learning rates to preserve pretrained features.
    """
    param_groups = []

    # Classification head: full LR
    param_groups.append({
        'params': model.head.parameters(),
        'lr': lr,
        'weight_decay': weight_decay
    })

    # Encoder layers: layer-wise decay
    num_layers = len(model.encoder.blocks)
    for i, block in enumerate(model.encoder.blocks):
        # Decay increases for earlier layers
        layer_lr = lr * (layer_decay ** (num_layers - i - 1))
        param_groups.append({
            'params': block.parameters(),
            'lr': layer_lr,
            'weight_decay': weight_decay
        })

    # Patch embedding: smallest LR
    param_groups.append({
        'params': model.encoder.patch_embed.parameters(),
        'lr': lr * (layer_decay ** num_layers),
        'weight_decay': weight_decay
    })

    return param_groups


def adjust_learning_rate(optimizer, epoch, args, num_epochs):
    """Cosine LR schedule with warmup."""
    if epoch < args.warmup_epochs:
        # Linear warmup
        lr_scale = (epoch + 1) / args.warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - args.warmup_epochs) / (num_epochs - args.warmup_epochs)
        lr_scale = 0.5 * (1 + np.cos(np.pi * progress))

    for param_group in optimizer.param_groups:
        base_lr = param_group['lr']
        param_group['lr'] = base_lr * lr_scale


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    return total_loss / len(dataloader), 100. * correct / total


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

    return 100. * correct / total


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device (support MPS for Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load pretrained encoder
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    encoder = mae_vit_tiny(img_size=64, patch_size=16)
    encoder.load_state_dict(checkpoint['model'], strict=False)
    encoder = encoder.to(device)

    # Create fine-tuning model
    print(f"Creating fine-tuning model with {args.pool_type} pooling")
    model = FineTuneClassifier(encoder, num_classes=100, pool_type=args.pool_type)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.2f}M (entire model)")

    # Create dataloaders
    print("Loading CIFAR-100...")
    train_loader, val_loader = get_cifar100_loaders(args.data_fraction, args.batch_size, img_size=64)

    # Create optimizer with layer-wise LR decay
    param_groups = get_parameter_groups(model, args.lr, args.weight_decay, args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups)

    # Training loop
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # Adjust LR
        adjust_learning_rate(optimizer, epoch, args, args.epochs)

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Evaluate
        val_acc = evaluate(model, val_loader, device)
        print(f"Val Acc: {val_acc:.2f}%")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': best_acc,
                'args': args
            }, output_dir / 'finetune_best.pth')

    print(f"\nFine-tuning completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'finetune_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data fraction: {args.data_fraction}\n")
        f.write(f"Pool type: {args.pool_type}\n")
        f.write(f"Layer decay: {args.layer_decay}\n")
        f.write(f"Best validation accuracy: {best_acc:.2f}%\n")

    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
