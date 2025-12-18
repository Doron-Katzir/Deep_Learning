"""
MAE Pretraining Script

This script handles pretraining of Masked Autoencoders with various masking strategies.
Designed for single-GPU training with TinyImageNet dataset.

Usage:
    python pretrain_mae.py --config configs/pretrain_baseline.yaml
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from models.mae_vit import mae_vit_tiny
from masking import create_mask_generator
from datasets.tinyimagenet import TinyImageNet


def parse_args():
    parser = argparse.ArgumentParser(description='MAE Pretraining')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--mask_ratio', type=float, default=None, help='Override mask ratio (e.g., 0.5, 0.75, 0.9)')
    parser.add_argument('--mask_type', type=str, default=None, help='Override mask type (random, grid, saliency)')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(config: Dict[str, Any], is_train: bool = True) -> DataLoader:
    """
    Create dataloader for TinyImageNet.

    Args:
        config: Configuration dictionary
        is_train: If True, return training dataloader, else validation

    Returns:
        DataLoader for TinyImageNet
    """
    from torchvision import transforms

    # Data augmentation for MAE (minimal as per paper)
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                config['model']['img_size'],
                scale=config['augmentation']['random_resized_crop']['scale'],
                ratio=config['augmentation']['random_resized_crop']['ratio']
            ),
            transforms.RandomHorizontalFlip() if config['augmentation']['horizontal_flip'] else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['augmentation']['normalize']['mean'],
                std=config['augmentation']['normalize']['std']
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(config['model']['img_size']),
            transforms.CenterCrop(config['model']['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['augmentation']['normalize']['mean'],
                std=config['augmentation']['normalize']['std']
            )
        ])

    # Create TinyImageNet dataset
    split = 'train' if is_train else 'val'
    dataset = TinyImageNet(
        root=config['data']['data_path'],
        split=split,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=is_train,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=is_train
    )

    return loader


# Removed get_mask function - now using MaskGenerator classes


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    mask_generator,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    writer: SummaryWriter
):
    """Train for one epoch."""
    model.train()

    num_batches = len(dataloader)
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        # Generate mask using configured strategy
        B = images.shape[0]
        num_patches = (config['model']['img_size'] // config['model']['patch_size']) ** 2

        # For saliency masking, pass images; for others, images=None is fine
        mask = mask_generator.generate(B, num_patches, device, images)

        # Forward pass with automatic mixed precision
        # Use appropriate autocast for device type
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'  # MPS uses CPU context
        use_amp = config['training']['use_amp'] and device.type == 'cuda'  # Only CUDA supports AMP well

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            loss, pred, mask = model(images, mask)

        # Backward pass
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Logging
        if batch_idx % config['training']['log_interval'] == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch}][{batch_idx}/{num_batches}] "
                  f"Loss: {loss.item():.4f} LR: {lr:.6f}")

            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/lr', lr, global_step)

    avg_loss = total_loss / num_batches
    writer.add_scalar('train/loss_epoch', avg_loss, epoch)

    return avg_loss


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, config: Dict[str, Any]):
    """Cosine learning rate schedule with warmup."""
    max_epochs = config['training']['epochs']
    warmup_epochs = config['training']['warmup_epochs']
    base_lr = config['training']['lr']
    min_lr = config['training']['min_lr']

    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    config: Dict[str, Any],
    output_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': config
    }

    # Save regular checkpoint
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = output_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)

    # Keep only last checkpoint and best
    if epoch > 0 and (epoch - 1) % config['training']['save_interval'] != 0:
        prev_checkpoint = output_dir / f'checkpoint_epoch_{epoch-1:03d}.pth'
        if prev_checkpoint.exists():
            prev_checkpoint.unlink()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override parameters from command line
    if args.mask_ratio is not None:
        config['masking']['mask_ratio'] = args.mask_ratio
        print(f"Override: mask_ratio = {args.mask_ratio}")
    if args.mask_type is not None:
        config['masking']['type'] = args.mask_type
        print(f"Override: mask_type = {args.mask_type}")
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        print(f"Override: epochs = {args.epochs}")

    # Auto-generate output directory based on experiment config
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create descriptive subdirectory: mask_type_mask_ratio
        mask_type = config['masking']['type']
        mask_ratio = config['masking']['mask_ratio']
        exp_name = f"{mask_type}_mask_{mask_ratio}"
        output_dir = Path(config['system']['output_dir']) / exp_name
        print(f"Auto-generated output directory: {output_dir}")

    log_dir = output_dir / 'logs'  # Logs inside experiment directory
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    set_seed(config['system']['seed'])

    # Setup device (support MPS for Apple Silicon)
    if config['system']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif config['system']['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Note: MPS device detected, using Apple Silicon GPU")
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create model
    print("Creating model...")
    model = mae_vit_tiny(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        in_chans=config['model']['in_chans'],
        norm_pix_loss=config['model']['norm_pix_loss']
    )
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=config['training']['betas']
    )

    # Create gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=config['training']['use_amp'])

    # Create mask generator
    print(f"Creating mask generator: {config['masking']['type']} (ratio={config['masking']['mask_ratio']})")
    mask_generator = create_mask_generator(
        mask_type=config['masking']['type'],
        mask_ratio=config['masking']['mask_ratio']
    )

    # Create dataloader
    print("Loading dataset...")
    train_loader = get_dataloader(config, is_train=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    print(f"Starting training for {config['training']['epochs']} epochs...")
    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, config['training']['epochs']):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, config)

        # Train one epoch
        epoch_start = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, mask_generator, device, epoch, config, writer
        )
        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch}] completed in {epoch_time:.2f}s - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if epoch % config['training']['save_interval'] == 0 or epoch == config['training']['epochs'] - 1:
            save_checkpoint(model, optimizer, scaler, epoch, config, output_dir, is_best)

    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()
