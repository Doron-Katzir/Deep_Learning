"""
Plot Training Curves from TensorBoard Logs

Extracts reconstruction loss from TensorBoard event files and creates
comparison plots across different mask ratios.

Usage:
    python -m evaluation.plot_training_curves
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard_logs(log_dir: Path, tag: str = 'train/loss_epoch'):
    """
    Parse TensorBoard logs and extract scalar values.

    Args:
        log_dir: Path to TensorBoard log directory
        tag: Scalar tag to extract (e.g., 'train/loss_epoch')

    Returns:
        steps: List of step/epoch numbers
        values: List of corresponding values
    """
    # Find event files
    event_files = list(log_dir.glob('events.out.tfevents.*'))

    if not event_files:
        print(f"WARNING: No event files found in {log_dir}")
        return [], []

    # Load the event file
    ea = event_accumulator.EventAccumulator(
        str(event_files[0]),
        size_guidance={event_accumulator.SCALARS: 0}  # Load all scalars
    )
    ea.Reload()

    # Get available tags
    available_tags = ea.Tags()['scalars']

    if tag not in available_tags:
        print(f"WARNING: Tag '{tag}' not found in {log_dir}")
        print(f"Available tags: {available_tags}")
        return [], []

    # Extract scalar events
    scalar_events = ea.Scalars(tag)

    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]

    return steps, values


def plot_loss_comparison(
    losses_dict: dict,
    output_path: Path,
    title: str = "MAE Reconstruction Loss vs. Training Epoch",
    xlabel: str = "Epoch",
    ylabel: str = "Reconstruction Loss (MSE)"
):
    """
    Plot loss curves for multiple mask ratios.

    Args:
        losses_dict: Dict mapping mask_ratio -> (epochs, losses)
        output_path: Where to save the plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))

    # Plot each mask ratio
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    markers = ['o', 's', '^']

    for idx, (mask_ratio, (epochs, losses)) in enumerate(sorted(losses_dict.items())):
        plt.plot(
            epochs,
            losses,
            label=f'Mask Ratio {mask_ratio:.0%}',
            color=colors[idx],
            marker=markers[idx],
            markevery=max(1, len(epochs) // 20),  # Show ~20 markers
            linewidth=2,
            markersize=6,
            alpha=0.8
        )

    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Format axes
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def print_summary_statistics(losses_dict: dict):
    """Print summary statistics for each mask ratio."""
    print("\n" + "="*70)
    print("RECONSTRUCTION LOSS SUMMARY")
    print("="*70)
    print(f"{'Mask Ratio':<15} {'Initial Loss':<15} {'Final Loss':<15} {'Best Loss':<15}")
    print("-"*70)

    for mask_ratio, (epochs, losses) in sorted(losses_dict.items()):
        if len(losses) > 0:
            initial_loss = losses[0]
            final_loss = losses[-1]
            best_loss = min(losses)

            print(f"{mask_ratio:<15.0%} {initial_loss:<15.4f} {final_loss:<15.4f} {best_loss:<15.4f}")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot Training Curves from TensorBoard Logs')
    parser.add_argument('--log_base_dir', type=str, default='checkpoints',
                        help='Base directory containing checkpoint folders')
    parser.add_argument('--mask_ratios', nargs='+', type=float, default=[0.5, 0.75, 0.9],
                        help='Mask ratios to plot')
    parser.add_argument('--tag', type=str, default='train/loss_epoch',
                        help='TensorBoard tag to extract')
    parser.add_argument('--output', type=str, default='results/analysis/reconstruction_loss_comparison.png',
                        help='Output path for plot')
    args = parser.parse_args()

    print("="*70)
    print("TENSORBOARD LOG PARSER")
    print("="*70)

    # Parse logs for each mask ratio
    losses_dict = {}

    for mask_ratio in args.mask_ratios:
        log_dir = Path(args.log_base_dir) / f'random_mask_{mask_ratio}' / 'logs'

        print(f"\nParsing logs for mask_ratio={mask_ratio}")
        print(f"  Log directory: {log_dir}")

        if not log_dir.exists():
            print(f"  WARNING: Directory does not exist, skipping...")
            continue

        epochs, losses = parse_tensorboard_logs(log_dir, tag=args.tag)

        if len(epochs) > 0:
            print(f"  Found {len(epochs)} epochs of data")
            print(f"  Loss range: {min(losses):.4f} - {max(losses):.4f}")
            losses_dict[mask_ratio] = (epochs, losses)
        else:
            print(f"  No data found for tag '{args.tag}'")

    if not losses_dict:
        print("\nERROR: No valid loss data found for any mask ratio!")
        return

    # Print summary statistics
    print_summary_statistics(losses_dict)

    # Create plot
    print("Creating comparison plot...")
    plot_loss_comparison(
        losses_dict=losses_dict,
        output_path=Path(args.output)
    )

    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    main()
