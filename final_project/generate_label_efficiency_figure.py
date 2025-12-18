"""
Generate Publication-Quality Label Efficiency Bar Chart

Creates a clean, minimal bar chart suitable for academic papers.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_label_efficiency_chart():
    """Generate label efficiency bar chart."""

    # Data
    mask_ratios = ['0.5', '0.75']
    labels_10pct = [5.33, 5.87]
    labels_100pct = [13.17, 14.75]

    # Figure setup - LaTeX-friendly size (3.5 inches wide for single column)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Bar positions
    x = np.arange(len(mask_ratios))
    width = 0.35

    # Muted, paper-friendly colors
    color_10 = '#7C9EB2'   # Muted blue-gray
    color_100 = '#C77B58'  # Muted orange-brown

    # Create bars
    bars1 = ax.bar(x - width/2, labels_10pct, width,
                   label='10% Labels', color=color_10,
                   edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, labels_100pct, width,
                   label='100% Labels', color=color_100,
                   edgecolor='black', linewidth=0.8)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Labels and title
    ax.set_xlabel('Mask Ratio', fontsize=11, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Label Efficiency under Linear Probing on CIFAR-100',
                 fontsize=12, fontweight='bold', pad=15)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(mask_ratios, fontsize=10)

    # Y-axis - set appropriate range
    ax.set_ylim(0, 18)
    ax.set_yticks(np.arange(0, 19, 3))
    ax.tick_params(axis='y', labelsize=10)

    # Legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.95,
              fontsize=10, edgecolor='black', fancybox=False)

    # Grid - subtle
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = Path('results/analysis/label_efficiency.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"✓ Label efficiency chart saved to: {output_path}")

    # Also save as PDF for LaTeX
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {pdf_path}")

    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("LABEL EFFICIENCY SUMMARY")
    print("="*60)
    print(f"{'Mask Ratio':<15} {'10% Labels':<15} {'100% Labels':<15} {'Gain':<10}")
    print("-"*60)
    for i, ratio in enumerate(mask_ratios):
        gain = labels_100pct[i] - labels_10pct[i]
        print(f"{ratio:<15} {labels_10pct[i]:<15.2f}% {labels_100pct[i]:<15.2f}% +{gain:.2f}%")
    print("="*60)

    # Key observations
    print("\nKey Observations:")
    print(f"• Mask ratio 0.75 outperforms 0.5 at both label fractions")
    print(f"• With 10% labels: 0.75 achieves {labels_10pct[1]:.2f}% vs 0.5's {labels_10pct[0]:.2f}%")
    print(f"• With 100% labels: 0.75 achieves {labels_100pct[1]:.2f}% vs 0.5's {labels_100pct[0]:.2f}%")
    retention_50 = (labels_10pct[0] / labels_100pct[0]) * 100
    retention_75 = (labels_10pct[1] / labels_100pct[1]) * 100
    print(f"• Label retention at 10%: 0.5 = {retention_50:.1f}%, 0.75 = {retention_75:.1f}%")


if __name__ == '__main__':
    create_label_efficiency_chart()
