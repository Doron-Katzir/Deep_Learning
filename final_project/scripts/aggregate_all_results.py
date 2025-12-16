"""
Aggregate All Experimental Results

This script collects results from all experiments and creates comprehensive
tables and plots for the research paper.

Usage:
    python scripts/aggregate_all_results.py --results_dir results/
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def parse_result_file(result_file: Path) -> dict:
    """Parse a results text file."""
    with open(result_file, 'r') as f:
        content = f.read()

    results = {}

    # Extract checkpoint path to determine mask type and ratio
    ckpt_match = re.search(r'Checkpoint: .*/(.*_mask_[\d.]+)/', content)
    if ckpt_match:
        exp_name = ckpt_match.group(1)
        parts = exp_name.split('_mask_')
        results['mask_type'] = parts[0]
        results['mask_ratio'] = float(parts[1])

    # Extract data fraction
    frac_match = re.search(r'Data fraction: ([\d.]+)', content)
    if frac_match:
        results['data_fraction'] = float(frac_match.group(1))

    # Extract accuracy
    acc_match = re.search(r'Best validation accuracy: ([\d.]+)%', content)
    if acc_match:
        results['accuracy'] = float(acc_match.group(1))

    return results


def aggregate_downstream_results(results_dir: Path) -> pd.DataFrame:
    """Aggregate linear probe and fine-tuning results."""
    records = []

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Linear probe results
        linear_file = exp_dir / 'linear_probe' / 'linear_probe_results.txt'
        if linear_file.exists():
            result = parse_result_file(linear_file)
            result['method'] = 'Linear Probe'
            records.append(result)

        # Fine-tuning results
        finetune_file = exp_dir / 'finetune' / 'finetune_results.txt'
        if finetune_file.exists():
            result = parse_result_file(finetune_file)
            result['method'] = 'Fine-tuning'
            records.append(result)

    df = pd.DataFrame(records)
    return df


def plot_main_results(df: pd.DataFrame, output_dir: Path):
    """Create main results plot: accuracy vs mask ratio for each strategy."""
    # Filter for full data (data_fraction = 1.0)
    df_full = df[df['data_fraction'] == 1.0].copy()

    if df_full.empty:
        print("No full-data results found for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Linear Probing
    ax = axes[0]
    df_linear = df_full[df_full['method'] == 'Linear Probe']

    for mask_type in df_linear['mask_type'].unique():
        df_mask = df_linear[df_linear['mask_type'] == mask_type]
        df_mask = df_mask.sort_values('mask_ratio')
        ax.plot(df_mask['mask_ratio'], df_mask['accuracy'],
               marker='o', linewidth=2, markersize=8, label=mask_type.capitalize())

    ax.set_xlabel('Mask Ratio', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Linear Probing Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Fine-tuning
    ax = axes[1]
    df_finetune = df_full[df_full['method'] == 'Fine-tuning']

    for mask_type in df_finetune['mask_type'].unique():
        df_mask = df_finetune[df_finetune['mask_type'] == mask_type]
        df_mask = df_mask.sort_values('mask_ratio')
        ax.plot(df_mask['mask_ratio'], df_mask['accuracy'],
               marker='o', linewidth=2, markersize=8, label=mask_type.capitalize())

    ax.set_xlabel('Mask Ratio', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Fine-tuning Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'main_results.png', dpi=300, bbox_inches='tight')
    print(f"Main results plot saved to {output_dir / 'main_results.png'}")


def create_results_table(df: pd.DataFrame, output_dir: Path):
    """Create LaTeX-ready results table."""
    # Filter for full data
    df_full = df[df['data_fraction'] == 1.0].copy()

    if df_full.empty:
        print("No full-data results found for table")
        return

    # Pivot table
    table = df_full.pivot_table(
        index=['mask_type', 'mask_ratio'],
        columns='method',
        values='accuracy'
    )

    # Save as CSV
    csv_path = output_dir / 'results_table.csv'
    table.to_csv(csv_path)
    print(f"Results table saved to {csv_path}")

    # Save as LaTeX
    latex_path = output_dir / 'results_table.tex'
    with open(latex_path, 'w') as f:
        f.write(table.to_latex(float_format="%.2f"))
    print(f"LaTeX table saved to {latex_path}")

    # Print to console
    print("\n" + "="*80)
    print("MAIN RESULTS TABLE")
    print("="*80)
    print(table.to_string(float_format="%.2f"))
    print("="*80 + "\n")


def create_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Create grouped bar chart comparing all methods."""
    # Filter for full data and mask_ratio=0.75 (most common)
    df_comparison = df[(df['data_fraction'] == 1.0) & (df['mask_ratio'] == 0.75)].copy()

    if df_comparison.empty:
        print("No data for comparison plot")
        return

    # Pivot for plotting
    pivot = df_comparison.pivot(index='mask_type', columns='method', values='accuracy')

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Masking Strategy', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Downstream Performance Comparison (Mask Ratio = 0.75)', fontsize=14, fontweight='bold')
    ax.legend(title='Method', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_dir / 'comparison_barplot.png', dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_dir / 'comparison_barplot.png'}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate all results')
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print("Aggregating all experimental results...")

    # Aggregate downstream results
    df = aggregate_downstream_results(results_dir)

    if df.empty:
        print("No results found!")
        return

    # Create analysis directory
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    create_results_table(df, analysis_dir)
    plot_main_results(df, analysis_dir)
    create_comparison_plot(df, analysis_dir)

    # Save master CSV
    master_csv = analysis_dir / 'all_results.csv'
    df.to_csv(master_csv, index=False)
    print(f"Master results CSV saved to {master_csv}")

    print("\n" + "="*80)
    print("RESULT AGGREGATION COMPLETED!")
    print("="*80)
    print(f"All outputs saved to: {analysis_dir}")


if __name__ == '__main__':
    main()
