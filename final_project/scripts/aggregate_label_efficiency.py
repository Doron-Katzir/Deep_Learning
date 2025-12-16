"""
Aggregate Label Efficiency Results

This script collects results from label efficiency experiments and creates
a CSV summary and plot showing accuracy vs data fraction.

Usage:
    python scripts/aggregate_label_efficiency.py --results_dir results/label_efficiency
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_results_file(results_file: Path) -> dict:
    """Parse a results text file and extract key metrics."""
    with open(results_file, 'r') as f:
        content = f.read()

    # Extract data fraction
    data_frac_match = re.search(r'Data fraction: ([\d.]+)', content)
    data_frac = float(data_frac_match.group(1)) if data_frac_match else None

    # Extract accuracy
    acc_match = re.search(r'Best validation accuracy: ([\d.]+)%', content)
    accuracy = float(acc_match.group(1)) if acc_match else None

    return {
        'data_fraction': data_frac,
        'accuracy': accuracy
    }


def aggregate_results(results_dir: Path) -> pd.DataFrame:
    """Aggregate all results from directory."""
    records = []

    # Find all result files
    for method in ['linear_probe', 'finetune']:
        method_dirs = sorted(results_dir.glob(f'{method}_*'))

        for method_dir in method_dirs:
            results_file = method_dir / f'{method}_results.txt'
            if results_file.exists():
                result = parse_results_file(results_file)
                result['method'] = method
                records.append(result)

    df = pd.DataFrame(records)
    df = df.sort_values(['method', 'data_fraction'])

    return df


def plot_label_efficiency(df: pd.DataFrame, output_path: Path):
    """Create label efficiency plot."""
    plt.figure(figsize=(10, 6))

    # Plot curves for each method
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        label = method.replace('_', ' ').title()
        plt.plot(method_df['data_fraction'] * 100, method_df['accuracy'],
                marker='o', linewidth=2, markersize=8, label=label)

    plt.xlabel('Training Data Fraction (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Label Efficiency: Accuracy vs Training Data Fraction', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')

    # Format x-axis
    plt.xticks([1, 5, 10, 100], ['1%', '5%', '10%', '100%'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate label efficiency results')
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Aggregate results
    print(f"Aggregating results from {results_dir}...")
    df = aggregate_results(results_dir)

    if df.empty:
        print("No results found!")
        return

    # Save CSV
    csv_path = results_dir / 'label_efficiency_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Print table
    print("\n" + "="*60)
    print("Label Efficiency Results")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)

    # Create plot
    plot_path = results_dir / 'label_efficiency_curve.png'
    plot_label_efficiency(df, plot_path)

    print(f"\nAggregation complete!")


if __name__ == '__main__':
    main()
