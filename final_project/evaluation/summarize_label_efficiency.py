"""
Summarize Label Efficiency Results

Collects and formats label efficiency experiment results into a summary table.

Usage:
    python -m evaluation.summarize_label_efficiency
"""

import re
from pathlib import Path


def parse_result_file(result_file: Path) -> float:
    """
    Parse linear probe result file and extract accuracy.

    Args:
        result_file: Path to linear_probe_results.txt

    Returns:
        accuracy: Best validation accuracy (or -1 if not found)
    """
    if not result_file.exists():
        return -1.0

    with open(result_file, 'r') as f:
        content = f.read()

    match = re.search(r'Best validation accuracy: ([\d.]+)%', content)
    if match:
        return float(match.group(1))

    return -1.0


def main():
    # Results directory
    results_dir = Path('results/label_efficiency')

    # Mask ratios tested
    mask_ratios = [0.5, 0.75]

    # Data fractions tested
    label_fractions = [
        (0.1, '10pct'),
        (1.0, '100pct')
    ]

    # Collect results
    results = {}

    for ratio in mask_ratios:
        results[ratio] = {}

        for fraction, pct_str in label_fractions:
            result_file = results_dir / f'mask_{ratio}_labels_{pct_str}' / 'linear_probe_results.txt'
            accuracy = parse_result_file(result_file)
            results[ratio][fraction] = accuracy

    # Generate summary report
    output_file = Path('results/analysis/label_efficiency_summary.txt')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LABEL EFFICIENCY ANALYSIS - MAE PRETRAINING\n")
        f.write("=" * 80 + "\n\n")

        f.write("Experimental Setup:\n")
        f.write("-" * 80 + "\n")
        f.write("• Pretrained encoders: MAE with random masking\n")
        f.write("• Mask ratios tested: 0.5, 0.75\n")
        f.write("• Evaluation protocol: Linear probing on CIFAR-100\n")
        f.write("• Training labels: 10% and 100% of CIFAR-100 training set\n")
        f.write("• Hyperparameters: 100 epochs, batch_size=256, lr=0.001\n")
        f.write("\n\n")

        f.write("Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Mask Ratio':<15} {'10% Labels':<15} {'100% Labels':<15} {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")

        for ratio in mask_ratios:
            acc_10 = results[ratio][0.1]
            acc_100 = results[ratio][1.0]

            if acc_10 > 0 and acc_100 > 0:
                improvement = acc_100 - acc_10
                f.write(f"{ratio:<15.0%} {acc_10:<15.2f}% {acc_100:<15.2f}% {improvement:+.2f}%\n")
            else:
                acc_10_str = f"{acc_10:.2f}%" if acc_10 > 0 else "Not run"
                acc_100_str = f"{acc_100:.2f}%" if acc_100 > 0 else "Not run"
                f.write(f"{ratio:<15.0%} {acc_10_str:<15} {acc_100_str:<15} -\n")

        f.write("=" * 80 + "\n\n")

        f.write("Observations:\n")
        f.write("-" * 80 + "\n")

        # Calculate label efficiency metrics
        for ratio in mask_ratios:
            acc_10 = results[ratio][0.1]
            acc_100 = results[ratio][1.0]

            if acc_10 > 0 and acc_100 > 0:
                retention = (acc_10 / acc_100) * 100 if acc_100 > 0 else 0
                f.write(f"• Mask ratio {ratio:.0%}: With only 10% labels, retains {retention:.1f}% of full performance\n")

        f.write("\n")

        # Add interpretation
        f.write("\nInterpretation:\n")
        f.write("-" * 80 + "\n")
        f.write("Label efficiency measures how well pretrained representations generalize\n")
        f.write("with limited labeled data. Higher retention percentage indicates better\n")
        f.write("quality pretrained features that require less labeled data for downstream tasks.\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"Summary saved to: {output_file}")

    # Print to console
    print("\n" + "=" * 80)
    with open(output_file, 'r') as f:
        print(f.read())


if __name__ == '__main__':
    main()
