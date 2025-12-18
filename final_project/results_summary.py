"""
Results Summary Script

Collects and displays results from all experiments.
"""

import re
from pathlib import Path

ratios = [0.5, 0.75, 0.9]

print("\n" + "="*60)
print("RESULTS COMPARISON - Random Masking")
print("="*60)
print(f"{'Mask Ratio':<15} {'Linear Probe':<15} {'Fine-tuning':<15}")
print("-"*60)

for ratio in ratios:
    # Linear probe
    lp_file = f"results/random_{ratio}/linear_probe/linear_probe_results.txt"
    if Path(lp_file).exists():
        with open(lp_file) as f:
            content = f.read()
            match = re.search(r'Best validation accuracy: ([\d.]+)%', content)
            lp_acc = match.group(1) if match else "N/A"
    else:
        lp_acc = "Not run"

    # Fine-tuning
    ft_file = f"results/random_{ratio}/finetune/finetune_results.txt"
    if Path(ft_file).exists():
        with open(ft_file) as f:
            content = f.read()
            match = re.search(r'Best validation accuracy: ([\d.]+)%', content)
            ft_acc = match.group(1) if match else "N/A"
    else:
        ft_acc = "Not run"

    lp_str = f"{lp_acc}%" if lp_acc != "Not run" else lp_acc
    ft_str = f"{ft_acc}%" if ft_acc != "Not run" else ft_acc

    print(f"{ratio:<15} {lp_str:<15} {ft_str:<15}")

print("="*60 + "\n")
