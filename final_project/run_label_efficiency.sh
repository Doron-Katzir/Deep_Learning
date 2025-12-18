#!/bin/bash

# Label Efficiency Analysis
# Tests pretrained MAE encoders with reduced labels

# Activate virtual environment
source venv/bin/activate

echo "========================================="
echo "LABEL EFFICIENCY ANALYSIS"
echo "========================================="
echo ""

# Create output directory
mkdir -p results/label_efficiency

# Mask ratios to test
ratios=(0.5 0.75)

# Data fractions to test
fractions=(0.1 1.0)

for ratio in "${ratios[@]}"; do
    echo "========================================"
    echo "Mask Ratio: ${ratio}"
    echo "========================================"

    for fraction in "${fractions[@]}"; do
        echo ""
        echo "--- Training with ${fraction} ($(echo "$fraction * 100" | bc)%) of labels ---"

        # Convert fraction to percentage for output directory
        percent=$(echo "$fraction * 100" | bc | cut -d. -f1)

        python linear_probe.py \
            --checkpoint checkpoints/random_mask_${ratio}/checkpoint_best.pth \
            --data_fraction ${fraction} \
            --output_dir results/label_efficiency/mask_${ratio}_labels_${percent}pct \
            --epochs 100 \
            --batch_size 256 \
            --lr 0.001
    done

    echo ""
done

echo ""
echo "========================================="
echo "GENERATING SUMMARY"
echo "========================================="

# Create summary file
python -m evaluation.summarize_label_efficiency

echo ""
echo "========================================="
echo "LABEL EFFICIENCY ANALYSIS COMPLETED!"
echo "========================================="
echo "Results saved to: results/analysis/label_efficiency_summary.txt"
