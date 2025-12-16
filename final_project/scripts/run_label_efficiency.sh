#!/bin/bash

# Label Efficiency Experiment Script
#
# This script runs label efficiency experiments by varying the amount of
# training data (1%, 5%, 10%, 100%) for both linear probing and fine-tuning.
#
# Usage:
#   bash scripts/run_label_efficiency.sh <checkpoint_path> <output_dir>
#
# Example:
#   bash scripts/run_label_efficiency.sh checkpoints/random_mask_0.75/checkpoint_best.pth results/label_efficiency

set -e  # Exit on error

# Parse arguments
CHECKPOINT=$1
OUTPUT_DIR=$2

if [ -z "$CHECKPOINT" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash scripts/run_label_efficiency.sh <checkpoint_path> <output_dir>"
    exit 1
fi

echo "========================================="
echo "Label Efficiency Experiments"
echo "========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Data fractions to test
FRACTIONS=(0.01 0.05 0.10 1.0)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run linear probing experiments
echo "Running Linear Probing experiments..."
for frac in "${FRACTIONS[@]}"; do
    echo ""
    echo "--- Linear Probe: ${frac} data fraction ---"
    python linear_probe.py \
        --checkpoint "$CHECKPOINT" \
        --data_fraction "$frac" \
        --output_dir "${OUTPUT_DIR}/linear_probe_${frac}" \
        --epochs 100 \
        --lr 0.1 \
        --batch_size 256
done

# Run fine-tuning experiments
echo ""
echo "Running Fine-tuning experiments..."
for frac in "${FRACTIONS[@]}"; do
    echo ""
    echo "--- Fine-tune: ${frac} data fraction ---"
    python finetune.py \
        --checkpoint "$CHECKPOINT" \
        --data_fraction "$frac" \
        --output_dir "${OUTPUT_DIR}/finetune_${frac}" \
        --epochs 100 \
        --lr 1e-4 \
        --batch_size 128 \
        --layer_decay 0.65
done

# Aggregate results
echo ""
echo "Aggregating results..."
python scripts/aggregate_label_efficiency.py --results_dir "$OUTPUT_DIR"

echo ""
echo "Label efficiency experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
