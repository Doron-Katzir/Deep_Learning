#!/bin/bash

# Master Experiment Script
#
# This script runs the complete experimental pipeline for the MAE research project:
# 1. Pretraining with all masking strategies and ratios
# 2. Downstream evaluation (linear probe + fine-tuning)
# 3. Representation analysis (CKA, t-SNE, attention maps)
# 4. Result aggregation
#
# Usage:
#   bash scripts/run_all_experiments.sh
#
# Estimated time: 3-5 days on single GPU
# Recommended: Run in tmux/screen session

set -e

echo "========================================="
echo "MAE Structured Masking - Full Pipeline"
echo "========================================="
echo "Started at: $(date)"
echo ""

# Configuration
DATA_ROOT="./data/tiny-imagenet-200"
CHECKPOINT_ROOT="./checkpoints"
RESULTS_ROOT="./results"

# Check if TinyImageNet exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: TinyImageNet not found at $DATA_ROOT"
    echo "Please download and extract TinyImageNet first."
    echo "See: python -m datasets.tinyimagenet"
    exit 1
fi

# Create directories
mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT"

# =============================================================================
# PHASE 1: PRETRAINING
# =============================================================================
echo ""
echo "PHASE 1: PRETRAINING"
echo "===================="
echo ""

# Masking strategies to test
MASK_TYPES=("random" "grid" "saliency")
MASK_RATIOS=(0.5 0.75 0.9)

for mask_type in "${MASK_TYPES[@]}"; do
    for mask_ratio in "${MASK_RATIOS[@]}"; do
        exp_name="${mask_type}_mask_${mask_ratio}"
        echo "--- Training: $exp_name ---"

        # Create config dynamically (copy from baseline and modify)
        config_file="configs/pretrain_${exp_name}.yaml"

        # Train model
        python pretrain_mae.py \
            --config "$config_file" \
            --output_dir "${CHECKPOINT_ROOT}/${exp_name}"

        echo "Completed: $exp_name"
        echo ""
    done
done

echo "Pretraining phase completed!"

# =============================================================================
# PHASE 2: DOWNSTREAM EVALUATION
# =============================================================================
echo ""
echo "PHASE 2: DOWNSTREAM EVALUATION"
echo "==============================="
echo ""

for mask_type in "${MASK_TYPES[@]}"; do
    for mask_ratio in "${MASK_RATIOS[@]}"; do
        exp_name="${mask_type}_mask_${mask_ratio}"
        checkpoint="${CHECKPOINT_ROOT}/${exp_name}/checkpoint_best.pth"

        if [ ! -f "$checkpoint" ]; then
            echo "WARNING: Checkpoint not found: $checkpoint"
            continue
        fi

        echo "--- Evaluating: $exp_name ---"

        # Linear probing
        python linear_probe.py \
            --checkpoint "$checkpoint" \
            --data_fraction 1.0 \
            --output_dir "${RESULTS_ROOT}/${exp_name}/linear_probe" \
            --epochs 100

        # Fine-tuning
        python finetune.py \
            --checkpoint "$checkpoint" \
            --data_fraction 1.0 \
            --output_dir "${RESULTS_ROOT}/${exp_name}/finetune" \
            --epochs 100

        echo "Completed evaluation: $exp_name"
        echo ""
    done
done

echo "Downstream evaluation completed!"

# =============================================================================
# PHASE 3: LABEL EFFICIENCY (selected models)
# =============================================================================
echo ""
echo "PHASE 3: LABEL EFFICIENCY EXPERIMENTS"
echo "======================================"
echo ""

# Run label efficiency for baseline models (mask_ratio=0.75)
for mask_type in "${MASK_TYPES[@]}"; do
    exp_name="${mask_type}_mask_0.75"
    checkpoint="${CHECKPOINT_ROOT}/${exp_name}/checkpoint_best.pth"

    if [ ! -f "$checkpoint" ]; then
        echo "WARNING: Checkpoint not found: $checkpoint"
        continue
    fi

    echo "--- Label efficiency: $exp_name ---"
    bash scripts/run_label_efficiency.sh "$checkpoint" "${RESULTS_ROOT}/${exp_name}/label_efficiency"
    echo ""
done

echo "Label efficiency experiments completed!"

# =============================================================================
# PHASE 4: REPRESENTATION ANALYSIS
# =============================================================================
echo ""
echo "PHASE 4: REPRESENTATION ANALYSIS"
echo "================================="
echo ""

# CKA comparison (compare all models at mask_ratio=0.75)
echo "--- CKA Analysis ---"
checkpoints=()
labels=()
for mask_type in "${MASK_TYPES[@]}"; do
    exp_name="${mask_type}_mask_0.75"
    checkpoint="${CHECKPOINT_ROOT}/${exp_name}/checkpoint_best.pth"
    if [ -f "$checkpoint" ]; then
        checkpoints+=("$checkpoint")
        labels+=("$mask_type")
    fi
done

if [ ${#checkpoints[@]} -gt 1 ]; then
    python -m evaluation.cka \
        --checkpoints "${checkpoints[@]}" \
        --labels "${labels[@]}" \
        --output "${RESULTS_ROOT}/analysis/cka_comparison.png"
fi

# t-SNE visualizations
echo "--- t-SNE Visualizations ---"
for mask_type in "${MASK_TYPES[@]}"; do
    exp_name="${mask_type}_mask_0.75"
    checkpoint="${CHECKPOINT_ROOT}/${exp_name}/checkpoint_best.pth"

    if [ ! -f "$checkpoint" ]; then
        continue
    fi

    python -m evaluation.visualize_embeddings \
        --checkpoint "$checkpoint" \
        --method tsne \
        --output "${RESULTS_ROOT}/analysis/tsne_${mask_type}.png" \
        --num_samples 5000
done

echo "Representation analysis completed!"

# =============================================================================
# PHASE 5: RESULT AGGREGATION
# =============================================================================
echo ""
echo "PHASE 5: RESULT AGGREGATION"
echo "==========================="
echo ""

python scripts/aggregate_all_results.py --results_dir "$RESULTS_ROOT"

echo ""
echo "========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================="
echo "Results saved to: $RESULTS_ROOT"
echo "Completed at: $(date)"
