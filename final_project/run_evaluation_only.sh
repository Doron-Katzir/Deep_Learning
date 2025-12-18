#!/bin/bash

# Evaluation Pipeline (Skip Pretraining)
# Use this when you already have trained checkpoints

echo "========================================="
echo "EVALUATION PIPELINE (No Retraining)"
echo "========================================="
echo "Using existing checkpoints in checkpoints/"
echo ""

# Check if checkpoints exist
if [ ! -f "checkpoints/random_mask_0.5/checkpoint_best.pth" ]; then
    echo "ERROR: Checkpoint not found for mask_ratio 0.5"
    echo "Please train models first with pretrain_mae.py"
    exit 1
fi

echo "========== PHASE 2A: LINEAR PROBING =========="
for ratio in 0.5 0.75 0.9; do
    echo ""
    echo "--- Linear Probing: mask_ratio ${ratio} ---"
    python linear_probe.py \
        --checkpoint checkpoints/random_mask_${ratio}/checkpoint_best.pth \
        --output_dir results/random_${ratio}/linear_probe \
        --epochs 100 \
        --batch_size 256
done

echo ""
echo "========== PHASE 2B: FINE-TUNING =========="
for ratio in 0.5 0.75 0.9; do
    echo ""
    echo "--- Fine-tuning: mask_ratio ${ratio} ---"
    python finetune.py \
        --checkpoint checkpoints/random_mask_${ratio}/checkpoint_best.pth \
        --output_dir results/random_${ratio}/finetune \
        --epochs 100 \
        --batch_size 128
done

echo ""
echo "========== PHASE 3: ANALYSIS =========="

# Create analysis directory
mkdir -p results/analysis

# CKA comparison
echo "--- CKA Analysis ---"
python -m evaluation.cka \
    --checkpoints \
        checkpoints/random_mask_0.5/checkpoint_best.pth \
        checkpoints/random_mask_0.75/checkpoint_best.pth \
        checkpoints/random_mask_0.9/checkpoint_best.pth \
    --labels "0.5" "0.75" "0.9" \
    --output results/analysis/cka_comparison.png

# t-SNE visualization
echo "--- t-SNE Visualization ---"
python -m evaluation.visualize_embeddings \
    --checkpoint checkpoints/random_mask_0.75/checkpoint_best.pth \
    --method tsne \
    --output results/analysis/tsne.png \
    --num_samples 5000

echo ""
echo "========== RESULTS SUMMARY =========="
python results_summary.py

echo ""
echo "========================================="
echo "EVALUATION COMPLETED!"
echo "========================================="
echo "Results saved to: results/"
echo "Figures saved to: results/analysis/"
