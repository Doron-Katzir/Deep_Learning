  #!/bin/bash

  echo "========== PHASE 1: PRETRAINING =========="
  caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml --mask_ratio 0.5
  caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml --mask_ratio 0.75
  caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml --mask_ratio 0.9

  echo "========== PHASE 2A: LINEAR PROBING =========="
  for ratio in 0.5 0.75 0.9; do
      python linear_probe.py \
          --checkpoint checkpoints/random_mask_${ratio}/checkpoint_best.pth \
          --output_dir results/random_${ratio}/linear_probe \
          --epochs 100
  done

  echo "========== PHASE 2B: FINE-TUNING =========="
  for ratio in 0.5 0.75 0.9; do
      python finetune.py \
          --checkpoint checkpoints/random_mask_${ratio}/checkpoint_best.pth \
          --output_dir results/random_${ratio}/finetune \
          --epochs 100
  done

  echo "========== PHASE 4: ANALYSIS =========="
  python -m evaluation.cka \
      --checkpoints \
          checkpoints/random_mask_0.5/checkpoint_best.pth \
          checkpoints/random_mask_0.75/checkpoint_best.pth \
          checkpoints/random_mask_0.9/checkpoint_best.pth \
      --labels "0.5" "0.75" "0.9" \
      --output results/analysis/cka_comparison.png

  python -m evaluation.visualize_embeddings \
      --checkpoint checkpoints/random_mask_0.75/checkpoint_best.pth \
      --method tsne \
      --output results/analysis/tsne.png

  echo "========== DONE! =========="
  python results_summary.py