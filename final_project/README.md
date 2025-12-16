# Masked Autoencoders with Structured Masking

**A Master's-Level Research Project on MAE Representation Learning**

This repository reproduces and extends Masked Autoencoders (MAE) [He et al., CVPR 2022] to study how the **structure and ratio of patch masking** affect representation quality and label efficiency under limited compute constraints.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Research Questions](#research-questions)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Analysis](#analysis)
- [Reproducing Results](#reproducing-results)
- [Key Design Choices](#key-design-choices)
- [Results Summary](#results-summary)
- [Citation](#citation)

---

## 🎯 Project Overview

### Motivation
Masked Autoencoders have shown remarkable success in self-supervised learning, but most research focuses on **random masking**. This project investigates whether **structured masking patterns** can improve representation learning efficiency.

### Key Constraints
- **Single GPU**: All experiments designed for 24GB VRAM (e.g., RTX 3090)
- **One semester timeline**: ~100-150 epochs pretraining
- **Model**: ViT-Tiny (5.7M parameters)
- **Pretraining**: TinyImageNet (200 classes, 100K images)
- **Evaluation**: CIFAR-100 (100 classes)

### Masking Strategies Studied

| Strategy | Description | Hypothesis |
|----------|-------------|------------|
| **Random** | MAE baseline - uniform random masking | Strong baseline from original paper |
| **Grid** | Deterministic periodic patterns (checkerboard, stripes) | Tests spatial inductive bias |
| **Saliency** | Content-aware masking via edge detection | Harder reconstruction → better features |

### Mask Ratios Tested
- **0.5** (50% masked)
- **0.75** (75% masked) - MAE default
- **0.9** (90% masked)

---

## 🔬 Research Questions

1. **How does masking structure affect representation quality?**
   - Measured via linear probing and fine-tuning accuracy
   - Analyzed with CKA similarity across models

2. **Which masking strategy is most label-efficient?**
   - Evaluated at 1%, 5%, 10%, 100% of training labels

3. **What is the optimal mask ratio for each strategy?**
   - Trade-off between task difficulty and information preservation

---

## 📁 Repository Structure

```
final_project/
├── configs/                  # YAML configuration files
│   ├── pretrain_baseline.yaml
│   ├── pretrain_grid.yaml
│   └── pretrain_saliency.yaml
├── datasets/                 # Dataset loaders
│   └── tinyimagenet.py
├── models/                   # Model architectures
│   └── mae_vit.py           # ViT-Tiny MAE implementation
├── masking/                  # Masking strategies
│   ├── mask_generator.py    # Base interface
│   ├── random_mask.py
│   ├── grid_mask.py
│   ├── saliency_mask.py
│   └── visualize.py         # Mask visualization
├── evaluation/               # Analysis tools
│   ├── cka.py               # Centered Kernel Alignment
│   ├── visualize_embeddings.py  # t-SNE/PCA
│   └── visualize_attention.py   # Attention maps
├── scripts/                  # Experiment orchestration
│   ├── run_all_experiments.sh
│   ├── run_label_efficiency.sh
│   ├── aggregate_label_efficiency.py
│   └── aggregate_all_results.py
├── pretrain_mae.py          # Pretraining script
├── linear_probe.py          # Linear evaluation
├── finetune.py              # Fine-tuning evaluation
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## 🔧 Installation

### Prerequisites
- Python 3.11+
- CUDA 11.8+ (for GPU support)
- 24GB VRAM GPU recommended

### Setup

```bash
# Clone repository
cd final_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📦 Dataset Setup

### TinyImageNet

```bash
# Download TinyImageNet
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

# Extract
unzip tiny-imagenet-200.zip -d ./data/

# Verify structure
python -m datasets.tinyimagenet
```

Expected structure:
```
data/tiny-imagenet-200/
├── train/
│   ├── n01443537/
│   │   └── images/
│   └── ...
├── val/
│   ├── images/
│   └── val_annotations.txt
└── wnids.txt
```

### CIFAR-100
CIFAR-100 will be downloaded automatically on first use.

---

## 🚀 Usage

### Training

#### Single Experiment

```bash
# Random masking (baseline)
python pretrain_mae.py --config configs/pretrain_baseline.yaml

# Grid masking
python pretrain_mae.py --config configs/pretrain_grid.yaml

# Saliency-guided masking
python pretrain_mae.py --config configs/pretrain_saliency.yaml
```

#### Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir logs/

# Check GPU usage
watch -n 1 nvidia-smi
```

### Evaluation

#### Linear Probing

```bash
python linear_probe.py \
    --checkpoint checkpoints/random_mask_0.75/checkpoint_best.pth \
    --data_fraction 1.0 \
    --output_dir results/linear_probe
```

#### Fine-tuning

```bash
python finetune.py \
    --checkpoint checkpoints/random_mask_0.75/checkpoint_best.pth \
    --data_fraction 1.0 \
    --output_dir results/finetune
```

#### Label Efficiency

```bash
bash scripts/run_label_efficiency.sh \
    checkpoints/random_mask_0.75/checkpoint_best.pth \
    results/label_efficiency
```

### Analysis

#### CKA Similarity

```bash
python -m evaluation.cka \
    --checkpoints \
        checkpoints/random_mask_0.75/checkpoint_best.pth \
        checkpoints/grid_mask_0.75/checkpoint_best.pth \
        checkpoints/saliency_mask_0.75/checkpoint_best.pth \
    --labels Random Grid Saliency \
    --output results/analysis/cka_comparison.png
```

#### t-SNE Visualization

```bash
python -m evaluation.visualize_embeddings \
    --checkpoint checkpoints/random_mask_0.75/checkpoint_best.pth \
    --method tsne \
    --output results/analysis/tsne.png \
    --num_samples 5000
```

#### Attention Maps

```bash
python -m evaluation.visualize_attention \
    --checkpoint checkpoints/random_mask_0.75/checkpoint_best.pth \
    --image_path data/sample_image.jpg \
    --output results/analysis/attention.png
```

---

## 🔄 Reproducing Results

### Full Pipeline

Run all experiments end-to-end:

```bash
bash scripts/run_all_experiments.sh
```

This will:
1. Train 9 models (3 strategies × 3 mask ratios) - **~3-5 days**
2. Run downstream evaluation (linear probe + fine-tuning)
3. Conduct label efficiency experiments
4. Perform representation analysis (CKA, t-SNE)
5. Aggregate and plot all results

### Quick Test Run

For testing the pipeline on limited resources:

```bash
# Train for 10 epochs only
python pretrain_mae.py \
    --config configs/pretrain_baseline.yaml \
    --output_dir checkpoints/test_run

# Override config
# Edit configs/pretrain_baseline.yaml: set epochs to 10
```

---

## 🧠 Key Design Choices

### Architecture: ViT-Tiny

| Component | Specification |
|-----------|---------------|
| Embed dim | 192 |
| Depth | 12 layers |
| Heads | 3 |
| Decoder dim | 512 |
| Decoder depth | 4 layers |
| **Total params** | **5.7M** |

**Rationale**: Smallest ViT variant that still demonstrates representation learning, fits on single GPU with reasonable batch size.

### Masking Implementation

All masking strategies:
- Preserve **exact mask ratio** for fair comparison
- Are **GPU-accelerated** (no CPU bottleneck)
- Support **batch-wise generation**
- Are **reproducible** with fixed seed

### Training Protocol

- **Optimizer**: AdamW
- **LR**: 1.5e-4 (base for batch size 128)
- **Schedule**: Cosine decay with warmup
- **Mixed precision**: FP16 for memory efficiency
- **Normalization**: Normalize pixel targets per patch

### Evaluation Protocol

**Linear Probing**:
- Freeze encoder, train linear head only
- SGD optimizer, cosine LR schedule
- 100 epochs

**Fine-tuning**:
- Update all weights
- Layer-wise LR decay (0.65)
- AdamW optimizer
- 100 epochs

---

## 📊 Results Summary

*(Results will be populated after running experiments)*

### Main Results

| Masking Strategy | Mask Ratio | Linear Probe | Fine-tuning |
|------------------|------------|--------------|-------------|
| Random | 0.75 | TBD | TBD |
| Grid | 0.75 | TBD | TBD |
| Saliency | 0.75 | TBD | TBD |

### Label Efficiency

*(Curve showing accuracy vs. data fraction)*

### Representation Similarity (CKA)

*(Heatmap showing inter-model CKA scores)*

---

## 📝 Configuration

All experiments are configured via YAML files. Key parameters:

```yaml
# Example: configs/pretrain_baseline.yaml

masking:
  type: random          # random, grid, saliency
  mask_ratio: 0.75      # 0.5, 0.75, 0.9

training:
  epochs: 100
  lr: 1.5e-4
  batch_size: 128
  use_amp: true         # Mixed precision

data:
  data_path: ./data/tiny-imagenet-200
```

---

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size in config
batch_size: 64  # instead of 128

# Or use gradient accumulation (modify pretrain_mae.py)
```

### TinyImageNet Not Found

```bash
# Verify download
ls data/tiny-imagenet-200/

# Re-download if corrupted
rm -rf data/tiny-imagenet-200
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ./data/
```

### Checkpoint Not Loading

```bash
# Check checkpoint exists
ls checkpoints/*/checkpoint_best.pth

# Verify checkpoint format
python -c "import torch; print(torch.load('path/to/checkpoint.pth').keys())"
```

---

## 🧪 Testing Mask Generators

Run sanity checks on all masking strategies:

```bash
python -m masking.visualize
```

This will:
- Verify mask ratios are exact
- Check binary values (0 or 1)
- Test reproducibility
- Visualize mask patterns

---

## 📚 References

```bibtex
@inproceedings{he2022mae,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{kornblith2019cka,
  title={Similarity of Neural Network Representations Revisited},
  author={Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
  booktitle={ICML},
  year={2019}
}
```

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email@university.edu]

---

## ⚖️ License

This project is for academic research purposes.

---

## 🙏 Acknowledgments

- Original MAE paper and code by Facebook AI Research
- TinyImageNet dataset from Stanford CS231n
- CIFAR-100 dataset from the University of Toronto

---

**Built with academic rigor and research best practices.**
