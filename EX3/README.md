# Exercise 3 - Semi-Supervised Learning with M1 VAE

Implementation of the M1 scheme from "Semi-supervised Learning with Deep Generative Models" by Kingma et al., applied to Fashion MNIST and MNIST.

## Model Architecture
- **Latent dimension**: 50
- **Encoder**: 784 → 600 → 600 → 50 (softplus activation)
- **Decoder**: 50 → 600 → 600 → 784 (softplus activation)
- **SVM Kernel**: RBF (Radial Basis Function)
- **Optimizer**: RMSprop (lr=0.0003, momentum=0.1, alpha=0.999)
- **Weight initialization**: N(0, 0.001²)
- **Bernoulli sampling** during VAE training

## How to Train and Test

### Train VAE (Step 1)

Train VAE on Fashion MNIST:
```bash
python q3.py --mode train_vae --epochs 100 --dataset fashion_mnist --save_dir ./checkpoints_fashion
```

Train VAE on MNIST:
```bash
python q3.py --mode train_vae --epochs 100 --dataset mnist --save_dir ./checkpoints_mnist
```

### Run Experiments (Step 2)

Run all experiments (100, 600, 1000, 3000 labels) for Fashion MNIST:
```bash
python q3.py --mode run_all --vae_checkpoint ./checkpoints_fashion/vae_final.pt --dataset fashion_mnist --save_dir ./checkpoints_fashion
```

Run all experiments for MNIST:
```bash
python q3.py --mode run_all --vae_checkpoint ./checkpoints_mnist/vae_final.pt --dataset mnist --save_dir ./checkpoints_mnist
```

### Compare Both Datasets

Run comparison for both datasets:
```bash
python q3.py --mode compare --datasets both
```

Run comparison for Fashion MNIST only:
```bash
python q3.py --mode compare --datasets fashion
```

Run comparison for MNIST only:
```bash
python q3.py --mode compare --datasets mnist
```

## Output Files

**After VAE training:**
- `checkpoints_{dataset}/vae_final.pt` - Trained VAE model
- `checkpoints_{dataset}/training_curves.png` - Loss curves (log scale)
- `checkpoints_{dataset}/reconstructions_epoch_*.png` - Sample reconstructions

**After SVM training:**
- `checkpoints_{dataset}/svm_rbf_{n_labels}labels.pkl` - Trained SVM models
- `checkpoints_{dataset}/results_{n_labels}labels.txt` - Detailed results
- `checkpoints_{dataset}/all_results.csv` - Summary of all experiments

**After comparison:**
- `combined_results.csv` - Side-by-side comparison table

## Paper Baseline (MNIST with M1+TSVM)
- 100 labels: 11.82% error
- 600 labels: 5.72% error
- 1000 labels: 4.24% error
- 3000 labels: 3.49% error

## Requirements
```bash
pip install torch torchvision numpy scikit-learn pandas matplotlib tqdm
```
