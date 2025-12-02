# Exercise 3 - Semi-Supervised Learning with M1 VAE

Implementation of the M1 scheme from "Semi-supervised Learning with Deep Generative Models" by Kingma et al., applied to Fashion MNIST and MNIST.

## Implementation Details

### Model Architecture
- **Latent dimension**: 50
- **Encoder**: 784 → 600 → 600 → 50 (with softplus activation)
- **Decoder**: 50 → 600 → 600 → 784 (with softplus activation)
- **SVM Kernel**: RBF (Radial Basis Function)

### Key Features
- Weight initialization: N(0, 0.001²) as in paper
- Optimizer: RMSprop (lr=0.0003, momentum=0.1, alpha=0.999)
- Bernoulli sampling during VAE training
- Fixed seed (42) for reproducibility
- Equal samples per class for labeled data

## How to Train the Model

### Step 1: Train the VAE

Train the VAE on all data (labeled + unlabeled):

**Fashion MNIST (default):**
```bash
python q3.py --mode train_vae --epochs 50 --dataset fashion_mnist
```

**MNIST (for comparison with paper):**
```bash
python q3.py --mode train_vae --epochs 50 --dataset mnist
```

**Output:**
- Saved model: `checkpoints/vae_final.pt`
- Training curves: `checkpoints/training_curves.png`
- Reconstructions: `checkpoints/reconstructions_epoch_{10,20,30,40,50}.png`

### Step 2: Train SVM and Evaluate

After training the VAE, train SVM classifiers with different numbers of labeled samples.

**Important:** Use the same dataset that was used for VAE training!

#### Option A: Run all experiments at once

**Fashion MNIST:**
```bash
python q3.py --mode run_all --vae_checkpoint ./checkpoints/vae_final.pt --dataset fashion_mnist
```

**MNIST:**
```bash
python q3.py --mode run_all --vae_checkpoint ./checkpoints/vae_final.pt --dataset mnist
```

This will automatically run experiments with 100, 600, 1000, and 3000 labels and generate a comparison table.

#### Option B: Run individual experiments

**Fashion MNIST:**
```bash
python q3.py --mode train_svm --vae_checkpoint ./checkpoints/vae_final.pt --n_labels 100 --dataset fashion_mnist
python q3.py --mode train_svm --vae_checkpoint ./checkpoints/vae_final.pt --n_labels 600 --dataset fashion_mnist
python q3.py --mode train_svm --vae_checkpoint ./checkpoints/vae_final.pt --n_labels 1000 --dataset fashion_mnist
python q3.py --mode train_svm --vae_checkpoint ./checkpoints/vae_final.pt --n_labels 3000 --dataset fashion_mnist
```

**MNIST:**
```bash
python q3.py --mode train_svm --vae_checkpoint ./checkpoints/vae_final.pt --n_labels 100 --dataset mnist
python q3.py --mode train_svm --vae_checkpoint ./checkpoints/vae_final.pt --n_labels 600 --dataset mnist
python q3.py --mode train_svm --vae_checkpoint ./checkpoints/vae_final.pt --n_labels 1000 --dataset mnist
python q3.py --mode train_svm --vae_checkpoint ./checkpoints/vae_final.pt --n_labels 3000 --dataset mnist
```

## How to Test the Model

To test a trained model, the SVM training script automatically evaluates on the test set and reports:
- Training accuracy
- Test accuracy
- Test error rate
- Per-class classification report

Results are saved to:
- SVM model: `checkpoints/svm_rbf_{n_labels}labels.pkl`
- Results text file: `checkpoints/results_{n_labels}labels.txt`

## SVM Kernel

This implementation uses the **RBF (Radial Basis Function)** kernel for the SVM classifier, which is well-suited for the high-dimensional latent features extracted by the VAE.

## Requirements

```bash
pip install torch torchvision numpy scikit-learn pandas matplotlib tqdm
```

## File Structure

```
EX3/
├── q3.py                      # Main implementation (all-in-one)
├── compare_datasets.py        # Script to compare MNIST vs Fashion MNIST
├── README.md                  # This file
├── FashionMNIST/              # Fashion MNIST dataset (provided)
├── MNIST/                     # MNIST dataset (auto-downloaded)
├── checkpoints_mnist/         # MNIST models and results
│   ├── vae_final.pt
│   ├── svm_rbf_*labels.pkl
│   └── all_results.csv
├── checkpoints_fashion/       # Fashion MNIST models and results
│   ├── vae_final.pt
│   ├── svm_rbf_*labels.pkl
│   └── all_results.csv
└── combined_results.csv       # Combined comparison table
```

## Expected Results

### Paper Baseline (MNIST with M1+TSVM)
- 100 labels: 11.82% error
- 600 labels: 5.72% error
- 1000 labels: 4.24% error
- 3000 labels: 3.49% error

### Our Results

**MNIST with M1+SVM:** Expected to be slightly worse than paper (due to using regular SVM instead of transductive SVM)

**Fashion MNIST with M1+SVM:** Expected to be significantly worse than paper's MNIST results due to:
1. Fashion MNIST is more challenging than MNIST
2. Assignment uses regular SVM (paper uses transductive SVM)

Training with both datasets allows for better insight and comparison with the paper's results.

## Comparing Both Datasets

To get comprehensive results and compare both MNIST and Fashion MNIST side-by-side:

### Step 1: Train VAE models for both datasets

```bash
# Train VAE on MNIST
python q3.py --mode train_vae --epochs 50 --dataset mnist --save_dir ./checkpoints_mnist

# Train VAE on Fashion MNIST
python q3.py --mode train_vae --epochs 50 --dataset fashion_mnist --save_dir ./checkpoints_fashion
```

### Step 2: Run comparison script

```bash
python compare_datasets.py \
    --mnist_vae ./checkpoints_mnist/vae_final.pt \
    --fashion_vae ./checkpoints_fashion/vae_final.pt \
    --mnist_save_dir ./checkpoints_mnist \
    --fashion_save_dir ./checkpoints_fashion
```

This will:
1. Run all experiments (100, 600, 1000, 3000 labels) on both datasets
2. Generate a combined comparison table showing:
   - Paper results (MNIST with M1+TSVM)
   - Our MNIST results (M1+SVM)
   - Our Fashion MNIST results (M1+SVM)
3. Calculate average differences between methods
4. Save combined results to `combined_results.csv`

**Output example:**
```
N Labels     | Paper (M1+TSVM)    | MNIST (M1+SVM)     | Fashion MNIST (M1+SVM)
-------------|--------------------|--------------------|-----------------------
100          | 11.82              | 13.45              | 18.23
600          | 5.72               | 6.89               | 11.45
1000         | 4.24               | 5.12               | 9.87
3000         | 3.49               | 4.23               | 8.34
```

## Notes

- The VAE is trained on all 60,000 training images (unsupervised)
- Only the SVM uses labeled data (semi-supervised approach)
- Labeled samples are selected with equal representation from each class
- Results are deterministic when using the same seed (42)
- Training on both datasets provides better insight into model performance and dataset difficulty
