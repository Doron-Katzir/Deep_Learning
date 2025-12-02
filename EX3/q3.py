import os
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Device configuration (silent - will print in main)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class Encoder(nn.Module):
    """
    Encoder network q_φ(z|x)
    Maps input x to latent distribution parameters (μ, log σ²)
    """
    def __init__(self, input_dim=784, hidden_dim=600, latent_dim=50):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.softplus(self.fc1(x))
        h2 = F.softplus(self.fc2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network p_θ(x|z)
    Maps latent z back to input space
    """
    def __init__(self, latent_dim=50, hidden_dim=600, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h1 = F.softplus(self.fc1(z))
        h2 = F.softplus(self.fc2(h1))
        x_recon = torch.sigmoid(self.fc_out(h2))
        return x_recon


class M1_VAE(nn.Module):
    """
    Complete M1 VAE model
    """
    def __init__(self, input_dim=784, hidden_dim=600, latent_dim=50):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def encode(self, x):
        """Encode input to latent representation (using mean)"""
        mu, _ = self.encoder(x)
        return mu

    def loss_function(self, x_recon, x, mu, logvar):
        """VAE loss = Reconstruction loss + KL divergence"""
        x = x.view(x.size(0), -1)
        x_recon = x_recon.view(x_recon.size(0), -1)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div, recon_loss, kl_div


def init_weights(m):
    """
    Initialize weights as in Kingma et al. paper (Section 4.4):
    - Weights: N(0, 0.001²)
    - Biases: 0
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ============================================================================
# VAE TRAINING FUNCTIONS
# ============================================================================

def train_vae_epoch(model, train_loader, optimizer, epoch, use_bernoulli=True):
    """
    Train VAE for one epoch
    """
    model.train()
    train_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)

        # Bernoulli sampling as in paper
        if use_bernoulli:
            data = torch.bernoulli(data)

        optimizer.zero_grad()
        x_recon, mu, logvar, z = model(data)
        loss, recon_loss, kl_loss = model.loss_function(x_recon, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        recon_loss_total += recon_loss.item()
        kl_loss_total += kl_loss.item()

        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() / len(data):.2f}',
                'recon': f'{recon_loss.item() / len(data):.2f}',
                'kl': f'{kl_loss.item() / len(data):.2f}'
            })

    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_loss_total / len(train_loader.dataset)
    avg_kl = kl_loss_total / len(train_loader.dataset)
    return avg_loss, avg_recon, avg_kl


def test_vae_epoch(model, test_loader):
    """Evaluate VAE on test set"""
    model.eval()
    test_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            x_recon, mu, logvar, z = model(data)
            loss, recon_loss, kl_loss = model.loss_function(x_recon, data, mu, logvar)
            test_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()

    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon = recon_loss_total / len(test_loader.dataset)
    avg_kl = kl_loss_total / len(test_loader.dataset)
    return avg_loss, avg_recon, avg_kl


def save_reconstructions(model, test_loader, save_path, epoch):
    """Save reconstruction examples"""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:8].to(device)
        x_recon, _, _, _ = model(data)

        fig, axes = plt.subplots(2, 8, figsize=(12, 3))
        for i in range(8):
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', rotation=0, labelpad=40)

            axes[1, i].imshow(x_recon[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Reconstructed', rotation=0, labelpad=40)

        plt.suptitle(f'Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def train_vae(args):
    """Main function to train VAE"""
    dataset_name = "MNIST" if args.dataset == "mnist" else "Fashion MNIST"
    print("="*70)
    print(f"Training M1 VAE on {dataset_name}")
    print("="*70)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    transform = transforms.Compose([transforms.ToTensor()])

    dataset_class = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    train_dataset = dataset_class(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(
        root=args.data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Initialize model
    model = M1_VAE(
        input_dim=784, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim
    ).to(device)

    model.apply(init_weights)
    print("Applied paper's weight initialization: N(0, 0.001²)")

    print(f"\nModel architecture:")
    print(f"Input dim: 784")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.999, momentum=0.1)
    print(f"Using RMSProp optimizer (lr={args.lr}, alpha=0.999, momentum=0.1)")

    # Training loop
    train_losses = []
    test_losses = []

    print("\nStarting training...")
    print(f"Using Bernoulli sampling: {args.use_bernoulli}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon, train_kl = train_vae_epoch(model, train_loader, optimizer, epoch, args.use_bernoulli)
        test_loss, test_recon, test_kl = test_vae_epoch(model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.2f} (Recon: {train_recon:.2f}, KL: {train_kl:.2f}) | '
              f'Test Loss: {test_loss:.2f} (Recon: {test_recon:.2f}, KL: {test_kl:.2f})')

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'vae_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')

            recon_path = os.path.join(args.save_dir, f'reconstructions_epoch_{epoch}.png')
            save_reconstructions(model, test_loader, recon_path, epoch)

    # Save final model
    final_model_path = os.path.join(args.save_dir, 'vae_final.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f'\nSaved final model: {final_model_path}')

    # Plot training curves with log scale
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('VAE Training Curves')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
    plt.close()
    print(f'Saved training curves')


# ============================================================================
# SVM TRAINING FUNCTIONS
# ============================================================================

def select_labeled_samples(dataset, n_labels, seed=42):
    """Select n_labels samples with equal representation from each class"""
    np.random.seed(seed)
    n_classes = 10
    samples_per_class = n_labels // n_classes

    class_indices = [[] for _ in range(n_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected_indices = []
    for class_idx in range(n_classes):
        class_samples = np.random.choice(
            class_indices[class_idx], size=samples_per_class, replace=False
        )
        selected_indices.extend(class_samples)

    np.random.shuffle(selected_indices)
    return selected_indices


def extract_features(model, data_loader, desc="Extracting features"):
    """Extract latent features from data using trained VAE encoder"""
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc=desc):
            data = data.to(device)
            features = model.encode(data)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    return features, labels


def train_svm(args):
    """Train SVM on VAE latent features"""
    dataset_name = "MNIST" if args.dataset == "mnist" else "Fashion MNIST"
    print("="*70)
    print(f"Training SVM with {args.n_labels} labeled samples on {dataset_name}")
    print("="*70)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Kernel: {args.kernel}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_class = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    train_dataset = dataset_class(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(
        root=args.data_dir, train=False, download=True, transform=transform
    )

    # Select labeled samples
    print(f"\nSelecting {args.n_labels} labeled samples (seed={args.seed})...")
    labeled_indices = select_labeled_samples(train_dataset, args.n_labels, seed=args.seed)
    labeled_dataset = Subset(train_dataset, labeled_indices)

    print(f"Labeled training samples: {len(labeled_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    labeled_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Load VAE model
    print(f"\nLoading VAE model from {args.vae_checkpoint}...")
    model = M1_VAE(
        input_dim=784, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim
    ).to(device)
    model.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
    model.eval()
    print("VAE model loaded successfully")

    # Extract features
    print("\nExtracting features from labeled training data...")
    train_features, train_labels = extract_features(model, labeled_loader, "Labeled training set")
    print(f"Extracted features shape: {train_features.shape}")

    print("\nExtracting features from test data...")
    test_features, test_labels = extract_features(model, test_loader, "Test set")
    print(f"Test features shape: {test_features.shape}")

    # Train SVM
    print(f"\nTraining SVM with {args.kernel} kernel (C={args.C})...")
    svm = SVC(kernel=args.kernel, C=args.C, random_state=args.seed, verbose=True)
    svm.fit(train_features, train_labels)
    print("SVM training completed")

    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_predictions = svm.predict(train_features)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")

    print("\nEvaluating on test set...")
    test_predictions = svm.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_error = (1 - test_accuracy) * 100

    print(f"\n{'='*60}")
    print(f"Results for {args.n_labels} labeled samples:")
    print(f"{'='*60}")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test error rate: {test_error:.2f}%")
    print(f"{'='*60}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions,
                                target_names=['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']))

    # Save SVM model
    os.makedirs(args.save_dir, exist_ok=True)
    svm_path = os.path.join(args.save_dir, f'svm_{args.kernel}_{args.n_labels}labels.pkl')
    with open(svm_path, 'wb') as f:
        pickle.dump(svm, f)
    print(f"\nSaved SVM model: {svm_path}")

    # Save results
    results_path = os.path.join(args.save_dir, f'results_{args.n_labels}labels.txt')
    with open(results_path, 'w') as f:
        f.write(f"M1 VAE + SVM Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Number of labeled samples: {args.n_labels}\n")
        f.write(f"SVM kernel: {args.kernel}\n")
        f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n")
        f.write(f"Test error rate: {test_error:.2f}%\n")
    print(f"Saved results: {results_path}")

    return test_error


# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================

def run_all_experiments(args):
    """Run all experiments with 100, 600, 1000, 3000 labels"""
    if not os.path.exists(args.vae_checkpoint):
        print(f"Error: VAE checkpoint not found at {args.vae_checkpoint}")
        print("Please train the VAE first using: python q3.py --mode train_vae")
        return

    dataset_name = "MNIST" if args.dataset == "mnist" else "Fashion MNIST"
    print("="*70)
    print("M1 VAE Semi-Supervised Learning Experiments")
    print(f"Dataset: {dataset_name}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"SVM kernel: {args.kernel}")
    print("="*70)

    n_labels_list = [100, 600, 1000, 3000]
    results = {}

    # Temporarily save original n_labels
    original_n_labels = getattr(args, 'n_labels', None)

    for n_labels in n_labels_list:
        args.n_labels = n_labels
        try:
            error_rate = train_svm(args)
            results[n_labels] = error_rate
        except Exception as e:
            print(f"Error running experiment with {n_labels} labels: {e}")
            results[n_labels] = None

    # Restore original n_labels
    if original_n_labels is not None:
        args.n_labels = original_n_labels

    # Generate results table
    print("\n" + "="*70)
    print(f"FINAL RESULTS - {dataset_name} (M1 VAE + SVM)")
    print("="*70)
    print(f"{'Number of Labels':<20} | {'Test Error Rate (%)':<20}")
    print("-"*70)
    for n_labels, error_rate in results.items():
        if error_rate is not None:
            print(f"{n_labels:<20} | {error_rate:<20.2f}")
        else:
            print(f"{n_labels:<20} | {'N/A':<20}")
    print("="*70)

    # Save results to CSV
    results_df = pd.DataFrame([
        {'n_labels': n, 'test_error_rate': err}
        for n, err in results.items()
    ])
    csv_path = os.path.join(args.save_dir, 'all_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Comparison with paper
    print("\n" + "="*70)
    print("COMPARISON WITH PAPER (Table 1)")
    print("="*70)
    our_label = f"Ours ({dataset_name})"
    print(f"{'Labels':<10} | {'Paper (MNIST M1+TSVM)':<25} | {our_label:<25}")
    print("-"*70)

    paper_results = {100: 11.82, 600: 5.72, 1000: 4.24, 3000: 3.49}

    for n_labels in n_labels_list:
        paper_err = paper_results.get(n_labels, 'N/A')
        our_err = results.get(n_labels, 'N/A')
        if isinstance(our_err, float):
            print(f"{n_labels:<10} | {paper_err:<20} | {our_err:<25.2f}")
        else:
            print(f"{n_labels:<10} | {paper_err:<20} | {our_err:<25}")
    print("="*70)
    print("Note: Paper uses MNIST with transductive SVM (M1+TSVM)")
    print(f"      Our implementation uses {dataset_name} with regular SVM (M1)")
    print("="*70)


# ============================================================================
# COMPARE DATASETS
# ============================================================================

def compare_datasets(args):
    """Compare results between different datasets"""
    import subprocess

    # Determine which datasets to run
    datasets_to_run = []
    if args.datasets in ['fashion', 'both']:
        datasets_to_run.append(('fashion_mnist', args.fashion_vae, args.fashion_save_dir, 'Fashion MNIST'))
    if args.datasets in ['mnist', 'both']:
        datasets_to_run.append(('mnist', args.mnist_vae, args.mnist_save_dir, 'MNIST'))

    print("="*70)
    if args.datasets == 'both':
        print("M1 VAE COMPARISON: MNIST vs Fashion MNIST")
    elif args.datasets == 'mnist':
        print("M1 VAE: MNIST")
    else:
        print("M1 VAE: Fashion MNIST")
    print("="*70)

    # Check if VAE checkpoints exist
    missing_checkpoints = []
    for dataset, vae_path, save_dir, name in datasets_to_run:
        if not os.path.exists(vae_path):
            missing_checkpoints.append(f"{name} VAE: {vae_path}")

    if missing_checkpoints:
        print("\n⚠️  Missing VAE checkpoints:")
        for cp in missing_checkpoints:
            print(f"  - {cp}")
        print("\nPlease train the VAE models first:")
        for dataset, vae_path, save_dir, name in datasets_to_run:
            if not os.path.exists(vae_path):
                print(f"  python q3.py --mode train_vae --epochs 50 --dataset {dataset} --save_dir {save_dir}")
        return

    # Run experiments on selected datasets
    all_results = {}
    for dataset, vae_path, save_dir, name in datasets_to_run:
        print(f"\n{'='*70}")
        print(f"Running experiments on {name}")
        print(f"{'='*70}\n")

        cmd = [
            'python', 'q3.py',
            '--mode', 'run_all',
            '--vae_checkpoint', vae_path,
            '--dataset', dataset,
            '--save_dir', save_dir
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        if result.returncode != 0:
            print(f"\n⚠️  Experiments failed for {name}. Please check the output above.")
            return

        # Extract results
        csv_path = os.path.join(save_dir, 'all_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results = {}
            for _, row in df.iterrows():
                results[int(row['n_labels'])] = float(row['test_error_rate'])
            all_results[name] = results
        else:
            print(f"\n⚠️  Could not load results for {name}. Please check if experiments completed successfully.")
            return

    # Paper results (from Table 1)
    paper_results = {100: 11.82, 600: 5.72, 1000: 4.24, 3000: 3.49}
    n_labels_list = [100, 600, 1000, 3000]

    # Generate results table
    print("\n" + "="*90)
    if args.datasets == 'both':
        print("COMBINED RESULTS TABLE - MNIST vs Fashion MNIST")
        print("="*90)
        print(f"{'N Labels':<12} | {'Paper (M1+TSVM)':<18} | {'MNIST (M1+SVM)':<18} | {'Fashion MNIST (M1+SVM)':<22}")
    elif args.datasets == 'mnist':
        print("RESULTS TABLE - MNIST")
        print("="*90)
        print(f"{'N Labels':<12} | {'Paper (M1+TSVM)':<18} | {'MNIST (M1+SVM)':<18}")
    else:
        print("RESULTS TABLE - Fashion MNIST")
        print("="*90)
        print(f"{'N Labels':<12} | {'Paper (M1+TSVM)':<18} | {'Fashion MNIST (M1+SVM)':<22}")
    print("-"*90)

    for n_labels in n_labels_list:
        paper = paper_results.get(n_labels, float('nan'))
        line = f"{n_labels:<12} | {paper:<18.2f}"

        if 'MNIST' in all_results:
            mnist = all_results['MNIST'].get(n_labels, float('nan'))
            line += f" | {mnist:<18.2f}"

        if 'Fashion MNIST' in all_results:
            fashion = all_results['Fashion MNIST'].get(n_labels, float('nan'))
            line += f" | {fashion:<22.2f}"

        print(line)

    print("="*90)

    # Calculate average differences
    if args.datasets == 'both' and 'MNIST' in all_results and 'Fashion MNIST' in all_results:
        print("\nAnalysis:")
        print("-"*90)

        mnist_results = all_results['MNIST']
        fashion_results = all_results['Fashion MNIST']

        # MNIST vs Paper
        mnist_diffs = [mnist_results.get(n, 0) - paper_results.get(n, 0) for n in n_labels_list]
        avg_mnist_diff = sum(mnist_diffs) / len(mnist_diffs)
        print(f"MNIST (M1+SVM) vs Paper (M1+TSVM):    {avg_mnist_diff:+.2f}% average difference")

        # Fashion MNIST vs MNIST
        fashion_diffs = [fashion_results.get(n, 0) - mnist_results.get(n, 0) for n in n_labels_list]
        avg_fashion_diff = sum(fashion_diffs) / len(fashion_diffs)
        print(f"Fashion MNIST vs MNIST:               {avg_fashion_diff:+.2f}% average difference")

        # Fashion MNIST vs Paper
        fashion_paper_diffs = [fashion_results.get(n, 0) - paper_results.get(n, 0) for n in n_labels_list]
        avg_fashion_paper_diff = sum(fashion_paper_diffs) / len(fashion_paper_diffs)
        print(f"Fashion MNIST vs Paper:               {avg_fashion_paper_diff:+.2f}% average difference")

        print("-"*90)
    elif args.datasets == 'mnist' and 'MNIST' in all_results:
        print("\nAnalysis:")
        print("-"*90)
        mnist_results = all_results['MNIST']
        mnist_diffs = [mnist_results.get(n, 0) - paper_results.get(n, 0) for n in n_labels_list]
        avg_mnist_diff = sum(mnist_diffs) / len(mnist_diffs)
        print(f"MNIST (M1+SVM) vs Paper (M1+TSVM):    {avg_mnist_diff:+.2f}% average difference")
        print("-"*90)
    elif args.datasets == 'fashion' and 'Fashion MNIST' in all_results:
        print("\nAnalysis:")
        print("-"*90)
        fashion_results = all_results['Fashion MNIST']
        fashion_paper_diffs = [fashion_results.get(n, 0) - paper_results.get(n, 0) for n in n_labels_list]
        avg_fashion_paper_diff = sum(fashion_paper_diffs) / len(fashion_paper_diffs)
        print(f"Fashion MNIST vs Paper:               {avg_fashion_paper_diff:+.2f}% average difference")
        print("-"*90)

    # Save results
    if args.datasets == 'both' and 'MNIST' in all_results and 'Fashion MNIST' in all_results:
        combined_df = pd.DataFrame({
            'n_labels': n_labels_list,
            'paper_m1_tsvm': [paper_results[n] for n in n_labels_list],
            'mnist_m1_svm': [all_results['MNIST'][n] for n in n_labels_list],
            'fashion_mnist_m1_svm': [all_results['Fashion MNIST'][n] for n in n_labels_list]
        })
        combined_csv = 'combined_results.csv'
        combined_df.to_csv(combined_csv, index=False)
        print(f"\nCombined results saved to: {combined_csv}")

    print("\nNotes:")
    print("  - Paper uses MNIST with transductive SVM (M1+TSVM)")
    print("  - Our implementation uses regular SVM (M1+SVM)")
    if args.datasets == 'both':
        print("  - Fashion MNIST is more challenging than MNIST")
    print("="*90)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Question 3: M1 VAE Semi-Supervised Learning')

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train_vae', 'train_svm', 'run_all', 'compare'],
                        help='Mode: train_vae, train_svm, run_all, or compare')

    # VAE training parameters
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs for VAE training')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate (paper uses 0.0003)')
    parser.add_argument('--latent_dim', type=int, default=50, help='latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=600, help='hidden layer dimension')
    parser.add_argument('--use_bernoulli', action='store_true', default=True,
                        help='use Bernoulli sampling during training (as in paper)')
    parser.add_argument('--no_bernoulli', dest='use_bernoulli', action='store_false',
                        help='disable Bernoulli sampling')

    # SVM training parameters
    parser.add_argument('--vae_checkpoint', type=str, default='./checkpoints/vae_final.pt',
                        help='path to trained VAE model')
    parser.add_argument('--n_labels', type=int, choices=[100, 600, 1000, 3000],
                        help='number of labeled samples (for train_svm mode)')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf', 'poly'],
                        help='SVM kernel')
    parser.add_argument('--C', type=float, default=1.0, help='SVM regularization parameter')

    # General parameters
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='dataset to use (mnist or fashion_mnist)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='directory to save models')
    parser.add_argument('--data_dir', type=str, default='.', help='directory for dataset (contains MNIST/FashionMNIST folder)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # Compare mode parameters
    parser.add_argument('--datasets', type=str, default='both', choices=['fashion', 'mnist', 'both'],
                        help='Which dataset(s) to run in compare mode: fashion, mnist, or both (default: both)')
    parser.add_argument('--mnist_vae', type=str, default='./checkpoints_mnist/vae_final.pt',
                        help='Path to MNIST VAE checkpoint (for compare mode)')
    parser.add_argument('--fashion_vae', type=str, default='./checkpoints/vae_final.pt',
                        help='Path to Fashion MNIST VAE checkpoint (for compare mode)')
    parser.add_argument('--mnist_save_dir', type=str, default='./checkpoints_mnist',
                        help='Save directory for MNIST results (for compare mode)')
    parser.add_argument('--fashion_save_dir', type=str, default='./checkpoints',
                        help='Save directory for Fashion MNIST results (for compare mode)')

    args = parser.parse_args()

    # Print device info once
    device_name = str(device).upper().replace("MPS", "MPS (Apple Silicon)").replace("CUDA", "CUDA").replace("CPU", "CPU")
    print(f"Using device: {device}")

    # Execute based on mode
    if args.mode == 'train_vae':
        train_vae(args)
    elif args.mode == 'train_svm':
        if args.n_labels is None:
            print("Error: --n_labels is required for train_svm mode")
            print("Choose from: 100, 600, 1000, 3000")
            return
        train_svm(args)
    elif args.mode == 'run_all':
        run_all_experiments(args)
    elif args.mode == 'compare':
        compare_datasets(args)


if __name__ == '__main__':
    main()
