# Deep Learning - Exercise 1 (Practical Part)
# Author: Daniel Toberman
# Description: Implementation and comparison of LeNet-5 variants on FashionMNIST with TensorBoard logging

import os
import urllib.request
import gzip
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ------------------------------------------------------------
# 1. Device setup (MPS → CUDA → CPU)
# ------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Metal) backend")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

# ------------------------------------------------------------
# 2. Download Fashion-MNIST from GitHub (Zalando Research)
# ------------------------------------------------------------
def download_fashion_mnist_from_github(root="./data/fashion_mnist"):
    base_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = {
        "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte",
    }
    os.makedirs(root, exist_ok=True)
    for fname_gz, fname in files.items():
        out_path = os.path.join(root, fname)
        if not os.path.exists(out_path):
            print(f"Downloading {fname_gz} from Zalando GitHub...")
            url = base_url + fname_gz
            urllib.request.urlretrieve(url, out_path + ".gz")
            with gzip.open(out_path + ".gz", "rb") as f_in:
                with open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(out_path + ".gz")

# download data
print("Checking dataset...")
download_fashion_mnist_from_github()

# ------------------------------------------------------------
# 3. Dataset (FashionMNIST)
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# ------------------------------------------------------------
# 4. LeNet-5 architecture (MNIST-suited variant)
# ------------------------------------------------------------
class LeNet5(nn.Module):
    """LeNet-5 variant adapted for MNIST/FashionMNIST (1×28×28 input)."""
    def __init__(self, dropout=False, batchnorm=False):
        super().__init__()
        layers = [
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6) if batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        ]
        self.features = nn.Sequential(*layers)

        fc_layers = [
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Linear(84, 10)
        ]
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------
# 5. Training & evaluation utilities
# ------------------------------------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum / total, 100 * correct / total


def train_model(name, model, trainloader, testloader, optimizer, criterion, epochs=15):
    writer = SummaryWriter(log_dir=f"runs/lenet5_{name.lower().replace(' ', '_')}")
    train_acc_list, test_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        _, test_acc = evaluate(model, testloader, criterion)

        writer.add_scalar('Accuracy/Train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/Test', test_acc, epoch+1)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"[{name}] Epoch {epoch+1:02d}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    writer.close()
    return train_acc_list, test_acc_list

# ------------------------------------------------------------
# 6. Run experiments
# ------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
results = {}

configs = [
    ("Baseline", LeNet5(), optim.Adam),
    ("Dropout", LeNet5(dropout=True), optim.Adam),
    ("Weight Decay", LeNet5(), lambda params, lr: optim.Adam(params, lr=lr, weight_decay=1e-4)),
    ("BatchNorm", LeNet5(batchnorm=True), optim.Adam)
]

for name, model, opt_fn in configs:
    model = model.to(device)
    optimizer = opt_fn(model.parameters(), lr=1e-3) if callable(opt_fn) else opt_fn(model.parameters(), lr=1e-3)
    train_acc, test_acc = train_model(name, model, trainloader, testloader, optimizer, criterion, epochs=15)
    results[name] = (train_acc, test_acc)

# ------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------
print("\nFinal Accuracies (%):")
print("Model\tTrain\tTest")
for label, (train_acc, test_acc) in results.items():
    print(f"{label}\t{train_acc[-1]:.2f}\t{test_acc[-1]:.2f}")

print("\nRun 'tensorboard --logdir=runs' to view the training and test accuracy curves.")
