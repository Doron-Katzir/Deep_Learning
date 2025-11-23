import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Reproducibility
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# Model
class LeNet5(nn.Module):
    def __init__(self, use_dropout=False, use_batchnorm=False):
        super().__init__()
        self.use_bn = use_batchnorm
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # keep 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)            # 28->24

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(6)
            self.bn2 = nn.BatchNorm2d(16)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 1x28x28 -> conv1 -> 6x28x28 -> pool -> 6x14x14
        # -> conv2(5x5) -> 16x10x10 -> pool -> 16x5x5 -> 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn: x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.use_bn: x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        if self.use_dropout: x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Data
def get_dataloaders(batch_size=128):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.FashionMNIST(root="./data", train=True, transform=tfm, download=True)
    test_ds  = datasets.FashionMNIST(root="./data", train=False, transform=tfm, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

# Eval helpers
@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    total = 0; correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return 100.0 * correct / total

# Training
def train_model(name, model, train_loader, test_loader, device, num_epochs=25, weight_decay=0.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    model.to(device)
    writer = SummaryWriter(log_dir=os.path.join("runs", name))
    train_acc_hist, test_acc_hist = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        avg_train_loss = running_loss / max(1, num_batches)
        tr_acc = evaluate_acc(model, train_loader, device)
        te_acc = evaluate_acc(model, test_loader, device)
        train_acc_hist.append(tr_acc); test_acc_hist.append(te_acc)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", tr_acc, epoch)
        writer.add_scalar("Accuracy/Test", te_acc, epoch)

        print(f"[{name}] Epoch {epoch:02d}/{num_epochs}  "
              f"TrainLoss={avg_train_loss:.4f}  TrainAcc={tr_acc:.2f}%  TestAcc={te_acc:.2f}%")

    writer.close()
    return train_acc_hist, test_acc_hist, model

# Plotting
def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def plot_curve(epochs, values, title, ylabel, save_path):
    plt.figure()
    plt.plot(epochs, values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_train_vs_test(epochs, train_values, test_values, title, save_path):
    plt.figure()
    plt.plot(epochs, train_values, label="Train")
    plt.plot(epochs, test_values, label="Test")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)
    train_loader, test_loader = get_dataloaders(batch_size=128)

    configs = [{"name": "Baseline",    "use_dropout": False, "use_batchnorm": False, "weight_decay": 0.0},
        {"name": "Dropout",     "use_dropout": True,  "use_batchnorm": False, "weight_decay": 0.0},
        {"name": "WeightDecay", "use_dropout": False, "use_batchnorm": False, "weight_decay": 5e-4},
        {"name": "BatchNorm",   "use_dropout": False, "use_batchnorm": True,  "weight_decay": 0.0},]

    ensure_dir("plots")
    ensure_dir("plots_combined")
    ensure_dir("checkpoints")

    results = []
    num_epochs = 25
    epochs = list(range(1, num_epochs + 1))

    for cfg in configs:
        model = LeNet5(use_dropout=cfg["use_dropout"], use_batchnorm=cfg["use_batchnorm"])
        tr_hist, te_hist, model = train_model(
            cfg["name"], model, train_loader, test_loader, device,
            num_epochs=num_epochs, weight_decay=cfg["weight_decay"])

        # Save weights
        torch.save(model.state_dict(), os.path.join("checkpoints", f"{cfg['name']}.pt"))

        # 8 required plots
        plot_curve(epochs, tr_hist, f"{cfg['name']} - Train Accuracy", "Accuracy (%)",
                   os.path.join("plots", f"{cfg['name']}_train_acc.png"))
        plot_curve(epochs, te_hist, f"{cfg['name']} - Test Accuracy", "Accuracy (%)",
                   os.path.join("plots", f"{cfg['name']}_test_acc.png"))

        # 4 combined plots
        plot_train_vs_test(epochs, tr_hist, te_hist,
                           f"{cfg['name']} - Train vs Test Accuracy",
                           os.path.join("plots_combined", f"{cfg['name']}_train_vs_test.png"))

        # Final accuracies for table
        results.append([cfg["name"], f"{tr_hist[-1]:.2f}", f"{te_hist[-1]:.2f}"])

    # Save results table
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Final Train Accuracy (%)", "Final Test Accuracy (%)"])
        writer.writerows(results)

if __name__ == "__main__":
    main()