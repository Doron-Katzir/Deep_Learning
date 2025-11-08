import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------
# 1. Model
# -------------------------
class LeNet5(nn.Module):
    def __init__(self, use_dropout=False, use_batchnorm=False):
        super(LeNet5, self).__init__()
        self.use_bn = use_batchnorm
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # keep 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 28->24->12->8 etc.

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(6)
            self.bn2 = nn.BatchNorm2d(16)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # After conv/pool: 1x28x28 -> conv1 -> 6x28x28 -> pool -> 6x14x14
        # -> conv2 (5x5) -> 16x10x10 -> pool -> 16x5x5 -> 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------
# 2. Data
# -------------------------
def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

# -------------------------
# 3. Eval helper
# -------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

# -------------------------
# 4. Training
# -------------------------
def train_model(config_name, model, train_loader, test_loader, device, num_epochs=15,
                weight_decay=0.0, logdir="runs"):
    writer = SummaryWriter(log_dir=os.path.join(logdir, config_name))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # IMPORTANT PART:
        # PDF says for dropout: measure train accuracy WITHOUT dropout.
        # Easiest: always evaluate train in eval() mode so it's correct for every config.
        train_eval_loss, train_eval_acc = evaluate(model, train_loader, device)

        # test
        test_loss, test_acc = evaluate(model, test_loader, device)

        epoch_loss = running_loss / total

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(train_eval_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # TensorBoard
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Loss/Train_evalmode", train_eval_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_eval_acc, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)

        print(f"[{config_name}] Epoch {epoch}/{num_epochs} "
              f"TrainAcc(eval)={train_eval_acc:.2f}% TestAcc={test_acc:.2f}%")

    writer.close()
    return history

# -------------------------
# 5. Plotting helpers (to satisfy “8 graphs” + “recommended 4”)
# -------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_single_metric(epochs, values, title, ylabel, save_path):
    plt.figure()
    plt.plot(epochs, values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_train_test_together(epochs, train_vals, test_vals, title, ylabel, save_path):
    plt.figure()
    plt.plot(epochs, train_vals, label="Train")
    plt.plot(epochs, test_vals, label="Test")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------------
# 6. Main
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=128)

    # 4 configs required by PDF
    configs = [
        {
            "name": "Baseline",
            "use_dropout": False,
            "use_batchnorm": False,
            "weight_decay": 0.0
        },
        {
            "name": "Dropout",
            "use_dropout": True,
            "use_batchnorm": False,
            "weight_decay": 0.0
        },
        {
            "name": "WeightDecay",
            "use_dropout": False,
            "use_batchnorm": False,
            "weight_decay": 5e-4
        },
        {
            "name": "BatchNorm",
            "use_dropout": False,
            "use_batchnorm": True,
            "weight_decay": 0.0
        }
    ]

    all_results = []
    plots_dir = "plots"
    ckpt_dir = "checkpoints"
    ensure_dir(plots_dir)
    ensure_dir(ckpt_dir)

    num_epochs = 15
    epochs = list(range(1, num_epochs + 1))

    for cfg in configs:
        model = LeNet5(
            use_dropout=cfg["use_dropout"],
            use_batchnorm=cfg["use_batchnorm"]
        )
        history = train_model(
            cfg["name"],
            model,
            train_loader,
            test_loader,
            device,
            num_epochs=num_epochs,
            weight_decay=cfg["weight_decay"],
            logdir="runs"
        )

        # save model weights (PDF: “…and how to test it with the saved weights.”)
        save_path = os.path.join(ckpt_dir, f"{cfg['name']}.pt")
        torch.save(model.state_dict(), save_path)

        # store final results for table
        final_train_acc = history["train_acc"][-1]
        final_test_acc = history["test_acc"][-1]
        all_results.append({
            "name": cfg["name"],
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc
        })

        # ---- PLOTS PER CONFIG (PDF wants 8 graphs total) ----
        # 1) train acc
        plot_single_metric(
            epochs,
            history["train_acc"],
            title=f"{cfg['name']} - Train Accuracy",
            ylabel="Accuracy (%)",
            save_path=os.path.join(plots_dir, f"{cfg['name']}_train_acc.png")
        )
        # 2) test acc
        plot_single_metric(
            epochs,
            history["test_acc"],
            title=f"{cfg['name']} - Test Accuracy",
            ylabel="Accuracy (%)",
            save_path=os.path.join(plots_dir, f"{cfg['name']}_test_acc.png")
        )

        # OPTIONAL BUT RECOMMENDED IN PDF: train+test together (accuracy)
        plot_train_test_together(
            epochs,
            history["train_acc"],
            history["test_acc"],
            title=f"{cfg['name']} - Train vs Test Accuracy",
            ylabel="Accuracy (%)",
            save_path=os.path.join(plots_dir, f"{cfg['name']}_acc_both.png")
        )

        # ALSO PLOT LOSSES (nice for convergence)
        plot_train_test_together(
            epochs,
            history["train_loss"],
            history["test_loss"],
            title=f"{cfg['name']} - Train vs Test Loss",
            ylabel="Loss",
            save_path=os.path.join(plots_dir, f"{cfg['name']}_loss_both.png")
        )

    # -------------------------
    # 7. Results table (8 numbers)
    # -------------------------
    # Save csv
    import csv
    csv_path = "results.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Final Train Accuracy (%)", "Final Test Accuracy (%)"])
        for r in all_results:
            writer.writerow([r["name"], f"{r['final_train_acc']:.2f}", f"{r['final_test_acc']:.2f}"])

    # Pretty print
    print("\nFinal Accuracies (for report):")
    print("{:<12} {:>12} {:>12}".format("Model", "Train (%)", "Test (%)"))
    for r in all_results:
        print("{:<12} {:>12.2f} {:>12.2f}".format(
            r["name"], r["final_train_acc"], r["final_test_acc"]
        ))

    # -------------------------
    # 8. Mini README (what PDF asked for)
    # -------------------------
    print("\n========== README ==========")
    print("How to train all 4 settings:")
    print("    python Deep_Learning_Ex1.py")
    print("\nOutputs:")
    print("  - TensorBoard logs: runs/")
    print("  - Model weights: checkpoints/Baseline.pt, Dropout.pt, WeightDecay.pt, BatchNorm.pt")
    print("  - Plots (for the PDF submission): plots/*.png  (8+ plots)")
    print("  - Results table: results.csv")
    print("\nHow to test with saved weights (example):")
    print("    model = LeNet5(...);")
    print("    model.load_state_dict(torch.load('checkpoints/Baseline.pt', map_location='cpu'))")
    print("    model.eval();  # then run on data")
    print("============================")

if __name__ == "__main__":
    main()