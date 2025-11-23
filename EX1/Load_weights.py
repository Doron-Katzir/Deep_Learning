import torch
from Deep_Learning_Ex1 import LeNet5, evaluate_acc, get_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)

    _, test_loader = get_dataloaders(batch_size=128)

    configs = [
        {"name": "Baseline",
            "weights": "checkpoints/Baseline.pt",
            "use_dropout": False,
            "use_batchnorm": False},
        {"name": "Dropout",
            "weights": "checkpoints/Dropout.pt",
            "use_dropout": True,
            "use_batchnorm": False},
        {"name": "WeightDecay",
            "weights": "checkpoints/WeightDecay.pt",
            "use_dropout": False,
            "use_batchnorm": False},
        {"name": "BatchNorm",
            "weights": "checkpoints/BatchNorm.pt",
            "use_dropout": False,
            "use_batchnorm": True}]

    results = []

    for cfg in configs:
        model = LeNet5(
            use_dropout=cfg["use_dropout"],
            use_batchnorm=cfg["use_batchnorm"])
        state_dict = torch.load(cfg["weights"], map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        acc = evaluate_acc(model, test_loader, device)
        print(f"{cfg['name']} Test Accuracy: {acc:.2f}%")
        results.append((cfg["name"], acc))

if __name__ == "__main__":
    main()