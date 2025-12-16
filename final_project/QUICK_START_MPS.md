# 🚀 Quick Start - Running on Apple Silicon (MPS)

## ✅ Changes Made for MPS

1. **Updated `pretrain_mae.py`**:
   - ✅ Automatic MPS device detection
   - ✅ Fixed mixed precision for MPS compatibility
   - ✅ Proper autocast handling

2. **Created MPS-optimized configs**:
   - ✅ `configs/pretrain_baseline_mps.yaml` (Random masking)
   - ✅ `configs/pretrain_grid_mps.yaml` (Grid masking)
   - ✅ `configs/pretrain_saliency_mps.yaml` (Saliency masking)

3. **MPS-specific settings**:
   - `device: mps`
   - `batch_size: 64` (reduced from 128)
   - `num_workers: 0` (required for MPS)
   - `pin_memory: false` (not needed for MPS)
   - `use_amp: false` (MPS doesn't fully support AMP)

---

## 🏃 Run Now (3 Simple Steps)

### **Step 1: Verify MPS Works**

```bash
cd /Users/danieltoberman/Documents/git/Deep_Learning/final_project

python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected: `MPS available: True`

### **Step 2: Download Dataset**

```bash
# Download TinyImageNet (if not already done)
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ./data/

# Verify
ls data/tiny-imagenet-200/train/ | wc -l
# Should show: 200
```

### **Step 3: Run Training**

```bash
# Quick test (5 epochs, ~2-3 hours)
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml

# Full training (100 epochs, ~2-3 days)
# Use caffeinate to prevent sleep:
caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
```

---

## 📊 Monitor Training

### **Terminal 1: Training**
```bash
caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
```

### **Terminal 2: TensorBoard**
```bash
tensorboard --logdir logs/
# Open: http://localhost:6006
```

### **Terminal 3: GPU Monitor**
```bash
# Monitor GPU usage
sudo powermetrics --samplers gpu_power -i 1000

# Or use Activity Monitor:
# Open Activity Monitor > Window > GPU History
```

---

## ⚙️ Adjust for Your Mac

### **Find Your Mac's Memory**
```bash
system_profiler SPHardwareDataType | grep Memory
```

### **Recommended Batch Sizes**

| Mac Model | Memory | Batch Size |
|-----------|--------|------------|
| M1 (8GB) | 8GB | 32 |
| M1 Pro/M2 | 16GB | 64 |
| M1 Max/M2 Pro | 32GB | 96-128 |
| M2 Max/M3 | 48GB+ | 128 |

**To change**: Edit the config file

```yaml
data:
  batch_size: 32  # Change this number
```

---

## 🎯 Quick Test Before Full Training

Edit `configs/pretrain_baseline_mps.yaml`:

```yaml
training:
  epochs: 5  # Just 5 epochs for testing (~2 hours)
```

Then run:
```bash
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
```

If this works, change back to 100 epochs for full training.

---

## 📈 Expected Training Times

| Epochs | Time (M1/M2) | Time (M1 Pro/Max) |
|--------|--------------|-------------------|
| 5 | 2-3 hours | 1.5-2 hours |
| 50 | 1-1.5 days | 15-20 hours |
| 100 | 2-3 days | 1.5-2 days |

---

## 🐛 Common Issues

### **"MPS backend out of memory"**
**Fix**: Reduce batch size
```yaml
data:
  batch_size: 32  # or even 16
```

### **"num_workers > 0 not supported on MPS"**
**Fix**: Already set in MPS configs
```yaml
data:
  num_workers: 0
```

### **Mac goes to sleep during training**
**Fix**: Use `caffeinate`
```bash
caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
```

---

## 🎯 After Training

### **Evaluate the Model**

```bash
# Linear probing
python linear_probe.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --output_dir results/linear_probe

# Fine-tuning
python finetune.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --output_dir results/finetune
```

---

## 📁 Where to Find Outputs

```
final_project/
├── checkpoints/
│   └── checkpoint_best.pth        # Best model
├── logs/
│   └── events.out.tfevents.*      # TensorBoard logs
└── results/
    ├── linear_probe/
    └── finetune/
```

---

## 💡 Pro Tips

1. **Close Chrome/Safari** - They use GPU memory
2. **Plug in your Mac** - Better performance
3. **Good cooling** - Prevents thermal throttling
4. **Use `caffeinate`** - Prevents sleep during long training
5. **Start with 5 epochs** - Test before committing to 100

---

## ✅ You're Ready!

Run this now:
```bash
cd /Users/danieltoberman/Documents/git/Deep_Learning/final_project
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
```

See full details in `RUN_ON_MPS.md` 📚
