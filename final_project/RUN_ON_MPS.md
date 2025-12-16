# Running MAE on Apple Silicon (MPS)

## 🍎 MPS-Specific Setup

### **What Changed for MPS**

1. **Device Detection**: Automatic MPS detection in `pretrain_mae.py`
2. **Mixed Precision**: Disabled (MPS doesn't fully support AMP yet)
3. **Data Loading**:
   - `num_workers: 0` (avoid multiprocessing issues)
   - `pin_memory: false` (not needed for MPS)
4. **Batch Size**: Reduced to 64 (adjust based on your RAM)

### **System Requirements**

- macOS 12.3+ (Monterey or later)
- Apple Silicon (M1, M1 Pro, M1 Max, M2, M3, etc.)
- PyTorch 2.0+ with MPS support
- 16GB+ unified memory recommended

---

## 🚀 Quick Start for MPS

### **1. Verify MPS is Available**

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import torch; print(f'MPS built: {torch.backends.mps.is_built()}')"
```

Expected output:
```
MPS available: True
MPS built: True
```

### **2. Download TinyImageNet**

```bash
cd /Users/danieltoberman/Documents/git/Deep_Learning/final_project

# Download dataset
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ./data/
```

### **3. Run Training with MPS Config**

```bash
# Use the MPS-optimized config
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
```

---

## ⚙️ Performance Expectations

### **Training Speed (M1 Pro/Max/M2)**

- **Batch size 64**: ~2-3 seconds per batch
- **1 epoch**: ~30-40 minutes (TinyImageNet)
- **100 epochs**: ~50-70 hours (~2-3 days)

### **Memory Usage**

- **Model**: ~500MB
- **Batch 64**: ~4-6GB unified memory
- **Batch 128**: ~8-10GB (if you have 32GB+ RAM)

### **Comparison to CUDA**

- MPS is typically **60-70%** the speed of NVIDIA RTX 3090
- Still **5-10x faster** than CPU-only training

---

## 🔧 Adjusting Settings for Your Mac

### **For M1/M2 Base (8GB)**

Edit `configs/pretrain_baseline_mps.yaml`:

```yaml
data:
  batch_size: 32  # Reduce to 32
  num_workers: 0

training:
  epochs: 50  # Consider fewer epochs for testing
```

### **For M1 Pro/Max (16-32GB)**

```yaml
data:
  batch_size: 64  # Default, works well
  num_workers: 0
```

### **For M2 Pro/Max/Ultra (32GB+)**

```yaml
data:
  batch_size: 128  # Can try larger batches
  num_workers: 0
```

---

## 🐛 Common MPS Issues & Fixes

### **Issue 1: "MPS backend out of memory"**

**Solution**: Reduce batch size

```yaml
data:
  batch_size: 32  # or even 16
```

### **Issue 2: Training is slow**

**Causes**:
- Other apps using GPU
- Swap memory being used

**Solutions**:
```bash
# Close other apps
# Monitor memory:
activity monitor  # Check "Memory" tab

# Reduce batch size if swap is being used
```

### **Issue 3: "Operation not supported on MPS"**

Some operations aren't supported on MPS yet. If you encounter this:

**Temporary workaround**:
```yaml
system:
  device: cpu  # Fall back to CPU for that operation
```

### **Issue 4: Multiprocessing errors with DataLoader**

**Solution**: Already fixed in MPS config
```yaml
data:
  num_workers: 0  # Must be 0 for MPS
```

---

## 📊 Monitoring Training on Mac

### **Monitor GPU Usage**

```bash
# Terminal 1: Run training
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml

# Terminal 2: Monitor GPU
sudo powermetrics --samplers gpu_power -i 1000

# Or use Activity Monitor GUI:
# Activity Monitor > Window > GPU History
```

### **Check Memory Usage**

```bash
# Memory pressure
memory_pressure

# Detailed memory stats
vm_stat
```

### **TensorBoard**

```bash
# In separate terminal
tensorboard --logdir logs/

# Open: http://localhost:6006
```

---

## 🎯 Recommended Workflow for MPS

### **1. Quick Test (30 min)**

```yaml
# Edit configs/pretrain_baseline_mps.yaml
training:
  epochs: 5
```

```bash
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
```

### **2. Overnight Training (8-12 hours)**

```yaml
training:
  epochs: 50
```

### **3. Full Training (2-3 days)**

```yaml
training:
  epochs: 100
```

**Tip**: Use `caffeinate` to prevent Mac from sleeping:

```bash
caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
```

---

## 🔍 Evaluation Scripts (Also MPS-Compatible)

All evaluation scripts automatically detect and use MPS:

```bash
# Linear probing
python linear_probe.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --batch_size 128 \
    --output_dir results/linear_probe

# Fine-tuning
python finetune.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --batch_size 64 \
    --output_dir results/finetune
```

---

## 💡 Tips for Faster Training on MPS

1. **Close unnecessary apps** - Free up unified memory
2. **Use smaller batch sizes** - Better GPU utilization
3. **Disable Chrome/Safari** - They use GPU memory
4. **Keep Mac plugged in** - Full performance mode
5. **Good ventilation** - Prevent thermal throttling
6. **Monitor temps**: Use iStat Menus or similar

---

## 📈 Expected Results on MPS

Training time for full pipeline (3 strategies × 3 ratios):

| Mac Model | Unified Memory | Time per Model | Total Time |
|-----------|---------------|----------------|------------|
| M1 Base | 8GB | ~4 days | ~36 days |
| M1 Pro | 16GB | ~3 days | ~27 days |
| M1 Max | 32GB | ~2.5 days | ~22 days |
| M2 Pro | 16GB | ~2.5 days | ~22 days |
| M3 Max | 48GB | ~2 days | ~18 days |

**Recommendation**: Start with 1-2 strategies instead of all 9 models

---

## 🎯 Practical Research Strategy for MPS

Given the longer training times, consider:

### **Option 1: Focus on One Strategy**

```bash
# Just random masking with 3 ratios
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
# Edit config: mask_ratio: 0.5
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
# Edit config: mask_ratio: 0.9
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
```

### **Option 2: Reduce Epochs**

```yaml
training:
  epochs: 50  # Instead of 100
```

### **Option 3: Compare Strategies at One Ratio**

```bash
# All strategies at mask_ratio=0.75
python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml
python pretrain_mae.py --config configs/pretrain_grid_mps.yaml
python pretrain_mae.py --config configs/pretrain_saliency_mps.yaml
```

---

## ✅ Verification Checklist

Before starting full training:

- [ ] MPS available: `torch.backends.mps.is_available() == True`
- [ ] TinyImageNet downloaded and extracted
- [ ] Test run (5 epochs) completed successfully
- [ ] Enough free disk space (~10GB for checkpoints)
- [ ] Mac stays awake (`caffeinate` or System Settings)
- [ ] Monitoring set up (TensorBoard or Activity Monitor)

---

## 🆘 Still Having Issues?

Check PyTorch version:
```bash
python -c "import torch; print(torch.__version__)"
# Should be 2.0.0 or higher
```

Update PyTorch if needed:
```bash
pip install --upgrade torch torchvision
```

---

**Ready to train on your Mac! 🚀**
