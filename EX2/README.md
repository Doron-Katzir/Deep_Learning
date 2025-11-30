# Deep Learning Exercise 2 - Penn Treebank Language Modeling

Implementation of the "small" Zaremba model for next-word prediction on the Penn Treebank dataset.

## Training Commands

### 1. LSTM without Dropout

```bash
python Deep_Learning_Ex2.py --data ./PTB --model lstm --dropout 0.0 --epochs 13 --lr 1.0 --nonmono 5 --lr_decay 2.0 --save_dir ./checkpoints/lstm_nodrop
```

**Target:** Validation perplexity < 125

---

### 2. LSTM with Dropout

```bash
python Deep_Learning_Ex2.py --data ./PTB --model lstm --dropout 0.5 --epochs 75 --lr 0.25 --nonmono 60 --lr_decay 1.2 --save_dir ./checkpoints/lstm_drop
```

**Target:** Validation perplexity < 100

---

### 3. GRU without Dropout

```bash
python Deep_Learning_Ex2.py --data ./PTB --model gru --dropout 0.0 --epochs 13 --lr 1.0 --nonmono 5 --lr_decay 2.0 --save_dir ./checkpoints/gru_nodrop
```

**Target:** Validation perplexity < 125

---

### 4. GRU with Dropout

```bash
python Deep_Learning_Ex2.py --data ./PTB --model gru --dropout 0.5 --epochs 75 --lr 0.25 --nonmono 60 --lr_decay 1.2 --save_dir ./checkpoints/gru_drop
```

**Target:** Validation perplexity < 100

---

## Testing with Saved Weights

To evaluate a trained model on train/validation/test sets:

### LSTM without Dropout
```bash
python evaluate.py --data ./PTB --model lstm --dropout 0.0 --checkpoint ./checkpoints/lstm_nodrop/lstm_drop0.00_best.pt
```

### LSTM with Dropout
```bash
python evaluate.py --data ./PTB --model lstm --dropout 0.5 --checkpoint ./checkpoints/lstm_drop/lstm_drop0.50_best.pt
```

### GRU without Dropout
```bash
python evaluate.py --data ./PTB --model gru --dropout 0.0 --checkpoint ./checkpoints/gru_nodrop/gru_drop0.00_best.pt
```

### GRU with Dropout
```bash
python evaluate.py --data ./PTB --model gru --dropout 0.5 --checkpoint ./checkpoints/gru_drop/gru_drop0.50_best.pt
```

---

## Output Files

Each training run generates:
- **Checkpoint files:** `{save_dir}/{model}_drop{dropout}_best.pt` (best model based on validation)
- **Convergence plot:** `{save_dir}/{model}_drop{dropout}.png` (train and test perplexity)
- **Results CSV:** `{save_dir}/results.csv` (final perplexities)
- **TensorBoard logs:** `./runs/{model}_drop{dropout}/` (view with `tensorboard --logdir ./runs`)
