import os
import math
import time
import argparse
import csv
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import trange

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# -----------------------
# Data helpers (PTB)
# -----------------------
def read_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()[1:].split(' ')

def build_vocab(train_path):
    words = read_words(train_path)
    unique_words = sorted(set(words))
    vocab = {w: i for i, w in enumerate(unique_words)}
    itos = {i: w for w, i in vocab.items()}
    return vocab, itos

def file_to_tensor(path, vocab):
    words = read_words(path)
    # Map words to indices, use vocab['<unk>'] for unknown words
    unk_idx = vocab.get('<unk>', 0)  # Default to 0 if no <unk>
    ids = [vocab.get(w, unk_idx) for w in words]
    return torch.tensor(ids, dtype=torch.long)

def batchify(data, batch_size, device):
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous().to(device)
    return data

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

# -----------------------
# Model
# -----------------------
class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, emb_size=200, hidden_size=200, num_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if self.rnn_type == 'lstm':
            self.rnns = [nn.LSTM(hidden_size, hidden_size) for _ in range(num_layers)]
        elif self.rnn_type == 'gru':
            self.rnns = [nn.GRU(hidden_size, hidden_size) for _ in range(num_layers)]
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        self.rnns = nn.ModuleList(self.rnns)

        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        # Initialize ALL parameters uniformly
        for param in self.parameters():
            nn.init.uniform_(param, -initrange, initrange)

    def forward(self, input, hidden):
        # input: seq_len x batch
        x = self.encoder(input)
        x = self.dropout(x)

        # Apply each RNN layer with dropout between them
        new_hidden = []
        for i, rnn in enumerate(self.rnns):
            x, h = rnn(x, hidden[i])
            new_hidden.append(h)
            x = self.dropout(x)

        decoded = self.decoder(x.view(x.size(0)*x.size(1), x.size(2)))
        return decoded, new_hidden

    def init_hidden(self, batch_size):
        # Return list of hidden states, one per layer
        if self.rnn_type == 'lstm':
            return [(torch.zeros(1, batch_size, self.hidden_size, device=device),
                     torch.zeros(1, batch_size, self.hidden_size, device=device))
                    for _ in range(self.num_layers)]
        else:
            return [torch.zeros(1, batch_size, self.hidden_size, device=device)
                    for _ in range(self.num_layers)]

# -----------------------
# Training and evaluation
# -----------------------
def repackage_hidden(h):
    # h is now a list of hidden states (one per layer)
    if isinstance(h, list):
        return [repackage_hidden(v) for v in h]
    elif isinstance(h, torch.Tensor):
        return h.detach()
    else:
        # Tuple for LSTM (h, c)
        return tuple(v.detach() for v in h)

def evaluate(model, data_source, criterion, bptt):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    # Use the actual batch size from the data (data_source.size(1))
    batch_size = data_source.size(1)
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0)-1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)  # Sum of per-token losses
            total_loss += loss.item()
            total_tokens += len(targets)  # Count actual tokens
            hidden = repackage_hidden(hidden)
    avg_loss = total_loss / total_tokens  # Average per token
    return avg_loss

def train_epoch(model, train_data, optimizer, criterion, bptt, batch_size, clip, writer, epoch):
    model.train()
    total_loss = 0.0
    hidden = model.init_hidden(batch_size)
    iters = 0
    seqs = range(0, train_data.size(0)-1, bptt)
    pbar = trange(len(seqs), desc=f"Epoch {epoch}", leave=False)
    for idx in pbar:
        i = idx * bptt
        data, targets = get_batch(train_data, i, bptt)
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # criterion with reduction='sum' returns sum of losses
        total_loss += loss.item()
        iters += len(targets)

        if (idx+1) % 50 == 0:
            cur_loss = total_loss / iters
            cur_ppl = math.exp(cur_loss)
            pbar.set_postfix({'ppl': f"{cur_ppl:.2f}"})

    avg_loss = total_loss / iters  # iters = total_tokens
    return avg_loss

# -----------------------
# Utilities: plotting and CSV
# -----------------------
def plot_curves(results_dir, label, train_vals, test_vals):
    # save as png; user can also view tensorboard
    plt.figure()
    epochs = range(1, len(train_vals)+1)
    plt.plot(epochs, train_vals, label='Train')
    plt.plot(epochs, test_vals, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title(label)
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(results_dir, f"{label.replace(' ', '_')}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def save_summary_csv(save_dir, rows):
    csv_path = os.path.join(save_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model','dropout','epoch','train_ppl','valid_ppl','test_ppl'])
        for r in rows:
            writer.writerow(r)
    print("Saved summary csv:", csv_path)

# -----------------------
# Main experiment runner
# -----------------------
def run_experiment(args):
    # 1) prepare data & vocab
    train_path = os.path.join(args.data, 'ptb.train.txt')
    valid_path = os.path.join(args.data, 'ptb.valid.txt')
    test_path  = os.path.join(args.data, 'ptb.test.txt')
    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Please place train.txt, valid.txt, test.txt in --data directory (download from Moodle).")

    print("Building vocab from train...")
    vocab, itos = build_vocab(train_path)
    vocab_size = len(vocab)
    print("Vocab size:", vocab_size)

    train_tensor = file_to_tensor(train_path, vocab)
    valid_tensor = file_to_tensor(valid_path, vocab)
    test_tensor  = file_to_tensor(test_path, vocab)

    train_data = batchify(train_tensor, args.batch_size, device)
    valid_data = batchify(valid_tensor, args.batch_size, device)
    test_data  = batchify(test_tensor, args.batch_size, device)

    # 2) model (small Zaremba: 2 layers, 200 hidden units)
    model = RNNModel(args.model, vocab_size, emb_size=200, hidden_size=200, num_layers=2, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # prepare logging/checkpoint folders
    run_name = f"{args.model}_drop{args.dropout:.2f}"
    log_dir = os.path.join(args.logdir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    best_val = None
    train_ppls = []
    valid_ppls = []
    test_ppls = []
    rows_for_csv = []

    # training loop
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        train_loss = train_epoch(model, train_data, optimizer, criterion, args.bptt, args.batch_size, args.clip, writer, epoch)
        val_loss = evaluate(model, valid_data, criterion, args.bptt)
        test_loss = evaluate(model, test_data, criterion, args.bptt)
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        test_ppl = math.exp(test_loss)
        train_ppls.append(train_ppl)
        valid_ppls.append(val_ppl)
        test_ppls.append(test_ppl)

        writer.add_scalar('Perplexity/Train', train_ppl, epoch)
        writer.add_scalar('Perplexity/Valid', val_ppl, epoch)
        writer.add_scalar('Perplexity/Test', test_ppl, epoch)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d} | Train ppl {train_ppl:.2f} | Val ppl {val_ppl:.2f} | Test ppl {test_ppl:.2f} | Time {elapsed:.1f}s")

        # save checkpoint if best
        save_path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch}.pt")
        torch.save(model.state_dict(), save_path)

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.save_dir, f"{run_name}_best.pt")
            torch.save(model.state_dict(), best_path)

        # simple learning rate schedule for SGD
        if args.optimizer == 'sgd' and epoch > args.nonmono:
            for g in optimizer.param_groups:
                g['lr'] /= args.lr_decay
            print(f"Decayed learning rate. New lr: {optimizer.param_groups[0]['lr']:.4f}")

    # evaluate best model on train/valid/test
    print("Loading best model:", best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))
    # evaluate on train/valid/test
    train_loss_final = evaluate(model, train_data, criterion, args.bptt)
    val_loss_final = evaluate(model, valid_data, criterion, args.bptt)
    test_loss_final = evaluate(model, test_data, criterion, args.bptt)
    train_ppl_final = math.exp(train_loss_final)
    val_ppl_final = math.exp(val_loss_final)
    test_ppl_final = math.exp(test_loss_final)

    print("Final Perplexities:")
    print(f"Train: {train_ppl_final:.2f} | Val: {val_ppl_final:.2f} | Test: {test_ppl_final:.2f}")

    # save plots and csv summary
    results_dir = args.save_dir
    plot_curves(results_dir, run_name, train_ppls, test_ppls)
    rows_for_csv.append([args.model, args.dropout, args.epochs, train_ppl_final, val_ppl_final, test_ppl_final])
    save_summary_csv(results_dir, rows_for_csv)

    writer.add_scalar('Perplexity/Test', test_ppl_final)
    writer.close()

    # print table (single row)
    print("\nSummary (model, dropout, train_ppl, val_ppl, test_ppl):")
    print(f"{args.model}\t{args.dropout}\t{train_ppl_final:.2f}\t{val_ppl_final:.2f}\t{test_ppl_final:.2f}")

# -----------------------
# Argument parser
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--model', choices=['lstm', 'gru'], default='lstm')
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=9, help='total epochs ')
    p.add_argument('--batch_size', type=int, default=20)
    p.add_argument('--bptt', type=int, default=35, help='sequence length')
    p.add_argument('--lr', type=float, default=1.0, help='initial learning rate')
    p.add_argument('--optimizer', choices=['sgd','adam'], default='sgd')
    p.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--logdir', type=str, default='./runs')
    p.add_argument('--nonmono', type=int, default=5, help='start LR decay after epoch 5')
    p.add_argument('--lr_decay', type=float, default=2.0, help='LR decay divisor (divide by 2.0)')
    return p.parse_args()

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    run_experiment(args)
