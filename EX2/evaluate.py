import os
import math
import argparse
import torch
import torch.nn as nn
from Deep_Learning_Ex2 import (
    RNNModel, build_vocab, file_to_tensor, batchify,
    evaluate, device
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path with train.txt valid.txt test.txt')
    parser.add_argument('--model', choices=['lstm', 'gru'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint (.pt file)')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout used during training (for model architecture)')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for evaluation')
    parser.add_argument('--bptt', type=int, default=35, help='sequence length')
    args = parser.parse_args()

    # Load data
    train_path = os.path.join(args.data, 'ptb.train.txt')
    valid_path = os.path.join(args.data, 'ptb.valid.txt')
    test_path = os.path.join(args.data, 'ptb.test.txt')

    print("Building vocab from train...")
    vocab, itos = build_vocab(train_path)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    train_tensor = file_to_tensor(train_path, vocab)
    valid_tensor = file_to_tensor(valid_path, vocab)
    test_tensor = file_to_tensor(test_path, vocab)

    train_data = batchify(train_tensor, args.batch_size, device)
    valid_data = batchify(valid_tensor, args.batch_size, device)
    test_data = batchify(test_tensor, args.batch_size, device)

    # Load model (small Zaremba: 2 layers, 200 hidden units)
    print(f"Loading model from {args.checkpoint}")
    model = RNNModel(args.model, vocab_size, emb_size=200, hidden_size=200, num_layers=2, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='sum')

    # Evaluate
    print("Evaluating...")
    train_loss = evaluate(model, train_data, criterion, args.bptt)
    valid_loss = evaluate(model, valid_data, criterion, args.bptt)
    test_loss = evaluate(model, test_data, criterion, args.bptt)

    train_ppl = math.exp(train_loss)
    valid_ppl = math.exp(valid_loss)
    test_ppl = math.exp(test_loss)

    print("\nResults:")
    print(f"Train Perplexity: {train_ppl:.2f}")
    print(f"Valid Perplexity: {valid_ppl:.2f}")
    print(f"Test Perplexity:  {test_ppl:.2f}")

if __name__ == "__main__":
    main()
