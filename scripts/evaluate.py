import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from torch.utils.data import DataLoader

from src.data.dataset import DNADataset
from src.models.blt_cnn import BLTCNNHybridFusion
from src.training.metrics import evaluate_model, calculate_per_class_metrics


def generate_data(num_samples=100, seq_len=100):
    np.random.seed(42)
    bases = ['A', 'T', 'G', 'C']
    mapping = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}

    sequences = []
    for _ in range(num_samples):
        seq = ''.join(np.random.choice(bases, seq_len))
        sequences.append(seq)

    def encode(seq):
        enc = [mapping.get(b, 0) for b in seq]
        if len(enc) < seq_len:
            enc += [0] * (seq_len - len(enc))
        return enc[:seq_len]

    X = np.array([encode(s) for s in sequences])

    y = np.zeros((num_samples, 3))
    for i in range(num_samples):
        if 'G' in sequences[i][10:20]:
            y[i, 0] = 1
        if 'T' in sequences[i][20:30]:
            y[i, 1] = 1
        if 'C' in sequences[i][30:40]:
            y[i, 2] = 1

    return X, y, sequences


def main():
    print("=" * 60)
    print("BLT-CNN Evaluation")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Device: {device}")

    # Load model
    print("\n[2] Loading model...")
    model = BLTCNNHybridFusion(
        vocab_size=5,
        embedding_dim=64,
        num_classes=3,
        num_filters=256,
        kernel_size=9,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ).to(device)

    model_path = 'models/blt_cnn_final.pth'
    if not os.path.exists(model_path):
        print("    Model not found. Run scripts/train.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"    Loaded from {model_path}")

    # Data
    print("\n[3] Loading test data...")
    X, y, sequences = generate_data(num_samples=100, seq_len=100)
    test_loader = DataLoader(DNADataset(X, y), batch_size=32)
    print(f"    Test samples: {len(X)}")

    # Evaluate
    print("\n[4] Evaluating...")
    metrics = evaluate_model(model, test_loader, device)

    # Per class
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            preds = model(seqs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    binary_preds = (all_preds > 0.5).astype(int)

    class_names = ['Methicillin', 'Ciprofloxacin', 'Vancomycin']
    per_class = calculate_per_class_metrics(all_labels, binary_preds, class_names)

    # Print
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"\nPer-Class:")
    for name, m in per_class.items():
        print(f"  {name:15s} | "
              f"P: {m['precision']:.4f} | "
              f"R: {m['recall']:.4f} | "
              f"F1: {m['f1']:.4f}")

    # Save
    print("\n[5] Saving results...")
    os.makedirs('results', exist_ok=True)
    results = {
        'overall': {k: float(v) for k, v in metrics.items()},
        'per_class': {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in per_class.items()
        }
    }
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("    Saved to results/evaluation_results.json")
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()