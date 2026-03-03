import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import DNADataset
from src.models.blt_cnn import BLTCNNHybridFusion
from src.training.losses import BioConstraintLoss
from src.training.metrics import evaluate_model


def generate_data(num_samples=500, seq_len=100):
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

    return X, y


def main():
    print("=" * 60)
    print("BLT-CNN Training")
    print("=" * 60)

    # Data
    print("\n[1] Generating data...")
    X, y = generate_data(num_samples=500, seq_len=100)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(DNADataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader  = DataLoader(DNADataset(X_test,  y_test),  batch_size=32)

    print(f"    Train: {X_train.shape}  Test: {X_test.shape}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[2] Device: {device}")

    # Model
    print("\n[3] Building model...")
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
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss + optimizer
    criterion = BioConstraintLoss(mutation_weight=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\n[4] Training...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for seqs, labels in train_loader:
            seqs   = seqs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(seqs)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate_model(model, test_loader, device)
        print(f"    Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f}")

    # Save
    print("\n[5] Saving model...")
    torch.save(model.state_dict(), 'models/blt_cnn_final.pth')
    print("    Saved to models/blt_cnn_final.pth")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()