import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from torch.utils.data import DataLoader

from src.data.dataset import DNADataset
from src.models.blt_cnn import BLTCNNHybridFusion
from src.training.losses import BioConstraintLoss
from src.training.metrics import evaluate_model, calculate_per_class_metrics


def load_real_data():
    """Load preprocessed real genome data"""
    X = np.load("data/processed/sequences.npy")
    y = np.load("data/processed/labels.npy")

    with open("data/processed/genome_ids.txt") as f:
        genome_ids = [line.strip() for line in f.readlines()]

    return X, y, genome_ids


def main():
    print("=" * 60)
    print("BLT-CNN Training on Real S. aureus Genomes")
    print("=" * 60)

    # Load real data
    print("\n[1] Loading real genome data...")
    X, y, genome_ids = load_real_data()
    print(f"    Sequences: {X.shape}")
    print(f"    Labels:    {y.shape}")
    print(f"    Genomes:   {genome_ids}")

    # Split data
    print("\n[2] Splitting data...")
    n = len(X)
    train_end = int(0.7 * n)
    val_end   = int(0.85 * n)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val   = X[train_end:val_end]
    y_val   = y[train_end:val_end]

    X_test  = X[val_end:]
    y_test  = y[val_end:]

    print(f"    Train: {X_train.shape}")
    print(f"    Val:   {X_val.shape}")
    print(f"    Test:  {X_test.shape}")

    # Dataloaders
    train_loader = DataLoader(DNADataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader   = DataLoader(DNADataset(X_val,   y_val),   batch_size=8)
    test_loader  = DataLoader(DNADataset(X_test,  y_test),  batch_size=8)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[3] Device: {device}")

    # Model
    print("\n[4] Building model...")
    model = BLTCNNHybridFusion(
        vocab_size=5,
        embedding_dim=64,
        num_classes=3,
        num_filters=128,
        kernel_size=9,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ).to(device)
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss + optimizer
    criterion = BioConstraintLoss(mutation_weight=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
    )

    # Training
    print("\n[5] Training...")
    epochs = 20
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Train
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

        avg_train_loss = total_loss / len(train_loader)

        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss    = 1 - val_metrics['accuracy']

        # Scheduler step
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])

        print(f"    Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/blt_cnn_real_best.pth')
            print(f"             -> Saved best model!")

    # Final test evaluation
    print("\n[6] Final Test Evaluation...")
    model.load_state_dict(torch.load('models/blt_cnn_real_best.pth'))
    test_metrics = evaluate_model(model, test_loader, device)

    # Per class metrics
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            preds = model(seqs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds    = np.vstack(all_preds)
    all_labels   = np.vstack(all_labels)
    binary_preds = (all_preds > 0.5).astype(int)

    class_names = ['Methicillin', 'Ciprofloxacin', 'Vancomycin']
    per_class   = calculate_per_class_metrics(all_labels, binary_preds, class_names)

    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS ON REAL GENOMES")
    print("=" * 60)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"\nPer-Class:")
    for name, m in per_class.items():
        print(f"  {name:15s} | "
              f"P: {m['precision']:.4f} | "
              f"R: {m['recall']:.4f} | "
              f"F1: {m['f1']:.4f}")

    # Save results
    print("\n[7] Saving results...")
    os.makedirs('results', exist_ok=True)

    results = {
        'dataset': 'Real S. aureus genomes',
        'num_genomes': len(X),
        'overall': {k: float(v) for k, v in test_metrics.items()},
        'per_class': {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in per_class.items()
        },
        'history': {
            k: [float(v) for v in vals]
            for k, vals in history.items()
        }
    }

    with open('results/real_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("    Saved to results/real_results.json")
    print("\n" + "=" * 60)
    print("Training on Real Data Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()