import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.dataset import DNADataset
from src.models.blt_cnn import BLTCNNHybridFusion
from src.training.losses import BioConstraintLoss
from src.training.metrics import evaluate_model, calculate_per_class_metrics


def load_card_data():
    """Load processed CARD data"""
    X = np.load("data/processed/card_sequences.npy")
    y = np.load("data/processed/card_labels.npy")

    with open("data/processed/card_gene_names.txt") as f:
        gene_names = [line.strip() for line in f.readlines()]

    return X, y, gene_names


def main():
    print("=" * 60)
    print("BLT-CNN Training on CARD Database")
    print("=" * 60)

    # Load CARD data
    print("\n[1] Loading CARD data...")
    X, y, gene_names = load_card_data()
    print(f"    Sequences: {X.shape}")
    print(f"    Labels:    {y.shape}")
    print(f"    Methicillin:   {int(y[:,0].sum())} resistant")
    print(f"    Ciprofloxacin: {int(y[:,1].sum())} resistant")
    print(f"    Vancomycin:    {int(y[:,2].sum())} resistant")

    # Split data
    print("\n[2] Splitting data (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"    Train: {X_train.shape}")
    print(f"    Val:   {X_val.shape}")
    print(f"    Test:  {X_test.shape}")

    # Dataloaders
    train_loader = DataLoader(
        DNADataset(X_train, y_train), 
        batch_size=64, 
        shuffle=True
    )
    val_loader  = DataLoader(DNADataset(X_val,  y_val),  batch_size=64)
    test_loader = DataLoader(DNADataset(X_test, y_test), batch_size=64)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[3] Device: {device}")

    # Model
    print("\n[4] Building model...")
    model = BLTCNNHybridFusion(
        vocab_size=5,
        embedding_dim=64,
        num_classes=3,
        num_filters=256,
        kernel_size=9,
        num_heads=8,
        num_layers=2,
        dropout=0.2
    ).to(device)
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss + optimizer
    criterion = BioConstraintLoss(mutation_weight=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training
    print("\n[5] Training...")
    epochs        = 20
    best_val_loss = float('inf')
    history       = {'train_loss': [], 'val_loss': [], 'val_f1': []}

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

        avg_loss = total_loss / len(train_loader)

        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss    = 1 - val_metrics['f1']

        scheduler.step(val_loss)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])

        print(f"    Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, 'models/blt_cnn_card_best.pth')

    # Final test evaluation
    print("\n[6] Final Test Evaluation...")
    model.load_state_dict(torch.load('models/blt_cnn_card_best.pth'))
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
    per_class   = calculate_per_class_metrics(
        all_labels, binary_preds, class_names
    )

    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS ON CARD DATABASE")
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
        'dataset':     'CARD Database',
        'num_genes':   len(X),
        'overall':     {k: float(v) for k, v in test_metrics.items()},
        'per_class':   {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in per_class.items()
        },
        'history': {
            k: [float(v) for v in vals]
            for k, vals in history.items()
        }
    }

    with open('results/card_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("    Saved to results/card_results.json")
    print("\n" + "=" * 60)
    print("Training on CARD Data Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()