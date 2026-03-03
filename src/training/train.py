"""
Main Training Script for BLT-CNN
===============================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import DNADataset
from src.models.blt_cnn import BLTCNNHybridFusion
from src.training.losses import BioConstraintLoss
from src.training.metrics import evaluate_model


def generate_sample_data(num_samples=500, seq_length=100):
    """Generate sample training data"""
    np.random.seed(42)
    
    # Generate random DNA sequences
    bases = ['A', 'T', 'G', 'C']
    sequences = []
    for _ in range(num_samples):
        seq = ''.join(np.random.choice(bases, seq_length))
        sequences.append(seq)
    
    # Encode sequences
    def encode(seq):
        mapping = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}
        encoded = [mapping.get(b, 0) for b in seq]
        if len(encoded) < seq_length:
            encoded = encoded + [0] * (seq_length - len(encoded))
        return encoded[:seq_length]
    
    X = np.array([encode(s) for s in sequences])
    
    # Generate labels
    y = np.zeros((num_samples, 3))
    for i in range(num_samples):
        if 'G' in sequences[i][10:20]:
            y[i, 0] = 1
        if 'T' in sequences[i][20:30]:
            y[i, 1] = 1
        if 'C' in sequences[i][30:40]:
            y[i, 2] = 1
        if np.random.random() < 0.2:
            y[i, np.random.randint(0, 3)] = 1 - y[i, np.random.randint(0, 3)]
    
    return X, y


def main():
    print("=" * 60)
    print("BLT-CNN Training")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating data...")
    X, y = generate_sample_data(num_samples=500, seq_length=100)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create datasets
    train_dataset = DNADataset(X_train, y_train)
    test_dataset = DNADataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[2] Device: {device}")
    
    # Model
    print("\n[3] Creating model...")
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
    
    # Loss and optimizer
    criterion = BioConstraintLoss(mutation_weight=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\n[4] Training...")
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        
        print(f"    Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f}")
    
    # Save model
    print("\n[5] Saving model...")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/blt_cnn_final.pth')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()