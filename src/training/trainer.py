"""
Trainer for BLT-CNN Model
=========================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os


class Trainer:
    """
    Trainer for BLT-CNN model
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda',
                 save_dir: str = 'models'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for sequences, labels in tqdm(self.train_loader, desc="Training"):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in tqdm(self.val_loader, desc="Validation"):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                predictions = self.model(sequences)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                pred_binary = (predictions > 0.5).float()
                correct += (pred_binary == labels).float().sum().item()
                total += labels.numel()
        
        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy
    
    def train(self, epochs: int):
        """Train model for specified epochs"""
        best_val_loss = float('inf')
        
        print(f"Training on {self.device}")
        print(f"Epochs: {epochs}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                print(f"  -> Saved best model!")
        
        print("-" * 50)
        print("Training complete!")
        
        return self.history
    
    def save_model(self, filename: str):
        """Save model"""
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)