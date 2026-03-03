"""
Loss Functions for BLT-CNN
==========================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BioConstraintLoss(nn.Module):
    """
    Bio-Constraint Loss Function
    
    Prevents biologically implausible predictions by adding
    mutation-based regularization.
    
    Reference: DeepARG (Arango-Argoty et al., 2018)
    """
    
    def __init__(self, mutation_weight: float = 0.3):
        super().__init__()
        self.mutation_weight = mutation_weight
        
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                mutation_penalty: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate loss
        
        Args:
            predictions: (batch, num_classes) - predicted probabilities
            targets: (batch, num_classes) - true labels
            mutation_penalty: (batch,) - entropy-based mutation indicator
            
        Returns:
            loss: scalar
        """
        # Binary cross-entropy
        bce = F.binary_cross_entropy(predictions, targets)
        
        # Add mutation constraint if provided
        if mutation_penalty is not None:
            # Penalize predictions in high-entropy (mutable) regions
            mutation_loss = (mutation_penalty.mean() * predictions.mean())
            total = bce + self.mutation_weight * mutation_loss
        else:
            total = bce
            
        return total


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Reference: Lin et al., 2017
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()