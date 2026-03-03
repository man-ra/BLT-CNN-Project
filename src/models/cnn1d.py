"""
1D CNN for Local Motif Detection
================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DLocalMotif(nn.Module):
    """
    1D CNN for local resistance motif detection
    
    Uses codon-level resolution (kernel size = 9 for 3 codons)
    """
    
    def __init__(self, 
                 input_dim: int = 64,
                 num_filters: int = 256,
                 kernel_size: int = 9):
        super().__init__()
        
        # First conv layer
        self.conv1 = nn.Conv1d(
            input_dim, 
            num_filters, 
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # Second conv layer
        self.conv2 = nn.Conv1d(
            num_filters, 
            num_filters // 2, 
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(num_filters // 2)
        
        # Pooling and dropout
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (batch, seq_len, input_dim)
            
        Returns:
            Output (batch, seq_len // 2, num_filters // 2)
        """
        # Transpose for Conv1d: (batch, channels, length)
        x = x.transpose(1, 2)
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Pooling
        x = self.pool(x)
        x = self.dropout(x)
        
        # Transpose back: (batch, length, channels)
        x = x.transpose(1, 2)
        
        return x