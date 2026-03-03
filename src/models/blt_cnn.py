"""
Complete BLT-CNN Hybrid Fusion Model
==================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blt_patcher import EntropyPatcher
from .cnn1d import CNN1DLocalMotif
from .transformer import TransformerEncoder


class BLTCNNHybridFusion(nn.Module):
    """
    BLT-CNN: Hybrid Fusion Architecture for Antibiotic Resistance Prediction
    
    Combines:
    - BLT Entropy Patching (Meta AI, 2024)
    - 1D CNN for local motifs
    - Transformer for global context
    - Bio-Constraint Loss
    
    Reference: Byte Latent Transformer (Meta AI, 2024)
    """
    
    def __init__(self, 
                 vocab_size: int = 5,
                 embedding_dim: int = 64,
                 num_classes: int = 3,
                 num_filters: int = 256,
                 kernel_size: int = 9,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        
        super().__init__()
        
        # DNA Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # BLT Entropy Patching
        self.entropy_patcher = EntropyPatcher(
            embedding_dim=embedding_dim,
            patch_size_high=3,
            patch_size_low=12,
            entropy_threshold=1.5
        )
        
        # CNN for local motifs
        self.cnn = CNN1DLocalMotif(
            input_dim=embedding_dim,
            num_filters=num_filters,
            kernel_size=kernel_size
        )
        
        # Transformer for global context
        self.transformer = TransformerEncoder(
            d_model=num_filters // 2,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Fusion layer
        fusion_dim = (num_filters // 2) + embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, num_filters),
            nn.LayerNorm(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_filters // 2, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass
        
        Args:
            x: Input DNA sequence (batch, seq_len)
            
        Returns:
            predictions: (batch, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # BLT Entropy Patching
        blt_features, entropy = self.entropy_patcher(x)
        # blt_features: (batch, embedding_dim)
        
        # CNN for local features
        cnn_out = self.cnn(embedded)
        # cnn_out: (batch, seq_len//2, num_filters//2)
        
        # Transformer for global context
        transformer_out = self.transformer(cnn_out)
        # transformer_out: (batch, seq_len//2, num_filters//2)
        
        # Global average pooling
        global_features = transformer_out.mean(dim=1)
        # global_features: (batch, num_filters//2)
        
        # Fusion: concatenate CNN features and BLT features
        fused = torch.cat([global_features, blt_features], dim=1)
        # fused: (batch, num_filters//2 + embedding_dim)
        
        # Classification
        fused = self.fusion(fused)
        output = self.classifier(fused)
        
        return output