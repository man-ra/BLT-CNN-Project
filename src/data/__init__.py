"""
Data processing module for BLT-CNN
=================================

Contains:
- dataset: PyTorch Dataset class
- encoding: DNA sequence encoders
- entropy: Shannon entropy calculations
"""

from .dataset import DNADataset
from .encoding import DNAEncoder
from .entropy import ShannonEntropy

__all__ = ['DNADataset', 'DNAEncoder', 'ShannonEntropy']