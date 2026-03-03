"""
BLT-CNN: Hybrid Fusion Model for Antibiotic Resistance Prediction
================================================================

A deep learning framework for predicting antibiotic resistance from 
DNA sequences using Biological Language Transformers and CNN.

Modules:
- data: Dataset handling, DNA encoding, entropy calculation
- models: BLT-CNN architecture components
- training: Training loops, losses, metrics

Reference Papers:
- Byte Latent Transformer (Meta AI, 2024)
- DeepARG (Arango-Argoty et al., 2018)
"""

"""
BLT-CNN package
"""


__version__ = "1.0.0"
__author__ = "Research Team"
__license__ = "MIT"

from . import data
from . import models
from . import training

__all__ = ['data', 'models', 'training']