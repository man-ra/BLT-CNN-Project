# BLT-CNN: Antibiotic Resistance Prediction

Hybrid deep learning architecture combining BLT Entropy Patching, 
1D CNN, and Transformer Encoder for antibiotic resistance prediction 
from genomic sequences.

## Results
- Accuracy: 93.34%
- AUC: 98.55%
- F1: 94.18%
- Dataset: CARD Database (4,005 sequences)

## Architecture
- BLT Entropy Patcher (adaptive patching)
- 1D CNN (local motif detection)
- Transformer Encoder (global context)

## Run
pip install -r requirements.txt
python app_api.py