# BLT-CNN: Hybrid Fusion Model for Antibiotic Resistance Prediction

This project is a research-style implementation of a **hybrid fusion model** to predict antibiotic resistance from pathogen DNA.

## What is implemented (core idea)

- **Entropy-based patching** (BLT-inspired): adapt patch size based on Shannon entropy  
- **1D CNN**: learns local DNA motifs  
- **Transformer Encoder**: learns long-range/global dependencies  
- **Fusion Head**: combines CNN+Transformer features with BLT patch features  
- **Multi-label output**: predicts resistance for multiple antibiotics

## How to run (demo with synthetic data)

From the project root:

```powershell
pip install -r requirements.txt
python scripts/train.py
python scripts/evaluate.py