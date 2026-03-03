"""
Evaluation Metrics for BLT-CNN
=============================
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            predictions = model(sequences)
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, binary_preds),
        'precision': precision_score(all_labels, binary_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, binary_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, binary_preds, average='weighted', zero_division=0),
    }
    
    # AUC (only if more than one class)
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_preds, average='weighted')
    except:
        metrics['auc'] = 0.0
    
    return metrics


def calculate_per_class_metrics(y_true, y_pred, class_names=None):
    """
    Calculate per-class metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary of per-class metrics
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(y_true.shape[1])]
    
    metrics = {}
    
    for i, name in enumerate(class_names):
        metrics[name] = {
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
        }
    
    return metrics