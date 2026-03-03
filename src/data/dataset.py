"""
DNA Dataset for PyTorch Training
==============================
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class DNADataset(Dataset):
    """
    PyTorch Dataset for DNA sequences
    
    Args:
        sequences: Encoded DNA sequences (N, seq_len)
        labels: Binary labels (N, num_classes)
        transform: Optional transform function
    """
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, 
                 transform=None):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return sequence, label


class ResistanceDataset(Dataset):
    """
    Dataset for antibiotic resistance prediction
    
    Args:
        fasta_file: Path to FASTA file
        label_file: Path to label CSV
        encoder: DNA encoder
        max_len: Maximum sequence length
    """
    
    def __init__(self, fasta_file: str, label_file: str = None,
                 max_len: int = 100):
        self.sequences = []
        self.labels = []
        self.max_len = max_len
        
        # Load sequences
        self._load_fasta(fasta_file)
        
        # Load labels if provided
        if label_file:
            self._load_labels(label_file)
    
    def _load_fasta(self, fasta_file: str):
        """Load sequences from FASTA file"""
        from Bio import SeqIO
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq).upper()
            
            # Encode
            encoded = self._encode(seq)
            self.sequences.append(encoded)
    
    def _encode(self, sequence: str) -> np.ndarray:
        """Encode DNA sequence"""
        mapping = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}
        
        encoded = [mapping.get(base, 0) for base in sequence]
        
        # Pad or truncate
        if len(encoded) < self.max_len:
            encoded = encoded + [0] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
            
        return np.array(encoded)
    
    def _load_labels(self, label_file: str):
        """Load labels from CSV"""
        import pandas as pd
        
        df = pd.read_csv(label_file)
        
        # Assuming columns: sample_id, antibiotic1, antibiotic2, ...
        antibiotics = [col for col in df.columns if col != 'sample_id']
        
        for _, row in df.iterrows():
            label = [int(row[ab]) for ab in antibiotics]
            self.labels.append(label)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.LongTensor(self.sequences[idx])
        
        if self.labels:
            label = torch.FloatTensor(self.labels[idx])
        else:
            label = torch.zeros(1)
            
        return sequence, label