"""
DNA Sequence Encoding
====================
"""

import numpy as np


class DNAEncoder:
    """DNA sequence encoder"""
    
    ENCODING = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}
    
    ONE_HOT = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'C': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    
    @staticmethod
    def encode(sequence: str, max_len: int = 100) -> np.ndarray:
        encoded = [DNAEncoder.ENCODING.get(base.upper(), 0) 
                   for base in sequence]
        
        if len(encoded) < max_len:
            encoded = encoded + [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
            
        return np.array(encoded)
    
    @staticmethod
    def one_hot(sequence: str, max_len: int = 100) -> np.ndarray:
        encoded = np.zeros((max_len, 4))
        
        for i, base in enumerate(sequence[:max_len]):
            if base.upper() in DNAEncoder.ONE_HOT:
                encoded[i] = DNAEncoder.ONE_HOT[base.upper()]
                
        return encoded