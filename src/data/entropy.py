"""
Shannon Entropy Calculation for DNA Sequences
============================================
"""

import numpy as np
from collections import Counter


class ShannonEntropy:
    """
    Calculate Shannon entropy for DNA sequence analysis
    
    Reference: Shannon, C.E. (1948). A Mathematical Theory of Communication.
    """
    
    @staticmethod
    def calculate(sequence: str) -> float:
        """Calculate Shannon entropy for entire sequence"""
        counts = Counter(sequence.upper())
        total = len(sequence)
        
        probs = [count / total for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return entropy
    
    @staticmethod
    def calculate_window(sequence: str, window_size: int = 10) -> np.ndarray:
        """Calculate sliding window entropy"""
        entropy = []
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            ent = ShannonEntropy.calculate(window)
            entropy.append(ent)
            
        return np.array(entropy)
    
    @staticmethod
    def get_hotspots(sequence: str, threshold: float = 1.5) -> list:
        """Identify high-entropy regions (mutation hotspots)"""
        window_entropy = ShannonEntropy.calculate_window(sequence, 10)
        hotspots = np.where(window_entropy > threshold)[0].tolist()
        return hotspots
    
    @staticmethod
    def create_patches(sequence: str, high_thr: float = 1.5, 
                       low_thr: float = 0.5) -> tuple:
        """Create entropy-based patches"""
        patches = []
        patch_types = []
        
        i = 0
        while i < len(sequence):
            remaining = sequence[i:]
            window = remaining[:10] if len(remaining) >= 10 else remaining
            entropy = ShannonEntropy.calculate(window)
            
            if entropy > high_thr:
                size = 3
                patch_type = 1
            elif entropy < low_thr:
                size = 12
                patch_type = 0
            else:
                size = 6
                patch_type = 0
            
            patch = remaining[:size]
            patches.append(patch)
            patch_types.append(patch_type)
            i += size
            
        return patches, patch_types