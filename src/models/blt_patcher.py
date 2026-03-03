# # """
# # BLT Entropy Patching Module
# # ==========================
# # Reference: Byte Latent Transformer (Meta AI, 2024)
# # """

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F


# # class EntropyPatcher(nn.Module):
# #     """
# #     BLT Entropy-Based Patching Module
    
# #     Implements entropy-guided dynamic patching:
# #     - High entropy (mutation hotspots) → 3-byte patches
# #     - Low entropy (conserved regions) → 12-byte patches
    
# #     Reference: Byte Latent Transformer (Meta AI, 2024)
# #     """
    
# #     def __init__(self, 
# #                  embedding_dim: int = 64,
# #                  patch_size_high: int = 3,
# #                  patch_size_low: int = 12,
# #                  entropy_threshold: float = 1.5):
# #         super().__init__()
        
# #         self.embedding_dim = embedding_dim
# #         self.patch_size_high = patch_size_high
# #         self.patch_size_low = patch_size_low
# #         self.entropy_threshold = entropy_threshold
        
# #     def calculate_entropy(self, sequence: torch.Tensor) -> torch.Tensor:
# #         """Calculate Shannon entropy for each position"""
# #         batch_size, seq_len = sequence.shape
# #         entropy = torch.zeros((batch_size, seq_len), dtype=torch.float32)
        
# #         for b in range(batch_size):
# #             for i in range(seq_len):
# #                 start = max(0, i - 4)
# #                 end = min(seq_len, i + 5)
# #                 window = sequence[b, start:end]
                
# #                 if window.numel() > 0:
# #                     unique, counts = torch.unique(window, return_counts=True)
# #                     probs = counts.float() / window.size(0)
# #                     ent = -torch.sum(probs * torch.log2(probs + 1e-10))
# #                     entropy[b, i] = ent
                    
# #         return entropy
    
# #     def forward(self, x: torch.Tensor):
# #         """
# #         Forward pass
        
# #         Args:
# #             x: Input sequence (batch, seq_len)
            
# #         Returns:
# #             blt_features: (batch, embedding_dim)
# #             entropy: (batch, seq_len)
# #         """
# #         batch_size = x.size(0)
        
# #         # Calculate entropy
# #         entropy = self.calculate_entropy(x)
        
# #         # Create patches based on entropy
# #         patch_features = []
        
# #         for b in range(batch_size):
# #             seq_features = []
# #             i = 0
            
# #             while i < x.size(1):
# #                 ent = entropy[b, i].item()
                
# #                 # Determine patch size based on entropy
# #                 if ent > self.entropy_threshold:
# #                     size = self.patch_size_high  # High entropy = small patch
# #                 else:
# #                     size = self.patch_size_low   # Low entropy = large patch
                
# #                 end = min(i + size, x.size(1))
# #                 patch = x[b, i:end]
                
# #                 # Mean pooling for patch representation
# #                 patch_emb = patch.float().mean() if patch.size(0) > 0 else 0
# #                 seq_features.append(patch_emb)
# #                 i = end
            
# #             patch_features.append(torch.stack(seq_features))
        
# #         # Pad to same number of patches
# #         max_patches = max(p.size(0) for p in patch_features)
        
# #         padded = []
# #         for p in patch_features:
# #             if p.size(0) < max_patches:
# #                 padded.append(F.pad(p, (0, max_patches - p.size(0))))
# #             else:
# #                 padded.append(p)
        
# #         # Average pool patches
# #         blt_features = torch.stack(padded).mean(dim=2)
        
# #         return blt_features, entropy

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class EntropyPatcher(nn.Module):
#     """
#     Entropy-guided patching (BLT-inspired).

#     Input:  x (batch, seq_len)
#     Output:
#       - blt_features: (batch, embedding_dim)
#       - entropy:      (batch, seq_len)
#     """

#     def __init__(
#         self,
#         embedding_dim: int = 64,
#         patch_size_high: int = 3,
#         patch_size_low: int = 12,
#         entropy_threshold: float = 1.5,
#         window: int = 9,
#     ):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.patch_size_high = patch_size_high
#         self.patch_size_low = patch_size_low
#         self.entropy_threshold = entropy_threshold
#         self.window = window

#         # Project scalar patch summary → embedding_dim
#         self.proj = nn.Sequential(
#             nn.Linear(1, embedding_dim),
#             nn.ReLU(),
#             nn.Linear(embedding_dim, embedding_dim),
#         )

#     @staticmethod
#     def _shannon(counts: torch.Tensor) -> torch.Tensor:
#         probs = counts / (counts.sum(dim=-1, keepdim=True) + 1e-12)
#         return -(probs * torch.log2(probs + 1e-12)).sum(dim=-1)

#     def calculate_entropy(self, x: torch.Tensor) -> torch.Tensor:
#         device = x.device
#         b, L = x.shape
#         K = max(int(x.max().item()) + 1, 5)

#         half = self.window // 2
#         xpad = F.pad(x, (half, half), value=0)

#         entropy = torch.zeros((b, L), dtype=torch.float32, device=device)
#         for i in range(L):
#             w = xpad[:, i: i + self.window]          # (b, window)
#             counts = torch.zeros((b, K), dtype=torch.float32, device=device)
#             for t in range(K):
#                 counts[:, t] = (w == t).float().sum(dim=1)
#             entropy[:, i] = self._shannon(counts)

#         return entropy

#     def forward(self, x: torch.Tensor):
#         b, L = x.shape
#         entropy = self.calculate_entropy(x)   # (b, L)

#         all_sample_vecs = []

#         for bi in range(b):
#             i = 0
#             patch_vecs = []

#             while i < L:
#                 ent_i = float(entropy[bi, i].item())
#                 psize = (
#                     self.patch_size_high
#                     if ent_i > self.entropy_threshold
#                     else self.patch_size_low
#                 )
#                 j = min(i + psize, L)
#                 patch = x[bi, i:j].float()

#                 # scalar summary → (1, 1)
#                 patch_mean = patch.mean().view(1, 1)

#                 # project to embedding_dim → (1, embedding_dim)
#                 vec = self.proj(patch_mean)          # (1, embedding_dim)
#                 patch_vecs.append(vec)
#                 i = j

#             # stack patches → (num_patches, embedding_dim)
#             patch_vecs = torch.cat(patch_vecs, dim=0)

#             # average over patches → (embedding_dim,)
#             sample_vec = patch_vecs.mean(dim=0)
#             all_sample_vecs.append(sample_vec)

#         # stack batch → (batch, embedding_dim)
#         blt_features = torch.stack(all_sample_vecs, dim=0)

#         return blt_features, entropy

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyPatcher(nn.Module):
    """
    Fast Entropy-Based Patching Module
    Uses vectorized entropy calculation instead of loops
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        patch_size_high: int = 3,
        patch_size_low: int = 12,
        entropy_threshold: float = 1.5,
        window: int = 9,
    ):
        super().__init__()
        self.embedding_dim   = embedding_dim
        self.patch_size_high = patch_size_high
        self.patch_size_low  = patch_size_low
        self.entropy_threshold = entropy_threshold
        self.window          = window

        self.proj = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def calculate_entropy_fast(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast vectorized entropy using one-hot counts
        No Python loops over positions
        """
        b, L = x.shape
        K    = 5  # vocab size: 0=N,1=A,2=T,3=G,4=C

        # One-hot: (b, L, K)
        x_clamp  = x.clamp(0, K-1)
        one_hot  = F.one_hot(x_clamp, num_classes=K).float()

        # Sliding window sum using conv1d
        # kernel of ones → sum counts in window
        kernel = torch.ones(K, 1, self.window, device=x.device)

        # (b, K, L) for conv
        one_hot_t = one_hot.permute(0, 2, 1)

        # Pad
        pad      = self.window // 2
        padded   = F.pad(one_hot_t, (pad, pad), value=0)

        # Count per window: (b, K, L)
        counts   = F.conv1d(padded, kernel, groups=K)

        # Normalize to probabilities
        total    = counts.sum(dim=1, keepdim=True).clamp(min=1e-12)
        probs    = counts / total

        # Shannon entropy: (b, L)
        entropy  = -(probs * torch.log2(probs + 1e-12)).sum(dim=1)

        return entropy

    def forward(self, x: torch.Tensor):
        b, L    = x.shape
        entropy = self.calculate_entropy_fast(x)  # (b, L)

        all_vecs = []

        for bi in range(b):
            i          = 0
            patch_vecs = []

            while i < L:
                ent_i = float(entropy[bi, i].item())
                psize = (
                    self.patch_size_high
                    if ent_i > self.entropy_threshold
                    else self.patch_size_low
                )
                j     = min(i + psize, L)
                patch = x[bi, i:j].float()

                patch_mean = patch.mean().view(1, 1)
                vec        = self.proj(patch_mean)
                patch_vecs.append(vec)
                i = j

            patch_vecs = torch.cat(patch_vecs, dim=0)
            sample_vec = patch_vecs.mean(dim=0)
            all_vecs.append(sample_vec)

        blt_features = torch.stack(all_vecs, dim=0)
        return blt_features, entropy