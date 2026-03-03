# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import numpy as np
# from Bio import SeqIO

# # Config
# RAW_DIR       = "data/raw"
# PROCESSED_DIR = "data/processed"
# SEQ_LEN       = 1000
# os.makedirs(PROCESSED_DIR, exist_ok=True)


# def load_genomes():
#     print("=" * 60)
#     print("Preprocessing Real S. aureus Genomes")
#     print("=" * 60)

#     sequences  = []
#     genome_ids = []

#     print("\n[1] Loading FASTA files...")
#     fasta_files = sorted([
#         f for f in os.listdir(RAW_DIR) if f.endswith(".fasta")
#     ])

#     for fname in fasta_files:
#         fpath = os.path.join(RAW_DIR, fname)
#         try:
#             # Use fasta-pearson to handle comment lines
#             records = list(SeqIO.parse(fpath, "fasta-pearson"))

#             if len(records) == 0:
#                 print(f"    Skipped (empty): {fname}")
#                 continue

#             record = records[0]
#             seq = str(record.seq).upper()

#             # Keep only valid bases
#             seq = ''.join(b if b in 'ATGCN' else 'N' for b in seq)

#             if len(seq) < 100:
#                 print(f"    Skipped (too short): {fname}")
#                 continue

#             sequences.append(seq)
#             genome_ids.append(record.id)
#             print(f"    Loaded: {fname} | {record.id} | {len(seq)} bp")

#         except Exception as e:
#             print(f"    Error {fname}: {e}")

#     print(f"\n    Total loaded: {len(sequences)} genomes")
#     return sequences, genome_ids


# def encode_sequences(sequences, seq_len=SEQ_LEN):
#     print(f"\n[2] Encoding sequences (first {seq_len} bp)...")
#     mapping = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}

#     encoded = []
#     for seq in sequences:
#         enc = [mapping.get(b, 0) for b in seq[:seq_len]]
#         if len(enc) < seq_len:
#             enc += [0] * (seq_len - len(enc))
#         encoded.append(enc[:seq_len])

#     X = np.array(encoded)
#     print(f"    Encoded shape: {X.shape}")
#     return X


# def create_labels(num_samples):
#     print("\n[3] Creating resistance labels...")
#     np.random.seed(42)

#     y = np.zeros((num_samples, 3))
#     for i in range(num_samples):
#         y[i, 0] = np.random.choice([0, 1], p=[0.4, 0.6])  # Methicillin
#         y[i, 1] = np.random.choice([0, 1], p=[0.5, 0.5])  # Ciprofloxacin
#         y[i, 2] = np.random.choice([0, 1], p=[0.8, 0.2])  # Vancomycin

#     print(f"    Labels shape:             {y.shape}")
#     print(f"    Methicillin resistant:    {int(y[:,0].sum())}/{num_samples}")
#     print(f"    Ciprofloxacin resistant:  {int(y[:,1].sum())}/{num_samples}")
#     print(f"    Vancomycin resistant:     {int(y[:,2].sum())}/{num_samples}")
#     return y


# def save_data(X, y, genome_ids):
#     print("\n[4] Saving processed data...")
#     np.save(os.path.join(PROCESSED_DIR, "sequences.npy"), X)
#     np.save(os.path.join(PROCESSED_DIR, "labels.npy"),    y)

#     with open(os.path.join(PROCESSED_DIR, "genome_ids.txt"), 'w') as f:
#         for gid in genome_ids:
#             f.write(gid + "\n")

#     print(f"    Saved sequences.npy  → shape {X.shape}")
#     print(f"    Saved labels.npy     → shape {y.shape}")
#     print(f"    Saved genome_ids.txt → {len(genome_ids)} IDs")


# def main():
#     sequences, genome_ids = load_genomes()

#     if len(sequences) == 0:
#         print("\nNo genomes loaded. Check data/raw/ folder.")
#         return

#     X = encode_sequences(sequences)
#     y = create_labels(len(sequences))
#     save_data(X, y, genome_ids)

#     print("\n" + "=" * 60)
#     print("Preprocessing Complete!")
#     print("=" * 60)
#     print("\nNext: Run scripts/train_real.py")


# if __name__ == "__main__":
#     main() 
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
from Bio import SeqIO

# Config
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
LABELS_FILE   = "data/labels/known_labels.json"
SEQ_LEN       = 1000
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_genomes():
    print("=" * 60)
    print("Preprocessing Real S. aureus Genomes")
    print("=" * 60)

    sequences  = []
    genome_ids = []

    print("\n[1] Loading FASTA files...")
    fasta_files = sorted([
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".fasta") or f.endswith(".fna")
    ])

    for fname in fasta_files:
        fpath = os.path.join(RAW_DIR, fname)
        try:
            records = list(SeqIO.parse(fpath, "fasta-pearson"))
            if len(records) == 0:
                continue

            record = records[0]
            seq = str(record.seq).upper()
            seq = ''.join(b if b in 'ATGCN' else 'N' for b in seq)

            if len(seq) < 100:
                continue

            sequences.append(seq)
            genome_ids.append(record.id)
            print(f"    Loaded: {record.id} | {len(seq)} bp")

        except Exception as e:
            print(f"    Error {fname}: {e}")

    print(f"\n    Total loaded: {len(sequences)} genomes")
    return sequences, genome_ids


def encode_sequences(sequences, seq_len=SEQ_LEN):
    print(f"\n[2] Encoding sequences (first {seq_len} bp)...")
    mapping = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}

    encoded = []
    for seq in sequences:
        enc = [mapping.get(b, 0) for b in seq[:seq_len]]
        if len(enc) < seq_len:
            enc += [0] * (seq_len - len(enc))
        encoded.append(enc[:seq_len])

    X = np.array(encoded)
    print(f"    Encoded shape: {X.shape}")
    return X


def load_real_labels(genome_ids):
    """Load real resistance labels from known_labels.json"""
    print("\n[3] Loading real resistance labels...")

    with open(LABELS_FILE) as f:
        known_labels = json.load(f)

    y = []
    matched   = 0
    unmatched = 0

    for gid in genome_ids:
        if gid in known_labels:
            y.append(known_labels[gid])
            matched += 1
        else:
            # Default: sensitive for all
            y.append([0, 0, 0])
            unmatched += 1
            print(f"    No label found for {gid} → defaulting to [0,0,0]")

    y = np.array(y, dtype=np.float32)

    print(f"    Matched:   {matched}/{len(genome_ids)}")
    print(f"    Unmatched: {unmatched}/{len(genome_ids)}")
    print(f"    Labels shape: {y.shape}")
    print(f"\n    Resistance Summary:")
    print(f"    Methicillin:   {int(y[:,0].sum())}/{len(genome_ids)} resistant")
    print(f"    Ciprofloxacin: {int(y[:,1].sum())}/{len(genome_ids)} resistant")
    print(f"    Vancomycin:    {int(y[:,2].sum())}/{len(genome_ids)} resistant")

    return y


def save_data(X, y, genome_ids):
    print("\n[4] Saving processed data...")
    np.save(os.path.join(PROCESSED_DIR, "sequences.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "labels.npy"),    y)

    with open(os.path.join(PROCESSED_DIR, "genome_ids.txt"), 'w') as f:
        for gid in genome_ids:
            f.write(gid + "\n")

    print(f"    Saved sequences.npy  → shape {X.shape}")
    print(f"    Saved labels.npy     → shape {y.shape}")
    print(f"    Saved genome_ids.txt → {len(genome_ids)} IDs")


def main():
    sequences, genome_ids = load_genomes()

    if len(sequences) == 0:
        print("\nNo genomes loaded!")
        return

    X = encode_sequences(sequences)
    y = load_real_labels(genome_ids)
    save_data(X, y, genome_ids)

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print("\nNext: Run scripts/train_real.py")


if __name__ == "__main__":
    main()