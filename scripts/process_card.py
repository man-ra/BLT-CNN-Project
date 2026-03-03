import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from Bio import SeqIO

# CARD files
CARD_DIR      = "data/card/card-data.tar"
PROCESSED_DIR = "data/processed"
SEQ_LEN       = 300

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Drug class mapping to our 3 labels
DRUG_CLASS_MAP = {
    # Methicillin (index 0)
    'penam':                  0,
    'penem':                  0,
    'cephalosporin':          0,
    'carbapenem':             0,
    'monobactam':             0,
    'methicillin':            0,
    'beta-lactam':            0,

    # Ciprofloxacin (index 1)
    'fluoroquinolone':        1,
    'quinolone':              1,
    'ciprofloxacin':          1,

    # Vancomycin (index 2)
    'glycopeptide':           2,
    'vancomycin':             2,
}


def load_aro_index():
    """Load ARO index to get drug class for each ARO number"""
    print("\n[1] Loading ARO index...")
    fpath = os.path.join(CARD_DIR, "aro_index.tsv")

    df = pd.read_csv(fpath, sep='\t', encoding='utf-8')
    print(f"    Columns: {list(df.columns)}")
    print(f"    Total entries: {len(df)}")

    # Build ARO → drug class mapping
    aro_to_drug = {}
    for _, row in df.iterrows():
        aro    = str(row['ARO Accession']).strip()
        drugs  = str(row['Drug Class']).lower()
        aro_to_drug[aro] = drugs

    print(f"    ARO entries loaded: {len(aro_to_drug)}")
    return aro_to_drug


def get_label(drug_class_str: str) -> list:
    """Convert drug class string to binary label"""
    label = [0, 0, 0]
    found = False

    for drug, idx in DRUG_CLASS_MAP.items():
        if drug in drug_class_str:
            label[idx] = 1
            found = True

    return label if found else None


def load_card_sequences(aro_to_drug: dict):
    """Load resistance gene sequences from CARD FASTA files"""
    print("\n[2] Loading CARD FASTA files...")

    fasta_files = [
        "nucleotide_fasta_protein_homolog_model.fasta",
        "nucleotide_fasta_protein_variant_model.fasta",
        "nucleotide_fasta_rRNA_gene_variant_model.fasta",
    ]

    sequences  = []
    labels     = []
    gene_names = []

    for fname in fasta_files:
        fpath = os.path.join(CARD_DIR, fname)

        if not os.path.exists(fpath):
            print(f"    Not found: {fname}")
            continue

        print(f"\n    Loading: {fname}")
        count   = 0
        skipped = 0

        for record in SeqIO.parse(fpath, "fasta"):
            seq = str(record.seq).upper()

            # Keep only DNA sequences
            if set(seq) - set('ATGCN'):
                skipped += 1
                continue

            # Extract ARO number from header
            # Format: gb|acc|strand|pos|ARO:XXXXXXX|GeneName [organism]
            header = record.description
            aro    = None

            for part in header.split('|'):
                if part.startswith('ARO:'):
                    aro = part.strip()
                    break

            if aro is None:
                skipped += 1
                continue

            # Get drug class
            drug_class = aro_to_drug.get(aro, '')
            label      = get_label(drug_class)

            if label is None:
                skipped += 1
                continue

            sequences.append(seq)
            labels.append(label)
            gene_names.append(f"{record.id}|{aro}")
            count += 1

        print(f"    Loaded:  {count}")
        print(f"    Skipped: {skipped}")

    print(f"\n    Total sequences: {len(sequences)}")
    return sequences, labels, gene_names


def encode_sequences(sequences, seq_len=SEQ_LEN):
    """Encode DNA sequences to integers"""
    print(f"\n[3] Encoding {len(sequences)} sequences (length={seq_len})...")
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


def main():
    print("=" * 60)
    print("Processing CARD Database")
    print("=" * 60)

    # Load ARO index
    aro_to_drug = load_aro_index()

    # Load sequences
    sequences, labels, gene_names = load_card_sequences(aro_to_drug)

    if len(sequences) == 0:
        print("\nNo sequences loaded!")
        return

    # Encode
    X = encode_sequences(sequences)
    y = np.array(labels, dtype=np.float32)

    # Summary
    print(f"\n[4] Dataset Summary:")
    print(f"    Total genes:       {len(sequences)}")
    print(f"    Methicillin:       {int(y[:,0].sum())} resistant genes")
    print(f"    Ciprofloxacin:     {int(y[:,1].sum())} resistant genes")
    print(f"    Vancomycin:        {int(y[:,2].sum())} resistant genes")

    # Save
    print(f"\n[5] Saving...")
    np.save(os.path.join(PROCESSED_DIR, "card_sequences.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "card_labels.npy"),    y)

    with open(os.path.join(PROCESSED_DIR, "card_gene_names.txt"), 'w') as f:
        for name in gene_names:
            f.write(name + "\n")

    print(f"    Saved card_sequences.npy → shape {X.shape}")
    print(f"    Saved card_labels.npy    → shape {y.shape}")

    print("\n" + "=" * 60)
    print("CARD Processing Complete!")
    print("=" * 60)
    print("\nNext: Run scripts/train_card.py")


if __name__ == "__main__":
    main()