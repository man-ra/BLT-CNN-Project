import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Bio import Entrez, SeqIO

Entrez.email = "your@email.com"
os.makedirs("data/raw", exist_ok=True)

# Verified S. aureus complete genome accessions only
ACCESSIONS = [
    "NC_002952",  # MRSA252
    "NC_003923",  # MW2
    "NC_007793",  # USA300
    "NC_009641",  # Newman
    "NC_010079",  # USA300 TCH1516
    "NC_013450",  # ED98
    "NC_017341",  # JKD6008
    "NC_017342",  # TCH60
    "NC_017343",  # ECT-R2
    "NC_017347",  # T0131
    "NC_021554",  # CA-347
    "NC_022226",  # CN1
    "NC_022442",  # SA957
    "CP000255",   # MRSA252
    "CP000703",   # USA300
    "CP001844",   # JH9
    "CP001845",   # JH1
    "CP003166",   # S0385
    "CP003667",   # 04-02981
    "CP006044",   # 6850
]


def verify_saureus(content: str) -> bool:
    """Check if sequence is S. aureus"""
    header = content.split('\n')[0].lower()
    return 'staphylococcus aureus' in header


def download_genomes():
    print("=" * 60)
    print("Downloading S. aureus Genomes from NCBI")
    print("=" * 60)

    # Delete wrong files first
    print("\n[0] Cleaning wrong files...")
    wrong = ["NC_010084.fasta", "NC_028761.fasta", "NC_030485.fasta"]
    for f in wrong:
        fpath = os.path.join("data/raw", f)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"    Deleted: {f}")

    print(f"\n[1] Using {len(ACCESSIONS)} verified S. aureus accessions")

    downloaded = 0
    skipped    = 0
    failed     = 0

    print("\n[2] Downloading...")
    for i, acc in enumerate(ACCESSIONS):
        output_file = f"data/raw/{acc}.fasta"

        # Skip if already downloaded
        if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
            print(f"    [{i+1}/{len(ACCESSIONS)}] Already exists: {acc}")
            downloaded += 1
            continue

        try:
            handle = Entrez.efetch(
                db="nucleotide",
                id=acc,
                rettype="fasta",
                retmode="text"
            )
            content = handle.read()
            handle.close()

            if not content or not content.strip().startswith(">"):
                failed += 1
                print(f"    [{i+1}/{len(ACCESSIONS)}] Empty: {acc}")
                time.sleep(0.5)
                continue

            # Verify it's S. aureus
            if not verify_saureus(content):
                skipped += 1
                header = content.split('\n')[0]
                print(f"    [{i+1}/{len(ACCESSIONS)}] Wrong species: {header[:60]}")
                time.sleep(0.4)
                continue

            with open(output_file, 'w') as f:
                f.write(content)

            downloaded += 1
            first_line = content.split('\n')[0]
            print(f"    [{i+1}/{len(ACCESSIONS)}] OK: {first_line[:60]}")

            time.sleep(0.4)

        except Exception as e:
            failed += 1
            print(f"    [{i+1}/{len(ACCESSIONS)}] Error {acc}: {e}")
            time.sleep(1)

    print("\n[3] Done!")
    print(f"    Downloaded:  {downloaded}")
    print(f"    Wrong species skipped: {skipped}")
    print(f"    Failed:      {failed}")
    print(f"    Saved to:    data/raw/")
    print(f"\n    Total valid files: {len([f for f in os.listdir('data/raw') if f.endswith('.fasta')])}")


if __name__ == "__main__":
    download_genomes()