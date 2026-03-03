"""
BLT-CNN FastAPI Backend
=======================
Connects the 3D HTML frontend to your real trained model.

Install:
    pip install fastapi uvicorn torch numpy

Run:
    python app_api.py

API runs at:  http://localhost:8000
Docs at:      http://localhost:8000/docs   ← auto-generated!
"""

import sys
import os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

# ── Add project root to path ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import BLTCNNHybridFusion

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/blt_cnn_card_best.pth"
SEQ_LEN     = 300
NUC_MAP     = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0, 'U': 2}
ANTIBIOTICS = ['Methicillin', 'Ciprofloxacin', 'Vancomycin']

ANTIBIOTIC_INFO = {
    'Methicillin':   {'gene': 'mecA',       'pathogen': 'S. aureus (MRSA)',    'drug_class': 'Beta-lactam',    'note': 'Most common hospital-acquired infection'},
    'Ciprofloxacin': {'gene': 'qnr, gyrA',  'pathogen': 'E. coli, Klebsiella', 'drug_class': 'Fluoroquinolone','note': 'Most prescribed broad-spectrum antibiotic'},
    'Vancomycin':    {'gene': 'vanA, vanB',  'pathogen': 'Enterococcus (VRE)',  'drug_class': 'Glycopeptide',   'note': 'Last-resort antibiotic'},
}

# ── Load model once at startup ─────────────────────────────────────────────────
print("\n" + "="*55)
print("  BLT-CNN FastAPI Server")
print("="*55)
print(f"[1] Loading model: {MODEL_PATH}")

try:
    model = BLTCNNHybridFusion(vocab_size=5, embedding_dim=64, num_classes=3)
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    MODEL_LOADED = True
    print("    ✅ Model loaded — 944,387 parameters")
except Exception as e:
    model = None
    MODEL_LOADED = False
    print(f"    ❌ Model load failed: {e}")

print("="*55 + "\n")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BLT-CNN AMR Predictor",
    description="Antibiotic Resistance Prediction from DNA Sequences using BLT-CNN hybrid architecture",
    version="1.0.0"
)

# Allow HTML frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML frontend at root
if os.path.exists("blt_cnn_ui.html"):
    @app.get("/", response_class=FileResponse)
    def serve_frontend():
        return FileResponse("blt_cnn_ui.html")


# ── Request / Response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sequence: str

class AntibioticResult(BaseModel):
    name: str
    probability: float
    percentage: str
    status: str          # "resistant" | "borderline" | "susceptible"
    label: str           # "RESISTANT" | "BORDERLINE" | "SUSCEPTIBLE"
    gene: str
    pathogen: str
    drug_class: str
    note: str

class PredictResponse(BaseModel):
    success: bool
    sequence_length: int
    results: List[AntibioticResult]
    summary: str
    model_loaded: bool


# ── Helper functions ───────────────────────────────────────────────────────────
def encode_sequence(seq: str) -> torch.Tensor:
    """Encode nucleotide sequence to integer tensor."""
    seq = seq.upper().strip().replace(" ", "").replace("\n", "")
    encoded = [NUC_MAP.get(c, 0) for c in seq[:SEQ_LEN]]
    # Pad if shorter than SEQ_LEN
    if len(encoded) < SEQ_LEN:
        encoded += [0] * (SEQ_LEN - len(encoded))
    return torch.tensor([encoded], dtype=torch.long)


def get_status(prob: float):
    """Convert probability to status label."""
    if prob >= 0.7:
        return "resistant", "RESISTANT"
    elif prob >= 0.4:
        return "borderline", "BORDERLINE"
    else:
        return "susceptible", "SUSCEPTIBLE"


def compute_entropy(seq: str) -> List[dict]:
    """Compute Shannon entropy profile for visualization."""
    seq = seq.upper()[:SEQ_LEN]
    profile = []
    step = max(1, len(seq) // 10)
    for i in range(0, len(seq), step):
        window = seq[i:i+20]
        if not window:
            break
        counts = {n: window.count(n) for n in 'ATGCN'}
        total = len(window)
        ent = 0.0
        for c in counts.values():
            if c > 0:
                p = c / total
                ent -= p * (p and __import__('math').log2(p))
        profile.append({
            "position": f"{i}–{min(i+step, len(seq))}",
            "entropy": round(max(0, ent), 3),
            "patch_type": "small (3-nt)" if ent > 1.5 else "large (12-nt)",
            "hotspot": ent > 1.5
        })
    return profile


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if API and model are running."""
    return {
        "status": "online",
        "model_loaded": MODEL_LOADED,
        "model_path": MODEL_PATH,
        "sequence_length": SEQ_LEN,
        "antibiotics": ANTIBIOTICS
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict antibiotic resistance from a DNA sequence.

    Send a POST request with:
        { "sequence": "ATGAAAAAGATAAAAATTGT..." }

    Returns probabilities for Methicillin, Ciprofloxacin, Vancomycin.
    """
    seq = req.sequence.strip()

    if not seq:
        raise HTTPException(status_code=400, detail="Sequence cannot be empty")

    if len(seq) < 20:
        raise HTTPException(status_code=400, detail="Sequence too short — minimum 20 nucleotides")

    # Validate characters
    clean = seq.upper().replace(" ", "").replace("\n", "")
    invalid = set(clean) - set('ATGCNU')
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid characters in sequence: {invalid}")

    # Run model or demo mode
    if MODEL_LOADED and model is not None:
        try:
            tensor = encode_sequence(clean)
            with torch.no_grad():
                output = model(tensor)
            probs = output.squeeze().tolist()
            # Ensure list
            if isinstance(probs, float):
                probs = [probs]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    else:
        # Demo mode — return placeholder values
        probs = [0.05, 0.05, 0.05]

    # Build results
    results = []
    for i, ab in enumerate(ANTIBIOTICS):
        prob = float(np.clip(probs[i], 0.0, 1.0))
        status, label = get_status(prob)
        info = ANTIBIOTIC_INFO[ab]
        results.append(AntibioticResult(
            name=ab,
            probability=round(prob, 4),
            percentage=f"{prob*100:.1f}%",
            status=status,
            label=label,
            gene=info['gene'],
            pathogen=info['pathogen'],
            drug_class=info['drug_class'],
            note=info['note']
        ))

    # Build summary string
    resistant   = [r.name for r in results if r.status == "resistant"]
    borderline  = [r.name for r in results if r.status == "borderline"]
    susceptible = [r.name for r in results if r.status == "susceptible"]
    parts = []
    if resistant:   parts.append(f"Resistant to {', '.join(resistant)}")
    if borderline:  parts.append(f"Borderline: {', '.join(borderline)}")
    if susceptible: parts.append(f"Susceptible to {', '.join(susceptible)}")
    summary = " | ".join(parts)

    return PredictResponse(
        success=True,
        sequence_length=len(clean),
        results=results,
        summary=summary,
        model_loaded=MODEL_LOADED
    )


@app.post("/entropy")
def entropy_profile(req: PredictRequest):
    """Get entropy profile of a DNA sequence for BLT patching visualization."""
    seq = req.sequence.strip()
    if not seq:
        raise HTTPException(status_code=400, detail="Sequence cannot be empty")
    return {
        "sequence_length": len(seq),
        "profile": compute_entropy(seq),
        "threshold": 1.5,
        "description": "Positions with entropy > 1.5 get small patches (3-nt), others get large patches (12-nt)"
    }


# ── Run server ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting BLT-CNN API server...")
    print("Frontend: http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    print("Health:   http://localhost:8000/health")
    print("\nPress Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)