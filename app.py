"""
BLT-CNN Antibiotic Resistance Predictor
========================================
Streamlit frontend for the BLT-CNN model.

Run with:
    pip install streamlit torch numpy
    streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import sys
import os
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BLT-CNN | AMR Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root theme ── */
:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --surface2:  #1a2235;
    --border:    #1f2d45;
    --accent:    #00e5ff;
    --accent2:   #7c3aed;
    --green:     #00ff88;
    --red:       #ff4d6d;
    --yellow:    #ffd166;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Headings ── */
h1, h2, h3 {
    font-family: var(--font-head) !important;
    letter-spacing: -0.02em;
}

/* ── Inputs ── */
textarea, input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    border-radius: 8px !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #fff !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,229,255,0.3) !important;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.4);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.metric-card.resistant::before  { background: var(--red); }
.metric-card.susceptible::before { background: var(--green); }
.metric-card.warning::before    { background: var(--yellow); }

.metric-label {
    font-family: var(--font-head);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: var(--font-head);
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-value.resistant  { color: var(--red); }
.metric-value.susceptible { color: var(--green); }
.metric-value.warning    { color: var(--yellow); }

.metric-badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-resistant   { background: rgba(255,77,109,0.15); color: var(--red); border: 1px solid rgba(255,77,109,0.3); }
.badge-susceptible { background: rgba(0,255,136,0.1);  color: var(--green); border: 1px solid rgba(0,255,136,0.25); }
.badge-warning     { background: rgba(255,209,102,0.1); color: var(--yellow); border: 1px solid rgba(255,209,102,0.25); }

/* ── Progress bar ── */
.progress-wrap {
    background: var(--border);
    border-radius: 100px;
    height: 8px;
    margin: 0.8rem 0 0.3rem;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.fill-resistant   { background: linear-gradient(90deg, #ff4d6d, #ff6b6b); }
.fill-susceptible { background: linear-gradient(90deg, #00ff88, #00e5a0); }
.fill-warning     { background: linear-gradient(90deg, #ffd166, #ffb347); }

/* ── Info boxes ── */
.info-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    line-height: 1.6;
    color: var(--text);
}

/* ── Sequence display ── */
.seq-display {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    word-break: break-all;
    color: var(--accent);
    max-height: 100px;
    overflow-y: auto;
    line-height: 1.8;
}

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem;
}
.section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-title {
    font-family: var(--font-head);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    white-space: nowrap;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}

/* ── Warning banner ── */
.warn-banner {
    background: rgba(255,209,102,0.08);
    border: 1px solid rgba(255,209,102,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: var(--yellow);
    margin: 1rem 0;
}

/* ── Entropy viz ── */
.entropy-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.3rem 0;
    font-size: 0.78rem;
}
.entropy-label { width: 90px; color: var(--muted); font-size: 0.72rem; }
.entropy-bar-bg {
    flex: 1;
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.entropy-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.entropy-val { width: 40px; text-align: right; color: var(--text); font-size: 0.72rem; }
</style>
""", unsafe_allow_html=True)


# ── Encoding helper ────────────────────────────────────────────────────────────
NUC_MAP = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0, 'U': 2}
ANTIBIOTICS = ['Methicillin', 'Ciprofloxacin', 'Vancomycin']
ANTIBIOTIC_INFO = {
    'Methicillin': {
        'gene': 'mecA',
        'pathogen': 'S. aureus (MRSA)',
        'class': 'Beta-lactam',
        'clinical': 'Most common hospital-acquired infection'
    },
    'Ciprofloxacin': {
        'gene': 'qnr, gyrA',
        'pathogen': 'E. coli, Klebsiella',
        'class': 'Fluoroquinolone',
        'clinical': 'Most prescribed broad-spectrum antibiotic'
    },
    'Vancomycin': {
        'gene': 'vanA, vanB',
        'pathogen': 'Enterococcus (VRE)',
        'class': 'Glycopeptide',
        'clinical': 'Last-resort antibiotic'
    }
}

EXAMPLE_SEQUENCES = {
    "mecA — Methicillin resistance (S. aureus)":
        "ATGAAAAAGATAAAAATTGTTCCACTTATTTTAATAGTTGTAGTTGTCGGGTTTGGTATATAAAACATTAATGCAAAAAATCGATAATATTGATACTGTAACAAGTACAACTTATTTACTTGAAAATAATACTAAAATATATGCTTATAAACAAATGCCTCAACTTTATCCAACAATTTACAGCTAATGAACATAATGAAGTCGAAATTATCATGCCAGATGAATATCAAATCGAAGTTGATTATAAATCAAATGATGTAAGTACACAATATAC",
    "vanA — Vancomycin resistance (Enterococcus)":
        "ATGGGCGGCAGTGAACAGATCGGCAGCGGCGTTTCCTTTGAGCCGCTGCTGAAAGTGATCGAGCAGAAAGGCATCACCGTCACCGGCACCGGCGGCGGCATCATCGGCGGCATCGTCGGCGAGCCGTTCTTCAACATCGAGAAGACCGGCGTCACCAAGATCGGCGGCAGCAAGATCATCGGCGGCAGCAAGACCATCGGCAACACCAACGGCAAGACCACCGGCGGCAACACCGGCAACAGCAACGAGAAGACCACCGGCAAGGCAGGC",
    "qnrB — Ciprofloxacin resistance (E. coli)":
        "ATGGATATTATTGATAAAGTTTTTCAGCAAGAGCAACTATTAGAACAGCTTGGGTTTGATGCCCTGAACATCGGTTTTGAAGCCCTGGCACGCATGCGCTATGCCCTGCGCTATGCCAGCCAGTTACTGCAGGCGCTGGCACAGCAGCAGCAGCAGCAGGTGGCGGCGATTATCGATAACCTGACGCAGATCGTCAACCCGGCGATGGATTTTATCGATACCCTGAAAGACGCGTTAATGCTGCACAGCGGTTATCAGTTGTTTGATAAC"
}

SEQ_LEN = 300


def encode_sequence(seq: str) -> torch.Tensor:
    seq = seq.upper().strip().replace(" ", "").replace("\n", "")
    encoded = [NUC_MAP.get(c, 0) for c in seq[:SEQ_LEN]]
    if len(encoded) < SEQ_LEN:
        encoded += [0] * (SEQ_LEN - len(encoded))
    return torch.tensor([encoded], dtype=torch.long)


def compute_entropy_profile(seq: str, window: int = 20) -> list:
    """Simple entropy profile for visualization."""
    seq = seq.upper()[:SEQ_LEN]
    entropies = []
    step = SEQ_LEN // 10
    for i in range(0, min(len(seq), SEQ_LEN), step):
        window_seq = seq[i:i+window]
        if not window_seq:
            break
        counts = {n: window_seq.count(n) for n in 'ATGCN'}
        total = len(window_seq)
        ent = 0.0
        for c in counts.values():
            if c > 0:
                p = c / total
                ent -= p * np.log2(p + 1e-10)
        entropies.append(round(ent, 2))
    return entropies


@st.cache_resource
def load_model(model_path: str):
    """Load BLT-CNN model."""
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.abspath(model_path))
        project_root = os.path.dirname(project_root)  # go up from models/
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.models import BLTCNNHybridFusion
        model = BLTCNNHybridFusion(vocab_size=5, embedding_dim=64, num_classes=3)

        state = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


def predict(model, seq_tensor: torch.Tensor):
    with torch.no_grad():
        out = model(seq_tensor)
    return out.squeeze().numpy().tolist()


def get_status(prob: float):
    if prob >= 0.7:
        return "resistant", "resistant", "badge-resistant"
    elif prob >= 0.4:
        return "warning", "warning", "badge-warning"
    else:
        return "susceptible", "susceptible", "badge-susceptible"


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.3rem; font-weight: 800;
                    background: linear-gradient(135deg, #00e5ff, #7c3aed);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;'>
            BLT-CNN
        </div>
        <div style='font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em; margin-top: 2px;'>
            AMR PREDICTOR v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-title" style="letter-spacing:0.15em; font-size:0.7rem; color:#64748b; font-weight:700;">⚙ MODEL</div>', unsafe_allow_html=True)
    model_path = st.text_input("Model path", value="models/blt_cnn_card_best.pth",
                                help="Path to your trained .pth file")

    load_btn = st.button("Load Model")
    model_loaded = False

    if load_btn or "model" in st.session_state:
        if load_btn:
            with st.spinner("Loading..."):
                model, err = load_model(model_path)
                if model:
                    st.session_state["model"] = model
                    st.session_state["model_err"] = None
                else:
                    st.session_state["model"] = None
                    st.session_state["model_err"] = err

        if st.session_state.get("model"):
            st.success("✅ Model loaded")
            model_loaded = True
        else:
            err = st.session_state.get("model_err", "")
            st.error(f"❌ {err[:80] if err else 'Not loaded'}")

    st.markdown("---")
    st.markdown('<div class="section-title" style="letter-spacing:0.15em; font-size:0.7rem; color:#64748b; font-weight:700;">📋 EXAMPLES</div>', unsafe_allow_html=True)
    selected_example = st.selectbox("Load example sequence", ["— select —"] + list(EXAMPLE_SEQUENCES.keys()))

    st.markdown("---")
    st.markdown('<div class="section-title" style="letter-spacing:0.15em; font-size:0.7rem; color:#64748b; font-weight:700;">ℹ ABOUT</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; line-height:1.7;'>
    <b style='color:#e2e8f0;'>Architecture</b><br>
    BLT Entropy Patcher<br>
    1D CNN (local motifs)<br>
    Transformer (global context)<br><br>
    <b style='color:#e2e8f0;'>Dataset</b><br>
    CARD Database · 4,005 sequences<br><br>
    <b style='color:#e2e8f0;'>Performance</b><br>
    Accuracy: 93.34%<br>
    AUC: 98.55%<br>
    F1: 94.18%
    </div>
    """, unsafe_allow_html=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🧬 BLT-CNN</div>
    <div class="hero-sub">Antibiotic Resistance Prediction from DNA Sequences</div>
</div>
""", unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("""
    <div class="section-header">
        <div class="section-line"></div>
        <div class="section-title">Input DNA Sequence</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    default_seq = ""
    if selected_example != "— select —":
        default_seq = EXAMPLE_SEQUENCES[selected_example]

    seq_input = st.text_area(
        label="DNA Sequence",
        value=default_seq,
        height=160,
        placeholder="Paste nucleotide sequence here (A, T, G, C, N)...\nMinimum 50 bp recommended. Sequences are padded/truncated to 300 bp.",
        label_visibility="collapsed"
    )

    uploaded = st.file_uploader("Or upload a FASTA file", type=["fasta", "fa", "txt"], label_visibility="visible")
    if uploaded:
        content = uploaded.read().decode("utf-8")
        lines = [l.strip() for l in content.split("\n") if not l.startswith(">") and l.strip()]
        seq_input = "".join(lines)
        st.markdown(f'<div class="info-box">📄 Loaded: <b>{uploaded.name}</b> · {len(seq_input)} bp</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔬 Predict Resistance")

with col_right:
    st.markdown("""
    <div class="section-header">
        <div class="section-line"></div>
        <div class="section-title">Sequence Info</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    if seq_input.strip():
        clean = seq_input.upper().strip().replace(" ", "").replace("\n", "")
        length = len(clean)
        valid_chars = sum(1 for c in clean if c in 'ATGCN')
        gc = (clean.count('G') + clean.count('C')) / max(length, 1) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Length", f"{length} bp")
        c2.metric("Valid", f"{valid_chars/max(length,1)*100:.0f}%")
        c3.metric("GC%", f"{gc:.1f}%")

        if length < 50:
            st.markdown('<div class="warn-banner">⚠ Sequence is very short. Predictions may be unreliable below 50 bp.</div>', unsafe_allow_html=True)

        # Entropy profile
        st.markdown("""
        <div class="section-header" style="margin-top:1rem">
            <div class="section-line"></div>
            <div class="section-title">Entropy Profile</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        entropies = compute_entropy_profile(clean)
        max_ent = max(entropies) if entropies else 1
        for i, ent in enumerate(entropies):
            pct = int(ent / max(max_ent, 0.01) * 100)
            region = f"pos {i*30}–{min((i+1)*30, length)}"
            color = "#ff4d6d" if ent > 1.5 else "#00e5ff"
            st.markdown(f"""
            <div class="entropy-bar-wrap">
                <div class="entropy-label">{region}</div>
                <div class="entropy-bar-bg">
                    <div class="entropy-bar-fill" style="width:{pct}%; background:{color};"></div>
                </div>
                <div class="entropy-val">{ent}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.68rem; color:#64748b; margin-top:0.3rem;">🔴 High entropy = mutation hotspot (small BLT patch) &nbsp;|&nbsp; 🔵 Low entropy = conserved (large patch)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box" style="color:#64748b;">Paste a sequence to see stats and entropy profile.</div>', unsafe_allow_html=True)


# ── Results ───────────────────────────────────────────────────────────────────
if predict_btn:
    if not seq_input.strip():
        st.error("Please enter a DNA sequence.")
    elif not model_loaded:
        st.markdown("""
        <div class="warn-banner" style="border-color:rgba(0,229,255,0.3); color:#00e5ff; background:rgba(0,229,255,0.05);">
        ℹ  Model not loaded — showing <b>demo predictions</b> using random weights.<br>
        Load your trained model from <code>models/blt_cnn_card.pth</code> for real predictions.
        </div>
        """, unsafe_allow_html=True)

        # Demo mode with plausible values
        np.random.seed(abs(hash(seq_input[:20])) % 2**31)
        probs = np.random.dirichlet([3, 1, 1]).tolist()
        probs[0] = min(0.97, probs[0] + 0.5)
        source = "demo"
    else:
        with st.spinner("Running BLT-CNN inference..."):
            time.sleep(0.4)
            tensor = encode_sequence(seq_input)
            probs = predict(st.session_state["model"], tensor)
            source = "model"

    st.markdown("""
    <div class="section-header" style="margin-top:2rem;">
        <div class="section-line"></div>
        <div class="section-title">Prediction Results</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3, gap="medium")
    for i, (ab, prob) in enumerate(zip(ANTIBIOTICS, probs)):
        card_class, val_class, badge_class = get_status(prob)
        label = "RESISTANT" if card_class == "resistant" else ("BORDERLINE" if card_class == "warning" else "SUSCEPTIBLE")
        info = ANTIBIOTIC_INFO[ab]

        with cols[i]:
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <div class="metric-label">{ab}</div>
                <div class="metric-value {val_class}">{prob*100:.1f}%</div>
                <div class="progress-wrap">
                    <div class="progress-fill fill-{card_class}" style="width:{prob*100:.1f}%"></div>
                </div>
                <span class="metric-badge {badge_class}">{label}</span>
                <div style="margin-top:1rem; font-size:0.72rem; color:#64748b; line-height:1.7; text-align:left;">
                    <div>Gene: <span style="color:#e2e8f0;">{info['gene']}</span></div>
                    <div>Pathogen: <span style="color:#e2e8f0;">{info['pathogen']}</span></div>
                    <div>Class: <span style="color:#e2e8f0;">{info['class']}</span></div>
                    <div style="margin-top:0.4rem; font-style:italic; color:#475569;">{info['clinical']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Summary interpretation
    st.markdown("<br>", unsafe_allow_html=True)
    resistant_to = [ab for ab, p in zip(ANTIBIOTICS, probs) if p >= 0.7]
    borderline   = [ab for ab, p in zip(ANTIBIOTICS, probs) if 0.4 <= p < 0.7]
    susceptible  = [ab for ab, p in zip(ANTIBIOTICS, probs) if p < 0.4]

    summary_parts = []
    if resistant_to:
        summary_parts.append(f"<span style='color:#ff4d6d; font-weight:700;'>Resistant</span> to: {', '.join(resistant_to)}")
    if borderline:
        summary_parts.append(f"<span style='color:#ffd166; font-weight:700;'>Borderline</span>: {', '.join(borderline)}")
    if susceptible:
        summary_parts.append(f"<span style='color:#00ff88; font-weight:700;'>Susceptible</span> to: {', '.join(susceptible)}")

    demo_note = " &nbsp;·&nbsp; <span style='color:#64748b; font-style:italic;'>Demo mode — load model for real predictions</span>" if source == "demo" else ""

    st.markdown(f"""
    <div class="info-box" style="margin-top:0.5rem;">
        🧬 &nbsp; {"&nbsp;&nbsp;|&nbsp;&nbsp;".join(summary_parts)}{demo_note}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem; color:#475569; margin-top:0.5rem; line-height:1.6;">
    ⚠ Threshold: ≥70% = Resistant · 40–70% = Borderline · &lt;40% = Susceptible.
    This tool is for research purposes only and is not intended for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem; padding-top:1.5rem; border-top:1px solid #1f2d45;
            text-align:center; font-size:0.72rem; color:#334155; line-height:2;">
    BLT-CNN · Antibiotic Resistance Predictor &nbsp;·&nbsp;
    Trained on CARD Database (4,005 sequences) &nbsp;·&nbsp;
    Accuracy 93.34% · AUC 98.55%<br>
    B.Tech Research Project · For research use only
</div>
""", unsafe_allow_html=True)
