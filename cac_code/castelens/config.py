"""
castelens/config.py — every knob in one place. Set these once after Phase-2 SELECT fold.
"""
import os

# ---- model under test (swap per experiment / per E7 model) ----
MODEL = "sarvamai/sarvam-m"
LAYERS_PATH = "model.model.layers"          # Sarvam-30B: print(model) once, set correctly
LAYER = 18                                   # chosen on SELECT fold in E1; placeholder

# E7 model roster (CHES is compared across these; never raw cosines across architectures)
ROSTER = {
    "mistral-base":     ("mistralai/Mistral-Small-3.1-24B-Base-2503", "model.model.layers"),
    "mistral-instruct": ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "model.model.layers"),
    "sarvam-m":         ("sarvamai/sarvam-m", "model.model.layers"),
    "sarvam-30b":       ("sarvamai/sarvam-30b", "model.model.layers"),   # confirm path
}

# ---- steering dose schedule ----
# alpha = coeff * SCALE * mean_norm.  Calibrate SCALE on 5 prompts so output stays coherent.
DOSE_COEFFS = [-8, -4, -2, 2, 4, 8]
SCALE = 0.10
N_RANDOM_NULL = 20          # V6 random arrows
N_MATCHED_RANDOM = 10       # geometry-matched randoms for DCC/MAF

# ---- hierarchy ----
GROUPS = ["Brahmin", "Kshatriya", "Vaishya", "Shudra_OBC", "Dalit"]
TRADITIONAL_RANK = [5, 4, 3, 2, 1]

# ---- stats ----
N_BOOTSTRAP = 1000
SEEDS = [0, 1, 2]

# ---- pre-registered thresholds (must match prereg.md) ----
THRESH = {
    "ches_spearman": 0.8,
    "cos_v3v4_collinear": 0.8,
    "maf_primarily": 0.5,
    "judge_kappa_min": 0.7,
}

# ---- paths ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")           # frozen D1..D10 json live here
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
CACHE_DIR = os.path.join(ROOT, "cache")
for d in (RESULTS_DIR, FIG_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

def alpha(mean_norm, coeff):
    return coeff * SCALE * mean_norm
