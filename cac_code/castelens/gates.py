"""
castelens/gates.py — the Phase-2 pilot gates. Run on Param-1 (free) first.
These decide scope. Two are hard; clearing them is the whole point of Day 15.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from .directions import build_direction, cosine
from .metrics import m4


def gate_B_cos_v3_v4(v3, v4):
    """GATE B: how collinear are disgust and anger here? > ~0.8 => switch DCC to V3_perp_V4 form."""
    c = cosine(v3, v4)
    print(f"[GATE B] cos(V3,V4) = {c:.3f}  -> {'COLLINEAR: use V3_perp_V4 headline' if c > 0.8 else 'separable: raw DCC ok'}")
    return c


def gate_C_v3_moves_m4(wrapper, v3, layer, neutral_prompts, disgust_words, anger_words, alpha):
    """
    GATE C (HARD): steering V3 must raise disgust-vocab (M4). If not, V3 is not a
    disgust direction -> drop the whole disgust arm. One run, before anything else.
    """
    base = m4(wrapper, neutral_prompts, disgust_words, anger_words)
    with wrapper.steering(v3, alpha, layer):
        steered = m4(wrapper, neutral_prompts, disgust_words, anger_words)
    delta = steered - base
    ok = delta > 0
    print(f"[GATE C] M4 base={base:.3f} steered={steered:.3f} delta={delta:+.3f}  -> {'PASS' if ok else 'FAIL: drop disgust arm'}")
    return ok, delta


def gate_1_v1_separates_testfold(wrapper, v1, layer, test_marked_prompts, test_neutral_prompts):
    """Check (1): does V1 separate UNSEEN test-fold names from neutral? AUC well above 0.5."""
    pm = wrapper.get_activations(test_marked_prompts, layer)[layer]
    pn = wrapper.get_activations(test_neutral_prompts, layer)[layer]
    v = (v1 / v1.norm())
    sm = (pm @ v).numpy()
    sn = (pn @ v).numpy()
    y = np.r_[np.ones(len(sm)), np.zeros(len(sn))]
    auc = roc_auc_score(y, np.r_[sm, sn])
    print(f"[CHECK 1] TEST-fold separation AUC = {auc:.3f}  -> {'green' if auc > 0.7 else 'weak'}")
    return auc


def gate_2_ses_2x2(wrapper, v1, layer, cells):
    """
    Check (2): caste axis follows caste AGAINST the wealth gradient.
    cells: dict with keys 'wealthy_brahmin','poor_brahmin','wealthy_dalit','poor_dalit'
           -> each a list of prompts. We project and confirm Brahmin>Dalit within each SES level.
    """
    v = v1 / v1.norm()
    proj = {k: (wrapper.get_activations(p, layer)[layer] @ v).mean().item() for k, p in cells.items()}
    wealthy_gap = proj["wealthy_brahmin"] - proj["wealthy_dalit"]
    poor_gap = proj["poor_brahmin"] - proj["poor_dalit"]
    ok = wealthy_gap > 0 and poor_gap > 0
    print(f"[CHECK 2] caste gap | wealthy={wealthy_gap:+.3f} poor={poor_gap:+.3f}  -> {'holds vs SES' if ok else 'SES confound'}")
    return ok, proj


# Example wiring (pseudo — fill prompts/words from frozen datasets):
# w = ModelWrapper("bharatgenai/Param-1-2.9B-Instruct")
# L = 14
# v1 = build_direction(w, d1_pos, d1_neg, L)
# v3 = build_direction(w, d4_pos, d4_neg, L)
# v4 = build_direction(w, d5_pos, d5_neg, L)
# alpha = 4.0 * w.mean_norm(d1_pos[:5], L)
# gate_B_cos_v3_v4(v3, v4)
# gate_C_v3_moves_m4(w, v3, L, neutral_prompts, disgust_words, anger_words, alpha)
