"""
E4 — Dissociation. (a) geometry: cos of each affect direction with V1 vs null.
(b) the 6-row cross-steering matrix (the figure that wins/loses the hackathon):
steer {V3,V4,V5,V7,matched-random,V6} -> read M1 and M3; steer V1 -> read M4.
Compute DCC. GATE B (cos V3,V4) reported here. -> Fig 3a.
Run:  python -m experiments.e4_dissociation
"""
import numpy as np
from castelens import ModelWrapper, cosine
from castelens.build_directions import build_all
from castelens.metrics import m1, m4, dcc_ci
from castelens import config, figures
from castelens.utils import save_result, load_data


def m3_proxy(w, items):
    """Untouchability proxy (judge-free): logp(refusal) - logp(acceptance) on D7c."""
    gaps = [w.logprob(it["prompt"], it["refuse"]) - w.logprob(it["prompt"], it["accept"]) for it in items]
    return float(np.mean(gaps)), np.array(gaps)


def run():
    w = ModelWrapper(config.MODEL, layers_path=config.LAYERS_PATH)
    L = config.LAYER
    data = {k: load_data(f"{k}_frozen.json") for k in ["D1", "D2", "D4", "D5", "D6", "D10"]}
    items = load_data("D7a_frozen.json")["items"]
    d7c = load_data("D7c_frozen.json")["items"]
    neutral = load_data("D7a_frozen.json")["neutral_prompts"]
    disg = load_data("lexicon.json")["disgust"]; ang = load_data("lexicon.json")["anger"]
    V = build_all(w, data, L)
    mn = w.mean_norm(data["D1"]["pos"][:8], L)
    a = config.alpha(mn, max(config.DOSE_COEFFS))

    # GATE B
    cos_v3v4 = cosine(V["V3"], V["V4"])
    print(f"  [GATE B] cos(V3,V4) = {cos_v3v4:.3f} -> "
          f"{'COLLINEAR: switch DCC to V3⊥V4 form' if cos_v3v4 > config.THRESH['cos_v3v4_collinear'] else 'separable'}")

    # geometry row
    geom = {k: cosine(V[k], V["V1"]) for k in ["V3", "V4", "V5", "V7"]}
    print("  cos with V1:", {k: round(v, 3) for k, v in geom.items()})

    base_m1, base_m1_gaps = m1(w, items)
    base_m3, base_m3_gaps = m3_proxy(w, d7c)

    rows = {"V3": V["V3"], "V4": V["V4"], "V5": V["V5"], "V7": V["V7"],
            "rand_matched": V["rand_matched"][0], "V6": V["V6"][0]}
    matrix, effects_m1 = [], {}
    for name, vec in rows.items():
        with w.steering(vec, a, L):
            _, g1 = m1(w, items)
            _, g3 = m3_proxy(w, d7c)
        e1 = g1 - base_m1_gaps; e3 = g3 - base_m3_gaps
        effects_m1[name] = e1
        matrix.append([float(e1.mean()), float(e3.mean())])
        print(f"  steer {name:13s} ΔM1={e1.mean():+.3f} ΔM3={e3.mean():+.3f}")

    # steer V1 -> read M4 (does caste raise disgust vocab? the converse direction)
    base_m4 = m4(w, neutral, disg, ang)
    with w.steering(V["V1"], a, L):
        m4_steered = m4(w, neutral, disg, ang)
    print(f"  steer V1 ΔM4(disgust-vocab) = {m4_steered - base_m4:+.3f}")

    # DCC (use raw form unless GATE B says collinear -> caller should swap V3 for V3⊥V4)
    dcc_pt, dcc_lo, dcc_hi = dcc_ci(effects_m1["V3"], effects_m1["V4"], effects_m1["V5"], effects_m1["V7"])
    print(f"  DCC = {dcc_pt:+.3f} CI[{dcc_lo:+.3f},{dcc_hi:+.3f}] excludes_0={dcc_lo > 0}")

    figures.heatmap(matrix, list(rows.keys()), ["ΔM1", "ΔM3"],
                    "E4: cross-steering (rows=steered, cols=outcome)", "fig3a_cross_steering.png")

    save_result("e4_dissociation", {
        "cos_v3v4": cos_v3v4, "collinear": cos_v3v4 > config.THRESH["cos_v3v4_collinear"],
        "cos_with_v1": geom, "cross_steer_matrix": matrix, "rows": list(rows.keys()),
        "dcc": [dcc_pt, dcc_lo, dcc_hi], "dcc_excludes_0": bool(dcc_lo > 0),
        "v1_raises_m4": float(m4_steered - base_m4),
    })


if __name__ == "__main__":
    run()
