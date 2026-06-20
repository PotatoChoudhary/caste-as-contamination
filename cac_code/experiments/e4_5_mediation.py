"""
E4.5 — Mediation (the title-maker). MAF = how much of caste's effect dies when you
strip its disgust component, vs stripping an equivalent random chunk.
  numerator   = β(V1⊥V3 -> M1)            (caste with disgust removed)
  denominator = median β(V1⊥r_ŵ -> M1)    (caste with matched-random removed)
  MAF = 1 - num/denom.  CI>0 => mediation; >0.5 => primarily. -> Fig 3b.
Run:  python -m experiments.e4_5_mediation
"""
import numpy as np
from castelens import ModelWrapper, orthogonalize
from castelens.build_directions import build_all
from castelens.metrics import m1, maf_ci, bootstrap_ci
from castelens import config, figures
from castelens.utils import save_result, load_data


def steer_effect_gaps(w, vec, alpha, L, items, base_gaps):
    with w.steering(vec, alpha, L):
        _, g = m1(w, items)
    return g - base_gaps


def run():
    w = ModelWrapper(config.MODEL, layers_path=config.LAYERS_PATH)
    L = config.LAYER
    data = {k: load_data(f"{k}_frozen.json") for k in ["D1", "D2", "D4", "D5", "D6", "D10"]}
    items = load_data("D7a_frozen.json")["items"]
    V = build_all(w, data, L)
    mn = w.mean_norm(data["D1"]["pos"][:8], L)
    a = config.alpha(mn, max(config.DOSE_COEFFS))
    base, base_gaps = m1(w, items)

    full = steer_effect_gaps(w, V["V1"], a, L, items, base_gaps)                 # full caste
    disgust_stripped = steer_effect_gaps(w, orthogonalize(V["V1"], V["V3"]), a, L, items, base_gaps)
    rand_rows = np.array([
        steer_effect_gaps(w, orthogonalize(V["V1"], r), a, L, items, base_gaps)
        for r in V["rand_matched"]
    ])

    maf_pt, maf_lo, maf_hi = maf_ci(disgust_stripped, rand_rows)
    print(f"  full caste ΔM1            = {full.mean():+.3f}")
    print(f"  disgust-stripped ΔM1      = {disgust_stripped.mean():+.3f}")
    print(f"  matched-random-stripped   = {rand_rows.mean(1).mean():+.3f} (median across {len(rand_rows)})")
    print(f"  MAF = {maf_pt:.3f} CI[{maf_lo:.3f},{maf_hi:.3f}]  "
          f"mediation={maf_lo > 0}  primarily={maf_pt > config.THRESH['maf_primarily']}")

    # converse: steer V3⊥V1 -> does caste bias still move above null? (disgust drives caste)
    conv = steer_effect_gaps(w, orthogonalize(V["V3"], V["V1"]), a, L, items, base_gaps)
    conv_ci = bootstrap_ci(np.mean, conv, n=config.N_BOOTSTRAP)
    print(f"  converse (V3⊥V1 -> M1) = {conv_ci[0]:+.3f} CI[{conv_ci[1]:+.3f},{conv_ci[2]:+.3f}]")

    pts = [float(full.mean()), float(rand_rows.mean(1).mean()), float(disgust_stripped.mean())]
    los = [float(np.percentile(full, 2.5)), float(np.percentile(rand_rows.mean(0), 2.5)),
           float(np.percentile(disgust_stripped, 2.5))]
    his = [float(np.percentile(full, 97.5)), float(np.percentile(rand_rows.mean(0), 97.5)),
           float(np.percentile(disgust_stripped, 97.5))]
    figures.bars_ci(["full caste", "rand-stripped", "disgust-stripped"], pts, los, his,
                    "ΔM1 (caste bias effect)", "E4.5: mediation — short last bar = the paper",
                    "fig3b_mediation.png")

    save_result("e4_5_mediation", {
        "maf": [maf_pt, maf_lo, maf_hi],
        "maf_mediation": bool(maf_lo > 0),
        "maf_primarily": bool(maf_pt > config.THRESH["maf_primarily"]),
        "converse_v3perp_v1": conv_ci,
        "bars": {"labels": ["full", "rand_stripped", "disgust_stripped"], "points": pts},
    })


if __name__ == "__main__":
    run()
