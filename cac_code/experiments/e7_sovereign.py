"""
E7 — Sovereign comparison. CHES across Base, Instruct, Sarvam-M, Sarvam-30B.
Shows what post-training sharpened, and from-scratch vs finetuned. -> Fig 4.
RULE: compare CHES SCORES across models, never raw cosines (not comparable across archs).
Run:  python -m experiments.e7_sovereign
"""
import numpy as np
from castelens import ModelWrapper, build_direction
from castelens.metrics import ches
from castelens import config, figures
from castelens.utils import save_result, load_data


def ches_for_model(model_id, layers_path, d1):
    w = ModelWrapper(model_id, layers_path=layers_path)
    n = len(w.layers)
    best = None
    for L in range(n):
        v = build_direction(w, d1["extract"]["marked"], d1["extract"]["neutral"], L)
        gm = [(w.get_activations(d1["test"]["groups"][g], L)[L] @ (v / v.norm())).mean().item()
              for g in config.GROUPS]
        mk = (w.get_activations(d1["test"]["marked"], L)[L] @ (v / v.norm())).numpy()
        nt = (w.get_activations(d1["test"]["neutral"], L)[L] @ (v / v.norm())).numpy()
        c = ches(gm, config.TRADITIONAL_RANK, mk, nt)[0]
        if best is None or c > best[1]:
            best = (L, c)
    del w
    return best     # (layer, ches)


def run():
    d1 = load_data("D1_frozen.json")
    results = {}
    for name, (mid, path) in config.ROSTER.items():
        try:
            L, c = ches_for_model(mid, path, d1)
            results[name] = {"best_layer": L, "ches": c}
            print(f"  {name:18s} CHES={c:.3f} @ L{L}")
        except Exception as e:
            print(f"  {name:18s} SKIPPED ({type(e).__name__}: {e})")

    names = list(results.keys())
    vals = [results[n]["ches"] for n in names]
    figures.bars_ci(names, vals, vals, vals, "CHES (best layer)",
                    "E7: caste-hierarchy encoding across sovereign models", "fig4_sovereign.png")
    save_result("e7_sovereign", {"per_model": results})


if __name__ == "__main__":
    run()
