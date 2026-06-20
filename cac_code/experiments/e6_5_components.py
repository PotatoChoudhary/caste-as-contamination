"""
E6.5 — Component attribution (earns or retires the word "circuit"). For each
attention and MLP block, measure how much it WRITES toward V1 vs V3. Same blocks
write both -> shared machinery ("circuit"). Different blocks -> "shared direction,
separate machinery" (often the more careful, better title). Either way you win.
Run:  python -m experiments.e6_5_components
"""
import numpy as np
from castelens import ModelWrapper
from castelens.build_directions import build_all
from castelens import config
from castelens.utils import save_result, load_data


def write_profile(comp_outputs, v, layers):
    """Mean projection of each (layer, component) output onto v_hat -> profile vector."""
    vh = (v / v.norm())
    prof = {}
    for L in layers:
        for c in ("attn", "mlp"):
            prof[f"L{L}.{c}"] = float((comp_outputs[L][c] @ vh).mean())
    return prof


def run():
    w = ModelWrapper(config.MODEL, layers_path=config.LAYERS_PATH)
    L = config.LAYER
    data = {k: load_data(f"{k}_frozen.json") for k in ["D1", "D2", "D4", "D5", "D6", "D10"]}
    probe = load_data("D7a_frozen.json")["neutral_prompts"]
    V = build_all(w, data, L)

    layers = list(range(len(w.layers)))
    comp = w.get_component_outputs(probe, layers)
    p1 = write_profile(comp, V["V1"], layers)
    p3 = write_profile(comp, V["V3"], layers)

    keys = list(p1.keys())
    a = np.array([p1[k] for k in keys]); b = np.array([p3[k] for k in keys])
    corr = float(np.corrcoef(a, b)[0, 1])

    # top writers for each, and their overlap
    top1 = set(np.array(keys)[np.argsort(-np.abs(a))[:8]])
    top3 = set(np.array(keys)[np.argsort(-np.abs(b))[:8]])
    overlap = len(top1 & top3) / len(top1 | top3)
    same_machinery = corr > 0.5 and overlap > 0.4

    print(f"  write-profile correlation(V1,V3) = {corr:+.3f}")
    print(f"  top-writer overlap (Jaccard)     = {overlap:.2f}")
    print(f"  verdict: {'SHARED machinery -> circuit' if same_machinery else 'SEPARATE machinery -> shared direction only'}")

    save_result("e6_5_components", {
        "profile_correlation": corr, "top_writer_overlap": overlap,
        "same_machinery": bool(same_machinery),
        "top_writers_v1": sorted(top1), "top_writers_v3": sorted(top3),
    })


if __name__ == "__main__":
    run()
