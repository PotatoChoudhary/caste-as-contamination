"""
E1 — Geometry. Build V1, select layer, compute CHES on the TEST fold, sweep CHES
across all layers. Proves P1 (the caste axis exists, in rank order). -> Fig 1.

Inputs (frozen, loaded from data/): D1 split into EXTRACT/SELECT/TEST with group tags.
Run:  python -m experiments.e1_geometry
"""
import numpy as np
from castelens import ModelWrapper, build_direction
from castelens.metrics import ches
from castelens import config, figures
from castelens.utils import save_result, load_data


def project(wrapper, prompts, v, layer):
    a = wrapper.get_activations(prompts, layer)[layer]
    return (a @ (v / v.norm())).numpy()


def run():
    w = ModelWrapper(config.MODEL, layers_path=config.LAYERS_PATH)
    d1 = load_data("D1_frozen.json")     # expects folds: extract/select/test, each {marked:[], neutral:[], groups:{g:[prompts]}}
    n_layers = len(w.layers)

    # --- sweep CHES across layers using EXTRACT to build, TEST to score ---
    ches_by_layer = []
    for L in range(n_layers):
        v = build_direction(w, d1["extract"]["marked"], d1["extract"]["neutral"], L)
        group_means = [project(w, d1["test"]["groups"][g], v, L).mean() for g in config.GROUPS]
        marked = project(w, d1["test"]["marked"], v, L)
        neutral = project(w, d1["test"]["neutral"], v, L)
        c, rho, auc = ches(group_means, config.TRADITIONAL_RANK, marked, neutral)
        ches_by_layer.append({"layer": L, "ches": c, "spearman": rho, "auc": auc})
        print(f"  L{L:02d}  CHES={c:.3f}  rho={rho:+.2f}  AUC={auc:.3f}")

    # --- choose layer on SELECT fold (best CHES) ---
    sel = []
    for L in range(n_layers):
        v = build_direction(w, d1["extract"]["marked"], d1["extract"]["neutral"], L)
        gm = [project(w, d1["select"]["groups"][g], v, L).mean() for g in config.GROUPS]
        mk = project(w, d1["select"]["marked"], v, L)
        nt = project(w, d1["select"]["neutral"], v, L)
        sel.append(ches(gm, config.TRADITIONAL_RANK, mk, nt)[0])
    chosen = int(np.argmax(sel))
    print(f"  chosen layer (SELECT fold) = {chosen}; update config.LAYER to this.")

    figures.line_ci(
        list(range(n_layers)),
        [[r["ches"] for r in ches_by_layer]],
        [[r["ches"] for r in ches_by_layer]],
        [[r["ches"] for r in ches_by_layer]],
        ["CHES (TEST fold)"], "layer", "CHES", "E1: caste hierarchy encoding by layer", "fig1_ches_by_layer.png")

    save_result("e1_geometry", {
        "chosen_layer": chosen,
        "ches_at_chosen": ches_by_layer[chosen],
        "ches_by_layer": ches_by_layer,
        "passes_threshold": ches_by_layer[chosen]["spearman"] >= config.THRESH["ches_spearman"],
    })


if __name__ == "__main__":
    run()
