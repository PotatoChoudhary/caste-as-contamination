"""
E2 — Confound battery (the named AUDIT deliverable). Reports a SURVIVAL FRACTION
for each control: how much of the naive caste signal is left after the control.
If the signal is largely socioeconomic, that's title row 5 — a winning headline.

survival_fraction = controlled_signal / naive_signal   (clipped to [0,1])
Run:  python -m experiments.e2_confounds
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from castelens import ModelWrapper, build_direction, cosine
from castelens import config
from castelens.utils import save_result, load_data


def auc_signal(w, v, layer, marked, neutral):
    """Separation strength as (AUC - 0.5)*2, in [0,1]."""
    vh = v / v.norm()
    sm = (w.get_activations(marked, layer)[layer] @ vh).numpy()
    sn = (w.get_activations(neutral, layer)[layer] @ vh).numpy()
    y = np.r_[np.ones(len(sm)), np.zeros(len(sn))]
    return float((roc_auc_score(y, np.r_[sm, sn]) - 0.5) * 2)


def survival(controlled, naive):
    return float(np.clip(controlled / naive, 0, 1)) if naive > 1e-9 else 0.0


def run():
    w = ModelWrapper(config.MODEL, layers_path=config.LAYERS_PATH)
    L = config.LAYER
    d1 = load_data("D1_frozen.json")
    d3 = load_data("D3_frozen.json")     # SES 2x2 cells

    # naive signal: train and test on EXTRACT (the optimistic, leaky number)
    v_naive = build_direction(w, d1["extract"]["marked"], d1["extract"]["neutral"], L)
    naive = auc_signal(w, v_naive, L, d1["extract"]["marked"], d1["extract"]["neutral"])

    rows = []

    # control (a): held-out TEST names (generalization)
    s_test = auc_signal(w, v_naive, L, d1["test"]["marked"], d1["test"]["neutral"])
    rows.append(("held-out TEST names", s_test, survival(s_test, naive)))

    # control (b): above random-cosine null. cos(V1, random) ~ 0; report margin over null band.
    g_null = [cosine(v_naive, __import__("torch").randn(w.d_model)) for _ in range(200)]
    null_band = float(np.percentile(np.abs(g_null), 95))
    rows.append(("random-cosine null (|cos| 95pct)", null_band, None))

    # control (c): SES 2x2 — caste gap retained within each wealth level
    def proj_mean(prompts):
        return float((w.get_activations(prompts, L)[L] @ (v_naive / v_naive.norm())).mean())
    wealthy_gap = proj_mean(d3["wealthy_brahmin"]) - proj_mean(d3["wealthy_dalit"])
    poor_gap = proj_mean(d3["poor_brahmin"]) - proj_mean(d3["poor_dalit"])
    ses_retained = min(wealthy_gap, poor_gap) / max(wealthy_gap, poor_gap, 1e-9)
    rows.append(("SES 2x2 (gap retained vs wealth)", float(ses_retained), float(np.clip(ses_retained, 0, 1))))

    # control (d): within-region matched names only
    s_region = auc_signal(w, v_naive, L, d1["test"].get("marked_region", d1["test"]["marked"]),
                          d1["test"].get("neutral_region", d1["test"]["neutral"]))
    rows.append(("within-region matched", s_region, survival(s_region, naive)))

    print(f"  naive signal = {naive:.3f}")
    for name, val, surv in rows:
        s = "n/a" if surv is None else f"{surv:.2f}"
        print(f"  {name:38s} value={val:.3f}  survival={s}")

    # headline flip: if everything collapses near the null, the proxy title wins
    survivals = [s for _, _, s in rows if s is not None]
    confound_wins = bool(survivals and np.mean(survivals) < 0.3)

    save_result("e2_confounds", {
        "naive_signal": naive,
        "controls": [{"control": n, "value": v, "survival": s} for n, v, s in rows],
        "mean_survival": float(np.mean(survivals)) if survivals else None,
        "confound_wins": confound_wins,
    })
    if confound_wins:
        print("  >> KILL-SWITCH: signal is largely a proxy. Pivot to title row 5 (audit headline).")


if __name__ == "__main__":
    run()
