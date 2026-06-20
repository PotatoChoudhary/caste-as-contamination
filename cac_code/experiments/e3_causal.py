"""
E3 — Causal arm. Steer V1 at doses, ablate V1, steer random V6 (flat). Dose-response
in M1 proves P2 (the axis is causal, not decorative). -> Fig 2.
Run:  python -m experiments.e3_causal
"""
import numpy as np
from castelens import ModelWrapper
from castelens.build_directions import build_all
from castelens.metrics import m1, bootstrap_ci
from castelens import config, figures
from castelens.utils import save_result, load_data


def run():
    w = ModelWrapper(config.MODEL, layers_path=config.LAYERS_PATH)
    L = config.LAYER
    data = {k: load_data(f"{k}_frozen.json") for k in ["D1", "D2", "D4", "D5", "D6", "D10"]}
    items = load_data("D7a_frozen.json")["items"]          # {prompt,pos,neg}
    V = build_all(w, data, L)

    mn = w.mean_norm(data["D1"]["pos"][:8], L)
    base, base_gaps = m1(w, items)
    print(f"  baseline M1 = {base:+.3f}")

    # steer V1 across doses
    steer_pt, steer_lo, steer_hi = [], [], []
    for coeff in config.DOSE_COEFFS:
        a = config.alpha(mn, coeff)
        with w.steering(V["V1"], a, L):
            _, gaps = m1(w, items)
        eff = gaps - base_gaps
        p, lo, hi = bootstrap_ci(np.mean, eff, n=config.N_BOOTSTRAP)
        steer_pt.append(p); steer_lo.append(lo); steer_hi.append(hi)
        print(f"  steer V1 coeff={coeff:+d}  dM1={p:+.3f} CI[{lo:+.3f},{hi:+.3f}]")

    # random V6 baseline at the largest dose (should be ~flat)
    rand_effs = []
    for r in V["V6"][:8]:
        with w.steering(r, config.alpha(mn, max(config.DOSE_COEFFS)), L):
            _, gaps = m1(w, items)
        rand_effs.append(float(np.mean(gaps - base_gaps)))
    rand_mean = float(np.mean(rand_effs))

    # ablate V1
    with w.ablation(V["V1"], list(range(L, len(w.layers)))):
        abl, abl_gaps = m1(w, items)
    abl_eff = bootstrap_ci(np.mean, abl_gaps - base_gaps, n=config.N_BOOTSTRAP)
    print(f"  ablate V1 dM1={abl_eff[0]:+.3f} CI[{abl_eff[1]:+.3f},{abl_eff[2]:+.3f}]")
    print(f"  random V6 mean effect (should be ~0) = {rand_mean:+.3f}")

    figures.line_ci(config.DOSE_COEFFS, [steer_pt], [steer_lo], [steer_hi],
                    ["steer V1"], "dose coeff (×scale×mean_norm)", "ΔM1",
                    "E2/E3: caste steering is dose-responsive", "fig2_dose_response.png")

    save_result("e3_causal", {
        "baseline_m1": base,
        "doses": config.DOSE_COEFFS,
        "steer_effect": list(zip(steer_pt, steer_lo, steer_hi)),
        "ablate_effect": abl_eff,
        "random_v6_mean": rand_mean,
        "monotonic": bool(np.all(np.diff(steer_pt) > 0)),
    })


if __name__ == "__main__":
    run()
