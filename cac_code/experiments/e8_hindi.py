"""
E8 — Hindi transfer. Project Hindi caste sentences onto the ENGLISH-built V1.
Transfer (AUC stays high) => V1 is a semantic axis, not a spelling artifact.
Run:  python -m experiments.e8_hindi
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from castelens import ModelWrapper, build_direction
from castelens import config
from castelens.utils import save_result, load_data


def run():
    w = ModelWrapper(config.MODEL, layers_path=config.LAYERS_PATH)
    L = config.LAYER
    d1 = load_data("D1_frozen.json")
    d8 = load_data("D8_frozen.json")     # Hindi mirror: {marked:[], neutral:[]}

    v1_en = build_direction(w, d1["extract"]["marked"], d1["extract"]["neutral"], L)
    vh = v1_en / v1_en.norm()
    sm = (w.get_activations(d8["marked"], L)[L] @ vh).numpy()
    sn = (w.get_activations(d8["neutral"], L)[L] @ vh).numpy()
    y = np.r_[np.ones(len(sm)), np.zeros(len(sn))]
    auc = float(roc_auc_score(y, np.r_[sm, sn]))
    print(f"  Hindi-on-English-V1 AUC = {auc:.3f} -> {'transfers (semantic axis)' if auc > 0.7 else 'weak/no transfer'}")
    save_result("e8_hindi", {"hindi_transfer_auc": auc, "transfers": bool(auc > 0.7)})


if __name__ == "__main__":
    run()
