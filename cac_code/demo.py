"""
demo.py — Phase-4 demo stub (delta K). Stub on Param-1 now; point at the real
model Sunday. A judge types a sentence, sees its caste score + hierarchy position.
    python demo.py "Sharma reviewed the application"
"""
import sys
import torch
from castelens import ModelWrapper, build_direction
from castelens.datasets import GROUPS, TRADITIONAL_RANK

MODEL = "bharatgenai/Param-1-2.9B-Instruct"   # swap to sarvamai/sarvam-m on Sunday
LAYER = 14                                    # set to the SELECT-fold best layer

# minimal pos/neg prompt lists to build V1 (replace with frozen D1 EXTRACT fold)
POS = ["Sharma walked in and", "Trivedi applied and"]
NEG = ["The person walked in and", "The individual applied and"]


def main():
    sentence = sys.argv[1] if len(sys.argv) > 1 else "Sharma reviewed the application"
    w = ModelWrapper(MODEL)
    v1 = build_direction(w, POS, NEG, LAYER)

    act = w.get_activations([sentence], LAYER)[LAYER][0]
    score = float(act @ (v1 / v1.norm()))

    # group reference projections (build once from frozen D1 group prompts)
    ref = {g: float(w.get_activations([f"{g} reviewed the application"], LAYER)[LAYER][0] @ v1)
           for g in GROUPS}
    nearest = min(ref, key=lambda g: abs(ref[g] - score))

    print(f"\nsentence : {sentence}")
    print(f"caste-axis score : {score:+.3f}")
    print(f"nearest group    : {nearest}")
    print("group reference scores:", {g: round(v, 2) for g, v in ref.items()})


if __name__ == "__main__":
    main()
