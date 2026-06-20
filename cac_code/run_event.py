"""
run_event.py — the weekend driver. Runs experiments in priority order, honoring the
Saturday-noon kill-switch (if E2 says the signal is a proxy, you still win — pivot title).
Run subsets:  python run_event.py e1 e2 e3 e4 e45
              python run_event.py all
"""
import sys
import importlib

ORDER = {
    "e1":  "experiments.e1_geometry",        # Fri/Sat  -> P1, Fig1
    "e2":  "experiments.e2_confounds",       # Sat AM   -> audit, kill-switch
    "e3":  "experiments.e3_causal",          # Sat mid  -> P2, Fig2
    "e4":  "experiments.e4_dissociation",    # Sat PM   -> DCC, GateB, Fig3a
    "e45": "experiments.e4_5_mediation",     # Sat PM   -> MAF, Fig3b (title-maker)
    "e5":  "experiments.e5_prompt_ablation", # gaps
    "e7":  "experiments.e7_sovereign",       # Sun AM   -> Fig4
    "e65": "experiments.e6_5_components",    # Sun AM   -> circuit vs separate
    "e8":  "experiments.e8_hindi",           # Sun AM
}
TIER1 = ["e1", "e2", "e3", "e4", "e45"]      # ship these or nothing


def main(which):
    if which == ["all"]:
        which = list(ORDER.keys())
    for key in which:
        print(f"\n===== {key.upper()} ({ORDER[key]}) =====")
        try:
            importlib.import_module(ORDER[key]).run()
        except Exception as e:
            print(f"  !! {key} FAILED: {type(e).__name__}: {e}")
            if key in TIER1:
                print("  TIER-1 experiment failed — fix before continuing, do not skip.")
                break

    # kill-switch check after E2
    if "e2" in which:
        try:
            from castelens.utils import load_result
            if load_result("e2_confounds").get("confound_wins"):
                print("\n>>> KILL-SWITCH: pivot to the socioeconomic-proxy headline (title row 5). "
                      "This is a WIN, not a failure. Reframe intro accordingly.")
        except Exception:
            pass

    print("\nDone. Sunday 10:00 -> run:  python title_decision.py")


if __name__ == "__main__":
    main(sys.argv[1:] or TIER1)
