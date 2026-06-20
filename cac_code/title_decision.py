"""
title_decision.py — Sunday 10:00. Reads results/*.json and picks the title from the
PRE-REGISTERED table. Pure logic; self-tests with no model:  python title_decision.py --selftest
"""
import sys

TITLES = {
    "circuit": "Indian Sovereign LLMs Implement Caste on a Disgust Circuit",
    "separate": "Shared Geometry, Separate Machinery: Caste and Disgust in Indian LLMs",
    "coupled": "Caste Bias Is Coupled to Disgust Geometry in Indian Sovereign LLMs",
    "own": "Caste Is Its Own Representation: The Purity Theory Fails Inside the Machine",
    "proxy": "The Caste Signal Is a Socioeconomic Proxy — An Audit Finding in Itself",
}


def decide(maf_lo, coupling_excludes_0, same_machinery, confound_wins):
    """Apply the five-branch pre-registered table. Returns (key, title)."""
    if confound_wins:
        return "proxy", TITLES["proxy"]
    if maf_lo > 0:                                   # MAF CI excludes 0 -> mediation
        key = "circuit" if same_machinery else "separate"
        return key, TITLES[key]
    if coupling_excludes_0:                          # coupling holds but no mediation
        return "coupled", TITLES["coupled"]
    return "own", TITLES["own"]


def from_results():
    from castelens.utils import load_result
    e2 = load_result("e2_confounds")
    e4 = load_result("e4_dissociation")
    e45 = load_result("e4_5_mediation")
    e65 = load_result("e6_5_components")
    key, title = decide(
        maf_lo=e45["maf"][1],
        coupling_excludes_0=e4["dcc_excludes_0"],
        same_machinery=e65["same_machinery"],
        confound_wins=e2["confound_wins"],
    )
    print(f"CHOSEN TITLE [{key}]:\n  {title}")
    return key, title


def _selftest():
    assert decide(0.3, True, True, False)[0] == "circuit"
    assert decide(0.3, True, False, False)[0] == "separate"
    assert decide(-0.1, True, False, False)[0] == "coupled"
    assert decide(-0.1, False, False, False)[0] == "own"
    assert decide(0.9, True, True, True)[0] == "proxy"      # confound overrides everything
    print("SELF-TEST PASS (all five branches)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        from_results()
