# CasteLens — pre-event code

Run order, one file at a time. What's verified vs what you must wire is stated plainly — no pretending untested model code is plug-and-play.

## Status: what actually runs right now

These **self-test on this machine, no models, no GPU** (I ran them):
- `python -m castelens.metrics` → CHES / DCC / MAF on synthetic data, all asserts pass.
- `python -m castelens.directions` → matched-random + orthogonalize math, asserts pass.
- `python -m castelens.datasets` → builds + freezes (sha256) whatever you fill in.

These are **correct scaffolding you must point at real weights** (I can't load Sarvam here to test the exact module paths):
- `castelens/wrapper.py`, `castelens/gates.py`, `demo.py`.
- The one thing to verify per architecture: the decoder-layer path. Default `model.model.layers` works for Mistral / Sarvam-M / Param-1. For Sarvam-30B, run `01_hook_proof.py`-style `print(model)` once and pass the right `layers_path=`.

## Order to run

| # | File | Phase | Needs GPU? |
|---|------|-------|-----------|
| 1 | `00_access_check.sh` | 0.1 — HF access, confirm Base downloadable | no |
| 2 | `01_hook_proof.py` | 0.3 — prove a hook fires on Param-1 | Kaggle free |
| 3 | `python -m castelens.metrics` | 1.2 — confirm metric math | no |
| 4 | `python -m castelens.directions` | 1.1 — confirm direction math | no |
| 5 | fill `castelens/datasets.py`, `python -m castelens.datasets` | 1.3 — build + freeze D1..D10 | no |
| 6 | wire + run `castelens/gates.py` on Param-1 | 2 — the pilot gates (B, C, checks 1–2) | Kaggle free |
| 7 | `03_preregister.sh` | 3 — commit + hash | no |
| 8 | `demo.py "some sentence"` | 4 — demo stub on Param-1 | Kaggle free |
| 9 | `05_download_weights.sh` | 5 — Thursday pre-download, 30B first | on the rented box |

## Install

```
pip install -U torch transformers accelerate numpy scipy scikit-learn
```

## The verbs (so you can read the wrapper)
- **steer**: `x ← x + α·v̂` — `with w.steering(v, alpha, layer):`
- **ablate**: `x ← x − (x·v̂)v̂` — `with w.ablation(v, layers):`
- **orthogonalize**: `directions.orthogonalize(v, w)` → part of `v` ⊥ `w`
- α: set as `frac * w.mean_norm(prompts, layer)`, not a raw number.

## What is NOT here, on purpose
- Real caste surname lists — fill from DECASTE (2505.14971) and Indian-BhED (2309.08573) appendices. I won't fabricate them.
- The full experiment runners (E1–E8) — those are the *event*, not pre-event. This repo gets you to a frozen, gate-cleared, pre-registered start line.

## Two gates that can kill the project (Phase 2, file 6)
- **Gate B** `cos(V3,V4)`: > ~0.8 → switch DCC headline to the `V3⊥V4` form.
- **Gate C** `V3→M4` (HARD): if steering V3 doesn't raise disgust-vocab, V3 isn't a disgust direction → drop the disgust arm, note it. One run, before anything downstream.
