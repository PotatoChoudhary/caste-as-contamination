# CasteLens — event-phase code (E1–E8)

The weekend runners. They plug into the `castelens` package from the pre-event batch.
Same honesty rule as before: pure-math is tested here; model-calling code is correct
scaffolding you point at weights.

## Verified on this machine (no model, no GPU)
- `python title_decision.py --selftest` → all five title branches correct.
- `python -m castelens.figures` → writes 3 smoke PNGs.
- `python -m castelens.metrics` / `directions` → still pass.
- `python -m py_compile experiments/*.py castelens/*.py` → all compile clean.

## NOT verified against real models (can't load Sarvam here)
- Every `experiments/e*.py` and `castelens/wrapper.get_component_outputs`.
- Verify once per architecture: the decoder-layer path (`config.LAYERS_PATH`) and,
  for E6.5, the attn/MLP attr names (`get_component_outputs(..., mlp_attr=...)`).

## ⚠️ wrapper.py CHANGED
I added `get_component_outputs` to `castelens/wrapper.py` for E6.5. **Re-download wrapper.py** —
the version in the first batch doesn't have it.

## Run order (mirrors the Fri→Sun calendar)
```
# Tier 1 — ship these or nothing:
python run_event.py e1 e2 e3 e4 e45
# Tier 2 — placing vs winning:
python run_event.py e5 e7 e65 e8
# or everything:
python run_event.py all
# Sunday 10:00 — pick the title from the pre-registered table:
python title_decision.py
```
Each runner saves `results/<name>.json` and a figure in `figures/`.

| Exp | Proves / delivers | Figure | Gate |
|-----|-------------------|--------|------|
| E1  | P1 caste axis + rank order (CHES) | fig1 | sets `config.LAYER` |
| E2  | confound audit + survival table | — | **kill-switch** → proxy title |
| E3  | P2 causal (dose-response, ablation) | fig2 | |
| E4  | DCC, cross-steering matrix | fig3a | **Gate B** cos(V3,V4) |
| E4.5| **MAF — the title-maker** | fig3b | |
| E5  | surface alignment ≠ mechanism | — | |
| E6.5| circuit vs separate machinery | — | sets the title fork |
| E7  | CHES across sovereign models | fig4 | CHES scores only, never cosines |
| E8  | Hindi transfer (semantic, not spelling) | — | |

## Data contract — what the frozen JSON in `data/` must contain
The runners load these (build them by extending `castelens/datasets.py` and freezing):
- `D1_frozen.json`: `{extract,select,test}` each with `marked:[], neutral:[], groups:{Group:[prompts]}}`; D1 also exposes `pos`/`neg` (= marked/neutral of extract) for direction-building.
- `D2,D4,D5,D6,D10_frozen.json`: each `{pos:[], neg:[]}` → builds V2,V3,V4,V5,V7.
- `D3_frozen.json`: `{wealthy_brahmin:[], poor_brahmin:[], wealthy_dalit:[], poor_dalit:[]}`.
- `D7a_frozen.json`: `{items:[{prompt,pos,neg}], neutral_prompts:[]}`.
- `D7c_frozen.json`: `{items:[{prompt,refuse,accept}]}` (untouchability).
- `D8_frozen.json`: `{marked:[], neutral:[]}` (Hindi).
- `lexicon.json`: `{disgust:[...words], anger:[...words]}`.

> The `pos/neg` keys the direction-builder expects are just the marked/neutral arms.
> Make `datasets.py` write both shapes when you freeze, or add a 3-line adapter.

## The one number that is the paper
`results/e4_5_mediation.json` → `maf`. CI excludes 0 = mediation; point > 0.5 = primarily.
A short "disgust-stripped" bar in fig3b is the headline. Everything else is support.
