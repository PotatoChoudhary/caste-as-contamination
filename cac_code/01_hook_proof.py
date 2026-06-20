"""
PHASE 0.3 — Prove a forward hook fires on Param-1 (free, on Kaggle GPU).
Paste into a Kaggle notebook with GPU ON. This is the foundational test:
if a hook doesn't fire here, nothing downstream works.

    pip install -U transformers accelerate
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "bharatgenai/Param-1-2.9B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()

# STEP 1 — print the module tree so you can SEE where the decoder layers live.
# For most HF decoder models it's model.model.layers ; verify here, don't assume.
print(model)
print("\n--- locating decoder layers ---")
try:
    layers = model.model.layers
    print("FOUND at model.model.layers, count =", len(layers))
except AttributeError:
    raise SystemExit("layers NOT at model.model.layers — read the tree above and fix the path")

# STEP 2 — register a forward hook on a mid layer, run one prompt, catch the tensor.
cache = {}
def hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out   # decoder layers often return a tuple
    cache["resid"] = h.detach()

mid = len(layers) // 2
handle = layers[mid].register_forward_hook(hook)

enc = tok("A Brahmin man walked into the office and", return_tensors="pt").to("cuda")
with torch.no_grad():
    model(**enc)
handle.remove()

print("\nHOOK FIRED. residual tensor shape:", tuple(cache["resid"].shape))
print("hidden size d_model =", model.config.hidden_size)
print(">> If you see a shape like [1, seq, d_model], Phase 0.3 PASSES.")
