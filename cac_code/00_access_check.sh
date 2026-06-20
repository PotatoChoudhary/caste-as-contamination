#!/usr/bin/env bash
# PHASE 0.1 — HuggingFace access. Run line by line.
# First accept the licenses IN THE BROWSER on these two pages (gated):
#   https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Base-2503
#   https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503

pip install -U "huggingface_hub[cli]"

huggingface-cli login            # paste a READ token
huggingface-cli whoami           # must print your username

# CRITICAL: confirm the BASE has real weights (not instruct-only).
# This lists files; you must see model-*.safetensors in the output.
huggingface-cli repo info mistralai/Mistral-Small-3.1-24B-Base-2503 --files 2>/dev/null \
  | grep -i "safetensors" || echo ">> NO SAFETENSORS FOUND ON BASE — fallback to Instruct-vs-Sarvam-M lineage, note it."

# Log the result so future-you remembers.
echo "Base downloadable check done: $(date)" >> ../PRE_EVENT_LOG.txt
