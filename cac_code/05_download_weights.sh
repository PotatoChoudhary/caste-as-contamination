#!/usr/bin/env bash
# PHASE 5 — pre-download all weights Thursday night. 30B FIRST (long pole, 129 GB).
# Target a persistent volume >= 400 GB. Run on the rented box, not your laptop.
set -e
export HF_HOME=/workspace/hf            # point at the big volume

dl () { huggingface-cli download "$1" --local-dir "/workspace/weights/$2" --local-dir-use-symlinks False; }

dl sarvamai/sarvam-30b                                  sarvam-30b      # 129 GB — start first
dl sarvamai/sarvam-m                                    sarvam-m
dl mistralai/Mistral-Small-3.1-24B-Base-2503            mistral-base
dl mistralai/Mistral-Small-3.1-24B-Instruct-2503        mistral-instruct
dl bharatgenai/Param-1-2.9B-Instruct                    param-1
# dl bharatgenai/Param2-17B-A2.4B-Thinking              param-2        # stretch only
echo "ALL WEIGHTS DOWNLOADED. df -h /workspace below:"
df -h /workspace
