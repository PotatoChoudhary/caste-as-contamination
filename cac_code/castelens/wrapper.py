"""
castelens/wrapper.py — the one ModelWrapper reused for every model.

Verify ONE thing per architecture: the path to the decoder-layer list.
  - Mistral / Sarvam-M / Param-1:  model.model.layers   (default)
  - Sarvam-30B (SarvamMoEForCausalLM): print(model) once, confirm the path,
    pass layers_path="model.layers" or whatever the tree shows.
Everything else is architecture-agnostic because we hook the residual stream
(the output of each decoder layer), not the internals.
"""
import torch
import numpy as np
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer


def _edit_output(out, fn):
    """Decoder layers may return a tuple (hidden, ...). Edit hidden only."""
    if isinstance(out, tuple):
        return (fn(out[0]),) + out[1:]
    return fn(out)


class ModelWrapper:
    def __init__(self, model_id, device="cuda", dtype=torch.bfloat16, layers_path="model.model.layers"):
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype, device_map=device
        ).eval()
        self.device = device
        self.layers = self._resolve(layers_path)
        self.d_model = self.model.config.hidden_size
        self.dtype = next(self.model.parameters()).dtype

    def _resolve(self, path):
        obj = self.model
        for p in path.split("."):
            obj = getattr(obj, p)
        return obj

    # ---------- READ: residual stream activations ----------
    @torch.no_grad()
    def get_activations(self, prompts, layers, position="last", batch_size=8):
        """Returns {layer: tensor[n_prompts, d_model]} (fp32, on CPU)."""
        if isinstance(layers, int):
            layers = [layers]
        out = {L: [] for L in layers}
        store = {}
        handles = []

        def mk(L):
            def h(m, i, o):
                store[L] = (o[0] if isinstance(o, tuple) else o).detach()
            return h

        for L in layers:
            handles.append(self.layers[L].register_forward_hook(mk(L)))
        try:
            self.tok.padding_side = "right"
            for s in range(0, len(prompts), batch_size):
                batch = prompts[s:s + batch_size]
                enc = self.tok(batch, return_tensors="pt", padding=True).to(self.device)
                self.model(**enc)
                last = enc.attention_mask.sum(1) - 1            # last real token index
                for L in layers:
                    h = store[L]
                    if position == "last":
                        sel = h[torch.arange(h.size(0)), last]
                    else:
                        sel = h[:, position, :]
                    out[L].append(sel.float().cpu())
        finally:
            for hd in handles:
                hd.remove()
        return {L: torch.cat(out[L], 0) for L in layers}

    @torch.no_grad()
    def mean_norm(self, prompts, layer):
        """Mean residual norm — use to set alpha = fraction * this."""
        a = self.get_activations(prompts, layer)[layer]
        return a.norm(dim=-1).mean().item()

    # ---------- WRITE: steering / ablation (context managers) ----------
    @contextmanager
    def steering(self, v, alpha, layers):
        """Add alpha * v_hat to the residual at given layer(s) for the duration."""
        if isinstance(layers, int):
            layers = [layers]
        v = (v / v.norm()).to(self.device).to(self.dtype)
        handles = []

        def hook(m, i, o):
            return _edit_output(o, lambda h: h + alpha * v)

        for L in layers:
            handles.append(self.layers[L].register_forward_hook(hook))
        try:
            yield
        finally:
            for hd in handles:
                hd.remove()

    @contextmanager
    def ablation(self, v, layers):
        """Project v_hat OUT of the residual: x <- x - (x.v_hat) v_hat."""
        if isinstance(layers, int):
            layers = [layers]
        v = (v / v.norm()).to(self.device).to(self.dtype)
        handles = []

        def hook(m, i, o):
            def f(h):
                return h - (h @ v).unsqueeze(-1) * v
            return _edit_output(o, f)

        for L in layers:
            handles.append(self.layers[L].register_forward_hook(hook))
        try:
            yield
        finally:
            for hd in handles:
                hd.remove()

    # ---------- READ: per-component writes (E6.5 only) ----------
    @torch.no_grad()
    def get_component_outputs(self, prompts, layers, attn_attr="self_attn", mlp_attr="mlp",
                              position="last", batch_size=8):
        """
        Output of each layer's attention block and MLP block at the chosen position.
        Returns {layer: {'attn': tensor[n,d], 'mlp': tensor[n,d]}}.
        MoE MLP is ONE unit (post-routing combined output) — never per-expert.
        Verify attr names per arch: Mistral/Sarvam-M/Param-1 use 'self_attn'/'mlp';
        for Sarvam-30B confirm the MLP attr (may be 'mlp' / 'block_sparse_moe' / 'feed_forward').
        """
        if isinstance(layers, int):
            layers = [layers]
        store, handles = {}, []

        def mk(key):
            def h(m, i, o):
                store[key] = (o[0] if isinstance(o, tuple) else o).detach()
            return h

        for L in layers:
            handles.append(getattr(self.layers[L], attn_attr).register_forward_hook(mk((L, "attn"))))
            handles.append(getattr(self.layers[L], mlp_attr).register_forward_hook(mk((L, "mlp"))))
        out = {L: {"attn": [], "mlp": []} for L in layers}
        try:
            self.tok.padding_side = "right"
            for s in range(0, len(prompts), batch_size):
                enc = self.tok(prompts[s:s + batch_size], return_tensors="pt", padding=True).to(self.device)
                self.model(**enc)
                last = enc.attention_mask.sum(1) - 1
                for L in layers:
                    for comp in ("attn", "mlp"):
                        h = store[(L, comp)]
                        sel = h[torch.arange(h.size(0)), last] if position == "last" else h[:, position, :]
                        out[L][comp].append(sel.float().cpu())
        finally:
            for hd in handles:
                hd.remove()
        return {L: {c: torch.cat(out[L][c], 0) for c in ("attn", "mlp")} for L in layers}

    # ---------- READ: log-prob of a completion (judge-free metric core) ----------
    @torch.no_grad()
    def logprob(self, prompt, completion):
        """Sum log P(completion tokens | prompt). Higher = model prefers it."""
        full = self.tok(prompt + completion, return_tensors="pt").to(self.device)
        plen = self.tok(prompt, return_tensors="pt").input_ids.size(1)
        lp = torch.log_softmax(self.model(**full).logits, dim=-1)
        ids = full.input_ids[0]
        return sum(lp[0, p - 1, ids[p]].item() for p in range(plen, ids.size(0)))
