"""
castelens/directions.py — build the seven directions and the matched-random null.

Direction = difference of means, residual stream, last token, unit-normalized.
This file SELF-TESTS with random vectors (no model needed): run it directly.
    python -m castelens.directions
"""
import torch


def build_direction(wrapper, pos_prompts, neg_prompts, layer):
    """V = normalize(mean(pos activations) - mean(neg activations))."""
    pos = wrapper.get_activations(pos_prompts, layer)[layer]
    neg = wrapper.get_activations(neg_prompts, layer)[layer]
    v = pos.mean(0) - neg.mean(0)
    return v / v.norm()


def cosine(a, b):
    a = a / a.norm()
    b = b / b.norm()
    return float((a @ b).item())


def orthogonalize(v, w):
    """v with all of w removed: returns the part of v perpendicular to w (unit)."""
    w = w / w.norm()
    out = v - (v @ w) * w
    return out / out.norm()


def geometry_matched_random(v1, v3, n=10, seed=0):
    """
    n random arrows that overlap v1 EXACTLY as much as v3 does.
    r = c*v1_hat + sqrt(1-c^2)*w,  c = cos(v1,v3), w random unit perpendicular to v1.
    Kills 'you moved caste by proxy' — the control has identical caste-overlap.
    """
    g = torch.Generator().manual_seed(seed)
    v1 = v1 / v1.norm()
    v3 = v3 / v3.norm()
    c = float((v1 @ v3).item())
    d = v1.numel()
    out = []
    while len(out) < n:
        w = torch.randn(d, generator=g, dtype=v1.dtype)
        w = w - (w @ v1) * v1          # make perpendicular to v1
        if w.norm() < 1e-3:            # degenerate: w was ~parallel to v1, resample
            continue
        w = w / w.norm()
        r = c * v1 + (1 - c ** 2) ** 0.5 * w
        out.append(r / r.norm())
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    d = 5120
    v1 = torch.randn(d); v1 = v1 / v1.norm()
    v3 = torch.randn(d); v3 = v3 / v3.norm()
    c = cosine(v1, v3)
    rs = geometry_matched_random(v1, v3, n=10, seed=12345)
    errs = [abs(cosine(r, v1) - c) for r in rs]
    assert max(errs) < 1e-4, errs
    vp = orthogonalize(v1, v3)
    assert abs(cosine(vp, v3)) < 1e-5
    print("SELF-TEST PASS")
    print(f"  cos(v1,v3) = {c:.4f}; matched randoms reproduce it (max err {max(errs):.2e})")
    print(f"  cos(v1_perp_v3, v3) = {cosine(vp, v3):.2e}  (should be ~0)")
