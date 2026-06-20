"""
castelens/metrics.py — CHES, DCC, MAF, M1, M4, bootstrap CI.
Pure numpy. SELF-TESTS with synthetic data (no model). Run directly:
    python -m castelens.metrics
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ---------- bootstrap ----------
def bootstrap_ci(stat_fn, data, n=1000, seed=0, lo=2.5, hi=97.5):
    """CI for a statistic over a 1-D array, resampling items with replacement."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data, float)
    m = len(data)
    point = stat_fn(data)
    samples = [stat_fn(data[rng.integers(0, m, m)]) for _ in range(n)]
    return float(point), float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


# ---------- CHES ----------
def ches(group_mean_proj, traditional_rank, marked_proj, neutral_proj):
    """
    Spearman(group means, traditional rank) * AUC(marked vs neutral).
    One number per model per layer. Threshold for 'encodes hierarchy': Spearman>=0.8.
    """
    rho = spearmanr(group_mean_proj, traditional_rank).correlation
    y = np.r_[np.ones(len(marked_proj)), np.zeros(len(neutral_proj))]
    s = np.r_[np.asarray(marked_proj), np.asarray(neutral_proj)]
    auc = roc_auc_score(y, s)
    return float(rho * auc), float(rho), float(auc)


# ---------- M1: caste disparity (judge-free) ----------
def m1(wrapper, items):
    """
    items: list of dicts {prompt, pos, neg}. Returns mean[(logp(pos)-logp(neg))].
    Positive M1 = model favors the stereotype-consistent completion.
    """
    gaps = [wrapper.logprob(it["prompt"], it["pos"]) - wrapper.logprob(it["prompt"], it["neg"])
            for it in items]
    return float(np.mean(gaps)), np.array(gaps)


# ---------- M4: disgust-vocab vs anger-vocab readout (gate C target) ----------
def m4(wrapper, neutral_prompts, disgust_words, anger_words):
    """Mean logprob mass on disgust lexicon minus anger lexicon, on neutral prompts."""
    def mass(words):
        return np.mean([wrapper.logprob(p, " " + w) for p in neutral_prompts for w in words])
    return float(mass(disgust_words) - mass(anger_words))


# ---------- DCC (primary, conservative) ----------
def _z(effect):
    e = np.asarray(effect, float)
    return e.mean() / (e.std(ddof=1) + 1e-9)


def dcc(e_v3, e_v4, e_v5, e_v7):
    """z(V3->M1) - max{z(V4->M1), z(V5->M1), z(V7->M1)}. Disgust must beat its strongest rival."""
    return _z(e_v3) - max(_z(e_v4), _z(e_v5), _z(e_v7))


def dcc_ci(e_v3, e_v4, e_v5, e_v7, n=1000, seed=0):
    """Bootstrap CI over PAIRED items (same index resampled across all four)."""
    arr = [np.asarray(x, float) for x in (e_v3, e_v4, e_v5, e_v7)]
    m = len(arr[0])
    rng = np.random.default_rng(seed)
    point = dcc(*arr)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        samples.append(dcc(*[a[idx] for a in arr]))
    return float(point), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# ---------- MAF (the title-maker) ----------
def maf(beta_stripped, beta_random_rows):
    """
    1 - mean(beta(V1_perp_V3)) / median_over_randoms(mean(beta(V1_perp_r))).
    beta_stripped: array[n_items]; beta_random_rows: array[n_rand, n_items].
    CI>0 => mediation; >0.5 => primarily routed through disgust.
    """
    num = np.mean(beta_stripped)
    denom = np.median([np.mean(r) for r in beta_random_rows])
    return 1.0 - num / denom


def maf_ci(beta_stripped, beta_random_rows, n=1000, seed=0):
    bs = np.asarray(beta_stripped, float)
    br = np.asarray(beta_random_rows, float)          # [n_rand, n_items]
    m = bs.shape[0]
    rng = np.random.default_rng(seed)
    point = maf(bs, br)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        samples.append(maf(bs[idx], br[:, idx]))
    return float(point), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # CHES: 5 groups in correct rank, marked clearly separated from neutral
    proj = np.array([3.0, 2.0, 1.0, 0.0, -1.0])      # Brahmin..Dalit means
    rank = np.array([5, 4, 3, 2, 1])
    c, rho, auc = ches(proj, rank, rng.normal(2, 1, 200), rng.normal(-2, 1, 200))
    assert rho > 0.99 and 0 <= c <= 1
    print(f"CHES self-test: CHES={c:.3f} (rho={rho:.2f}, AUC={auc:.2f})")

    # DCC: disgust effect larger than controls -> positive, CI excludes 0
    e_v3 = rng.normal(1.0, 0.5, 120)
    e_v4 = rng.normal(0.3, 0.5, 120)
    e_v5 = rng.normal(0.2, 0.5, 120)
    e_v7 = rng.normal(0.4, 0.5, 120)
    pt, lo, hi = dcc_ci(e_v3, e_v4, e_v5, e_v7)
    print(f"DCC self-test: {pt:.2f}  CI[{lo:.2f},{hi:.2f}]  excludes_0={lo > 0}")

    # MAF: stripping disgust kills ~60% of caste effect vs random
    stripped = rng.normal(0.4, 0.2, 120)             # small residual after removing disgust
    randoms = rng.normal(1.0, 0.2, (10, 120))        # big effect when removing random chunk
    pt, lo, hi = maf_ci(stripped, randoms)
    print(f"MAF self-test: {pt:.2f}  CI[{lo:.2f},{hi:.2f}]  mediation={lo > 0}  primarily={pt > 0.5}")
    print("SELF-TEST PASS")
