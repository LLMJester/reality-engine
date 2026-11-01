"""Paired cluster bootstrap (spec §8.4), as v2 tournament/analyze.py: items are clustered by system, every comparison is
paired on identical items, and a difference counts only if its CI excludes zero."""
import numpy as np


def paired_ci(a, b, clusters, n_boot=4000, alpha=0.05, seed=0):
    """a, b: {item: score}; clusters: {item: system}. Returns {est, lo, hi, n_items, n_systems, improved}."""
    items = sorted(set(a) & set(b))
    if not items:
        return {"est": None, "lo": None, "hi": None, "n_items": 0, "n_systems": 0, "improved": False}
    systems = sorted({clusters[i] for i in items})
    by = {s: [i for i in items if clusters[i] == s] for s in systems}
    d = {i: a[i] - b[i] for i in items}
    est = float(np.mean([d[i] for i in items]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(systems), len(systems), replace=True)
        vals = [d[i] for k in pick for i in by[systems[k]]]
        boots.append(np.mean(vals))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"est": est, "lo": float(lo), "hi": float(hi), "n_items": len(items), "n_systems": len(systems),
            "improved": bool(lo > 0)}


def mean_ci(a, clusters, n_boot=4000, alpha=0.05, seed=0):
    zero = {i: 0.0 for i in a}
    r = paired_ci(a, zero, clusters, n_boot, alpha, seed)
    r.pop("improved")
    return r
