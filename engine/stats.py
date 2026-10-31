"""The engine's statistics. Code only, no AI; deterministic (fixed bootstrap seed).

Scores are y[design][item] in [0, 1]. Every comparison is paired: computed on the items every design compared has a
score for. Ranges are 95% ranges; bootstrap ranges resample clusters of items (the pack's cluster_of), and t tests
average within clusters and test across them (df = clusters - 1).

  compare(...)       means, ranges, the could-be-best set (Hsu's multiple comparisons with the best, studentised,
                     one-sided) and every lead over the control (one-sided bootstrap p, Holm-corrected)
  by_level(...)      the same means split by the pack's difficulty level
  retention(...)     the gated keep rule for a 2^m factorial: a change is kept only if its own effect survives Holm
                     with a positive estimate, and the design it gives then beats the current best head to head
  screen(...)        one-at-a-time screen of single-part changes against a fixed bar
  final_sequence(..) fixed-sequence (gatekeeping) tests on the sealed items

Ported from the run 5 analysis of "Describing Reality for AI" (rpp-v3 v3/analysis5.py); the replay test proves the
numbers are the same.
"""
import itertools
import json
import math

import numpy as np


class Retired(RuntimeError):
    """An analysis tried to read an item the pack has retired."""


class Stats:
    def __init__(self, cluster_of=None, level_of=None, boot=10_000, seed=20261028, alpha=0.05, retired=frozenset()):
        self.cluster_of = cluster_of or (lambda i: i)
        self.level_of = level_of or (lambda i: None)
        self.boot, self.seed, self.alpha = boot, seed, alpha
        self.retired = frozenset(retired)

    # ------------------------------------------------------------------ data
    def matrix(self, scores, designs, items):
        """Complete-case matrix Y[d, i] over the items every design has a score for: (Y, items used, items dropped)."""
        bad = [i for i in items if i in self.retired]
        if bad:
            raise Retired(f"{len(bad)} retired items, e.g. {bad[0]}")
        used = [i for i in items if all(scores.get(d, {}).get(i) is not None for d in designs)]
        Y = np.array([[float(scores[d][i]) for i in used] for d in designs]) if used else np.zeros((len(designs), 0))
        return Y, used, [i for i in items if i not in used]

    def _weights(self, items, boot=None, seed=None):
        """Bootstrap weights (boot x items): clusters drawn with replacement; every item of a drawn cluster counts."""
        cl = [self.cluster_of(i) for i in items]
        ids = sorted(set(cl))
        idx = {c: j for j, c in enumerate(ids)}
        member = np.zeros((len(ids), len(items)))
        for n, c in enumerate(cl):
            member[idx[c], n] = 1.0
        rng = np.random.default_rng(self.seed if seed is None else seed)
        counts = rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=boot or self.boot)
        return counts @ member

    def boot_means(self, Y, items, boot=None, seed=None):
        """boot x designs bootstrap means (the same draws for every design, so differences stay paired)."""
        W = self._weights(items, boot, seed)
        return (W @ Y.T) / W.sum(axis=1, keepdims=True)

    def _clusters(self, used):
        cl = [self.cluster_of(i) for i in used]
        ids = sorted(set(cl))
        member = np.zeros((len(ids), len(used)))
        for n, c in enumerate(cl):
            member[ids.index(c), n] = 1.0
        return ids, member / member.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ tests
    def holm(self, pvals):
        """{name: p} -> {name: {p, p_holm, reject}} (Holm step-down, family-wise error alpha)."""
        order = sorted(pvals, key=lambda k: pvals[k])
        m, out, running, stop = len(order), {}, 0.0, False
        for r, k in enumerate(order):
            adj = min(1.0, (m - r) * pvals[k])
            running = max(running, adj)
            rej = (not stop) and pvals[k] <= self.alpha / (m - r)
            stop = stop or not rej
            out[k] = {"p": round(pvals[k], 5), "p_holm": round(running, 5), "reject": bool(rej)}
        return out

    def compare(self, scores, designs, items, control=None):
        """Means, ranges, the could-be-best set and leads over the control for one item set."""
        Y, used, dropped = self.matrix(scores, designs, items)
        if not used:
            return {"n_items": 0, "dropped": len(dropped)}
        m = Y.mean(axis=1)
        B = self.boot_means(Y, used)
        cen = B - m
        # Hsu's MCB, studentised and one-sided: design d against the best other design; the margin is q95 x se(d),
        # q95 the 95th percentile of the largest studentised centred difference over all designs at once
        k = len(designs)
        best_j = [max((j for j in range(k) if j != d), key=lambda j: m[j]) if k > 1 else d for d in range(k)]
        best_other = np.array([m[best_j[d]] for d in range(k)])
        diff_cen = np.array([cen[:, best_j[d]] - cen[:, d] for d in range(k)])
        se = np.maximum(diff_cen.std(axis=1, ddof=1), 1e-9)
        q95 = float(np.percentile((diff_cen / se[:, None]).max(axis=0), 95)) if k > 1 else 0.0
        margins = q95 * se
        out = {"n_items": len(used), "dropped": len(dropped), "mcb_q95": round(q95, 4), "designs": {}}
        for d, name in enumerate(designs):
            out["designs"][name] = {"mean": round(float(m[d]), 4), "mean_exact": round(float(m[d]), 12),
                                    "range": ci(B[:, d]), "behind_best": round(float(best_other[d] - m[d]), 4),
                                    "mcb_margin": round(float(margins[d]), 4),
                                    "could_be_best": bool(best_other[d] - m[d] <= margins[d])}
        if control in designs:
            c = designs.index(control)
            pv, leads = {}, {}
            for d, name in enumerate(designs):
                if d == c:
                    continue
                diff = B[:, d] - B[:, c]
                leads[name] = {"lead": round(float(m[d] - m[c]), 4), "range": ci(diff)}
                pv[name] = p_greater(diff)
            h = self.holm(pv)
            for name in leads:
                leads[name].update(h[name])
            out["leads_over_control"] = leads
        out["ranking"] = sorted(designs, key=lambda n: (-out["designs"][n]["mean_exact"], n))
        return out

    def by_level(self, scores, designs, items):
        groups = {}
        for i in items:
            groups.setdefault(self.level_of(i), []).append(i)
        out = {}
        for g, its in sorted(groups.items(), key=lambda x: str(x[0])):
            if g is None:
                continue
            Y, used, _ = self.matrix(scores, designs, its)
            if not used:
                continue
            B = self.boot_means(Y, used, 2000)
            out[str(g)] = {n: {"mean": round(float(Y[d].mean()), 4), "range": ci(B[:, d]), "n": len(used)}
                           for d, n in enumerate(designs)}
        return out

    def rank_agreement(self, scores, designs, items_a, items_b):
        """Spearman correlation of the designs' means on two item sets, with a bootstrap range."""
        Ya, ua, _ = self.matrix(scores, designs, items_a)
        Yb, ub, _ = self.matrix(scores, designs, items_b)
        if not ua or not ub:
            return None
        a, b = list(Ya.mean(axis=1)), list(Yb.mean(axis=1))
        Ba, Bb = self.boot_means(Ya, ua, 2000, self.seed + 1), self.boot_means(Yb, ub, 2000, self.seed + 2)
        rs = np.array([spearman(Ba[j], Bb[j]) for j in range(2000)])
        rs = rs[np.isfinite(rs)]
        return {"spearman": round(spearman(a, b), 3), "range": ci(rs) if len(rs) else None}

    def paired_lead(self, scores, a, b, items):
        """Bootstrap lead of a over b: {lead, range, n_items, passes}."""
        Y, used, dropped = self.matrix(scores, [a, b], items)
        if not used:
            return {"n_items": 0}
        B = self.boot_means(Y, used)
        r = ci(B[:, 0] - B[:, 1])
        return {"lead": round(float(Y[0].mean() - Y[1].mean()), 4), "range": r, "n_items": len(used),
                "dropped": len(dropped), "passes": r[0] > 0}

    def paired_cluster(self, scores, a, b, items):
        """mean(a) - mean(b) on the items both have, averaged within clusters; t test across clusters."""
        Y, used, dropped = self.matrix(scores, [a, b], items)
        if not used:
            return {"n_items": 0, "dropped": len(dropped)}
        ids, M = self._clusters(used)
        out = t_summary(M @ (Y[0] - Y[1]))
        out.update(n_items=len(used), dropped=len(dropped), clusters=len(ids))
        return out

    # ------------------------------------------------------------------ factorials
    def factorial_tests(self, Y, used, factors, family):
        """Every main effect and two-way interaction of a 2^m factorial (plus three-way terms named in family): the
        per-item contrast mean(+) - mean(-), averaged within clusters, t test across clusters; Holm over family."""
        rows = np.array(factorial_levels(len(factors)), float)
        cols = {f: rows[:, j] for j, f in enumerate(factors)}
        for a, b in itertools.combinations(factors, 2):
            cols[f"{a}:{b}"] = cols[a] * cols[b]
        for name in family:
            p = name.split(":")
            if len(p) == 3 and all(x in factors for x in p):
                cols[name] = cols[p[0]] * cols[p[1]] * cols[p[2]]
        ids, M = self._clusters(used)
        out, pv = {}, {}
        for name, x in cols.items():
            r = t_summary(M @ ((x @ Y) / (len(x) / 2)))
            out[name] = {"effect": r["estimate"], "se": r["se"], "t": r["t"], "p": r["p"], "df": r["df"],
                         "range": r["range"], "confirmatory": name in family}
            if name in family:
                pv[name] = r["p"]
        for name, h in self.holm(pv).items():
            out[name]["holm"] = h
        return out

    def retention(self, scores, design_names, items, factors, family=None):
        """The gated keep rule. design_names: the 2^m designs in standard order (the first factor changes fastest);
        design_names[0] (no change) is the current best. Main effects that survive Holm with a positive estimate are
        kept; a two-way interaction is kept only if it survives Holm and both its main effects are kept; every other
        term counts as zero, and undecided parts stay as they are. d* maximises the kept terms (ties: fewer changes);
        a part stays changed only while its effect at the other chosen levels is positive. d* replaces the current
        best only if its direct paired contrast with the current best has its whole 95% range above zero."""
        family = list(family) if family is not None else \
            list(factors) + [f"{a}:{b}" for a, b in itertools.combinations(factors, 2)]
        Y, used, dropped = self.matrix(scores, design_names, items)
        if len(used) < 2:
            raise ValueError(f"only {len(used)} items complete across the {len(design_names)} designs")
        tests = self.factorial_tests(Y, used, factors, family)
        kept = [f for f in factors if (tests[f].get("holm") or {}).get("reject") and tests[f]["effect"] > 0]
        pairs = [(a, b) for a, b in itertools.combinations(factors, 2)
                 if a in kept and b in kept and (tests[f"{a}:{b}"].get("holm") or {}).get("reject")]

        def value(lv):
            return sum(tests[f]["effect"] / 2 * lv[f] for f in kept) + \
                sum(tests[f"{a}:{b}"]["effect"] / 2 * lv[a] * lv[b] for a, b in pairs)

        best = None
        for combo in itertools.product((-1, 1), repeat=len(kept)):
            lv = {f: -1 for f in factors}
            lv.update(zip(kept, combo))
            key = (round(value(lv), 12), -sum(v > 0 for v in lv.values()))
            if best is None or key > best[0]:
                best = (key, lv)
        lv = dict(best[1]) if best else {f: -1 for f in factors}
        changed = True
        while changed:                                    # no part changed solely through a negative interaction
            changed = False
            for f in kept:
                if lv[f] > 0:
                    cond = tests[f]["effect"] + sum(tests[f"{a}:{b}"]["effect"] * lv[b if a == f else a]
                                                    for a, b in pairs if f in (a, b))
                    if cond <= 0:
                        lv[f], changed = -1, True
        n = sum(1 << j for j, f in enumerate(factors) if lv[f] > 0)
        out = {"n_items": len(used), "dropped": len(dropped), "effects": tests, "kept_mains": kept,
               "kept_pairs": [f"{a}:{b}" for a, b in pairs], "candidate": design_names[n],
               "candidate_levels": {f: "+" if lv[f] > 0 else "-" for f in factors},
               "design_means": {d: round(float(Y[j].mean()), 4) for j, d in enumerate(design_names)},
               "observed_best": design_names[int(np.argmax(np.round(Y.mean(axis=1), 12)))]}
        if n == 0:
            out.update(final_design=design_names[0], replaces=False, direct=None, gain=0.0)
            return out
        direct = self.paired_cluster({d: scores[d] for d in (design_names[n], design_names[0])}, design_names[n],
                                     design_names[0], used)
        replaces = bool(direct.get("range") and direct["range"][0] is not None and direct["range"][0] > 0)
        out.update(direct=direct, replaces=replaces, final_design=design_names[n] if replaces else design_names[0],
                   gain=direct["estimate"] if replaces else 0.0)
        return out

    def screen(self, scores, current, candidates, items, gate=0.015, max_advance=4):
        """The one-at-a-time screen. candidates: {label: {"slot": s, "archetype": a, ...}}, each changing one slot of
        the current best. Per slot, the best candidate (largest paired difference; ties: lower archetype) advances if
        its difference is at least `gate`; at most max_advance slots go on (the largest differences). A fixed bar,
        not a significance test: the confirmation that follows is the test."""
        labels = list(candidates)
        out = {"gate": gate, "max_advance": max_advance, "candidates": {}, "slots": {}}
        for lab in labels:
            r = self.paired_cluster(scores, lab, current, items)
            Y, used, _ = self.matrix(scores, [lab, current], items)
            by = {}
            for lv in sorted({self.level_of(i) for i in used if self.level_of(i) is not None}, key=str):
                its = [j for j, i in enumerate(used) if self.level_of(i) == lv]
                by[str(lv)] = round(float((Y[0, its] - Y[1, its]).mean()), 4)
            out["candidates"][lab] = dict(candidates[lab], **r, by_level=by,
                                          mean=round(float(Y[0].mean()), 4) if used else None)
        for slot in dict.fromkeys(c["slot"] for c in candidates.values()):
            mine = [lab for lab in labels if candidates[lab]["slot"] == slot]
            best = max(mine, key=lambda lab: (out["candidates"][lab].get("estimate", -9),
                                              -candidates[lab].get("archetype", 9)))
            d = out["candidates"][best].get("estimate")
            out["slots"][str(slot)] = {"best": best, "difference": d, "passes_gate": bool(d is not None and d >= gate)}
        order = [str(s) for s in dict.fromkeys(c["slot"] for c in candidates.values())]
        passing = sorted((s for s, v in out["slots"].items() if v["passes_gate"]),
                         key=lambda s: (-out["slots"][s]["difference"], order.index(s)))
        adv = sorted(passing[:max_advance], key=order.index)
        for s, v in out["slots"].items():
            v["advances"] = s in adv
        out["advancing"] = {s: out["slots"][s]["best"] for s in adv}
        out["capped"] = passing[max_advance:]
        return out

    def final_sequence(self, scores, final, rivals, sealed_items):
        """Fixed-sequence tests on the sealed items, in the order given ({name: rival label}); each counts only if
        every earlier one passed (whole 95% range above zero, t across clusters). The bootstrap lead is reported
        beside each."""
        out, still = {}, True
        for name, other in rivals.items():
            r = self.paired_cluster(scores, final, other, sealed_items)
            r["boot"] = self.paired_lead(scores, final, other, sealed_items)
            r["rival"] = other
            r["passes"] = bool(r.get("range") and r["range"][0] is not None and r["range"][0] > 0)
            r["counted"] = bool(still and r["passes"])
            still = r["counted"]
            out[name] = r
        return out


# ---------------------------------------------------------------------- helpers
def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def p_greater(boot_diff):
    """One-sided percentile-bootstrap p for 'difference > 0' (the share of draws at or below 0)."""
    return float((1 + np.sum(np.asarray(boot_diff) <= 0)) / (len(boot_diff) + 1))


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=ranks) / counts)[inv]


def spearman(a, b):
    if len(a) <= 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _t_sf(t, df):
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except ImportError:
        z = abs(t) * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
        return float(math.erfc(z / math.sqrt(2)))


def _t_crit(df):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:
        return 1.96 + 2.4 / max(df, 1)


def t_summary(c):
    """Mean of per-cluster values c with its t test (df = clusters - 1) and 95% t range."""
    c = np.asarray(c, float)
    n = len(c)
    est = float(c.mean()) if n else float("nan")
    se = float(c.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    df = n - 1
    t = 0.0 if abs(est) < 1e-12 else (est / se if se and se > 0 else float("inf"))
    p = _t_sf(t, df) if df > 0 and math.isfinite(t) else (0.0 if df > 0 else 1.0)
    tc = _t_crit(df) if df > 0 else float("nan")
    return {"estimate": round(est, 4), "se": round(se, 4), "t": round(t, 3), "df": df, "p": round(p, 5),
            "range": [round(est - tc * se, 4), round(est + tc * se, 4)] if df > 0 else [None, None]}


def factorial_levels(m):
    """The 2^m designs in standard order (the first factor changes fastest): rows of -1/+1."""
    return [[1 if (n >> b) & 1 else -1 for b in range(m)] for n in range(2 ** m)]


def design_labels(m):
    return [f"D{n + 1:02d}" for n in range(2 ** m)]


def clean(x):
    """JSON-safe copy: numpy scalars to Python, NaN and infinity to None."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def dumps(x):
    return json.dumps(clean(x), indent=1, allow_nan=False, default=str)
