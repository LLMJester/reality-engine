"""Run 5 analysis (V3-042; docs/RUN5_DESIGN.md section 6). Code only, no AI; deterministic (fixed bootstrap seed).

Scores are y[design][item] in [0, 1] (the grade score of one run). Every comparison is paired: computed on the items
both (or all) designs answered. Ranges are 95% bootstrap ranges over clusters of items (a generated base system with
its four k, or a hand-written system with its three questions), BOOT resamples.

  stage1(...)      founders: means, could-be-best set (Hsu, bootstrap margin), leads over the control (Holm), by group,
                   by k, by family, notes-only vs tool-assisted; the BLIND baseline beside them; winner and runner-up
                   (highest means on the generated items; ties broken by the hand-written mean, then name); the same
                   on the hand-written items with rank agreement (Spearman) and the hand-written top founder
  stage2(...)      the 2^5 factorial: primary tests across base systems for 5 parts, 10 pairs and A:B:C, Holm over
                   the 9 pre-registered; secondary 16-df fit on the 32 means with a Lenth check; the final design is
                   the argmax of the fitted model (mains + pairs, + A:B:C if p < 0.05)
  final(...)       on the sealed items, in order: winner vs control, then final vs winner (if they differ); also vs
                   runner-up, hand-written top founder and control, and the recipe / compute split of a doubled budget
V3-044 (independent review, approved by the human 2026-10-29) changed stage 1, stage 2 and the final check as above.
"""
import itertools
import json
import math
import re

import numpy as np

BOOT = 10_000
SEED = 20261028
FACTORS = ("A", "B", "C", "D", "E")
PAIRS = tuple(itertools.combinations(FACTORS, 2))
CONFIRMATORY = ("A", "B", "C", "D", "E", "A:B", "A:C", "B:C", "A:B:C")      # V3-044: nine (was seven)
ALPHA = 0.05


# ---------------------------------------------------------------- data
def cluster_of(item_id):
    m = re.match(r"^K\d+x(\d+)$", item_id)
    if m:
        return "b" + m.group(1)
    m = re.match(r"^(H\d+|HP\d+)-\d+$", item_id)
    return m.group(1) if m else item_id


def k_of(item_id):
    m = re.match(r"^K(\d+)x", item_id)
    return int(m.group(1)) if m else None


def matrix(scores, designs, items):
    """Complete-case matrix Y[d, i] over the items every design has a score for. Returns (Y, items_used, dropped).
    V3-058: raises v3.run5_sets.Retired for any item of the retired sealed set."""
    from v3.run5_sets import check_not_retired
    check_not_retired(items)
    used = [i for i in items if all(scores.get(d, {}).get(i) is not None for d in designs)]
    Y = np.array([[float(scores[d][i]) for i in used] for d in designs]) if used else np.zeros((len(designs), 0))
    return Y, used, [i for i in items if i not in used]


def _weights(items, boot=BOOT, seed=SEED):
    """Bootstrap weights (boot x n_items): clusters resampled with replacement; every item of a drawn cluster counts."""
    cl = [cluster_of(i) for i in items]
    ids = sorted(set(cl))
    idx = {c: j for j, c in enumerate(ids)}
    member = np.zeros((len(ids), len(items)))
    for n, c in enumerate(cl):
        member[idx[c], n] = 1.0
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=boot)       # boot x clusters
    return counts @ member                                                          # boot x items


def boot_means(Y, items, boot=BOOT, seed=SEED):
    """boot x designs bootstrap means (same resamples for every design, so differences stay paired)."""
    W = _weights(items, boot, seed)
    return (W @ Y.T) / W.sum(axis=1, keepdims=True)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def holm(pvals):
    """{name: p} -> {name: (p, adjusted p, reject)} (Holm step-down, family-wise error ALPHA)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, running, stop = len(order), {}, 0.0, False
    for r, k in enumerate(order):
        adj = min(1.0, (m - r) * pvals[k])
        running = max(running, adj)
        rej = (not stop) and pvals[k] <= ALPHA / (m - r)
        stop = stop or not rej
        out[k] = {"p": round(pvals[k], 5), "p_holm": round(running, 5), "reject": bool(rej)}
    return out


def _p_greater(boot_diff, obs=None):
    """One-sided percentile-bootstrap p for 'difference > 0' (the share of draws at or below 0), consistent with the
    percentile range ci() (code review, 2026-10-29)."""
    return float((1 + np.sum(np.asarray(boot_diff) <= 0)) / (len(boot_diff) + 1))


# ---------------------------------------------------------------- stage 1
def compare(scores, designs, items, control=None, boot=BOOT):
    """Means, ranges, could-be-best set and leads over the control for one item set."""
    Y, used, dropped = matrix(scores, designs, items)
    if not used:
        return {"n_items": 0, "dropped": len(dropped)}
    m = Y.mean(axis=1)
    B = boot_means(Y, used, boot)
    cen = B - m                                   # centred bootstrap
    # Hsu's multiple comparisons with the best, studentised and one-sided (code review, 2026-10-29): design d is
    # compared with the best other design j*(d); the margin for d is q95 x se(d), where q95 is the 95th percentile of
    # the largest studentised centred difference over all designs (simultaneous for all designs at once)
    k = len(designs)
    best_j = [max((j for j in range(k) if j != d), key=lambda j: m[j]) if k > 1 else d for d in range(k)]
    best_other = np.array([m[best_j[d]] for d in range(k)])
    diff_cen = np.array([cen[:, best_j[d]] - cen[:, d] for d in range(k)])          # designs x boot
    se = np.maximum(diff_cen.std(axis=1, ddof=1), 1e-9)
    q95 = float(np.percentile((diff_cen / se[:, None]).max(axis=0), 95)) if k > 1 else 0.0
    margins = q95 * se
    out = {"n_items": len(used), "dropped": len(dropped), "mcb_q95": round(q95, 4), "designs": {}}
    for d, name in enumerate(designs):
        # mean_exact: rounding noise below 1e-12 is a tie
        out["designs"][name] = {"mean": round(float(m[d]), 4), "mean_exact": round(float(m[d]), 12),
                                "range": ci(B[:, d]),
                                "behind_best": round(float(best_other[d] - m[d]), 4),
                                "mcb_margin": round(float(margins[d]), 4),
                                "could_be_best": bool(best_other[d] - m[d] <= margins[d])}
    if control in designs:
        c = designs.index(control)
        pv, leads = {}, {}
        for d, name in enumerate(designs):
            if d == c:
                continue
            diff = B[:, d] - B[:, c]
            obs = float(m[d] - m[c])
            leads[name] = {"lead": round(obs, 4), "range": ci(diff)}
            pv[name] = _p_greater(diff, obs)
        h = holm(pv)
        for name in leads:
            leads[name].update(h[name])
        out["leads_over_control"] = leads
    order = sorted(designs, key=lambda n: (-out["designs"][n]["mean_exact"], n))
    out["ranking"] = order
    return out


def by_subset(scores, designs, items, key):
    groups = {}
    for i in items:
        groups.setdefault(key(i), []).append(i)
    out = {}
    for g, its in sorted(groups.items(), key=lambda x: str(x[0])):
        Y, used, _ = matrix(scores, designs, its)
        if not used:
            continue
        B = boot_means(Y, used, 2000)
        out[str(g)] = {n: {"mean": round(float(Y[d].mean()), 4), "range": ci(B[:, d]), "n": len(used)}
                       for d, n in enumerate(designs)}
    return out


def _rankdata(a):
    """Average ranks (ties share the mean of their ranks)."""
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


def group_contrast(scores, groups, items, a="notes only", b=("fixed calculator", "own code")):
    """Pre-registered (V3-044): mean of the notes-only founders minus mean of the tool-assisted founders (each founder
    weighted equally within its side), paired by item; cluster-bootstrap range and two-sided p."""
    left = [f for f in groups.get(a, [])]
    right = [f for g in b for f in groups.get(g, [])]
    if not left or not right:
        return None
    Y, used, _ = matrix(scores, left + right, items)
    if not used:
        return None
    c = Y[:len(left)].mean(axis=0) - Y[len(left):].mean(axis=0)            # per item
    W = _weights(used)
    B = (W @ c) / W.sum(axis=1)
    obs = float(c.mean())
    p = float(min(1.0, 2 * (1 + min(np.sum(B <= 0), np.sum(B >= 0))) / (len(B) + 1)))   # two-sided percentile
    return {"notes_only": left, "tool_assisted": right, "difference": round(obs, 4), "range": ci(B), "p": round(p, 5),
            "n_items": len(used)}


def stage1(scores, founders, gen_items, hand_items=(), control="A0P", groups=None, families=None, forms=None,
           baseline="BLIND"):
    """V3-044: the BLIND baseline (no notes) is reported beside the founders, never chosen and never in the Holm
    family; ties on the generated mean are broken by the hand-written mean, then by name; the hand-written top founder
    is recorded (it joins the final check if it is not the winner or the runner-up)."""
    base = baseline if baseline in founders else None
    founders = [f for f in founders if f != baseline]
    gen = compare(scores, founders, gen_items, control)
    out = {"generated": gen}
    h = compare(scores, founders, list(hand_items), control) if hand_items else {}
    hm = {f: (h.get("designs") or {}).get(f, {}).get("mean_exact", 0.0) for f in founders}
    order = sorted((gen.get("designs") or {}),             # exact means: rounding can make false ties (code review)
                   key=lambda f: (-gen["designs"][f]["mean_exact"], -hm.get(f, 0.0), f))
    gen["ranking"] = order
    out["winner"] = order[0] if order else None
    out["runner_up"] = order[1] if len(order) > 1 else None
    out["tie_at_top"] = bool(len(order) > 1 and gen["designs"][order[1]]["could_be_best"])
    out["by_k"] = by_subset(scores, founders, gen_items, k_of)
    if families:
        out["by_family"] = by_subset(scores, founders, gen_items, lambda i: families.get(i, "?"))
    if groups:
        groups = {g: [f for f in fs if f in founders] for g, fs in groups.items()}
        out["groups"] = {g: round(float(np.mean([gen["designs"][f]["mean_exact"] for f in fs if f in gen["designs"]])), 4)
                         for g, fs in groups.items() if any(f in gen["designs"] for f in fs)}
        out["notes_vs_tools"] = group_contrast(scores, groups, gen_items)
    if base:
        out["baseline"] = {"design": base,
                           "generated": compare(scores, [base, control], gen_items, control) if control else None,
                           "hand": compare(scores, [base, control], list(hand_items), control) if hand_items else None}
    if hand_items:
        out["hand"] = h
        hr = sorted((h.get("designs") or {}), key=lambda f: (-h["designs"][f]["mean_exact"], f))
        h["ranking"] = hr
        out["hand_top"] = hr[0] if hr else None
        if h.get("n_items") and gen.get("n_items"):
            a = [gen["designs"][f]["mean_exact"] for f in founders]
            b = [h["designs"][f]["mean_exact"] for f in founders]
            Yg, ug, _ = matrix(scores, founders, gen_items)
            Yh, uh, _ = matrix(scores, founders, list(hand_items))
            Bg, Bh = boot_means(Yg, ug, 2000, SEED + 1), boot_means(Yh, uh, 2000, SEED + 2)
            rs = np.array([spearman(Bg[j], Bh[j]) for j in range(2000)])
            rs = rs[np.isfinite(rs)]
            out["rank_agreement"] = {"spearman": round(spearman(a, b), 3), "range": ci(rs) if len(rs) else None}
    if forms:
        out["own_forms"] = forms
    return out


# ---------------------------------------------------------------- stage 2: the 2^5 factorial
def design_table():
    """The 32 designs in standard order (A changes fastest): list of {factor: -1/+1}."""
    return [{f: (1 if (n >> b) & 1 else -1) for b, f in enumerate(FACTORS)} for n in range(32)]


def model_matrix(rows, with_higher=False):
    cols, names = [np.ones(len(rows))], ["intercept"]
    for f in FACTORS:
        cols.append(np.array([r[f] for r in rows], float))
        names.append(f)
    for a, b in PAIRS:
        cols.append(np.array([r[a] * r[b] for r in rows], float))
        names.append(f"{a}:{b}")
    if with_higher:
        for k in (3, 4, 5):
            for combo in itertools.combinations(FACTORS, k):
                cols.append(np.array([np.prod([r[f] for f in combo]) for r in rows], float))
                names.append(":".join(combo))
    return np.column_stack(cols), names


def _t_sf(t, df):
    """Two-sided p for a t statistic (scipy if present, else a normal approximation corrected for df)."""
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except ImportError:
        z = abs(t) * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
        return float(math.erfc(z / math.sqrt(2)))


def fit_factorial(ybar):
    """The pre-registered fit on 32 design means (standard order): {effects, residual_df, residual_sd, lenth,
    fitted, best}. Effects are mean(+) - mean(-); t tests on the 16 residual df; Holm over CONFIRMATORY."""
    rows = design_table()
    X, names = model_matrix(rows)
    beta, *_ = np.linalg.lstsq(X, ybar, rcond=None)
    resid = ybar - X @ beta
    df = len(ybar) - X.shape[1]                                   # 32 - 16 = 16
    s2 = float(resid @ resid / df)
    se_beta = math.sqrt(s2 / len(ybar))                           # orthogonal +-1 columns: X'X = 32 I
    tcrit = 2.12                                                  # t(0.975, 16)
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.975, df))
    except ImportError:
        pass
    effects, pv = {}, {}
    for j, n in enumerate(names[1:], 1):
        eff = 2 * beta[j]                                         # mean(+) - mean(-)
        se = 2 * se_beta
        t = 0.0 if abs(eff) < 1e-12 else (eff / se if se > 0 else float("inf"))
        p = _t_sf(t, df)
        effects[n] = {"effect": round(float(eff), 4), "se": round(se, 4),
                      "range": [round(float(eff - tcrit * se), 4), round(float(eff + tcrit * se), 4)],
                      "t": round(float(t), 3), "p": round(p, 5),
                      "confirmatory": n in CONFIRMATORY}
        if n in CONFIRMATORY:
            pv[n] = p
    h = holm(pv)
    for n in h:
        effects[n]["holm"] = h[n]
    # Lenth's pseudo standard error over the 16 higher-order contrasts (half-normal check)
    Xh, hn = model_matrix(rows, with_higher=True)
    bh = np.linalg.lstsq(Xh, ybar, rcond=None)[0]
    higher = {hn[j]: float(2 * bh[j]) for j in range(16, 32)}
    absc = np.abs(np.array(list(higher.values())))
    s0 = 1.5 * np.median(absc)
    pse = 1.5 * np.median(absc[absc < 2.5 * s0]) if np.any(absc < 2.5 * s0) else s0
    me = 2.57 * pse                                               # Lenth margin, ~16 contrasts (t(0.975, d=16/3))
    standout = {k: round(v, 4) for k, v in higher.items() if abs(v) > me}
    fitted = np.round(X @ beta, 12)                               # five mains + ten pairs
    return {"effects": effects, "residual_df": df, "residual_sd": round(math.sqrt(s2), 4),
            "lenth": {"pse": round(float(pse), 4), "margin": round(float(me), 4),
                      "higher": {k: round(v, 4) for k, v in higher.items()}, "stand_out": standout},
            "fitted": fitted, "best": int(np.argmax(fitted)), "X": X, "names": names}


ABC = "A:B:C"


def effect_columns(rows):
    """{effect name: +-1 column over the 32 designs}: the five parts, the ten pairs and A:B:C."""
    cols = {f: np.array([r[f] for r in rows], float) for f in FACTORS}
    for a, b in PAIRS:
        cols[f"{a}:{b}"] = cols[a] * cols[b]
    cols[ABC] = cols["A"] * cols["B"] * cols["C"]
    return cols


def cluster_tests(Y, used, boot=BOOT):
    """Primary stage 2 inference (V3-044): for every effect, the per-item contrast mean(+) - mean(-) over the 32
    designs, averaged within each base system; a t test across base systems (df = clusters - 1) and a cluster
    bootstrap range. Holm over CONFIRMATORY."""
    rows = design_table()
    cl = [cluster_of(i) for i in used]
    ids = sorted(set(cl))
    member = np.zeros((len(ids), len(used)))
    for n, c in enumerate(cl):
        member[ids.index(c), n] = 1.0
    member /= member.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(SEED + 7)
    draws = rng.integers(0, len(ids), size=(boot, len(ids)))
    out, pv = {}, {}
    df = len(ids) - 1
    for n, x in effect_columns(rows).items():
        c_item = (x @ Y) / 16.0                                  # mean(+) - mean(-) on each item
        c = member @ c_item                                      # per base system
        est = float(c.mean())
        se = float(c.std(ddof=1) / math.sqrt(len(c))) if len(c) > 1 else float("nan")
        t = 0.0 if abs(est) < 1e-12 else (est / se if se and se > 0 else float("inf"))
        p = _t_sf(t, df) if df > 0 else 1.0
        bm = c[draws].mean(axis=1)
        out[n] = {"effect": round(est, 4), "se": round(se, 4), "range": ci(bm), "t": round(t, 3), "p": round(p, 5),
                  "df": df, "confirmatory": n in CONFIRMATORY}
        if n in CONFIRMATORY:
            pv[n] = p
    for n, h in holm(pv).items():
        out[n]["holm"] = h
    return out


def stage2(scores, design_names, items, boot=BOOT):
    """design_names: the 32 design ids in standard order (design_table()). V3-044: primary tests are across base
    systems (cluster_tests); the 16-df fit on the 32 means is secondary. The final design is the argmax of the
    fitted model with five mains and ten pairs, plus A:B:C if its primary test has p < 0.05."""
    rows = design_table()
    Y, used, dropped = matrix(scores, design_names, items)
    if len(used) < 2:
        raise ValueError(f"stage 2: only {len(used)} items complete across the 32 designs")
    ybar = Y.mean(axis=1)
    primary = cluster_tests(Y, used, min(boot, 10_000))
    f = fit_factorial(ybar)                                      # secondary: 16 df from the higher interactions
    X, names = f["X"], f["names"]
    B = boot_means(Y, used, min(boot, 4000))
    Bb = np.linalg.lstsq(X, B.T, rcond=None)[0]                   # 16 x boot
    for j, n in enumerate(names[1:], 1):
        f["effects"][n]["boot_range"] = ci(2 * Bb[j])
    with_abc = primary[ABC]["p"] < 0.05
    Xf = np.column_stack([X, effect_columns(rows)[ABC]]) if with_abc else X
    beta = np.linalg.lstsq(Xf, ybar, rcond=None)[0]
    beta[np.abs(beta) < 1e-12] = 0.0                     # code review: noise below 1e-12 is no effect
    fitted = np.round(Xf @ beta, 12)
    best = int(np.argmax(fitted))                        # the first maximum: a tie keeps the lower design number
    return {"n_items": len(used), "dropped": len(dropped), "design_means": dict(zip(design_names, map(
        lambda v: round(float(v), 4), ybar))), "effects": primary, "fit_includes_abc": bool(with_abc),
        "secondary_16df": {"effects": f["effects"], "residual_df": f["residual_df"], "residual_sd": f["residual_sd"],
                           "lenth": f["lenth"]},
        "residual_df": primary["A"]["df"], "residual_sd": f["residual_sd"], "lenth": f["lenth"],
        "fitted": dict(zip(design_names, map(lambda v: round(float(v), 4), fitted))),
        "final_design": design_names[best], "final_levels": rows[best],
        "observed_best": design_names[int(np.argmax(np.round(ybar, 12)))]}


# ---------------------------------------------------------------- final check
def paired_lead(scores, a, b, items, boot=BOOT):
    Y, used, dropped = matrix(scores, [a, b], items)
    if not used:
        return {"n_items": 0}
    B = boot_means(Y, used, boot)
    lead = float(Y[0].mean() - Y[1].mean())
    r = ci(B[:, 0] - B[:, 1])
    return {"lead": round(lead, 4), "range": r, "n_items": len(used), "dropped": len(dropped), "passes": r[0] > 0}


def final(scores, final_design, winner, runner_up, control, sealed_items, hand_items=(), hand_top=None,
          winner_e2=None):
    """V3-044, in order on the sealed items:
    confirm 1  stage 1 winner - control > 0 (the stage 1 result holds on unseen items)
    confirm 2  final design - stage 1 winner > 0, only if the final design differs from the winner
    also: final vs runner-up, vs the hand-written top founder, vs control; and, when the final design has the doubled
    thinking budget, the split of its lead into recipe (final - winner with double turns) and compute (winner with
    double turns - winner). The same comparisons on the hand-written items."""
    same = final_design == winner
    out = {"final_design": final_design, "same_as_winner": same, "sealed": {}, "hand": {}, "confirm": {}}
    pairs = [("winner_vs_control", winner, control)] if winner != control else []
    if not same:
        pairs.append(("vs_winner", final_design, winner))
    for name, other in (("vs_runner_up", runner_up), ("vs_hand_top", hand_top), ("vs_control", control)):
        if other and other not in (final_design, winner if name == "vs_hand_top" else None):
            pairs.append((name, final_design, other))
    split = []
    if winner_e2:                                         # the recipe / compute split is on the sealed items only
        split = [("recipe_part", final_design, winner_e2), ("compute_part", winner_e2, winner)]
    for name, a, b in pairs + split:
        if a and b and a != b:
            out["sealed"][name] = paired_lead(scores, a, b, sealed_items)
            if hand_items and (name, a, b) not in split:
                out["hand"][name] = paired_lead(scores, a, b, list(hand_items))
    if winner_e2 == final_design:
        out["sealed"]["recipe_part"] = {"lead": 0.0, "range": [0.0, 0.0], "n_items": len(sealed_items),
                                        "passes": False, "note": "the final design is design 17 (budget only)"}
    c1 = out["sealed"].get("winner_vs_control")
    out["confirm"]["1_winner_beats_control"] = bool(c1 and c1["passes"])
    out["confirm"]["2_final_beats_winner"] = None if same else bool(out["sealed"]["vs_winner"]["passes"])
    designs = [d for d in dict.fromkeys([final_design, winner, runner_up, hand_top, winner_e2, control]) if d]
    out["sealed_means"] = compare(scores, designs, sealed_items, control).get("designs")
    return out


# ---------------------------------------------------------------- V3-058: stages 3 to N (docs/RUN5_STAGES_DESIGN.md)
SCREEN_GATE = 0.015          # a slot's best candidate advances if it beats the current best by at least this
MAX_ADVANCE = 4              # at most four slots go on to the confirmation (2^4 = 16 designs)
CONVERGE = 0.005             # a confirmed gain below this ends the stages


def _clusters(used):
    """(cluster ids, member matrix normalised by row): per-item values -> per-base-system means."""
    cl = [cluster_of(i) for i in used]
    ids = sorted(set(cl))
    member = np.zeros((len(ids), len(used)))
    for n, c in enumerate(cl):
        member[ids.index(c), n] = 1.0
    return ids, member / member.sum(axis=1, keepdims=True)


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


def paired_cluster(scores, a, b, items):
    """mean(a) - mean(b) on the items both have, averaged within base systems; t test across them."""
    Y, used, dropped = matrix(scores, [a, b], items)
    if not used:
        return {"n_items": 0, "dropped": len(dropped)}
    ids, M = _clusters(used)
    out = t_summary(M @ (Y[0] - Y[1]))
    out.update(n_items=len(used), dropped=len(dropped), clusters=len(ids))
    return out


def factorial_levels(m):
    """The 2^m designs in standard order (the first factor changes fastest): list of {factor index: -1/+1}."""
    return [[1 if (n >> b) & 1 else -1 for b in range(m)] for n in range(2 ** m)]


def factorial_tests(Y, used, factors, family):
    """Every main effect and two-way interaction of a 2^m factorial (plus three-way terms named in the family), as
    the per-item contrast mean(+) - mean(-) averaged within base systems, t test across base systems; Holm over
    `family` (the pre-registered confirmatory terms)."""
    m = len(factors)
    rows = np.array(factorial_levels(m), float)                    # 2^m x m
    cols = {f: rows[:, j] for j, f in enumerate(factors)}
    for a, b in itertools.combinations(factors, 2):
        cols[f"{a}:{b}"] = cols[a] * cols[b]
    for name in family:
        parts = name.split(":")
        if len(parts) == 3 and all(p in factors for p in parts):
            cols[name] = cols[parts[0]] * cols[parts[1]] * cols[parts[2]]
    ids, M = _clusters(used)
    out, pv = {}, {}
    for name, x in cols.items():
        c = M @ ((x @ Y) / (len(x) / 2))
        r = t_summary(c)
        out[name] = {"effect": r["estimate"], "se": r["se"], "t": r["t"], "p": r["p"], "df": r["df"],
                     "range": r["range"], "confirmatory": name in family}
        if name in family:
            pv[name] = r["p"]
    for name, h in holm(pv).items():
        out[name]["holm"] = h
    return out


def retention(scores, design_names, items, factors, family=None):
    """The gated baseline retention rule (V3-058, docs/RUN5_STAGES_DESIGN.md section 3). design_names: the 2^m designs
    in standard order; design_names[0] (all minus) is the current best. Main effects that survive Holm with a
    positive estimate are kept; a two-way interaction is kept only if it survives Holm and both its main effects are
    kept (marginality); every other term is zero; undecided factors stay minus. d* maximises the kept terms (ties:
    fewer changes, then the lower design number); a factor stays plus only while its conditional effect at the
    other chosen levels is positive. d* replaces the current best if and only if its direct paired contrast with the
    current best across base systems has its 95% range above zero."""
    m = len(factors)
    family = list(family) if family is not None else \
        list(factors) + [f"{a}:{b}" for a, b in itertools.combinations(factors, 2)]
    Y, used, dropped = matrix(scores, design_names, items)
    if len(used) < 2:
        raise ValueError(f"only {len(used)} items complete across the {len(design_names)} designs")
    tests = factorial_tests(Y, used, factors, family)
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
    while changed:                                        # no factor plus solely through a negative interaction
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
    direct = paired_cluster({d: scores[d] for d in (design_names[n], design_names[0])}, design_names[n],
                            design_names[0], used)
    replaces = bool(direct.get("range") and direct["range"][0] is not None and direct["range"][0] > 0)
    out.update(direct=direct, replaces=replaces, final_design=design_names[n] if replaces else design_names[0],
               gain=direct["estimate"] if replaces else 0.0)
    return out


def screen(scores, current, candidates, items, gate=SCREEN_GATE, max_advance=MAX_ADVANCE):
    """The one-at-a-time screen (V3-058, section 5). candidates: {label: {"slot": s, "archetype": a, ...}}, each
    design changing one slot of the current best. Per candidate: its paired difference against the current best,
    averaged within base systems, with its t range, and the same by k (reported, not gated). Per slot, the best
    candidate c* (largest difference; ties: lower archetype) advances if its difference is at least `gate`; if more
    than max_advance slots advance, the ones with the largest differences go on."""
    labels = list(candidates)
    out = {"gate": gate, "max_advance": max_advance, "candidates": {}, "slots": {}}
    for lab in labels:
        r = paired_cluster(scores, lab, current, items)
        Y, used, _ = matrix(scores, [lab, current], items)
        by_k = {}
        for k in sorted({k_of(i) for i in used if k_of(i) is not None}):
            its = [j for j, i in enumerate(used) if k_of(i) == k]
            by_k[str(k)] = round(float((Y[0, its] - Y[1, its]).mean()), 4)
        out["candidates"][lab] = dict(candidates[lab], **r, by_k=by_k,
                                      mean=round(float(Y[0].mean()), 4) if used else None)
    for slot in sorted({c["slot"] for c in candidates.values()}):
        mine = [lab for lab in labels if candidates[lab]["slot"] == slot]
        best = max(mine, key=lambda lab: (out["candidates"][lab].get("estimate", -9),
                                          -candidates[lab].get("archetype", 9)))
        d = out["candidates"][best].get("estimate")
        out["slots"][str(slot)] = {"best": best, "difference": d, "passes_gate": bool(d is not None and d >= gate)}
    passing = sorted((s for s, v in out["slots"].items() if v["passes_gate"]),
                     key=lambda s: (-out["slots"][s]["difference"], int(s)))
    adv = sorted(passing[:max_advance], key=int)
    for s, v in out["slots"].items():
        v["advances"] = s in adv
    out["advancing"] = {s: out["slots"][s]["best"] for s in adv}
    out["capped"] = passing[max_advance:]
    return out


def final_stages(scores, final, facts, control, hand_top, runner_up, sealed_items, hand_items=()):
    """The final check after the stages (V3-058, section 8), on the fresh sealed set. Fixed sequence at alpha 0.05,
    largest expected effect first; each test counts only if every earlier one passed (95% cluster range above 0):
      1 final vs plain prose (A0P)   2 final vs the hand-written top founder   3 final vs the stage 1 runner-up
      4 final vs FACTS, only if the stages changed FACTS
    Also reported: FACTS vs plain prose, means by k, and on the hand-written items the final design vs FACTS as a
    paired test across the items (df = items - 1)."""
    seq = [(name, other) for name, other in (("1_vs_plain_prose", control), ("2_vs_hand_top", hand_top),
                                             ("3_vs_runner_up", runner_up)) if other and other != final]
    if final != facts:
        seq.append(("4_vs_facts", facts))
    out = {"final_design": final, "same_as_facts": final == facts, "sequence": {}, "hand": {}}
    still = True
    for name, other in seq:
        r = paired_cluster(scores, final, other, sealed_items)
        r["boot"] = paired_lead(scores, final, other, sealed_items)
        r["passes"] = bool(r.get("range") and r["range"][0] is not None and r["range"][0] > 0)
        r["counted"] = bool(still and r["passes"])
        still = r["counted"]
        out["sequence"][name] = r
    if final != facts:
        out["sequence"]["4_vs_facts"]["note"] = "did the stages improve FACTS?"
    if control != facts:
        out["facts_vs_plain_prose"] = paired_cluster(scores, facts, control, sealed_items)
    designs = [d for d in dict.fromkeys([final, facts, control, hand_top, runner_up]) if d]
    out["sealed_means"] = compare(scores, designs, sealed_items, control).get("designs")
    out["by_k"] = by_subset(scores, designs, sealed_items, k_of)
    if hand_items and final != facts:
        Y, used, dropped = matrix(scores, [final, facts], list(hand_items))
        if used:
            r = t_summary(Y[0] - Y[1])
            r.update(n_items=len(used), dropped=len(dropped))
            out["hand"]["vs_facts"] = r
    return out


def clean(x):
    """JSON-safe copy: numpy scalars to Python, NaN and infinity to None (a range that cannot be computed, e.g. with
    a single cluster, is reported as missing, never as NaN, which is not valid JSON)."""
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
