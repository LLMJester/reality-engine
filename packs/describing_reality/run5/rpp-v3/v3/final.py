"""Final stage (spec §13, PREREG): finalists → final hash lock → sealed sets → hypotheses H1-H4.

    python -m v3.final select      # finalists by the pre-registered rule (no AI)
    python -m v3.final lock        # FINAL_LOCK: finalists, code and set ids hashed; sealed sets become readable
    python -m v3.final run         # sealed sets × finalists + control × both builders (primary builder 3 repeats)
    python -m v3.final analyse     # H1-H4 with paired cluster-bootstrap CIs → runs/final/analysis.json
"""
import argparse
import json
import math
import re
import statistics
import sys
import time

from v3 import items as I
from v3 import stats
from v3.config import CFG, RUNS
from v3.evolve import primary, single, small_builder, small_ok
from v3.state import Store

OUT = RUNS / "final"
STRUCTURED = ("sectioned", "ontology")
KRE = re.compile(r"^K(\d+)x")
FINAL_LADDER_N, FINAL_LADDER_START = 15, 2_500_000      # 15 bases x k in (1, 2, 4, 8) = 60 sealed composite items
# V3-037: run 4 has fresh sealed sets (config final.sealed); run 3's are retired. Without the key: runs 1-3's sets
SEALED = (CFG.get("final") or {}).get("sealed") or {}
FINAL_LADDER_START = SEALED.get("ladder_start", FINAL_LADDER_START)


def select(data, k=3):
    """Up to k finalists from the NSGA-II front, distinct niches, best primary-builder e2e per niche (ties:
    buildability). Run 1 selected on Claude; run 2 on GPT-6 Luna (V3-030)."""
    ev, g = data["evals"], data["genomes"]
    best = {}
    for gid in data.get("front", []):
        if gid in data.get("controls", []):
            continue
        f = ev[gid]["full"][primary()]
        niche = (g[gid]["genes"]["core"]["kind"], g[gid]["genes"]["build_protocol"]["kind"])
        key = (f["e2e"] or 0, f["build"] or 0, gid)
        if niche not in best or key > best[niche][0]:
            best[niche] = (key, gid)
    ranked = [gid for _, gid in sorted(best.values(), key=lambda x: x[0], reverse=True)]
    fin = ranked[:k]
    # H5 needs a structured plain-language finalist: the best one replaces the last pick if none made the cut
    structured = [gid for gid in ranked if g[gid]["genes"]["core"]["kind"] in STRUCTURED]
    if structured and not set(structured) & set(fin):
        fin = fin[:k - 1] + structured[:1]
    # V3-023: the front's best design on the small builder is a finalist too (it replaces the last pick other than
    # the structured one), so H6 is tested on a design selected for the small builder
    from v3.evolve import small_builder
    sb = small_builder()
    small = sorted((gid for gid in data.get("front", []) if gid not in data.get("controls", [])
                    and (ev[gid]["full"].get(sb) or {}).get("e2e") is not None),
                   key=lambda x: (-ev[x]["full"][sb]["e2e"], x))
    if small and small[0] not in fin and k >= 2:
        if len(fin) >= k:
            keep = set(structured[:1]) & set(fin)
            drop = next(x for x in reversed(fin) if x not in keep)
            fin = [x for x in fin if x != drop]
        fin.append(small[0])
    return fin


def sealed_items():
    """Final sets, readable only after FINAL_LOCK exists (items.py enforces it for generated test seeds)."""
    if not I.FINAL_LOCK.exists():
        raise I.Sealed("final sets are sealed until `python -m v3.final lock`")
    from harness.run import split_ids, load_item
    from discovery.world.splits import seeds
    out = []
    for iid in (split_ids("heldout") if SEALED.get("heldout", True) else []):
        item, ctx = load_item(iid)
        out.append(dict(item, context=ctx, system=item.get("system_id") or item["id"], grading="ai", set="heldout"))
    for sd in seeds("test", 60, start=SEALED.get("generated_start", 0)):
        out.append(dict(I.generated(sd), set="generated_test"))
    # composite ladder for H5 (V3-006): the same test-split base seeds at every k, so the ladder is paired
    for base in I.ladder_seeds(FINAL_LADDER_N, start=FINAL_LADDER_START):
        for k in I.COMPOSITE_KS:
            out.append(dict(I.composite(k, base), set="composite_test"))
    return out


def run(store):
    from v3.episode import run as episode
    data = store.data
    lock = json.loads(I.FINAL_LOCK.read_text(encoding="utf-8"))
    arms = lock["finalists"] + data["controls"]
    # the primary builder runs every item finalist_repeats times; the other builder once. In run 2 (V3-030) that is
    # Luna 3 times and Claude once: Claude never took part in evolution, so its run measures transfer (H6)
    other = small_builder() if primary() == "claude" else "claude"
    for it in sealed_items():
        for gid in arms:
            g = data["genomes"][gid]
            if small_ok(g) or primary() == "claude":
                for rep in range(CFG["evolution"]["finalist_repeats"]):
                    episode(g, it, primary(), "final", rep=rep)
            if other == "claude" or small_ok(g):
                episode(g, it, other, "final")


def _scores(builder, gid, rep=None):
    out, sysm = {}, {}
    for p in (RUNS / "final" / builder / gid).glob("*/record.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        s = r.get("grade", {}).get("score")
        if s is None:
            continue
        out.setdefault(r["item"], []).append(s)
        sysm[r["item"]] = r["system"]
    return {i: statistics.mean(v) for i, v in out.items()}, sysm


def slope_ci(diff, clusters, n_boot=4000, seed=0):
    """OLS slope of per-item difference against log2(k), with a cluster-bootstrap 95% CI (clusters = systems)."""
    import numpy as np
    pts = [(math.log2(int(KRE.match(i).group(1))), d, clusters[i]) for i, d in diff.items()]
    if len({x for x, _, _ in pts}) < 2:
        return {"est": None, "lo": None, "hi": None, "n_items": len(pts)}

    def fit(ps):
        x = np.array([p[0] for p in ps]); y = np.array([p[1] for p in ps])
        return float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else float("nan")
    by = {}
    for p in pts:
        by.setdefault(p[2], []).append(p)
    keys, rng, boots = sorted(by), np.random.default_rng(seed), []
    for _ in range(n_boot):
        b = fit([p for j in rng.integers(0, len(keys), len(keys)) for p in by[keys[j]]])
        if b == b:
            boots.append(b)
    return {"est": fit(pts), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "n_items": len(pts)}


def analyse(store):
    data = store.data
    lock = json.loads(I.FINAL_LOCK.read_text(encoding="utf-8"))
    ctrl = data["controls"][0]
    res = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "finalists": lock["finalists"], "control": ctrl, "H1": {}, "H2": {}}
    for gid in lock["finalists"]:
        for h, b in (("H1", primary()), ("H2", "claude" if single() else small_builder())):
            a, sa = _scores(b, gid)
            c, sc = _scores(b, ctrl)
            res[h][gid] = stats.paired_ci(a, c, {**sc, **sa})
    # H3: best full-evaluation e2e on dev, last generation vs generation 0 (paired on full-set items)
    ev = data["evals"]
    seeds = [gid for gid, g in data["genomes"].items() if g["operator"] == "seed" and gid not in data["controls"]]
    pb = primary()
    best0 = max(seeds, key=lambda x: ev[x]["full"][pb]["e2e"] or 0)
    full = [gid for gid in ev if "full" in ev[gid] and data["genomes"][gid]["operator"] != "seed"]
    if full:
        bestN = max(full, key=lambda x: ev[x]["full"][pb]["e2e"] or 0)
        a, b = ev[bestN]["full"][pb], ev[best0]["full"][pb]
        res["H3"] = {"best_evolved": bestN, "best_seed": best0,
                     "ci": stats.paired_ci(a["scores"], b["scores"], {**b["systems"], **a["systems"]})}
    # H4: e2e vs buildability across fully evaluated designs (Spearman with a bootstrap CI over designs)
    pts = [(ev[g]["full"][pb]["build"], ev[g]["full"][pb]["e2e"]) for g in ev
           if "full" in ev[g] and (ev[g]["full"].get(pb) or {}).get("e2e") is not None]
    if len(pts) >= 5:
        from scipy.stats import spearmanr
        import numpy as np
        rng = np.random.default_rng(0)
        rho = spearmanr(*zip(*pts)).statistic
        boots = [spearmanr(*zip(*[pts[i] for i in rng.integers(0, len(pts), len(pts))])).statistic
                 for _ in range(2000)]
        boots = [b for b in boots if b == b]
        res["H4"] = {"rho": float(rho), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
                     "n_designs": len(pts)}
    # H5: best structured plain-language finalist vs free plain language on composite items, and the gap vs log2(k)
    structured = [gid for gid in lock["finalists"] if lock["genomes"][gid]["genes"]["core"]["kind"] in STRUCTURED]
    if structured:
        res["H5"] = {}
        for b in ("claude", small_builder()):
            a, sa = _scores(b, structured[0])
            c, sc = _scores(b, ctrl)
            comp = [i for i in set(a) & set(c) if KRE.match(i)]
            res["H5"][b] = {"design": structured[0],
                            "diff": stats.paired_ci({i: a[i] for i in comp}, {i: c[i] for i in comp}, {**sc, **sa}),
                            "slope_log2k": slope_ci({i: a[i] - c[i] for i in comp}, {**sc, **sa})}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "lock", "run", "analyse"])
    a = ap.parse_args(argv)
    store = Store()
    if a.cmd == "select":
        print(json.dumps(select(store.data)))
    elif a.cmd == "lock":
        from v3.prereg import snapshot
        fin = select(store.data)
        I.FINAL_LOCK.write_text(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "finalists": fin,
                                            "genomes": {g: store.data["genomes"][g] for g in fin},
                                            "hashes": snapshot(store.data.get("item_sets", {}))}, indent=1),
                                encoding="utf-8")
        print("final lock written; finalists:", fin)
    elif a.cmd == "run":
        store.acquire()
        try:
            run(store)
        finally:
            store.release()
    else:
        print(json.dumps(analyse(store), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
