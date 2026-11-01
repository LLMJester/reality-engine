"""Complexity-ladder calibration: how score, effort and description length change with k (subsystems per item).

    python -m v3.calibrate --k 1,2,4,8 --n 6 --builders claude,codex --genome A0P

Runs one seed genome on n composite items per k (v3.items.composite; the same n bases for every k, so K1 ⊂ K2 ⊂ K4 ⊂
K8 and the comparison is paired), 4 threads, Claude runs throttled like v3.run. Writes runs/<phase>/summary_<genome>.json.
"""
import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

from v3 import items as I
from v3 import stats
from v3.config import CFG, RUNS



def calib_items(ks, n, start=I.CALIB_START):
    """{k: [item id]}: the same n family-stratified bases for every k, disjoint from the evaluation ladder."""
    bases = I.ladder_seeds(n, start)
    return {k: [f"K{k}x{b}" for b in bases] for k in ks}


def summarise(recs, items):
    """recs: episode records; items: {item id: item}. {builder: {k: {mean, lo, hi, n, n_errors, output_tokens,
    description_chars}}}, CI by item-cluster bootstrap (v3.stats.mean_ci)."""
    out = {}
    for b in sorted({r["builder"] for r in recs}):
        out[b] = {}
        for k in sorted({items[r["item"]]["k"] for r in recs if r["builder"] == b}):
            rs = [r for r in recs if r["builder"] == b and items[r["item"]]["k"] == k]
            ok = [r for r in rs if not r.get("error") and r.get("grade", {}).get("score") is not None]
            scores = {r["item"]: r["grade"]["score"] for r in ok}
            ci = stats.mean_ci(scores, {i: i for i in scores}) if scores else {"est": None, "lo": None, "hi": None}
            toks = [r.get("output_tokens") or 0 for r in ok]
            out[b][str(k)] = {"mean": ci["est"], "lo": ci["lo"], "hi": ci["hi"], "n": len(scores),
                              "n_errors": len(rs) - len(ok),
                              "output_tokens": statistics.mean(toks) if toks else None,
                              "description_chars": statistics.mean(len(items[r["item"]]["context"]) for r in rs)}
    return out


def table(summary):
    rows = ["builder  k   n  mean   95% CI          out tokens  desc chars"]
    for b, per_k in summary.items():
        for k, s in per_k.items():
            ci = "-" if s["mean"] is None else f"{s['mean']:.3f}  [{s['lo']:.3f}, {s['hi']:.3f}]"
            tok = "-" if s["output_tokens"] is None else f"{s['output_tokens']:.0f}"
            rows.append(f"{b:<8} {k:>2} {s['n']:>3}  {ci:<22} {tok:>10}  {s['description_chars']:>10.0f}")
    return "\n".join(rows)


def run(genome, ks, n, builders, *, runner=None, threads=4, phase="calib"):
    if runner is None:
        from v3.episode import run as runner
    ids = calib_items(ks, n)
    items = {i: I.load(i) for per_k in ids.values() for i in per_k}

    def one(job):
        b, iid = job
        try:                       # Claude calls are throttled inside v3.episode (claude_gate)
            return runner(genome, items[iid], b, phase=phase)
        except Exception as e:
            return {"builder": b, "item": iid, "error": f"{type(e).__name__}: {e}"}

    jobs = [(b, i) for b in builders for k in ks for i in ids[k]]
    with ThreadPoolExecutor(threads) as ex:
        recs = list(ex.map(one, jobs))
    for (b, _), r in zip(jobs, recs):
        r.setdefault("builder", b)
    return summarise(recs, items)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="1,2,4,8")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--builders", default="claude,codex")
    ap.add_argument("--genome", default="A0P")
    ap.add_argument("--phase", default="calib", help="runs/<phase>/ and summary location")
    a = ap.parse_args(argv)
    from v3.genome.seeds import load_all
    ks = [int(x) for x in a.k.split(",")]
    builders = [b for b in a.builders.split(",") if b]
    summary = run(load_all()[a.genome], ks, a.n, builders, phase=a.phase)
    out = RUNS / a.phase / f"summary_{a.genome}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"genome": a.genome, "k": ks, "n": a.n, "builders": builders, "summary": summary},
                              indent=1), encoding="utf-8")
    print(table(summary))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
