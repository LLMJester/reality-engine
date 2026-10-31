"""The replay test: the engine's analysis of the run 5 replay must match run 5's own analysis, number by number.

    python scripts/compare_replay.py [--work work/run5-replay] [--run5 ../rpp-v3/runs/run5/analysis.json]

Compares every decision and every reported mean, range, effect and p value that both analyses have (step by step:
compare = stage 1, take apart = stage 2's retention rule, round 1 = stage 3, final = the final check on sealed2).
Prints each group's count of matches and every difference; exits 1 if anything differs.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TA = {"A": "layout", "B": "rules", "C": "answering", "D": "example", "E": "budget"}
BO = {"1": "layout", "2": "rules", "3": "strategy", "4": "answering", "5": "example"}


def ta_name(n):
    return ":".join(TA[x] for x in n.split(":"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work" / "run5-replay"))
    ap.add_argument("--run5", default=str(ROOT.parent / "rpp-v3" / "runs" / "run5" / "analysis.json"))
    a = ap.parse_args(argv)
    E = json.loads((pathlib.Path(a.work) / "analysis.json").read_text(encoding="utf-8"))
    R = json.loads(pathlib.Path(a.run5).read_text(encoding="utf-8"))
    groups, diffs = {}, []

    def same(group, what, x, y):
        ok = x == y or (isinstance(x, float) and isinstance(y, float) and abs(x - y) < 1e-9)
        groups.setdefault(group, [0, 0])[0 if ok else 1] += 1
        if not ok:
            diffs.append(f"{group}: {what}: engine {x!r} vs run 5 {y!r}")

    # ---- compare (stage 1)
    e, r = E["compare"], R["stage1"]
    for k in ("winner", "runner_up", "tie_at_top"):
        same("compare", k, e[k], r[k])
    same("compare", "hand_top", e["top"].get("hand"), r.get("hand_top"))
    for d, v in r["generated"]["designs"].items():
        for f in ("mean", "range", "behind_best", "mcb_margin", "could_be_best"):
            same("compare", f"{d} {f}", e["main"]["designs"][d][f], v[f])
    same("compare", "mcb_q95", e["main"]["mcb_q95"], r["generated"]["mcb_q95"])
    same("compare", "ranking", e["main"]["ranking"], r["generated"]["ranking"])
    for d, v in r["generated"]["leads_over_control"].items():
        for f in ("lead", "range", "p", "p_holm", "reject"):
            same("compare", f"lead {d} {f}", e["main"]["leads_over_control"][d][f], v[f])
    for d, v in r["hand"]["designs"].items():
        for f in ("mean", "range", "could_be_best"):
            same("compare", f"hand {d} {f}", e["also"]["hand"]["designs"][d][f], v[f])
    for k, v in r["by_k"].items():
        for d, x in v.items():
            same("compare", f"k={k} {d}", e["by_level"][f"k={k}"][d], x)
    same("compare", "rank agreement", e["rank_agreement"]["hand"], r["rank_agreement"])
    b = r["baseline"]["generated"]["designs"]["BLIND"]
    same("compare", "BLIND mean", e["baselines"]["BLIND"]["designs"]["BLIND"]["mean"], b["mean"])

    # ---- take apart (stage 2, the gated retention rule)
    e, r = E["takeapart"], R["stage2_retention"]
    for k in ("final_design", "replaces", "gain", "candidate", "observed_best", "n_items"):
        same("take apart", k, e[k], r[k])
    same("take apart", "kept_mains", e["kept_mains"], [TA[x] for x in r["kept_mains"]])
    for n, v in r["effects"].items():
        for f in ("effect", "se", "t", "p", "df", "range", "confirmatory"):
            same("take apart", f"{n} {f}", e["effects"][ta_name(n)][f], v[f])
        if "holm" in v:
            same("take apart", f"{n} holm", e["effects"][ta_name(n)]["holm"], v["holm"])
    for d, m in r["design_means"].items():
        same("take apart", f"{d} mean", e["design_means"][d], m)

    # ---- round 1 (stage 3): the screen and the confirmation
    e, r = E["round1_screen"], R["stage3_screen"]
    lab = {}
    for old, v in r["candidates"].items():
        lab[old] = f"{BO[str(v['slot'])]}.{v['archetype']}"
    for old, v in r["candidates"].items():
        for f in ("estimate", "se", "t", "p", "df", "range", "mean", "n_items", "clusters"):
            same("round 1 screen", f"{old} {f}", e["candidates"][lab[old]][f], v[f])
        same("round 1 screen", f"{old} by level", {k.replace("k=", ""): x for k, x in
                                                   e["candidates"][lab[old]]["by_level"].items()}, v["by_k"])
    same("round 1 screen", "advancing", e["advancing"], {BO[s]: lab[x] for s, x in r["advancing"].items()})
    e, r = E["round1_confirm"], R["stage3_confirm"]
    for k in ("final_design", "replaces", "gain", "candidate", "observed_best", "n_items"):
        same("round 1 confirm", k, e[k], r[k])
    for n, v in r["effects"].items():
        name = ":".join(BO[x.replace("slot", "")] for x in n.split(":"))
        for f in ("effect", "se", "t", "p", "range"):
            same("round 1 confirm", f"{n} {f}", e["effects"][name][f], v[f])
    same("round 1 confirm", "direct", (e.get("direct") or {}).get("estimate"), (r.get("direct") or {}).get("estimate"))
    for d, m in r["design_means"].items():
        same("round 1 confirm", f"{d} mean", e["design_means"][d], m)

    # ---- the final check on sealed2
    e, r = E["final"], R["final2"]
    names = {"1_vs_plain_prose": "1_vs_control", "2_vs_hand_top": "2_vs_top_hand", "3_vs_runner_up": "3_vs_runner_up"}
    for old, v in r["sequence"].items():
        x = e["sequence"][names[old]]
        for f in ("estimate", "se", "t", "p", "df", "range", "passes", "counted", "n_items", "clusters"):
            same("final", f"{old} {f}", x[f], v[f])
        for f in ("lead", "range", "passes"):
            same("final", f"{old} boot {f}", x["boot"][f], v["boot"][f])
    for d, v in r["sealed_means"].items():
        d2 = "FINAL" if d == r["final_design"] else d
        for f in ("mean", "range", "could_be_best"):
            same("final", f"sealed {d} {f}", e["sealed_means"][d2][f], v[f])
    for k, v in r["by_k"].items():
        for d, x in v.items():
            d2 = "FINAL" if d == r["final_design"] else d
            same("final", f"k={k} {d}", e["by_level"][f"k={k}"][d2], x)

    w = max(len(g) for g in groups)
    for g, (ok, bad) in groups.items():
        print(f"{g:<{w}}  {ok:>5} match  {bad:>3} differ")
    for d in diffs:
        print("  DIFFERS", d)
    total = sum(ok for ok, _ in groups.values())
    print(f"{'all' if not diffs else 'NOT all'}: {total} values match, {len(diffs)} differ")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
