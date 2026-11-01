"""Run 3's built-in finals (V3-034, PREREG "Run 3 rules"): run automatically when evolution stops.

    python -m v3.final3 pilot      # pilot of finals B and C on non-sealed validation items (A0P and SEC_A1)
    python -m v3.final3 all        # lock -> A -> B -> C -> analyse -> paper kit (what the loop runs; resumable)
    python -m v3.final3 analyse    # recompute runs/final/analysis3.json from the records

Finalist: the champion (one design, named in FINAL_LOCK before any sealed item is opened); the control is A0P.
  A  accuracy   sealed sets (v2 held-out, generated test items, composite test ladder) x champion and control x
                GPT-6 Luna, `repeats_luna` repeats. H1, H3, H4, H5, H7.
  B  interaction  `interaction_bases` systems from the test split, each built up over four turns (k = 1, 2, 4, 8):
                the builder updates its own representation from the last turn plus only the new subsystems' text
                (the earlier descriptions are gone), then answers every question so far from the updated
                representation alone. Luna, champion and control. H8.
  C  transfer   the composite test ladder x champion and control x each pre-named OpenRouter small model, once. H9.
Nothing here decides anything by AI; every pass/fail rule is in PREREG.md.
"""
import argparse
import json
import math
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from v3 import items as I
from v3 import stats
from v3.config import CFG, RUNS
from v3.state import Store

OUT = RUNS / "final"
FC = CFG.get("final") or {}
RUN_NO = 4 if CFG["evolution"].get("mode") == "directed4" else 3      # V3-037: run 4 reuses the finals
LUNA = "codex"
STRUCTURED = ("sectioned", "ontology")
CARRY = ("WHAT YOU BUILT IN THE PREVIOUS STEP. The earlier descriptions are no longer available; this representation "
         "is all that is left of them:\n\n{rep}\n\nWhen you build now, build the complete representation again: "
         "everything above, updated with the new information below. Do not drop anything about the earlier "
         "systems.\n\n")
PILOT_B_START, PILOT_C_START = 1_960_000, 1_975_000     # validation split, unused by runs 1-3 and the probe


class Paused(RuntimeError):
    pass


def check_pause():
    from v3.run import PAUSE
    if PAUSE.exists():
        from v3.run import Paused as RunPaused
        raise RunPaused("state/PAUSE present")


def base_of(iid):
    m = re.match(r"^(?:K\d+|G\d+)x(\d+)$", iid)
    return f"B{m.group(1)}" if m else iid


def transfer_models(lock=None):
    return (lock or {}).get("transfer_models") or FC.get("transfer_models") or {}


# ---------------------------------------------------------------- episodes with transient retries (V3-031)
def episode(g, it, builder, phase, rep=0):
    from v3.episode import run
    from v3.evolve import transient
    r = {}
    for attempt in range(3):
        check_pause()
        try:
            r = run(g, it, builder, phase, rep=rep)
        except Exception as e:                          # noqa: BLE001 - recorded like a failed episode
            if type(e).__name__ == "Paused":
                raise
            r = {"error": f"{type(e).__name__}: {e}"}
        if not transient(r):
            break
        time.sleep(CFG["evolution"].get("retry_wait_s", 30))
    return r


def pool_run(jobs, fn, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda j: fn(*j), jobs))


# ---------------------------------------------------------------- B: interaction chains
def rep_text(genome, rec):
    """What the builder built last turn, as text: the notes (prose), or the representation rendered by the design."""
    if not rec or rec.get("error"):
        return "(No representation was built in the previous step.)"
    if genome["genes"]["core"]["kind"] == "prose":
        return rec.get("notes") or "(No notes were written in the previous step.)"
    rep = rec.get("representation")
    if rep in (None, {}, ""):
        return "(No representation was built in the previous step.)"
    try:
        from v3.genome.render import render
        arm = render(genome)
        return arm.render(arm.load(rep, given_gold=True))
    except Exception:                                   # noqa: BLE001
        return rep.get("text") if isinstance(rep, dict) and "text" in rep else json.dumps(rep, indent=1)


def turn_items(seed):
    return [dict(t, item=dict(t["item"], id=f"G{t['turn']}x{seed}", system=f"B{seed}")) for t in I.growth_turns(seed)]


def chain(g, seed, builder, phase):
    """One system built up over four turns; returns the four records."""
    out, prev = [], None
    for t in turn_items(seed):
        it = t["item"]
        if t["turn"] > 1:
            it = dict(it, context=CARRY.format(rep=rep_text(g, prev)) + t["new_text"])
        prev = episode(g, it, builder, phase)
        out.append(prev)
    return out


# ---------------------------------------------------------------- lock and runs
def interaction_seeds(n=None, start=None):
    return I.ladder_seeds(n or FC.get("interaction_bases", 20), start=start or FC.get("interaction_start", 2_700_000))


def lock(store):
    from v3.prereg import snapshot
    d = store.data
    if I.FINAL_LOCK.exists():
        return json.loads(I.FINAL_LOCK.read_text(encoding="utf-8"))
    champ = d["champion"]
    doc = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": RUN_NO, "finalists": [champ], "primary": champ,
           "control": d["controls"][0], "genomes": {champ: d["genomes"][champ]},
           "transfer_models": transfer_models(), "interaction_start": FC.get("interaction_start", 2_700_000),
           "interaction_bases": FC.get("interaction_bases", 20), "repeats_luna": FC.get("repeats_luna", 3),
           "hashes": snapshot(d.get("item_sets", {}))}
    I.FINAL_LOCK.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    store.event(f"FINAL_LOCK written: champion {champ} vs control {doc['control']}; sealed sets now readable")
    store.save()
    return doc


def arms(store, lk):
    d = store.data
    return [d["genomes"][lk["primary"]], d["genomes"][lk["control"]]]


def run_a(store, lk):
    from v3.final import sealed_items
    items = sealed_items()
    jobs = [(g, it, LUNA, "final", r) for it in items for g in arms(store, lk) for r in range(lk["repeats_luna"])]
    store.event(f"final A (accuracy): {len(jobs)} Luna episodes on the sealed sets")
    store.save()
    pool_run(jobs, episode, CFG["budget"].get("codex_parallel_agents", 8))


def run_b(store, lk):
    seeds = interaction_seeds(lk["interaction_bases"], lk["interaction_start"])
    jobs = [(g, s, LUNA, "final_interaction") for s in seeds for g in arms(store, lk)]
    store.event(f"final B (interaction): {len(jobs)} four-turn chains on Luna")
    store.save()
    pool_run(jobs, chain, CFG["budget"].get("codex_parallel_agents", 8))


def composite_test_items():
    from v3.final import FINAL_LADDER_N, FINAL_LADDER_START
    return [dict(I.composite(k, b), set="composite_test") for b in I.ladder_seeds(FINAL_LADDER_N, start=FINAL_LADDER_START)
            for k in I.COMPOSITE_KS]


def run_c(store, lk):
    if not FC.get("transfer_ready"):     # V3-034 addendum: final C runs only once its settings are fixed (V3-035)
        store.event("final C skipped: transfer settings not fixed (final.transfer_ready is false); reported as not run")
        store.save()
        return
    items = composite_test_items()
    jobs = [(g, it, m, "final_transfer") for m in transfer_models(lk) for it in items for g in arms(store, lk)]
    store.event(f"final C (transfer): {len(jobs)} episodes on {len(transfer_models(lk))} models")
    store.save()
    pool_run(jobs, episode, FC.get("transfer_parallel", 6))


# ---------------------------------------------------------------- analysis
def scores(phase, builder, gid):
    """{item: mean score over repeats}, {item: cluster}, {item: [records]} for one design's records in a phase."""
    out, recs = {}, {}
    for p in (RUNS / phase / builder / gid).glob("*/record.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        s = r.get("grade", {}).get("score")
        if s is None:
            continue
        out.setdefault(r["item"], []).append(s)
        recs.setdefault(r["item"], []).append(r)
    clusters = {i: base_of(i) if re.match(r"^[KG]\d+x", i) else recs[i][0].get("system", i) for i in out}
    return {i: statistics.mean(v) for i, v in out.items()}, clusters, recs


def contrast_ci(diff, clusters, hi_sel, lo_sel, n_boot=4000, seed=0):
    """(mean diff over items with hi_sel) minus (mean over lo_sel), cluster bootstrap over clusters."""
    import numpy as np
    by = {}
    for i, v in diff.items():
        by.setdefault(clusters[i], []).append((i, v))
    keys = sorted(by)

    def est(ks):
        a = [v for k in ks for i, v in by[k] if hi_sel(i)]
        b = [v for k in ks for i, v in by[k] if lo_sel(i)]
        return (np.mean(a) - np.mean(b)) if a and b else float("nan")
    rng = np.random.default_rng(seed)
    boots = [est([keys[j] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(n_boot)]
    boots = [b for b in boots if b == b]
    e = est(keys)
    return {"est": float(e), "lo": float(np.quantile(boots, 0.025)) if boots else None,
            "hi": float(np.quantile(boots, 0.975)) if boots else None, "n_clusters": len(keys)}


def kof(i):
    m = re.match(r"^K(\d+)x", i)
    return int(m.group(1)) if m else None


def correct_by_question(rec):
    """{question id: correct?} for a script-graded record (harness only; used for retention in final B)."""
    it = I.load(re.sub(r"^G(\d)x", lambda m: f"K{I.GROWTH_KS[int(m.group(1)) - 1]}x", rec["item"]))
    key, got = I.answer_key(it), I.parse_answers(rec.get("final_raw"))
    return {q: got.get(q) == a for q, a in key.items()}


def h9(lk, champ, ctrl):
    """C: H9 per model and pooled (item x model pairs, clustered by base system). V3-035: a model with paired records
    on fewer than half the items (rate limits, failures) is unusable: reported and left out; H9 needs at least
    transfer_min_models usable models."""
    out = {"per_model": {}, "unusable": {}}
    pa, pb_, pcl = {}, {}, {}
    n_items = len(composite_test_items())
    for m, model in transfer_models(lk).items():
        x, cx, _ = scores("final_transfer", m, champ)
        y, cy, _ = scores("final_transfer", m, ctrl)
        common = set(x) & set(y)
        if len(common) < n_items / 2:
            out["unusable"][m] = {"model": model, "paired_items": len(common), "items": n_items}
            continue
        out["per_model"][m] = dict(stats.paired_ci({i: x[i] for i in common}, {i: y[i] for i in common},
                                                   {**cy, **cx}), model=model, paired_items=len(common))
        for i in common:
            pa[f"{m}|{i}"], pb_[f"{m}|{i}"], pcl[f"{m}|{i}"] = x[i], y[i], base_of(i)
    if len(out["per_model"]) < FC.get("transfer_min_models", 2):
        out.update(status="not run" if not pa else "too few usable models", passed=None)
        return out
    pooled = stats.paired_ci(pa, pb_, pcl)
    ahead = sum(1 for r in out["per_model"].values() if (r["est"] or 0) > 0)
    out.update(pooled=pooled, models_ahead=ahead, n_models=len(out["per_model"]),
               passed=bool(pooled["improved"] and ahead > len(out["per_model"]) / 2))
    return out


def analyse(store, lk=None):
    from scipy.stats import spearmanr
    import numpy as np
    d = store.data
    lk = lk or json.loads(I.FINAL_LOCK.read_text(encoding="utf-8"))
    champ, ctrl = lk["primary"], lk["control"]
    res = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": RUN_NO, "champion": champ, "control": ctrl}
    # A: H1 pooled over the sealed sets, Luna, mean over repeats; per set; H7; H5
    a, ca, ra = scores("final", LUNA, champ)
    b, cb, _ = scores("final", LUNA, ctrl)
    cl = {**cb, **ca}
    res["H1"] = stats.paired_ci(a, b, cl)
    sets = {}
    for i, recs in ra.items():
        sets.setdefault("composite_test" if re.match(r"^K\d+x", i) else "generated_test" if i.startswith("W")
                        else "heldout", []).append(i)
    res["H1_by_set"] = {s: stats.paired_ci({i: a[i] for i in ids if i in b}, {i: b[i] for i in ids if i in b}, cl)
                        for s, ids in sets.items()}
    comp = [i for i in set(a) & set(b) if kof(i)]
    diff = {i: a[i] - b[i] for i in comp}
    res["H7"] = dict(contrast_ci(diff, cl, lambda i: kof(i) == 8, lambda i: kof(i) == 1),
                     per_k={k: stats.paired_ci({i: a[i] for i in comp if kof(i) == k},
                                               {i: b[i] for i in comp if kof(i) == k}, cl) for k in I.COMPOSITE_KS})
    if lk["genomes"][champ]["genes"]["core"]["kind"] in STRUCTURED:
        from v3.final import slope_ci
        res["H5"] = {"diff": stats.paired_ci({i: a[i] for i in comp}, {i: b[i] for i in comp}, cl),
                     "slope_log2k": slope_ci(diff, cl)}
    else:
        res["H5"] = {"not_tested": "the champion's core is not a structured plain-language core"}
    # H3 (descriptive): the champion vs the best starting design on the full dev set, and the confirmation chain
    ev, pb = d["evals"], CFG["evolution"].get("primary_builder", "codex")
    seeds = [g for g, x in d["genomes"].items() if x["operator"] == "seed" and g not in d["controls"]
             and (ev.get(g, {}).get("full", {}).get(pb) or {}).get("e2e") is not None]
    if seeds and champ in ev:
        best0 = max(seeds, key=lambda x: ev[x]["full"][pb]["e2e"])
        fa, fb = ev[champ]["full"][pb], ev[best0]["full"][pb]
        res["H3"] = {"best_seed": best0, "ci_dev": stats.paired_ci(fa["scores"], fb["scores"],
                                                                     {**fb["systems"], **fa["systems"]}),
                     "champions": d.get("champions", []),
                     "confirmations": {g: [p for p in ps if p.get("kind") == "confirmation"]
                                       for g, ps in d.get("proofs", {}).items()
                                       if any(p.get("kind") == "confirmation" for p in ps)}}
    pts = [(ev[g]["full"][pb]["build"], ev[g]["full"][pb]["e2e"]) for g in ev
           if (ev[g].get("full", {}).get(pb) or {}).get("e2e") is not None]
    if len(pts) >= 5:
        rng = np.random.default_rng(0)
        boots = [spearmanr(*zip(*[pts[i] for i in rng.integers(0, len(pts), len(pts))])).statistic for _ in range(2000)]
        boots = [x for x in boots if x == x]
        res["H4"] = {"rho": float(spearmanr(*zip(*pts)).statistic), "lo": float(np.quantile(boots, 0.025)),
                     "hi": float(np.quantile(boots, 0.975)), "n_designs": len(pts)}
    # B: H8 (mean over all turns), compounding (turn 4 minus turn 1), per turn, retention
    a8, c8, r8 = scores("final_interaction", LUNA, champ)
    b8, d8, s8 = scores("final_interaction", LUNA, ctrl)
    cl8 = {**d8, **c8}
    turn = lambda i: int(i[1])                                         # noqa: E731
    common = set(a8) & set(b8)
    diff8 = {i: a8[i] - b8[i] for i in common}
    res["H8"] = {"mean_over_turns": stats.paired_ci({i: a8[i] for i in common}, {i: b8[i] for i in common}, cl8),
                 "compounding_t4_minus_t1": contrast_ci(diff8, cl8, lambda i: turn(i) == 4, lambda i: turn(i) == 1),
                 "per_turn": {t: stats.paired_ci({i: a8[i] for i in common if turn(i) == t},
                                                 {i: b8[i] for i in common if turn(i) == t}, cl8) for t in (1, 2, 3, 4)},
                 "retention": {gid: retention(recs) for gid, recs in ((champ, r8), (ctrl, s8))}}
    res["H9"] = h9(lk, champ, ctrl)
    res["passed"] = {"H1": res["H1"]["improved"], "H7": (res["H7"]["lo"] or 0) > 0,
                     "H8": res["H8"]["mean_over_turns"]["improved"],
                     "H8_compounding": (res["H8"]["compounding_t4_minus_t1"]["lo"] or 0) > 0, "H9": res["H9"]["passed"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis3.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


def retention(recs):
    """Of the questions answered correctly at a turn, the share still answered correctly at the next turn (same
    question, after the representation was rebuilt with new subsystems). Descriptive."""
    by_seed = {}
    for i, rs in recs.items():
        by_seed.setdefault(base_of(i), {})[int(i[1])] = rs[0]
    kept = total = 0
    for turns in by_seed.values():
        for t in (1, 2, 3):
            if t in turns and t + 1 in turns:
                before, after = correct_by_question(turns[t]), correct_by_question(turns[t + 1])
                for q, ok in before.items():
                    if ok and q in after:
                        total += 1
                        kept += after[q]
    return {"kept": kept, "of": total, "share": kept / total if total else None}


# ---------------------------------------------------------------- paper kit (white paper appendix)
def paper_kit(store, lk):
    """Everything the white paper's appendix needs to let readers try the champion: its genome, the exact system
    prompt and build instructions the builder gets, and one representation it wrote on a non-sealed dev item."""
    from v3.episode import BUILD_TASK, BUILD_TASK_PROSE, prompts
    from v3.genome.render import render
    d, kit = store.data, OUT / "paper_kit"
    kit.mkdir(parents=True, exist_ok=True)
    for role, gid in (("champion", lk["primary"]), ("control", lk["control"])):
        g = d["genomes"][gid]
        arm = render(g)
        example = None
        full = (d["evals"].get(gid, {}).get("full", {}).get(LUNA) or {}).get("scores") or {}
        for iid in sorted(full, key=lambda i: (-(full[i] or 0), i)):
            if iid.startswith("K2x"):
                example = iid
                break
        sysp, user = prompts(arm, dict(I.load(example) if example else {"context": "<description>", "question": ""},
                                       question=BUILD_TASK_PROSE if g["genes"]["core"]["kind"] == "prose" else BUILD_TASK))
        rec = None
        for p in RUNS.glob(f"*full/{LUNA}/{gid}/{example}/record.json"):
            rec = json.loads(p.read_text(encoding="utf-8"))
        (kit / f"{role}.json").write_text(json.dumps({
            "id": gid, "genome": g, "tools": [t["function"] for t in arm.tools()], "system_prompt": sysp,
            "build_instruction": BUILD_TASK_PROSE if g["genes"]["core"]["kind"] == "prose" else BUILD_TASK,
            "example_item": example, "example_user_prompt": user,
            "example_representation": rep_text(g, rec) if rec else None,
            "example_score": full.get(example)}, indent=1), encoding="utf-8")
    store.event(f"paper kit written to {kit.relative_to(RUNS.parent)}")


# ---------------------------------------------------------------- orchestration
def run_all(store):
    """Lock, then A, B, C, analysis and the paper kit. Resumable: finished episodes come back from their records."""
    lk = lock(store)
    d = store.data
    for name, fn in (("A", run_a), ("B", run_b), ("C", run_c)):
        if name in d.get("finals_done", []):
            continue
        fn(store, lk)
        d.setdefault("finals_done", []).append(name)
        store.event(f"final {name} complete")
        store.save()
    res = analyse(store, lk)
    paper_kit(store, lk)
    store.event(f"run {RUN_NO} finals complete: " + json.dumps(res["passed"]))
    store.save()
    return res


def pilot(n_b=3, n_c=4):
    """Finals B and C end to end on validation items (never sealed): A0P and SEC_A1, Luna chains and each transfer
    model (and the spare). Records go to runs/pilot_final_*; prints a short summary."""
    from v3.genome.seeds import load_all
    s = load_all()
    gs = [s["A0P"], s["SEC_A1"]]
    out = {}
    seeds_b = I.ladder_seeds(n_b, start=PILOT_B_START)
    recs = pool_run([(g, sd, LUNA, "pilot_final_interaction") for sd in seeds_b for g in gs], chain, 6)
    out["B"] = [{"design": gs[j % 2]["name"], "turns": [(r.get("grade") or {}).get("score") for r in rs],
                 "errors": [str(r.get("error"))[:80] for r in rs if r.get("error")]} for j, rs in enumerate(recs)]
    items = [dict(I.composite(k, b)) for b in I.ladder_seeds(1, start=PILOT_C_START) for k in I.COMPOSITE_KS][:n_c]
    models = transfer_models()
    out["C"] = {}
    for m, model in models.items():
        rs = pool_run([(g, it, m, "pilot_final_transfer") for it in items for g in gs], episode, 4)
        out["C"][m] = {"model": model, "scores": [(r.get("grade") or {}).get("score") for r in rs],
                       "built": sum(1 for r in rs if not r.get("error") and (r.get("validity") or {}).get("usable", True)),
                       "errors": [str(r.get("error"))[:120] for r in rs if r.get("error")]}
    OUT.mkdir(parents=True, exist_ok=True)
    (RUNS / "pilot_final_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "all", "analyse"])
    a = ap.parse_args(argv)
    if a.cmd == "pilot":
        pilot()
        return 0
    store = Store()
    if a.cmd == "analyse":
        print(json.dumps(analyse(store)["passed"], indent=1))
        return 0
    store.acquire()
    try:
        run_all(store)
    finally:
        store.save()
        store.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
