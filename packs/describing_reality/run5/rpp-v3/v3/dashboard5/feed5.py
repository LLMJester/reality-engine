"""Run 5 dashboard feed (V3-042; RUN5_BUILD_PLAN part B). Read-only: it reads the run 5 state, its records and the
usage logs, and never writes. build(mode) -> one JSON payload for index.html.

Live numbers are computed from the records written so far with v3.analysis5 (fewer bootstrap resamples than the
final analysis, so a refresh stays fast); the pre-registered results are the ones in runs/<phase>/analysis.json.
"""
import json
import os
import pathlib
import threading
import time

from v3 import analysis5 as A
from v3.config import RUNS, STATE_DIR

LIVE_BOOT = 1000
PHASE = {"real": "run5", "mini": "run5mini", "pilot": "run5pilot"}
STATE = {"real": "run5", "mini": "run5mini", "pilot": "run5pilot"}


class ScoreCache:
    """Scores of every record under runs/<phase>/codex, read once per file (records do not change once written; a
    voided attempt is renamed, and the new record.json is read when its mtime changes)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.seen = {}           # path -> (mtime, design id, item, score or None, extra)
        self.extra = {}          # phase -> {design id: {item: extra}} from the last scan (trust report)

    def scan(self, phase):
        root = RUNS / phase / "codex"
        out, ext = {}, {}
        self.extra[phase] = ext
        if not root.is_dir():
            return out
        with self.lock:
            for gdir in root.iterdir():
                if not gdir.is_dir():
                    continue
                for idir in gdir.iterdir():
                    p = idir / "record.json"
                    if (idir / "zero.json").exists():      # a design-induced failure, scored 0
                        out.setdefault(gdir.name, {})[idir.name] = 0.0
                        continue
                    try:
                        mt = p.stat().st_mtime
                    except OSError:
                        continue
                    hit = self.seen.get(str(p))
                    if not hit or hit[0] != mt or hit[4].get("v") != 2:   # v2 adds build pass and tokens
                        try:
                            r = json.loads(p.read_text(encoding="utf-8"))
                            ok = not r.get("error") and (r.get("grade") or {}).get("status") == "ok"
                            v = (r.get("validity") if not r.get("error") else None) or {}
                            hit = (mt, gdir.name, idir.name, r["grade"]["score"] if ok else None,
                                   {"v": 2, "t": mt, "wall": r.get("wall_s_total"), "tok": r.get("output_tokens"),
                                    "built": bool(v.get("usable") and v.get("executable", True)) if v else None})
                        except (OSError, ValueError, KeyError):
                            continue
                        self.seen[str(p)] = hit
                    out.setdefault(hit[1], {})[hit[2]] = hit[3]
                    ext.setdefault(hit[1], {})[hit[2]] = hit[4]
        return out


CACHE = ScoreCache()


def _json(p, default=None):
    try:
        return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _last_line(p):
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4000))
            lines = [ln for ln in f.read().decode("utf-8", "replace").splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def judge_spend(since_ts):
    total, n = 0.0, 0
    try:
        with open(RUNS / "ledger.jsonl", encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("purpose") == "v3_grade" and r.get("t", "") >= since_ts:
                    total += float(r.get("cost") or 0)
                    n += 1
    except OSError:
        pass
    return {"usd": round(total, 2), "calls": n}


def usage():
    cx = _last_line(RUNS / "codex_usage.jsonl") or {}
    cl = _last_line(RUNS / "claude_usage.jsonl") or {}
    return {"luna_weekly_percent": cx.get("weekly_used_percent"), "luna_resets": cx.get("weekly_resets_at") or
            cx.get("resets_at"), "claude_weekly_percent": cl.get("seven_day"), "claude_resets": cl.get("seven_day_resets")}


def _labels(mode_obj, st):
    """{label: genome} for the founders, and the stage 2 designs once the + texts exist."""
    from v3 import run5 as R
    out = {"founders": R.founder_genomes(mode_obj.founders), "stage2": {}}
    if st.get("winner") and (mode_obj.dir / "plus.json").exists():
        try:
            out["stage2"] = R.stage2_designs(mode_obj, st)
        except Exception:
            pass
    return out


def _by_label(raw, designs, items):
    return {lab: {i: raw.get(g["id"], {}).get(i) for i in items} for lab, g in designs.items()}


def live_stage1(mode_obj, raw, founders):
    from v3 import run5 as R
    founders = {k: v for k, v in founders.items() if k != R.BLIND}    # the baseline is never in the tests
    out = {}
    for name, items in (("generated", mode_obj.gen), ("hand", mode_obj.hand)):
        if not items:
            continue
        sc = _by_label(raw, founders, items)
        done = {lab: sum(v is not None for v in s.values()) for lab, s in sc.items()}
        res = A.compare(sc, list(founders), list(items), R.CONTROL if R.CONTROL in founders else None, LIVE_BOOT)
        res["done"] = done
        res["planned"] = len(items)
        out[name] = res
    if mode_obj.gen:
        out["by_k"] = A.by_subset(_by_label(raw, founders, mode_obj.gen), list(founders), list(mode_obj.gen), A.k_of)
    return out


def live_stage2(mode_obj, raw, s2):
    items = list(mode_obj.stage2_items)
    if not s2 or not items:
        return None
    sc = _by_label(raw, s2, items)
    done = {lab: sum(v is not None for v in s.values()) for lab, s in sc.items()}
    res = None
    common = [i for i in items if all(sc[lab].get(i) is not None for lab in s2)]
    if len(common) >= 2:                                 # code review: only with at least two items common to all 32
        try:
            res = A.stage2(sc, list(s2), common, boot=300)
        except Exception:
            res = None
    return {"done": done, "planned": len(items), "fit": res,
            "levels": {lab: dict(zip(A.FACTORS, ["+" if x > 0 else "-" for x in r.values()]))
                       for lab, r in zip(s2, A.design_table())}}


def fresh_cache():
    """server5 keeps CACHE across a live reload of this module; a cache made by code without the trust fields
    (no .extra) is replaced, so the next scan reads every record once more."""
    global CACHE
    if not hasattr(CACHE, "extra"):
        CACHE = ScoreCache()
    return CACHE


def _metrics(raw, ext, gid, items):
    """Score, score by k, notes-check pass share and output tokens for one design on the items it has done."""
    sc = {i: raw.get(gid, {}).get(i) for i in items}
    done = [i for i in items if sc[i] is not None]
    if not done:
        return None
    by_k = {}
    for i in done:
        by_k.setdefault(A.k_of(i), []).append(sc[i])
    ex = [ext.get(gid, {}).get(i) or {} for i in done]
    checked = [e["built"] for e in ex if e.get("built") is not None]
    toks = [e["tok"] for e in ex if isinstance(e.get("tok"), (int, float))]
    return {"n": len(done), "mean": round(sum(sc[i] for i in done) / len(done), 4),
            "by_k": {str(k): round(sum(v) / len(v), 4) for k, v in sorted(by_k.items()) if k is not None},
            "built": round(sum(checked) / len(checked), 4) if checked else None,
            "tokens": round(sum(toks) / len(toks)) if toks else None}


def _pair(raw, ext, a, b, items):
    common = [i for i in items if raw.get(a, {}).get(i) is not None and raw.get(b, {}).get(i) is not None]
    return {"before": _metrics(raw, ext, a, common), "after": _metrics(raw, ext, b, common)}


def before_after(m, st, analysis, raw, raw2, labs, s2live):
    """Trust report "before and after": the stage 1 winner as it was (design D01 of stage 2) against the stage 2
    verdict (V3-058: the gated retention rule), on the stage 2 items; each later stage that changed the current best,
    its old against its new current best on that stage's confirmation items; and after the final check, FACTS
    against the final design on the fresh sealed set. The retired sealed set is never read (V3-058)."""
    s2 = labs.get("stage2") or {}
    if not s2:
        return None
    fit = (analysis.get("stage2") or (s2live or {}).get("fit") or {})
    ret = analysis.get("stage2_retention") or {}
    if analysis.get("stage2") and not ret:
        return None                                       # stage 2 is done; its verdict (V3-058) is not recorded yet
    target = ret.get("final_design") or fit.get("final_design")
    if not target or target not in s2:
        return None
    win, tgt = s2["D01"]["id"], s2[target]["id"]
    out = {"target": target, "final": bool(ret), "levels": ret.get("candidate_levels") if ret else fit.get("final_levels"),
           "stage2": _pair(raw2, CACHE.extra.get(m.phase2, {}), win, tgt, m.stage2_items),
           "current_best": (st.get("best") or {}).get("label"), "stages": []}
    for n in sorted((st.get("stages") or {}), key=int):
        c = analysis.get(f"stage{n}_confirm") or {}
        if c.get("replaces"):
            ph = m.stage_phase(int(n), "confirm")
            try:
                from v3 import run5 as R
                designs = R.confirm_designs(m, st, int(n))[0]
            except Exception:
                continue
            rw = CACHE.scan(ph)
            out["stages"].append({"stage": int(n), "design": c["final_design"],
                                  **_pair(rw, CACHE.extra.get(ph, {}), designs["D01"]["id"],
                                          designs[c["final_design"]]["id"], m.stage_items(int(n), "confirm"))})
    f2 = analysis.get("final2")
    if f2 and st.get("best"):
        try:
            from v3 import run5 as R
            fid = R.best_genome(st)["id"]
            facts = labs["founders"][st["winner"]]["id"]
            out["sealed"] = _pair(raw, CACHE.extra.get(m.phase, {}), facts, fid, m.final_sealed)
        except Exception:
            pass
    return out


def _span(raw, ext, pairs):
    """(done, first, last) over (design id, item) pairs: runs with a score, and the earliest and latest record
    times (the records' own file times)."""
    done, ts = 0, []
    for gid, iid in pairs:
        if raw.get(gid, {}).get(iid) is not None:
            done += 1
            t = (ext.get(gid, {}).get(iid) or {}).get("t")
            if t:
                ts.append(t)
    fmt = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) if t else None
    return done, fmt(min(ts) if ts else None), fmt(max(ts) if ts else None)


def stage_tiles(m, st, raw, raw2, labs, analysis, ba):
    """Progress page tiles: one per stage plus the trust report, from the records (counts, first and last record
    time), the state and the analysis."""
    from v3 import run5 as R
    founders = labs["founders"]
    ext1, ext2 = CACHE.extra.get(m.phase, {}), CACHE.extra.get(m.phase2, {})
    s1_items = list(m.gen) + list(m.hand)
    d1, f1, l1 = _span(raw, ext1, [(g["id"], i) for g in founders.values() for i in s1_items])
    an1 = analysis.get("stage1") or {}
    w = st.get("winner")
    tiles = {"stage1": {"done": d1, "planned": len(founders) * len(s1_items), "first": f1, "last": l1,
                        "winner": w, "winner_mean": ((an1.get("generated") or {}).get("designs") or {}).get(w, {}).get("mean")}}
    ev = st.get("events") or []
    approved = next((e["t"] for e in ev if e["msg"].startswith("plus texts approved")), None)
    plus = _json(m.dir / "plus.json") or {}
    tiles["plus"] = {"approved": approved or ((m.dir / "PLUS_APPROVED").exists() or None), "hash": plus.get("hash"),
                     "written": next((e["t"] for e in reversed(ev) if e["msg"].startswith("plus texts written")), None)}
    s2 = labs.get("stage2") or {}
    d2, f2, l2 = _span(raw2, ext2, [(g["id"], i) for g in s2.values() for i in m.stage2_items])
    ret = analysis.get("stage2_retention") or {}
    tiles["stage2"] = {"done": d2, "planned": 32 * len(m.stage2_items), "first": f2, "last": l2,
                       "final_design": ret.get("final_design"), "fitted": (analysis.get("stage2") or {}).get("final_design")}
    # V3-058: the retired sealed set (a count of its record files only; no score is read) and stages 3 to N
    tiles["retired"] = retired_count(m)
    order = ["stage1", "plus", "stage2"]
    pos = stage_position(m, st)
    from v3 import run5_sets as R5
    for n in m.stage_numbers if m.name != "real" or R5._stage_file() else ():
        sd = st.get("stages", {}).get(str(n), {})
        doc = _json(m.dir / f"stage{n}" / "candidates.json") or {}
        k_w, k_s, k_c = f"s{n}_write", f"s{n}_screen", f"s{n}_confirm"
        order += [k_w, k_s, k_c]
        tiles[k_w] = {"title": f"Stage {n} prep", "sub": "20 candidates", "hash": doc.get("hash"),
                      "approved": (m.dir / f"stage{n}" / "APPROVED").exists(), "written": bool(doc)}
        ph = m.stage_phase(n, "screen")
        items = m.stage_items(n, "screen")
        try:
            sdes = R.screen_designs(m, st, n) if doc else {}
        except Exception:
            sdes = {}
        rw, ex = (CACHE.scan(ph), CACHE.extra.get(ph, {})) if sdes else ({}, {})
        dn, fs, ls = _span(rw, ex, [(g["id"], i) for g in sdes.values() for i in items])
        tiles[k_s] = {"title": f"Stage {n} screen", "sub": "21 designs, one part changed each", "done": dn,
                      "planned": 21 * len(items), "first": fs, "last": ls,
                      "advancing": (sd.get("screen") or {}).get("advancing")}
        adv = (sd.get("screen") or {}).get("advancing") or {}
        items_c = m.stage_items(n, "confirm")
        ph_c = m.stage_phase(n, "confirm")
        try:
            cdes = R.confirm_designs(m, st, n)[0] if adv else {}
        except Exception:
            cdes = {}
        rw, ex = (CACHE.scan(ph_c), CACHE.extra.get(ph_c, {})) if cdes else ({}, {})
        dn, fs, ls = _span(rw, ex, [(g["id"], i) for g in cdes.values() for i in items_c])
        tiles[k_c] = {"title": f"Stage {n} confirmation", "sub": f"2^{len(adv) or 'm'} designs", "done": dn,
                      "planned": (2 ** len(adv)) * len(items_c) if adv else None, "estimate": not adv,
                      "first": fs, "last": ls, "result": sd.get("confirm")}
        if pos > stage_rank(m, k_c) and str(n) not in (st.get("stages") or {}):   # the stages stopped before this one
            for k in (k_w, k_s, k_c):
                tiles[k].update(skipped=True, planned=None, estimate=False)
    # the final check: the final design, FACTS, plain prose, the hand-written top and the runner-up on the fresh
    # sealed set, and the final design on the hand-written items
    names = [d for d in dict.fromkeys((w, R.CONTROL, st.get("hand_top"), st.get("runner_up"))) if d and d in founders]
    ids = {founders[n]["id"] for n in names}
    try:
        fid = R.best_genome(st)["id"] if st.get("best") else None
    except Exception:
        fid = None
    if fid:
        ids.add(fid)
    sealed2 = m.final_sealed if m.final_sealed_name and (m.name != "real" or R5._stage_file()) else []
    hand_rerun = fid and fid != founders.get(w, {}).get("id")   # the stage 1 winner's hand-written runs are stage 1's
    pairs = [(g, i) for g in ids for i in sealed2] + ([(fid, i) for i in m.hand] if hand_rerun else [])
    d3, f3_, l3 = _span(raw, ext1, pairs) if pos >= stage_rank(m, "final2") else (0, None, None)
    planned3 = len(ids) * len(sealed2) + (len(m.hand) if hand_rerun else 0)
    order.append("final2")
    tiles["final2"] = {"title": "Final check", "sub": "Fresh sealed questions", "done": d3, "planned": planned3,
                       "first": f3_, "last": l3,
                       "verdicts": {k: v.get("counted") for k, v in ((analysis.get("final2") or {}).get("sequence") or {}).items()}}
    tiles["order"] = order
    tiles["position"] = pos
    tiles["trust"] = ba and {"target": ba["target"], "final": ba["final"], "levels": ba.get("levels"),
                             "current_best": ba.get("current_best"),
                             "stage2": {k: (ba["stage2"].get(k) or {}).get("mean") for k in ("before", "after")},
                             "n": (ba["stage2"].get("before") or {}).get("n"),
                             "sealed": ba.get("sealed") and {k: (ba["sealed"].get(k) or {}).get("mean") for k in ("before", "after")}}
    return tiles


def stage_rank(m, key):
    """Position of a tile key in the run's order (stage1, plus, stage2, then write/screen/confirm per stage, final2)."""
    order = ["stage1", "plus", "stage2"] + [f"s{n}_{p}" for n in m.stage_numbers for p in ("write", "screen", "confirm")]
    order.append("final2")
    return order.index(key) if key in order else len(order)


def stage_position(m, st):
    """Which tile is running now, from the state (done = past every tile)."""
    s, n = st.get("stage") or "stage1", st.get("n")
    if s == "done":
        return stage_rank(m, "final2") + 1
    if s in ("stage1", "plus", "stage2"):
        return stage_rank(m, s)
    if s in ("verdict2", "final"):
        return stage_rank(m, "stage2") + 1
    if s in ("write", "screen", "confirm"):
        return stage_rank(m, f"s{n}_{s}")
    if s == "examples":
        return stage_rank(m, f"s{n}_confirm")
    return stage_rank(m, "final2")


def retired_count(m):
    """V3-058: how many record files the retired sealed set has (file names only; no record is opened)."""
    if m.name != "real":
        return None
    from v3 import run5_sets as R5
    doc = _json(R5.RETIRED_MANIFEST) or {}
    return {"records": doc.get("records"), "manifest": doc.get("manifest_sha256"), "made": doc.get("made")}


def stages_payload(m, st, analysis):
    """Per stage: the candidates (texts, lengths, rationale), the screen (final or live) and the confirmation (final
    or live), for the stage tabs."""
    from v3 import run5 as R
    out = {}
    for n in m.stage_numbers:
        doc = _json(m.dir / f"stage{n}" / "candidates.json")
        if not doc:
            continue
        cands = doc["candidates"]
        sres = analysis.get(f"stage{n}_screen")
        live = False
        done_s, done_c, mean_s = {}, {}, {}                  # items done and mean per design (as stage 2's table shows)
        try:
            ph, items = m.stage_phase(n, "screen"), m.stage_items(n, "screen")
            rw = CACHE.scan(ph)
            sdes = R.screen_designs(m, st, n)
            done_s = {lab: sum(rw.get(g["id"], {}).get(i) is not None for i in items) for lab, g in sdes.items()}
            mean_s = {}
            for lab, g in sdes.items():                      # each design's mean on the items it has done
                v = [rw.get(g["id"], {}).get(i) for i in items]
                v = [x for x in v if x is not None]
                mean_s[lab] = round(sum(v) / len(v), 4) if v else None
        except Exception:
            pass
        if not sres:
            try:
                ph, items = m.stage_phase(n, "screen"), m.stage_items(n, "screen")
                designs = R.screen_designs(m, st, n)
                sc = _by_label(CACHE.scan(ph), designs, items)
                common = [i for i in items if all(sc[lab].get(i) is not None for lab in designs)]
                if len({A.cluster_of(i) for i in common}) >= 2:
                    sres = A.screen(sc, "CUR", {lab: {"slot": c["slot"], "archetype": c["archetype"],
                                                      "length": c["length"]} for lab, c in cands.items()}, common)
                    live = True
            except Exception:
                sres = None
        cres = analysis.get(f"stage{n}_confirm")
        clive, levels = False, None
        adv = ((st.get("stages") or {}).get(str(n)) or {}).get("screen", {}).get("advancing")
        if adv:
            try:
                designs, levels, slots = R.confirm_designs(m, st, n)
                rwc, items_c = CACHE.scan(m.stage_phase(n, "confirm")), m.stage_items(n, "confirm")
                done_c = {lab: sum(rwc.get(g["id"], {}).get(i) is not None for i in items_c)
                          for lab, g in designs.items()}
                if not cres:
                    ph, items = m.stage_phase(n, "confirm"), m.stage_items(n, "confirm")
                    sc = _by_label(CACHE.scan(ph), designs, items)
                    common = [i for i in items if all(sc[lab].get(i) is not None for lab in designs)]
                    if len({A.cluster_of(i) for i in common}) >= 2:
                        cres = A.retention(sc, list(designs), common, [f"slot{s}" for s in slots])
                        clive = True
            except Exception:
                pass
        out[str(n)] = {"candidates": {lab: {k: c.get(k) for k in ("slot", "archetype", "rationale", "length", "value")}
                                      for lab, c in cands.items()},
                       "hash": doc.get("hash"), "approved": (m.dir / f"stage{n}" / "APPROVED").exists(),
                       "examples": (_json(m.dir / f"stage{n}" / "examples.json") or {}).get("examples"),
                       "screen": sres, "screen_live": live, "confirm": cres, "confirm_live": clive,
                       "levels": levels or (cres or {}).get("levels"), "done_screen": done_s, "mean_screen": mean_s,
                       "done_confirm": done_c}
    return out


def build(mode="real"):
    from v3 import run5 as R
    fresh_cache()
    t0 = time.time()
    m = R.Mode(mode)
    st = _json(m.state_file, {}) or {}
    raw = CACHE.scan(m.phase)
    raw2 = CACHE.scan(m.phase2)                           # stage 2 runs in its own record folder (V3-044)
    labs = _labels(m, st)
    prog = st.get("progress") or {}
    rate = None
    if prog.get("t0") and prog.get("done", 0) > prog.get("done_at_t0", 0):
        rate = (prog["done"] - prog["done_at_t0"]) / max(1e-9, (time.time() - prog["t0"]) / 3600)
    planned_total = len(m.founders) * (len(m.gen) + len(m.hand)) + 32 * len(m.stage2_items) + \
        (4 * len(m.sealed) + len(m.hand) if m.sealed else 0)
    runs_done = sum(1 for r in (raw, raw2) for d in r.values() for v in d.values() if v is not None)
    analysis = _json(RUNS / m.phase / "analysis.json", {}) or {}
    plus = _json(m.dir / "plus.json")
    s2live = live_stage2(m, raw2, labs["stage2"])
    from v3 import run5_sets as R5
    has_sets = m.name != "real" or R5._stage_file() is not None      # V3-058: the stage item sets exist
    payload = {
        "mode": mode, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": st.get("stage") or "not started", "paused": m.pause.exists(),
        "pause_text": m.pause.read_text(encoding="utf-8", errors="replace")[:300] if m.pause.exists() else "",
        "waiting_for": st.get("waiting_for"), "winner": st.get("winner"), "runner_up": st.get("runner_up"),
        "final_design": st.get("final_design"),
        "progress": {"stage": prog.get("stage"), "planned": prog.get("planned"), "done": prog.get("done"),
                     "runs_per_hour": rate and round(rate, 1),
                     "eta_hours": round((prog["planned"] - prog["done"]) / rate, 1) if rate else None,
                     "runs_done_total": runs_done, "planned_total": planned_total},
        "failed": list((st.get("zeroed") or {}).values())[-50:], "n_failed": len(st.get("zeroed") or {}),
        "halt": st.get("halt_reason"), "hand_top": st.get("hand_top"),
        "events": (st.get("events") or [])[-30:],
        "usage": usage(), "judge": judge_spend(st.get("started", "9999")),
        "sizes": {"founders": len(m.founders), "generated": len(m.gen), "hand": len(m.hand),
                  "stage2_items": len(m.stage2_items), "sealed": len(m.sealed), "workers": R.WORKERS,
                  "stages": list(m.stage_numbers),
                  "sealed2": len(m.final_sealed) if has_sets and m.final_sealed_name else None,
                  "screen_items": len(m.stage_items(m.stage_numbers[0], "screen")) if has_sets and m.stage_numbers else None,
                  "confirm_items": len(m.stage_items(m.stage_numbers[0], "confirm")) if has_sets and m.stage_numbers else None},
        "groups": {g: [f for f in fs if f in labs["founders"]] for g, fs in R.GROUPS.items()},
        "founders": {lab: {"id": g["id"], "core": g["genes"]["core"]["kind"]} for lab, g in labs["founders"].items()},
        "stage1": live_stage1(m, raw, labs["founders"]),
        "stage2": s2live,
        "plus": plus and {k: plus.get(k) for k in ("A", "B", "C", "D", "E", "rationale", "hash", "winner")},
        "plus_approved": (m.dir / "PLUS_APPROVED").exists(),
        "evidence": (_json(m.dir / "plus_evidence.json") or {}).get("evidence"),   # trust report: why notes fail
        "evidence_text": (_json(m.dir / "plus_evidence.json") or {}).get("text"),  # stage 2 tab: what Opus was given
        "before_after": before_after(m, st, analysis, raw, raw2, labs, s2live),
        "analysis": analysis,
    }
    payload["tiles"] = t = stage_tiles(m, st, raw, raw2, labs, analysis, payload["before_after"])
    payload["stages"] = stages_payload(m, st, analysis) if m.stage2_items else {}
    payload["stage_n"], payload["current_best"] = st.get("n"), (st.get("best") or {}).get("label")
    payload["stage_briefs"] = {str(n): {"text": ev.get("text"), "t": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(
        (m.dir / f"stage{n}" / "evidence.json").stat().st_mtime))}
        for n in m.stage_numbers if (ev := _json(m.dir / f"stage{n}" / "evidence.json"))}   # what Opus was given
    stages = ["stage1"] + ([k for k in t["order"] if k not in ("stage1", "plus") and not k.endswith("_write")]
                           if m.stage2_items else [])
    payload["progress"]["runs_done_total"] = sum(t[s].get("done") or 0 for s in stages)   # the tiles' own counts
    payload["progress"]["planned_total"] = sum(t[s].get("planned") or 0 for s in stages)
    payload["build_secs"] = round(time.time() - t0, 2)
    return payload
