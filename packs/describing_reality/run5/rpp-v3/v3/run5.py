"""Run 5 engine (V3-042; docs/RUN5_DESIGN.md, docs/RUN5_BUILD_PLAN.md step 4). Replaces the breeding loop for run 5.

    python -m v3.run5 loop             the real run (refuses unless the pre-registration lock verifies)
    python -m v3.run5 mini             the mini end-to-end run (pilot items; never in results)
    python -m v3.run5 pilot            the format pilot (8 designs x 60 pilot items; never in results)
    python -m v3.run5 status [mini|pilot]

Every stage is a list of (design, item) pairs; every pair runs exactly once (GPT-6 Luna through Codex, the two-phase
episode), WORKERS at a time. All stages of one run share one record folder (runs/<phase>/codex/<design>/<item>), so a
pair met before is reused, never rerun: design 1 of stage 2 is the stage 1 winner (its records count), and the
founders' hand-written records count again in the final check. A failed run is retried twice, then left out and
reported, never scored 0. state/PAUSE stops new runs of the real run (running ones finish); the mini run and the
pilot use their own state/run5mini/PAUSE and state/run5pilot/PAUSE (state/PAUSE also guards the archived run 4
loop, so it stays in place while they run). Own-code designs wait while the Docker
sandbox is down; they never run code on the host.

Stages and handovers (all by fixed rules in code):
  stage1   15 founders x (360 generated + 500 hand-written)
  choose   winner and runner-up = highest means on the generated items (v3.analysis5.stage1)
  plus     Opus writes the five "+" texts (v3.plus5) with the winner's stage 1 evidence on the generated items
           (V3-056, v3.plus_evidence5; saved in plus_evidence.json); fit check; frozen; the human approves
           (state/run5/PLUS_APPROVED; automatic in the mini run)
  stage2   the 31 new factorial designs x 360 generated items
  verdict2 V3-058: the gated baseline retention rule on stage 2 (v3.analysis5.retention) names the current best
  for each stage n = 3, 4, 5 (V3-058, docs/RUN5_STAGES_DESIGN.md; v3.stages5):
    write    Opus writes 4 candidates for each of the 5 slots of the current best from the firewalled evidence
             (v3.plus_evidence5.stage_brief); mechanical checks; frozen; the human approves (state/run5/stage<n>/
             APPROVED) or sends slots back (REJECT)
    screen   the current best + the 20 candidates x the stage's 320 screen items (runs/run5_s<n>a); a slot's best
             candidate advances if it leads by at least +0.015; at most 4 slots
    examples Opus writes the worked examples the confirmation still needs (checked; the human approves
             EXAMPLES_APPROVED only if any were written)
    confirm  2^m factorial over the advancing slots x the stage's 480 confirmation items (runs/run5_s<n>b); the
             retention rule decides the new current best
  final2   the fresh sealed set (sealed2) is opened once; the final design, FACTS, plain prose, the hand-written top
           founder and the runner-up x the sealed items; the final design x the hand-written items; fixed-sequence
           tests (v3.analysis5.final_stages). The stages stop early when no slot passes the screen (futility) or the
           confirmed gain is under +0.005 (convergence).
  done     analysis written to runs/<phase>/analysis.json
The final check that ran before V3-058 (state "final", the old sealed set) stopped at 12:25 on 2026-10-30; that state
now hands over to verdict2, and the old sealed set's records are retired (v3.run5_sets.Retired).
"""
import argparse
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from v3.config import RUNS, STATE_DIR

WORKERS = 20
RETRIES = 2
BUILDER = "codex"
CONTROL = "A0P"
BLIND = "BLIND"                 # V3-044: the no-notes baseline (answer from the question alone); never chosen
FOUNDERS = ("A0P", "SEC_A1", "ONT_LOOSE", "SEC_FLOW", "G1C3", "A5", "FACTS", "STATES", "SPEC", "A2", "A3", "G1C1",
            "JSON", "MARKDOWN", "OWN", BLIND)
GROUPS = {"notes only": ["A0P", "SEC_A1", "ONT_LOOSE", "SEC_FLOW", "FACTS", "STATES", "SPEC", "JSON", "MARKDOWN",
                         "OWN"],
          "fixed calculator": ["G1C3", "A5", "A2"], "own code": ["A3", "G1C1"], "baseline": [BLIND]}
PILOT_DESIGNS = ("FACTS", "JSON", "MARKDOWN", "OWN", "A2", "A3", "G1C1", "A0P")
MINI_DESIGNS = ("A0P", "FACTS", "JSON", "A3", BLIND)
BACKOFF = (60, 300)             # seconds before the 2nd and 3rd attempt of a run that failed for an infrastructure reason
JUDGE_CAL_N = 100               # V3-044: stage 1 hand-written answers re-graded by Claude Opus (reported, not gating)


class Halted(RuntimeError):
    """An infrastructure failure persisted through the retries: the run stops for the human (never drops the item)."""
PAUSE = STATE_DIR / "PAUSE"


class Paused(RuntimeError):
    pass


class Mode:
    """Where a run keeps its state and records, and which designs and items it uses."""

    def __init__(self, name):
        from v3 import items as I
        from v3 import run5_sets as R5
        self.name = name
        self.dir = STATE_DIR / {"real": "run5", "mini": "run5mini", "pilot": "run5pilot"}[name]
        self.phase = {"real": "run5", "mini": "run5mini", "pilot": "run5pilot"}[name]
        S = R5.sets()
        if name == "real":
            self.founders, self.gen, self.hand = FOUNDERS, S["main"], I.hand5_ids(False)
            self.sealed_name, self.stage2_items = "sealed", S["main"]
        elif name == "mini":
            hp = sorted(I.hand5_ids(True))
            self.founders, self.gen, self.hand = MINI_DESIGNS, S["mini"], hp[:2]
            self.sealed_name, self.stage2_items = "mini_sealed", [S["mini"][0], S["mini"][4]]   # two bases
        else:
            self.founders, self.gen, self.hand = PILOT_DESIGNS, S["pilot"], []
            self.sealed_name, self.stage2_items = None, []
        self.sealed = S[self.sealed_name] if self.sealed_name else []
        self.auto_approve = name == "mini"
        # V3-058: the stages and the final check's fresh sealed set (the mini run: one stage on its own items)
        self.stage_numbers = R5.STAGES if name == "real" else (3,) if name == "mini" else ()
        self.final_sealed_name = {"real": "sealed2", "mini": "mini_sealed"}.get(name)

    def stage_items(self, n, part):
        from v3 import run5_sets as R5
        if self.name == "real":
            return R5.sets()[f"s{n}_{part}"]
        return list(self.gen)                             # the mini run: its own six items, in separate folders

    @property
    def final_sealed(self):
        from v3 import run5_sets as R5
        return R5.sets()[self.final_sealed_name] if self.final_sealed_name else []

    def stage_phase(self, n, part):
        return f"{self.phase}_s{n}{'a' if part == 'screen' else 'b'}"

    @property
    def pause(self):
        return PAUSE if self.name == "real" else self.dir / "PAUSE"

    @property
    def state_file(self):
        return self.dir / "state.json"

    @property
    def records(self):
        return RUNS / self.phase / BUILDER

    @property
    def phase2(self):
        """Stage 2 runs in its own record folder, so all 32 designs run fresh (V3-044: design 1 does not reuse the
        stage 1 winner's records)."""
        return self.phase + "_s2"


# ---------------------------------------------------------------- state
def load_state(mode):
    try:
        return json.loads(mode.state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"stage": "stage1", "events": [], "failed": {}}


SAVE_RETRIES = 20                                         # V3-055: up to about 10 s of a briefly locked state file


def save_state(mode, st):
    mode.dir.mkdir(parents=True, exist_ok=True)
    tmp = mode.state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    for attempt in range(SAVE_RETRIES):
        try:
            tmp.replace(mode.state_file)
            return
        except PermissionError:                           # Windows: the dashboard or OneDrive has it open for a moment
            if attempt == SAVE_RETRIES - 1:
                raise
            time.sleep(0.5)


def event(mode, st, msg):
    st.setdefault("events", []).append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "msg": msg})
    st["events"] = st["events"][-200:]
    save_state(mode, st)
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def founder_genomes(names):
    from v3.genome.seeds import load_all
    seeds = load_all()
    return {n: seeds[n] for n in names}


# ---------------------------------------------------------------- records and scores
def record(mode, gid, iid, phase=None):
    from v3.run5_sets import check_not_retired
    if (phase or mode.phase) == mode.phase:
        check_not_retired([iid])                          # V3-058: the retired sealed set is never read
    p = RUNS / (phase or mode.phase) / BUILDER / gid / iid / "record.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def score(mode, gid, iid, phase=None):
    """The run's score; 0.0 for a run that failed for a reason caused by the design (zero.json, V3-044); None if it
    has not run (or failed for an infrastructure reason and must run again)."""
    r = record(mode, gid, iid, phase)                     # raises Retired for the retired sealed set (V3-058)
    if r and not r.get("error") and (r.get("grade") or {}).get("status") == "ok":
        return r["grade"].get("score")
    if (RUNS / (phase or mode.phase) / BUILDER / gid / iid / "zero.json").exists():
        return 0.0
    return None


def scores(mode, designs, items, phase=None):
    """{design label: {item: score}} for the designs {label: genome}."""
    return {lab: {i: score(mode, g["id"], i, phase) for i in items} for lab, g in designs.items()}


def classify_failure(rec):
    """V3-044. "design": caused by the design meeting the item (a forbidden tool call, the episode running out of
    time) - scored 0 after the retries. "infra": anything else (connection, service, grader errors) - retried with
    backoff, then the run halts for the human. "config": the design cannot run at all - halt."""
    e = str(rec.get("error") or "")
    if "outside the approved Codex allowlist" in e:
        return "config"
    if "INTEGRITY" in e or "rc=-9: timeout" in e:      # code review: only the episode's own wall-clock limit
        return "design"
    return "infra"


def blind_episode(g, item, phase):
    """The BLIND baseline: no build phase; Luna answers the question with no description and no notes."""
    from v3.episode import _raw, record_dir
    from v3.genome.render import render
    from v3.grading.grade import grade
    d = record_dir(phase, BUILDER, g["id"], item["id"])
    path = d / "record.json"
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("grade", {}).get("status") == "ok" and not r.get("error"):
            return r
        path.rename(d / f"record.void{sum(1 for _ in d.glob('record.void*.json'))}.json")
    it = dict(item, context="(The original description is not available and no notes were written. Answer from the "
                            "question alone.)")
    t0 = time.time()
    rec = _raw(BUILDER, g, render(g), it, d / "answer")
    rec.update({"genome": g["id"], "item": item["id"], "system": item.get("system"), "phase": phase, "rep": 0,
                "bank": item.get("bank"), "wall_s_total": time.time() - t0, "mode": "blind"})
    rec["grade"] = {"status": "not_graded"} if rec.get("error") else grade(item, rec.get("final_raw"))
    d.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return rec


def default_runner(g, item, phase):
    from v3.episode import run as run_episode
    return blind_episode(g, item, phase) if g.get("name") == BLIND else run_episode(g, item, BUILDER, phase)


# ---------------------------------------------------------------- running jobs
def job_order(designs, items, seed="run5-order"):
    """Items in a fixed shuffled order; designs rotated per item, so a paused stage has every design at about the
    same number of items done."""
    its = list(items)
    random.Random(seed).shuffle(its)
    labs = list(designs)
    out = []
    for n, i in enumerate(its):
        for j in range(len(labs)):
            out.append((labs[(n + j) % len(labs)], i))
    return out


def own_code(g):
    return g["genes"]["core"]["kind"] in ("simulation", "probabilistic")


def run_jobs(mode, st, designs, items, label, runner=None, workers=WORKERS, phase=None):
    """Run every missing (design, item) pair in `phase` (default: the run's main record folder). Raises Paused on
    PAUSE or an allowance guard, Halted on a persistent infrastructure failure (V3-044)."""
    from v3 import items as I
    from v3 import sandbox
    phase = phase or mode.phase
    runner = runner or default_runner
    st.pop("pause_reason", None)
    st.pop("halt_reason", None)
    st.pop("waiting_for", None)
    todo = [(lab, i) for lab, i in job_order(designs, items) if score(mode, designs[lab]["id"], i, phase) is None]
    total = len(designs) * len(items)
    st["progress"] = {"stage": label, "planned": total, "done": total - len(todo), "t0": time.time(),
                      "done_at_t0": total - len(todo)}
    save_state(mode, st)
    stop, lock = threading.Event(), threading.Lock()
    cache = {}

    def item(iid):
        with lock:
            if iid not in cache:
                cache[iid] = I.load(iid)
            return cache[iid]

    def one(job):
        lab, iid = job
        g = designs[lab]
        if stop.is_set():
            return
        while own_code(g) and not sandbox.available():
            if mode.pause.exists() or stop.is_set():
                stop.set()
                return
            time.sleep(60)
        last, kind = None, None
        for attempt in range(1 + RETRIES):
            if mode.pause.exists() or stop.is_set():
                stop.set()
                return
            try:
                rec = runner(g, item(iid), phase)
            except Exception as e:                       # Paused from the allowance guards, or a crash
                if type(e).__name__ == "Paused":
                    stop.set()
                    with lock:
                        st["pause_reason"] = str(e)
                    return
                rec = {"error": f"{type(e).__name__}: {e}"}
            last = rec
            if not rec.get("error") and (rec.get("grade") or {}).get("status") == "ok":
                kind = None
                break
            kind = classify_failure(rec)
            if kind == "config":
                break
            if kind == "infra" and attempt < RETRIES:
                time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
        with lock:
            key = f"{g['id']}|{iid}"
            if kind == "design":                          # caused by the design: scored 0, never dropped
                d = RUNS / phase / BUILDER / g["id"] / iid
                d.mkdir(parents=True, exist_ok=True)
                (d / "zero.json").write_text(json.dumps({"reason": "design-induced failure after retries",
                                                         "error": str(last.get("error"))[:500]}), encoding="utf-8")
                st.setdefault("zeroed", {})[key] = {"design": lab, "item": iid, "error": str(last.get("error"))[:300]}
            elif kind in ("infra", "config"):             # the run halts; the pair runs again after a restart
                st["halt_reason"] = f"{kind} failure on {lab} x {iid}: {str(last.get('error'))[:300]}"
                stop.set()
            st["progress"]["done"] += 1
            if st["progress"]["done"] % 20 == 0:
                save_state(mode, st)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    save_state(mode, st)
    if stop.is_set():
        if st.get("halt_reason"):
            raise Halted(st["halt_reason"])
        raise Paused(st.get("pause_reason") or f"{mode.pause} present")
    missing = [(lab, i) for lab, i in job_order(designs, items) if score(mode, designs[lab]["id"], i, phase) is None]
    if missing:                                           # code review: never hand over with a hole in the data
        st["halt_reason"] = f"{len(missing)} pairs in {label} have no score (e.g. {missing[0]})"
        raise Halted(st["halt_reason"])
    return {"done": st["progress"]["done"]}


# ---------------------------------------------------------------- stages
def stage1(mode, st, runner=None):
    from v3 import analysis5 as A
    designs = founder_genomes(mode.founders)
    run_jobs(mode, st, designs, list(mode.gen) + list(mode.hand), "stage1", runner)
    sc = scores(mode, designs, list(mode.gen) + list(mode.hand))
    res = A.stage1(sc, list(designs), mode.gen, mode.hand, control=CONTROL if CONTROL in designs else None,
                   groups={g: [f for f in fs if f in designs] for g, fs in GROUPS.items()},
                   forms=own_forms(mode, designs), families=base_families(mode.gen))
    res["sigma"] = item_noise(sc, [d for d in designs if d != BLIND], mode.gen)
    res["cost"] = cost_report(mode, designs, list(mode.gen) + list(mode.hand))
    res["conditional"] = conditional_accuracy(mode, designs, mode.gen)
    write_analysis(mode, "stage1", res)
    if mode.hand:                                         # before the handover, so a restart cannot skip it
        try:
            from v3 import judgecal
            cal = judgecal.calibrate(mode, designs, [d for d in designs if d != BLIND], mode.hand,
                                     n=JUDGE_CAL_N if mode.name == "real" else 4)
            write_analysis(mode, "judge_calibration", cal)
            event(mode, st, f"judge calibration: {cal.get('n')} answers re-graded by Opus, mean |difference| "
                            f"{cal.get('mean_abs_diff')}, correlation {cal.get('correlation')}")
        except SystemExit as e:                           # the Claude allowance stop: pause, do not move on
            raise Paused(f"Claude allowance guard before the judge calibration: {e}")
        except Exception as e:                            # reported, never gating
            event(mode, st, f"judge calibration failed: {type(e).__name__}: {str(e)[:200]}")
    st.update(winner=res["winner"], runner_up=res["runner_up"], hand_top=res.get("hand_top"),
              stage="plus" if mode.stage2_items else "done")
    event(mode, st, f"stage 1 done: winner {res['winner']}, runner-up {res['runner_up']}"
                    f"{' (tied at the top)' if res['tie_at_top'] else ''}"
                    f"{'; hand-written top ' + str(res.get('hand_top')) if res.get('hand_top') else ''}")


def plus(mode, st, call=None):
    from v3 import plus5 as P5
    winner = founder_genomes([st["winner"]])[st["winner"]]
    mode.dir.mkdir(parents=True, exist_ok=True)
    pf = mode.dir / "plus.json"
    from v3 import plus_evidence5 as E
    ev = E.summary(mode, winner, record, score)
    ev_text = E.render(ev)
    ev_hash = hashlib.sha256(ev_text.encode()).hexdigest()[:16]
    if pf.exists() and not (mode.dir / "PLUS_APPROVED").exists():
        old = json.loads(pf.read_text(encoding="utf-8")).get("evidence_hash")
        if old != ev_hash:
            # V3-056: texts written blind are archived, not rejected, and written again. V3-057: so are texts written
            # from an evidence brief other than the current one (V3-056's brief said wrongly that the builder sees no
            # checker problems). Approved texts are never touched.
            kind = "blind" if old is None else "superseded"
            n = len(list(mode.dir.glob(f"plus_{kind}_*.json")))
            pf.replace(mode.dir / f"plus_{kind}_{n + 1}.json")
            event(mode, st, f"the + texts written {'without the stage 1 evidence' if old is None else 'from an earlier evidence brief'}"
                            f" were archived (plus_{kind}_{n + 1}.json); Opus writes new ones")
    if not pf.exists():
        (mode.dir / "plus_evidence.json").write_text(json.dumps({"evidence": ev, "text": ev_text}, indent=1),
                                                     encoding="utf-8")
        if call is None:                                  # the Claude allowance guard before Opus writes
            from v3.hand5.factory import claude_guard
            try:
                claude_guard()
            except SystemExit as e:
                raise Paused(f"Claude allowance guard before the + texts: {e}")
        probs = None
        for attempt in range(3):
            try:
                p = P5.write_plus(winner, call, probs, ev_text)
                probs = P5.fit_check(winner, p)
            except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError,
                    __import__("subprocess").TimeoutExpired) as e:
                # an unreadable reply, a failed or timed-out Opus call: a failed attempt (code review)
                probs = [f"the reply was not the JSON object asked for ({type(e).__name__}: {str(e)[:120]})"]
            if not probs:
                break
            event(mode, st, f"plus texts failed the fit check (attempt {attempt + 1}): {probs[:3]}")
        else:
            st["waiting_for"] = f"the human: the + texts failed the fit check three times ({probs[:3]})"
            save_state(mode, st)
            raise Halted("the + texts failed the fit check three times")
        p = P5.freeze(p)
        p["evidence_hash"] = ev_hash
        pf.write_text(json.dumps(p, indent=1), encoding="utf-8")
        event(mode, st, f"plus texts written and frozen (hash {p['hash']}); waiting for the human's approval")
    p = json.loads(pf.read_text(encoding="utf-8"))
    if P5.freeze({k: p[k] for k in P5.FACTORS} | {"winner": p.get("winner")})["hash"] != p["hash"]:
        raise RuntimeError("plus.json changed after it was frozen")
    if p.get("winner") != winner["id"]:
        raise RuntimeError(f"plus.json was written for {p.get('winner')}, not the stage 1 winner {winner['id']}")
    rej = mode.dir / "PLUS_REJECT"
    if rej.exists():                                      # V3-044: only for a written reason, at most twice
        reason = (rej.read_text(encoding="utf-8").strip().splitlines() or [""])[0].strip().lower()
        n_rej = len(list(mode.dir.glob("plus_rejected_*.json")))
        if reason not in P5.REJECT_REASONS or n_rej >= 2:
            st["waiting_for"] = (f"PLUS_REJECT not accepted ({'unknown reason' if reason not in P5.REJECT_REASONS else 'two rejections already'}); "
                                 f"allowed reasons: {', '.join(P5.REJECT_REASONS)}")
            save_state(mode, st)
            return False
        pf.replace(mode.dir / f"plus_rejected_{n_rej + 1}.json")
        rej.replace(mode.dir / f"plus_reject_reason_{n_rej + 1}.txt")
        event(mode, st, f"plus texts sent back by the human ({reason}); a fresh Opus session writes new ones")
        return plus(mode, st, call)
    approved = mode.dir / "PLUS_APPROVED"
    if mode.auto_approve and not approved.exists():
        approved.write_text("mini run: approved automatically\n", encoding="utf-8")
    if not approved.exists():
        st["waiting_for"] = "the human to read the five + texts (plus.json) and approve stage 2"
        save_state(mode, st)
        return False
    st.pop("waiting_for", None)
    st["stage"] = "stage2"
    event(mode, st, "plus texts approved: stage 2 starts")
    return True


def stage2_designs(mode, st):
    from v3 import plus5 as P5
    winner = founder_genomes([st["winner"]])[st["winner"]]
    p = json.loads((mode.dir / "plus.json").read_text(encoding="utf-8"))
    return {f"D{n + 1:02d}": P5.build(winner, p, n) for n in range(32)}


def stage2(mode, st, runner=None):
    from v3 import analysis5 as A
    designs = stage2_designs(mode, st)
    run_jobs(mode, st, designs, mode.stage2_items, "stage2", runner, phase=mode.phase2)
    res = A.stage2(scores(mode, designs, mode.stage2_items, mode.phase2), list(designs), mode.stage2_items)
    res["cost"] = cost_report(mode, designs, mode.stage2_items, mode.phase2)
    write_analysis(mode, "stage2", res)
    st.update(final_design=res["final_design"], stage="verdict2")
    event(mode, st, f"stage 2 done: fitted best design {res['final_design']} "
                    f"({''.join(f + ('+' if v > 0 else '-') for f, v in res['final_levels'].items())}); the gated "
                    f"retention rule decides next (V3-058)")


# ---------------------------------------------------------------- V3-058: stages 3 to N
def verdict2(mode, st):
    """The gated baseline retention rule on stage 2 (V3-058; it replaces the argmax of the fitted model, which picked
    D18). Its design becomes the current best for stage 3."""
    from v3 import analysis5 as A
    from v3 import run5_sets as R5
    from v3 import stages5 as G
    if mode.name == "real" and not R5.RETIRED_MANIFEST.exists():
        st["waiting_for"] = "the human: write the retired sealed set's manifest (scripts/retire_sealed_v3_058.py)"
        save_state(mode, st)
        raise Halted("the retired sealed set has no manifest (V3-058 section 12b)")
    designs = stage2_designs(mode, st)
    res = A.retention(scores(mode, designs, mode.stage2_items, mode.phase2), list(designs), mode.stage2_items,
                      A.FACTORS, family=A.CONFIRMATORY)
    write_analysis(mode, "stage2_retention", res)
    if st.get("final_design") and "final_design_fitted" not in st:
        st["final_design_fitted"] = st["final_design"]     # the old rule's pick, kept for the record
    g = designs[res["final_design"]]
    st.update(final_design=res["final_design"], n=mode.stage_numbers[0] if mode.stage_numbers else None,
              best={"label": f"stage 2 {res['final_design']}", "id": g["id"], "parts": G.parts_of(g),
                    "stage": 2, "evidence": {"phase": mode.phase2, "items": "stage2"}},
              stages={}, stage="write" if mode.stage_numbers else "final2")
    event(mode, st, f"stage 2 verdict (gated retention rule): {res['final_design']} "
                    f"({'replaces' if res['replaces'] else 'keeps'} the stage 1 winner; kept parts "
                    f"{res['kept_mains'] or 'none'}); stage {st['n']} next")


def stage_dir(mode, n):
    d = mode.dir / f"stage{n}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def base_genome(st):
    return founder_genomes([st["winner"]])[st["winner"]]


def best_genome(st):
    from v3 import stages5 as G
    base = base_genome(st)
    if st["best"]["id"] == base["id"]:
        return base
    g = G.compose(base, st["best"]["parts"], [base["id"]], st["best"]["label"], "run 5 current best")
    if g["id"] != st["best"]["id"]:
        raise RuntimeError(f"the current best rebuilds as {g['id']}, not {st['best']['id']}")
    return g


def evidence_items(mode, st):
    ev = st["best"]["evidence"]
    if ev["items"] == "stage2":
        return mode.stage2_items
    n, part = ev["items"]
    return mode.stage_items(n, part)


def history(mode, st):
    """Earlier candidates with their measured effects for the stage n writer (design texts and numbers only)."""
    out = []
    try:
        an = json.loads((RUNS / mode.phase / "analysis.json").read_text(encoding="utf-8"))
        plus = json.loads((mode.dir / "plus.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        an, plus = {}, {}
    eff = ((an.get("stage2") or {}).get("effects") or {})
    names = {"A": "layout", "B": "fact-writing rules", "C": "answering", "D": "worked example", "E": "tool-call budget"}
    for f, part in names.items():
        if f in plus and f in eff:
            text = {"A": (plus["A"].get("core_params") or plus["A"].get("layout_text")),
                    "B": plus["B"].get("guidance_text"), "C": plus["C"].get("answering_text"),
                    "D": (plus["D"].get("examples") or [""])[0], "E": f"max_turns {plus['E'].get('max_turns')}"}[f]
            from v3.stages5 import length
            out.append({"stage": 2, "part": part, "effect": eff[f]["effect"], "range": eff[f]["range"],
                        "length": length(text), "text": text, "full": True,
                        "verdict": "not kept (stage 2 changed every part at once, one version each)"})
    for n in sorted(st.get("stages", {}), key=int):
        s = st["stages"][n]
        if not s.get("screen"):
            continue
        cand = json.loads((stage_dir(mode, int(n)) / "candidates.json").read_text(encoding="utf-8"))["candidates"]
        res = an.get(f"stage{n}_screen") or {}
        for slot in sorted({c["slot"] for c in cand.values()}):
            mine = [lab for lab in cand if cand[lab]["slot"] == slot]
            worst = min(mine, key=lambda lab: (res["candidates"][lab].get("estimate") or 0))
            adv = (res.get("advancing") or {}).get(str(slot))
            for lab in mine:
                r = res["candidates"][lab]
                from v3.stages5 import SLOTS
                out.append({"stage": int(n), "part": SLOTS[slot], "archetype": cand[lab]["archetype"],
                            "description": cand[lab]["rationale"], "text": cand[lab]["value"],
                            "effect": r.get("estimate"), "range": r.get("range"), "length": cand[lab]["length"],
                            "verdict": "advanced to the confirmation" if lab == adv else "screened out",
                            "full": lab in (adv, worst)})
        c = an.get(f"stage{n}_confirm")
        if c:
            out.append({"stage": int(n), "part": "confirmation", "effect": (c.get("direct") or {}).get("estimate"),
                        "range": (c.get("direct") or {}).get("range"), "length": 0, "full": False,
                        "verdict": (f"design {c['final_design']} became the current best" if c.get("replaces") else
                                    "no design beat the current best; it stays")})
    return out


def write(mode, st, call=None):
    """Opus writes the stage's 20 candidates (slot 1 first, so slots 2-4 know every header they must not name);
    mechanical checks; frozen; then the human gate."""
    from v3 import plus_evidence5 as E
    from v3 import stages5 as G
    n = st["n"]
    d = stage_dir(mode, n)
    cf = d / "candidates.json"
    cur, g = st["best"]["parts"], best_genome(st)
    rej = d / "REJECT"
    if cf.exists() and rej.exists() and not (d / "APPROVED").exists():
        lines = [ln.strip() for ln in rej.read_text(encoding="utf-8").splitlines() if ln.strip()]
        parsed = [re.match(r"^slot\s+([1-5])\s*:\s*(.+)$", ln, re.I) for ln in lines]
        n_rej = len(list(d.glob("candidates_rejected_*.json")))
        if not parsed or not all(parsed) or any(p.group(2).strip().lower() not in G.REJECT_REASONS for p in parsed) \
                or n_rej >= 2:
            st["waiting_for"] = (f"REJECT not accepted ({'two rejections already' if n_rej >= 2 else 'write lines slot <k>: <reason>'}); "
                                 f"allowed reasons: {', '.join(G.REJECT_REASONS)}")
            save_state(mode, st)
            return False
        doc = json.loads(cf.read_text(encoding="utf-8"))
        cf.replace(d / f"candidates_rejected_{n_rej + 1}.json")
        rej.replace(d / f"reject_reason_{n_rej + 1}.txt")
        redo = {int(p.group(1)) for p in parsed}
        keep = {lab: c for lab, c in doc["candidates"].items() if c["slot"] not in redo}
        event(mode, st, f"stage {n}: slots {sorted(redo)} sent back by the human; fresh Opus sessions rewrite them")
        return _write(mode, st, call, keep, sorted(redo))
    if not cf.exists():
        return _write(mode, st, call, {}, sorted(G.SLOTS))
    doc = json.loads(cf.read_text(encoding="utf-8"))
    if G.freeze(doc)["hash"] != doc["hash"]:
        raise RuntimeError(f"stage {n} candidates changed after they were frozen")
    if doc["current"] != cur:
        raise RuntimeError(f"stage {n} candidates were written for another current best")
    approved = d / "APPROVED"
    if mode.auto_approve and not approved.exists():
        approved.write_text("mini run: approved automatically\n", encoding="utf-8")
    if not approved.exists():
        st["waiting_for"] = (f"the human to read the stage {n} candidates (state/run5/stage{n}/candidates.json) and "
                             f"approve them (APPROVED) or send slots back (REJECT)")
        save_state(mode, st)
        return False
    st.pop("waiting_for", None)
    st["stage"] = "screen"
    event(mode, st, f"stage {n} candidates approved: the screen starts")
    return True


def _write(mode, st, call, keep, slots):
    from v3 import plus_evidence5 as E
    from v3 import stages5 as G
    n = st["n"]
    d = stage_dir(mode, n)
    cur, g = st["best"]["parts"], best_genome(st)
    ph = st["best"]["evidence"]["phase"]
    ev = E.summary(mode, g, lambda m, gid, iid: record(m, gid, iid, ph), lambda m, gid, iid: score(m, gid, iid, ph),
                   items=evidence_items(mode, st))
    text = E.stage_brief(n, ev, history(mode, st))
    (d / "evidence.json").write_text(json.dumps({"evidence": ev, "text": text}, indent=1), encoding="utf-8")
    if call is None:
        from v3.hand5.factory import claude_guard
        try:
            claude_guard()
        except SystemExit as e:
            raise Paused(f"Claude allowance guard before the stage {n} candidates: {e}")
    cands = dict(keep)
    for slot in slots:
        forbidden = G.layout_names(cur, cands) if slot != 1 else ()
        new, probs = G.write_slot(slot, cur, g, text, forbidden, call, log=lambda m: event(mode, st, f"stage {n}: {m}"))
        if probs:
            st["waiting_for"] = f"the human: stage {n} slot {slot} failed the checks three times ({probs[:3]})"
            save_state(mode, st)
            raise Halted(f"stage {n} slot {slot} candidates failed the checks three times")
        cands.update(new)
    for slot in G.naming_clashes(cur, cands):             # a kept text that names a header of a rewritten layout
        cands = {lab: c for lab, c in cands.items() if c["slot"] != slot}
        new, probs = G.write_slot(slot, cur, g, text, G.layout_names(cur, cands), call,
                                  log=lambda m: event(mode, st, f"stage {n}: {m}"))
        if probs:
            st["waiting_for"] = f"the human: stage {n} slot {slot} failed the checks three times ({probs[:3]})"
            save_state(mode, st)
            raise Halted(f"stage {n} slot {slot} candidates failed the checks three times")
        cands.update(new)
    probs = G.check_screen(cur, g, cands)
    if probs:
        st["waiting_for"] = f"the human: the stage {n} screen designs do not all build ({probs[:3]})"
        save_state(mode, st)
        raise Halted(f"stage {n} screen designs: {probs[:2]}")
    cands = dict(sorted(cands.items()))
    doc = G.freeze({"stage": n, "current": cur, "current_id": g["id"], "candidates": cands,
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest()[:16]})
    (d / "candidates.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    event(mode, st, f"stage {n} candidates written and frozen (hash {doc['hash']}; lengths "
                    + ", ".join(f"{lab} {c['length']}" for lab, c in cands.items()) + "); waiting for the human")
    return write(mode, st, call)


def _candidates(mode, n):
    return json.loads((stage_dir(mode, n) / "candidates.json").read_text(encoding="utf-8"))["candidates"]


def _cube(mode, st, n):
    from v3 import stages5 as G
    cube = G.initial_cube(st["best"]["parts"], _candidates(mode, n))
    ef = stage_dir(mode, n) / "examples.json"
    if ef.exists():
        cube.update(json.loads(ef.read_text(encoding="utf-8"))["examples"])
    return cube


def screen_designs(mode, st, n):
    from v3 import stages5 as G
    cur, g, cands = st["best"]["parts"], best_genome(st), _candidates(mode, n)
    cube = _cube(mode, st, n)
    out = {"CUR": g}
    for lab, c in cands.items():
        out[lab] = G.compose(base_genome(st), G.design_parts(cur, {c["slot"]: lab}, cands, cube), [g["id"]],
                             f"S{n} {lab}", f"run 5 stage {n} screen: slot {c['slot']} archetype {c['archetype']}")
    return out


def screen(mode, st, runner=None):
    from v3 import analysis5 as A
    n = st["n"]
    designs = screen_designs(mode, st, n)
    items = mode.stage_items(n, "screen")
    ph = mode.stage_phase(n, "screen")
    run_jobs(mode, st, designs, items, f"stage{n}_screen", runner, phase=ph)
    cands = _candidates(mode, n)
    res = A.screen(scores(mode, designs, items, ph), "CUR",
                   {lab: {"slot": c["slot"], "archetype": c["archetype"], "length": c["length"]}
                    for lab, c in cands.items()}, items)
    res["cost"] = cost_report(mode, designs, items, ph)
    write_analysis(mode, f"stage{n}_screen", res)
    st["stages"][str(n)] = {"screen": {"advancing": res["advancing"], "capped": res["capped"]}}
    if not res["advancing"]:
        st["stage"] = "final2"
        event(mode, st, f"stage {n} screen: no slot's best candidate leads by +{A.SCREEN_GATE}; the stages stop "
                        f"(futility) and the final check follows")
        return
    st["stage"] = "examples"
    event(mode, st, f"stage {n} screen done: advancing " + ", ".join(f"slot {s} {lab}" for s, lab in
                                                                     res["advancing"].items()))


def examples(mode, st, call=None):
    from v3 import stages5 as G
    n = st["n"]
    d = stage_dir(mode, n)
    adv = {int(s): lab for s, lab in st["stages"][str(n)]["screen"]["advancing"].items()}
    cands = _candidates(mode, n)
    cube = _cube(mode, st, n)
    ef = d / "examples.json"
    need = G.needed_examples(adv, cands, cube)
    if need:
        if call is None:
            from v3.hand5.factory import claude_guard
            try:
                claude_guard()
            except SystemExit as e:
                raise Paused(f"Claude allowance guard before the stage {n} examples: {e}")
        written = json.loads(ef.read_text(encoding="utf-8"))["examples"] if ef.exists() else {}
        for key in need:
            l1, l2, l5 = key.split("|")
            target = G.design_parts(st["best"]["parts"], {1: l1, 2: l2}, cands, {})
            ex, probs = G.write_example(target, cube[G.example_key("0", "0", l5)], call,
                                        log=lambda m: event(mode, st, f"stage {n} example {key}: {m}"))
            if probs:
                st["waiting_for"] = f"the human: stage {n} example {key} failed the checks three times ({probs[:2]})"
                save_state(mode, st)
                raise Halted(f"stage {n} example {key} failed the checks three times")
            written[key] = ex
        doc = {"stage": n, "examples": written,
               "hash": hashlib.sha256(json.dumps(written, sort_keys=True).encode()).hexdigest()[:16]}
        ef.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        event(mode, st, f"stage {n}: {len(need)} worked examples written for the confirmation (hash {doc['hash']})")
    if ef.exists():
        approved = d / "EXAMPLES_APPROVED"
        if mode.auto_approve and not approved.exists():
            approved.write_text("mini run: approved automatically\n", encoding="utf-8")
        if not approved.exists():
            st["waiting_for"] = (f"the human to read the stage {n} examples (state/run5/stage{n}/examples.json) and "
                                 f"approve them (EXAMPLES_APPROVED)")
            save_state(mode, st)
            return False
    st.pop("waiting_for", None)
    st["stage"] = "confirm"
    event(mode, st, f"stage {n}: the confirmation starts")
    return True


def confirm_designs(mode, st, n):
    """The 2^m designs in standard order over the advancing slots (lowest slot changes fastest); D01 = the current
    best."""
    from v3 import stages5 as G
    adv = {int(s): lab for s, lab in st["stages"][str(n)]["screen"]["advancing"].items()}
    slots = sorted(adv)
    cur, g, cands, cube = st["best"]["parts"], best_genome(st), _candidates(mode, n), _cube(mode, st, n)
    out, levels = {}, {}
    for j, lab in enumerate(G.confirm_labels(len(slots))):
        lv = {s: (adv[s] if (j >> b) & 1 else "0") for b, s in enumerate(slots)}
        levels[lab] = lv
        out[lab] = g if j == 0 else G.compose(base_genome(st), G.design_parts(cur, lv, cands, cube), [g["id"]],
                                              f"S{n} {lab}", f"run 5 stage {n} confirmation: {lv}")
    return out, levels, slots


def confirm(mode, st, runner=None):
    from v3 import analysis5 as A
    from v3 import stages5 as G
    n = st["n"]
    designs, levels, slots = confirm_designs(mode, st, n)
    items = mode.stage_items(n, "confirm")
    ph = mode.stage_phase(n, "confirm")
    run_jobs(mode, st, designs, items, f"stage{n}_confirm", runner, phase=ph)
    res = A.retention(scores(mode, designs, items, ph), list(designs), items, [f"slot{s}" for s in slots])
    res["levels"] = levels
    res["cost"] = cost_report(mode, designs, items, ph)
    write_analysis(mode, f"stage{n}_confirm", res)
    st["stages"][str(n)]["confirm"] = {"final_design": res["final_design"], "replaces": res["replaces"],
                                       "gain": res["gain"]}
    if res["replaces"]:
        fd = designs[res["final_design"]]
        parts = G.design_parts(st["best"]["parts"], levels[res["final_design"]], _candidates(mode, n),
                               _cube(mode, st, n))
        st["best"] = {"label": f"stage {n} {res['final_design']}", "id": fd["id"], "parts": parts, "stage": n,
                      "evidence": {"phase": ph, "items": [n, "confirm"]}}
    else:
        st["best"]["evidence"] = {"phase": ph, "items": [n, "confirm"]}   # its fresh D01 records
    later = [k for k in mode.stage_numbers if k > n]
    if not res["replaces"] or res["gain"] < A.CONVERGE or not later:
        why = ("no design beat the current best" if not res["replaces"] else
               f"the gain {res['gain']} is under +{A.CONVERGE}" if res["gain"] < A.CONVERGE else "the last stage")
        st["stage"] = "final2"
        event(mode, st, f"stage {n} confirmation done: {res['final_design']}"
                        f"{' replaces the current best' if res['replaces'] else ''}; the stages stop ({why})")
        return
    st.update(n=later[0], stage="write")
    event(mode, st, f"stage {n} confirmation done: {res['final_design']} is the new current best (gain "
                    f"{res['gain']}); stage {later[0]} next")


def final2(mode, st, runner=None):
    """The final check on the fresh sealed set (V3-058 section 8): the final design, FACTS, plain prose, the
    hand-written top founder and the stage 1 runner-up; the final design on the hand-written items."""
    from v3 import analysis5 as A
    from v3 import run5_sets as R5
    fd = best_genome(st) if st.get("best") else base_genome(st)
    names = [x for x in dict.fromkeys((st["winner"], CONTROL, st.get("hand_top"), st.get("runner_up"))) if x]
    fo = founder_genomes(names)
    labels = {"FINAL": fd, **fo}
    designs = {}
    for lab, g in labels.items():                        # one run per genome
        if g["id"] not in {x["id"] for x in designs.values()}:
            designs[lab] = g
    if mode.final_sealed_name not in R5.opened():
        R5.open_sealed(mode.final_sealed_name, f"{mode.name} final check, after the stages (V3-058)")
        event(mode, st, f"sealed set '{mode.final_sealed_name}' opened ({len(mode.final_sealed)} items)")
    run_jobs(mode, st, designs, mode.final_sealed, "final2", runner)
    run_jobs(mode, st, {"FINAL": fd}, mode.hand, "final2_hand", runner)
    sc = scores(mode, labels, list(mode.final_sealed) + list(mode.hand))

    def lab(name):
        return None if not name else ("FINAL" if fo[name]["id"] == fd["id"] else name)
    res = A.final_stages(sc, "FINAL", lab(st["winner"]), lab(CONTROL), lab(st.get("hand_top")),
                         lab(st.get("runner_up")), mode.final_sealed, mode.hand)
    res["final_label"] = (st.get("best") or {}).get("label")
    write_analysis(mode, "final2", res)
    st["stage"] = "done"
    event(mode, st, "final check done: " + "; ".join(f"{k} lead {v.get('estimate')} {v.get('range')} "
                                                     f"{'counted' if v.get('counted') else 'no'}"
                                                     for k, v in res["sequence"].items()))


# ---------------------------------------------------------------- reporting helpers
def own_forms(mode, designs):
    """Which form OWN's notes took, item by item (v3.formclass), with k."""
    from v3.analysis5 import k_of
    from v3.formclass import classify
    if "OWN" not in designs:
        return None
    out = {}
    for iid in mode.gen:
        r = record(mode, designs["OWN"]["id"], iid)
        if r and r.get("notes"):
            f = classify(r["notes"])["form"]
            out.setdefault(str(k_of(iid)), {}).setdefault(f, 0)
            out[str(k_of(iid))][f] += 1
    return out


def base_families(items):
    """{item: the problem family of its base system} (a composite item's first member is its base system)."""
    import re as _re
    from discovery.world import make_instance
    out, cache = {}, {}
    for iid in items:
        m = _re.match(r"^K\d+x(\d+)$", iid)
        if m:
            b = int(m.group(1))
            if b not in cache:
                cache[b] = make_instance(b).family
            out[iid] = cache[b]
    return out


def cost_report(mode, designs, items, phase=None):
    """V3-044 secondary outcome: mean output tokens (build + answer) and wall-clock seconds per item, by design."""
    out = {}
    for lab, g in designs.items():
        toks, wall = [], []
        for iid in items:
            r = record(mode, g["id"], iid, phase)
            if r and not r.get("error"):
                toks.append(r.get("output_tokens") or 0)
                wall.append(r.get("wall_s_total") or 0)
        if toks:
            out[lab] = {"n": len(toks), "mean_output_tokens": round(sum(toks) / len(toks)),
                        "mean_wall_s": round(sum(wall) / len(wall), 1)}
    return out


def counts_as_built(rec):
    """A checked core's notes count as built when usable and executable (v3.episode.buildable), and the run did not
    fail."""
    v = (rec.get("validity") if not rec.get("error") else None) or {}
    return bool(v.get("usable") and v.get("executable", True))


def conditional_accuracy(mode, designs, items):
    """V3-044 secondary outcome: for checked cores, the build pass rate and the mean score on items whose notes passed
    their check, beside the unconditional mean (all items, design-induced failures as 0)."""
    out = {}
    for lab, g in designs.items():
        ok, all_s, built = [], [], 0
        for iid in items:
            s = score(mode, g["id"], iid)
            if s is None:
                continue
            all_s.append(s)
            r = record(mode, g["id"], iid) or {}
            if counts_as_built(r):                        # a voided run is not "built"
                built += 1
                ok.append(s)
        if all_s:
            out[lab] = {"n": len(all_s), "unconditional": round(sum(all_s) / len(all_s), 4),
                        "built": built if any((record(mode, g["id"], i) or {}).get("validity") for i in items)
                        else None, "conditional": round(sum(ok) / len(ok), 4) if ok else None}
    return out


def item_noise(sc, designs, items):
    """sigma: the residual sd of score = design + item (two-way additive), over complete items."""
    import numpy as np
    from v3.analysis5 import matrix
    Y, used, _ = matrix(sc, designs, items)
    if Y.shape[1] < 2 or Y.shape[0] < 2:
        return None
    R = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + Y.mean()
    return round(float(np.sqrt((R ** 2).sum() / ((Y.shape[0] - 1) * (Y.shape[1] - 1)))), 4)


def write_analysis(mode, part, res):
    from v3.analysis5 import dumps
    p = RUNS / mode.phase / "analysis.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = {}
    doc[part] = json.loads(dumps(res))
    doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- the loop
def loop(mode, runner=None, plus_call=None):
    st = load_state(mode)
    st.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))
    save_state(mode, st)
    try:
        while True:
            if mode.pause.exists():
                raise Paused(f"{mode.pause} present")
            s = st["stage"]
            if s == "stage1":
                stage1(mode, st, runner)
            elif s == "plus":
                if not plus(mode, st, plus_call):
                    if mode.name == "mini":
                        raise RuntimeError("mini run: plus approval did not happen")
                    time.sleep(300)
                    st = load_state(mode)
            elif s == "stage2":
                stage2(mode, st, runner)
            elif s in ("verdict2", "final"):              # "final": the check V3-058 stopped and retired
                verdict2(mode, st)
            elif s in ("write", "examples"):
                if not (write if s == "write" else examples)(mode, st, plus_call):
                    if mode.name == "mini":
                        raise RuntimeError(f"mini run: the {s} approval did not happen")
                    time.sleep(300)
                    st = load_state(mode)
            elif s == "screen":
                screen(mode, st, runner)
            elif s == "confirm":
                confirm(mode, st, runner)
            elif s == "final2":
                final2(mode, st, runner)
            else:
                event(mode, st, f"{mode.name} run complete")
                return st
    except Paused as e:
        event(mode, st, f"paused: {e}")
        return st
    except Halted as e:
        st["waiting_for"] = f"the human: the run halted on an infrastructure failure ({e}). Fix it, then restart."
        event(mode, st, f"halted: {e}")
        return st


def status(mode):
    st = load_state(mode)
    p = st.get("progress") or {}
    rate = None
    if p.get("t0") and p.get("done", 0) > p.get("done_at_t0", 0):
        rate = (p["done"] - p["done_at_t0"]) / max(1e-9, (time.time() - p["t0"]) / 3600)
    return {"mode": mode.name, "stage": st.get("stage"), "progress": p, "runs_per_hour": rate and round(rate, 1),
            "scored_zero": len(st.get("zeroed", {})), "halt": st.get("halt_reason"), "winner": st.get("winner"),
            "runner_up": st.get("runner_up"), "hand_top": st.get("hand_top"),
            "final_design": st.get("final_design"), "waiting_for": st.get("waiting_for"),
            "stage_n": st.get("n"), "current_best": (st.get("best") or {}).get("label"),
            "paused": mode.pause.exists(), "last_events": st.get("events", [])[-5:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("loop", "mini", "pilot", "status"))
    ap.add_argument("which", nargs="?", default="real")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(status(Mode(a.which)), indent=1, default=str))
        return 0
    if a.cmd == "loop":
        from v3 import prereg
        changed = prereg.verify()
        if changed:
            print(f"the pre-registration lock does not verify ({changed[:5]}): refusing to start run 5")
            return 2                                 # the runner stops on 2 (refused)
    loop(Mode({"loop": "real", "mini": "mini", "pilot": "pilot"}[a.cmd]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
