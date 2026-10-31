"""The first problem pack: "Describing Reality for AI" (run 5 of the Discovery Programme, October 2026).

The problem: a small model (GPT-6 Luna, through Codex) reads a long, messy description of a system without seeing any
questions and writes notes in the form a design prescribes; later it answers multiple-choice questions from its notes
alone. A candidate is a design (a "genome": note format, fact-writing rules, answering procedure, worked example,
budget); an item is a generated system at size k = 1, 2, 4 or 8 (K<k>x<base>), or a hand-written document
(H<n>-<q>); the score is the share of questions answered right, graded by script against the generator's key.

This pack is an adapter: the problem code itself (the designs, the generator, the two-phase episode, the grader, the
designer's prompts and checks) is the run 5 harness, which it imports from RPP_V3_HOME (default ../rpp-v3; the public
copy is describing-reality-for-ai/harness). The generator that makes the systems and their answer keys is not public:
with it, anyone could rebuild the answer keys of sealed test sets.

  ReplayPack  reads the scores run 5 saved, so the engine can rerun the whole of run 5's decisions with no model
              calls (the replay test: the engine must reach the same numbers)
  LivePack    runs the real thing: Luna through Codex for the episodes, Claude Opus as the designer
"""
import json
import os
import pathlib
import re
import sys
import threading

from engine.pack import Pack, Paused

HOME = pathlib.Path(os.environ.get("RPP_V3_HOME") or pathlib.Path(__file__).resolve().parents[3] / "rpp-v3").resolve()
if str(HOME) not in sys.path:
    sys.path.insert(0, str(HOME))

# run 5's parts. Step 4 (take apart) used factors A-E; step 5 (the stages) used slots 1-5.
TAKEAPART = {"layout": "A", "rules": "B", "answering": "C", "example": "D", "budget": "E"}
BUILDOUT = {"layout": 1, "rules": 2, "strategy": 3, "answering": 4, "example": 5}


def _v3():
    from v3 import analysis5, items, plus5, run5, run5_sets, stages5   # noqa: F401  (rpp-v3 on sys.path)
    return analysis5, items, plus5, run5, run5_sets, stages5


class DescribingReality(Pack):
    name = "describing-reality"
    description = "How should a small model write down a system it reads about, so it can answer questions later?"
    control = "A0P"

    def __init__(self, founders=None, sets=None, retire=True):
        self._founders = founders
        self._sets = sets or {}
        self._retire = retire

    # ------------------------------------------------------------------ candidates and items
    def founders(self):
        _, _, _, R, _, _ = _v3()
        return R.founder_genomes(self._founders or R.FOUNDERS)

    def cluster_of(self, item):
        return _v3()[0].cluster_of(item)

    def level_of(self, item):
        k = _v3()[0].k_of(item)
        return None if k is None else f"k={k}"

    def retired_items(self):
        return _v3()[4].retired_items() if self._retire else frozenset()

    @staticmethod
    def reduce(rec, score):
        """The engine's record of a run 5 episode: the score and the counts behind it, never the text."""
        g = (rec or {}).get("grade") or {}
        v = (rec or {}).get("validity") or {}
        return {"score": score, "answered": g.get("answered"), "n_probes": g.get("n_probes"),
                "built": bool(v.get("usable") and v.get("executable", True)) if v else None,
                "output_tokens": (rec or {}).get("output_tokens"), "wall_s": (rec or {}).get("wall_s_total")}

    def failure_type(self, r):
        if not r:
            return None
        if r.get("error"):
            return "run failed"
        if r.get("built") is False:
            return "notes failed their check"
        if r.get("answered") is not None and r.get("n_probes") and r["answered"] < r["n_probes"]:
            return "questions left unanswered"
        if (r.get("score") or 0) < 1:
            return "wrong answers"
        return None

    def evidence(self, candidate, records):
        rs = [r for r in records.values() if r and not r.get("error")]
        if not rs:
            return {}
        built = [r["built"] for r in rs if r.get("built") is not None]
        toks = [r["output_tokens"] for r in rs if r.get("output_tokens")]
        out = {}
        if built:
            out["Notes that passed their checker"] = f"{sum(built) / len(built):.0%}"
        if toks:
            out["Mean output tokens per case"] = round(sum(toks) / len(toks))
        return out

    # ------------------------------------------------------------------ parts
    def slots(self, kind):
        return list(TAKEAPART if kind == "takeapart" else BUILDOUT)

    def parts(self, candidate, kind):
        if kind == "takeapart":
            return {s: f + "-" for s, f in TAKEAPART.items()}       # the winner as it is (level "-")
        if candidate.get("stage_parts"):
            return json.loads(json.dumps(candidate["stage_parts"]))
        return _v3()[5].parts_of(candidate)

    def compose(self, base, parts, changes, kind, extras):
        _, _, P5, _, _, G = _v3()
        if kind == "takeapart":
            n = sum(1 << "ABCDE".index(TAKEAPART[s]) for s in changes)
            return P5.build(base, extras["plus"], n)
        if not changes:
            return base
        levels = {BUILDOUT[s]: a["orig_label"] for s, a in changes.items()}
        new = G.design_parts(parts, levels, extras["cands"], extras["cube"])
        g = G.compose(base, new, [base["id"]], "engine " + " ".join(f"{s}={a['orig_label']}"
                                                                   for s, a in changes.items()))
        g["stage_parts"] = new
        return g

    @staticmethod
    def alts_from(cands, slots):
        """{slot name: [alt, ...]} from run 5's candidates {label: {"slot": n, "archetype": a, "value": ...}}."""
        out = {s: [] for s in slots}
        for lab, c in sorted(cands.items(), key=lambda x: (x[1]["slot"], x[1]["archetype"])):
            s = next(k for k, v in BUILDOUT.items() if v == c["slot"])
            if s in out:
                out[s].append(dict(c, orig_label=lab))
        return out


class ReplayPack(DescribingReality):
    """Run 5 again, from its saved records: no model is called. Every run returns the score run 5 recorded for that
    design and item in the matching record folder; the designer returns the texts Opus wrote in run 5."""
    FOLDERS = {"main": "run5", "takeapart": "run5_s2"}

    def item_sets(self):
        _, I, _, _, R5, _ = _v3()
        s = R5.sets()
        out = {"main": s["main"], "hand": I.hand5_ids(False), "sealed2": s["sealed2"]}
        for n in R5.STAGES:
            out[f"s{n}_screen"], out[f"s{n}_confirm"] = s[f"s{n}_screen"], s[f"s{n}_confirm"]
        return out

    def phase(self, folder):
        m = re.fullmatch(r"round(\d+)_(screen|confirm)", folder)
        if m:
            return f"run5_s{int(m.group(1)) + 2}{'a' if m.group(2) == 'screen' else 'b'}"
        return self.FOLDERS[folder]

    def run(self, candidate, item, ctx):
        _, _, _, R, R5, _ = _v3()
        R5.check_not_retired([item])
        ph = self.phase(ctx["folder"])
        d = R.RUNS / ph / R.BUILDER / candidate["id"] / item
        try:
            rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec = None
        if rec and not rec.get("error") and (rec.get("grade") or {}).get("status") == "ok":
            return self.reduce(rec, rec["grade"].get("score"))
        if (d / "zero.json").exists():
            return dict(self.reduce(rec, None), error="run 5: design-induced failure", failure="design")
        return {"error": f"run 5 has no record for {candidate['id']} x {item} in {ph}", "failure": "config"}

    def write(self, kind, candidate, parts, brief, slots, per_slot):
        _, _, _, R, _, G = _v3()
        sd = R.STATE_DIR / "run5"
        if kind == "takeapart":
            plus = json.loads((sd / "plus.json").read_text(encoding="utf-8"))
            return {s: [{"value": plus[TAKEAPART[s]], "factor": TAKEAPART[s], "rationale": "run 5's + text"}]
                    for s in slots}, {"plus": plus}
        n = brief["round"] + 2                                  # engine round 1 = run 5 stage 3
        f = sd / f"stage{n}" / "candidates.json"
        if not f.exists():
            raise RuntimeError(f"run 5 never wrote stage {n}")
        doc = json.loads(f.read_text(encoding="utf-8"))
        if doc["current"] != parts:
            raise RuntimeError(f"run 5's stage {n} candidates were written for other parts")
        return self.alts_from(doc["candidates"], slots), {"cands": doc["candidates"], "stage": n,
                                                          "cube": G.initial_cube(parts, doc["candidates"])}

    def prepare(self, kind, candidate, parts, advancing, extras):
        _, _, _, R, _, _ = _v3()
        f = R.STATE_DIR / "run5" / f"stage{extras['stage']}" / "examples.json"
        if f.exists():
            extras = dict(extras, cube=dict(extras["cube"], **json.loads(f.read_text(encoding="utf-8"))["examples"]))
            return extras, True
        return extras, False


class LivePack(DescribingReality):
    """The real problem: every run is a two-phase episode of GPT-6 Luna through Codex, graded by script; the designer
    is Claude Opus with run 5's prompts and mechanical checks. Own-code designs need Docker and are best left out of a
    pilot. Records of the episodes go under <work>/pack_records."""

    def __init__(self, founders=None, sets=None, retire=True, item_sets_file=None):
        super().__init__(founders, sets, retire)
        self._file = item_sets_file
        self._cache, self._lock = {}, threading.Lock()

    def item_sets(self):
        if self._file:
            p = pathlib.Path(self._file)
            p = p if p.is_absolute() else pathlib.Path(__file__).resolve().parents[2] / p
            doc = json.loads(p.read_text(encoding="utf-8"))
            ks = doc.get("ks", [1, 2, 4, 8])
            return {n: [f"K{k}x{b}" for b in bases for k in ks] for n, bases in doc["sets"].items()}
        return dict(self._sets)

    def item(self, iid):
        with self._lock:
            if iid not in self._cache:
                self._cache[iid] = _v3()[1].load(iid)
            return self._cache[iid]

    def run(self, candidate, item, ctx):
        from v3.episode import run as episode
        _, _, _, R, _, _ = _v3()
        phase = str(pathlib.Path(ctx["dir"]).parents[2].parent / "pack_records" / ctx["folder"])
        if candidate.get("name") == R.BLIND:
            rec = R.blind_episode(candidate, self.item(item), phase)
        else:
            rec = episode(candidate, self.item(item), R.BUILDER, phase)
        if not rec.get("error") and (rec.get("grade") or {}).get("status") == "ok":
            return self.reduce(rec, rec["grade"].get("score"))
        kind = R.classify_failure(rec) if rec.get("error") else "infra"
        return dict(self.reduce(rec, None), error=str(rec.get("error") or rec.get("grade"))[:500], failure=kind)

    @staticmethod
    def guard(what):
        from v3.hand5.factory import claude_guard
        try:
            claude_guard()
        except SystemExit as e:
            raise Paused(f"Claude allowance guard before {what}: {e}")

    def write(self, kind, candidate, parts, brief, slots, per_slot):
        _, _, P5, _, _, G = _v3()
        from engine.runner import Halted
        self.guard("the designer")
        if kind == "takeapart":
            probs = None
            for attempt in range(3):
                try:
                    plus = P5.write_plus(candidate, None, probs, brief["text"])
                    probs = P5.fit_check(candidate, plus)
                except Exception as e:                    # an unreadable reply or a failed call: a failed attempt
                    probs = [f"the reply was not the JSON object asked for ({type(e).__name__}: {str(e)[:120]})"]
                if not probs:
                    break
            else:
                raise Halted(f"the take-apart versions failed the checks three times: {probs[:3]}")
            plus = P5.freeze(plus)
            return {s: [{"value": plus[TAKEAPART[s]], "factor": TAKEAPART[s], "rationale": "Opus"}]
                    for s in slots}, {"plus": plus}
        cands = {}
        nums = [BUILDOUT[s] for s in slots]
        for n in nums:
            forbidden = G.layout_names(parts, cands) if n != 1 else ()
            new, probs = G.write_slot(n, parts, candidate, brief["text"], forbidden)
            if probs:
                raise Halted(f"slot {n} versions failed the checks three times: {probs[:3]}")
            cands.update(new)
        for n in [x for x in G.naming_clashes(parts, cands) if x in nums]:
            cands = {lab: c for lab, c in cands.items() if c["slot"] != n}
            new, probs = G.write_slot(n, parts, candidate, brief["text"], G.layout_names(parts, cands))
            if probs:
                raise Halted(f"slot {n} versions failed the checks three times: {probs[:3]}")
            cands.update(new)
        probs = G.check_screen(parts, candidate, cands)
        if probs:
            raise Halted(f"the screen designs do not all build: {probs[:3]}")
        return self.alts_from(cands, slots), {"cands": cands, "cube": G.initial_cube(parts, cands)}

    def prepare(self, kind, candidate, parts, advancing, extras):
        _, _, _, _, _, G = _v3()
        from engine.runner import Halted
        adv = {BUILDOUT[s]: a["orig_label"] for s, a in advancing.items()}
        need = G.needed_examples(adv, extras["cands"], extras["cube"])
        if not need:
            return extras, False
        self.guard("the worked examples")
        cube = dict(extras["cube"])
        for key in need:
            l1, l2, l5 = key.split("|")
            target = G.design_parts(parts, {1: l1, 2: l2}, extras["cands"], {})
            ex, probs = G.write_example(target, cube[G.example_key("0", "0", l5)])
            if probs:
                raise Halted(f"example {key} failed the checks three times: {probs[:2]}")
            cube[key] = ex
        return dict(extras, cube=cube), True
