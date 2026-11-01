"""The run 5 engine end to end with a fake builder (no Luna, no Claude): every stage and handover of the mini run,
pause and resume without repeats, design-induced failures scored 0, infrastructure failures halting the run, the seal,
the BLIND baseline, stage 2 run fresh, the final check's extra designs, the plus fit check and rejection rules
(V3-042, V3-044)."""
import json
import zlib

import pytest

from v3 import items as I
from v3 import plus5 as P5
from v3 import run5 as R
from v3 import run5_sets as R5

from tests.stages5_fakes import FakeOpus

BASE = {"A0P": 0.45, "FACTS": 0.80, "JSON": 0.70, "A3": 0.40, "BLIND": 0.15}
FAIL = ("JSON", "K8x")          # JSON always times out on k = 8 items (a design-induced failure)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(R, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(R, "PAUSE", tmp_path / "state" / "PAUSE")
    monkeypatch.setattr(R, "BACKOFF", (0, 0))
    monkeypatch.setattr(R5, "OPENED", tmp_path / "state" / "opened.json")
    fake = {f"HP1-{n}": {"id": f"HP1-{n}", "system": "HP1", "question": "q", "context": "c", "grading": "ai",
                         "task_type": "prediction", "answer_key": {}} for n in (1, 2, 3)}
    monkeypatch.setattr(I, "_hand5", lambda: fake)
    monkeypatch.setattr(I, "load", lambda iid: fake[iid] if iid in fake else {"id": iid, "system": iid})
    monkeypatch.setattr("v3.judgecal.calibrate", lambda *a, **k: {"n": 0})
    return tmp_path


class FakeLuna:
    def __init__(self, mode, pause_after=None, infra_fail=None):
        self.mode, self.calls, self.pairs, self.pause_after, self.infra_fail = mode, 0, [], pause_after, infra_fail

    def __call__(self, g, item, phase):
        self.calls += 1
        self.pairs.append((g["id"], item["id"], phase))
        if self.pause_after and self.calls == self.pause_after:
            self.mode.pause.parent.mkdir(parents=True, exist_ok=True)
            self.mode.pause.write_text("test pause")
        name = g.get("name", "")
        if self.infra_fail and (name, item["id"]) == self.infra_fail:
            return {"error": "rc=1: connection reset by peer"}
        if name == FAIL[0] and item["id"].startswith(FAIL[1]):
            return {"error": "rc=-9: timeout"}
        noise = (zlib.crc32(f"{g['id']}|{item['id']}".encode()) % 100) / 1000
        if name in BASE:
            s = BASE[name]
        elif name.startswith("S3"):                     # stage 3 (V3-058): layout s1a2 and example s5a2 help
            gn = g["genes"]
            s = 0.70 + (0.20 if "EDGES" in gn["core"]["params"]["headers"] else 0) \
                + (0.20 if any("id P1" in e for e in gn["examples"]["items"]) else 0)
        else:                                           # stage 2 designs: C+ helps, E+ helps a little
            s = 0.78 + (0.06 if "C+" in name else 0) + (0.02 if "E+" in name else 0)
        rec = {"genome": g["id"], "item": item["id"], "grade": {"status": "ok", "score": min(1, s + noise)},
               "output_tokens": 100, "wall_s_total": 1.0}
        d = R.RUNS / phase / "codex" / g["id"] / item["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "record.json").write_text(json.dumps(rec))
        return rec


PLUS_REPLY = json.dumps({
    "A": {"core_params": {"headers": ["PARTS", "QUANTITIES", "CAUSES", "RULES", "TIMELINE", "UNKNOWNS"],
                          "tags": False, "granularity": "fine"}},
    "B": {"guidance_text": "Write one fact per bullet with the exact names, numbers and units from the text; mark "
                           "anything the text leaves open as unknown, and keep every condition attached to its fact."},
    "C": {"answering_text": "Find every note about the things the question names, trace causes step by step, "
                            "then answer with values and units."},
    "D": {"examples": ["## THINGS\n- pump P1: fills tank T1\n## VALUES\n- T1.level: 40% at start"],
          "examples_new_layout": ["## PARTS\n- pump P1: fills tank T1\n## QUANTITIES\n- T1.level: 40% at start"]},
    "rationale": {"A": "a", "B": "b", "C": "c", "D": "d"}})


def test_mini_run_end_to_end(env):
    mode = R.Mode("mini")
    assert len(mode.gen) == 6 and len(mode.hand) == 2 and len(mode.sealed) == 4
    assert all(R5.is_sealed(int(i.split("x")[1])) for i in mode.sealed)        # sealed before the final check
    luna = FakeLuna(mode, pause_after=5 * 8 + 4 + 10)    # early in stage 2 (stage 1 = 40 runs + 4 retries)
    opus = FakeOpus(PLUS_REPLY)
    st = R.loop(mode, runner=luna, plus_call=opus)
    assert st["stage"] == "stage2" and "paused" in st["events"][-1]["msg"]
    assert mode.pause.exists() and not R.PAUSE.exists()
    mode.pause.unlink()
    st = R.loop(mode, runner=luna, plus_call=opus)
    assert st["stage"] == "done"
    from collections import Counter
    c = Counter(luna.pairs)
    # every pair ran once per record folder, except the design-induced failures: 1 + RETRIES tries, then scored 0
    assert {k: v for k, v in c.items() if v > 1} and all(v in (1, 1 + R.RETRIES) for v in c.values())
    assert all(z["design"] == "JSON" for z in st["zeroed"].values()) and len(st["zeroed"]) >= 2
    k8 = next(i for i in mode.gen if i.startswith("K8x"))
    assert R.score(mode, R.founder_genomes(["JSON"])["JSON"]["id"], k8) == 0.0
    an = json.loads((env / "runs" / "run5mini" / "analysis.json").read_text())
    assert set(an) >= {"stage1", "stage2", "stage2_retention", "stage3_screen", "stage3_confirm", "final2"}
    assert "final" not in an                                                    # the old final check is gone (V3-058)
    s1 = an["stage1"]
    assert s1["generated"]["dropped"] == 0                                      # failures count as 0, not dropped
    assert st["winner"] == "FACTS" and "BLIND" not in s1["generated"]["designs"]  # BLIND is never chosen
    assert s1["baseline"]["design"] == "BLIND" and s1["notes_vs_tools"] is not None
    assert "cost" in s1 and "conditional" in s1
    # stage 2: all 32 designs ran fresh in their own record folder (design 1 = the winner, not reused)
    s2pairs = {(g, i) for g, i, ph in luna.pairs if ph == "run5mini_s2"}
    assert len(s2pairs) == 32 * 2
    winner_id = R.founder_genomes(["FACTS"])["FACTS"]["id"]
    assert (winner_id, mode.stage2_items[0]) in s2pairs
    assert an["stage2"]["final_levels"]["C"] == 1 and an["stage2"]["final_levels"]["E"] == 1
    # V3-058: the gated retention rule (two bases: nothing is significant, so FACTS stays), not the fitted argmax
    assert an["stage2_retention"]["final_design"] == "D01" and st["final_design_fitted"] == "D32"
    # stage 3: 21 screen designs on their own folder; slots 1 and 5 advance; the 2^2 confirmation needs one example
    sc = an["stage3_screen"]
    assert len(sc["candidates"]) == 20 and set(sc["advancing"]) == {"1", "5"}
    assert sc["advancing"] == {"1": "s1a2", "5": "s5a2"}
    assert {(g, i) for g, i, ph in luna.pairs if ph == "run5mini_s3a"} and \
        len({g for g, i, ph in luna.pairs if ph == "run5mini_s3a"}) == 21
    assert len({g for g, i, ph in luna.pairs if ph == "run5mini_s3b"}) == 4
    ex = json.loads((env / "state" / "run5mini" / "stage3" / "examples.json").read_text())["examples"]
    assert list(ex) == ["s1a2|0|s5a2"] and "## EDGES" in ex["s1a2|0|s5a2"]
    assert all(c["length"] > 0 for c in json.loads((env / "state" / "run5mini" / "stage3" / "candidates.json")
                                                    .read_text())["candidates"].values())
    # final check on the mini sealed set, in the fixed sequence
    f = an["final2"]
    assert "1_vs_plain_prose" in f["sequence"] and f["sequence"]["1_vs_plain_prose"]["counted"] in (True, False)
    assert (env / "state" / "run5mini" / "PLUS_APPROVED").exists()               # automatic in the mini run
    assert not any(R5.is_sealed(int(i.split("x")[1])) for i in mode.sealed)     # opened once at the final check
    # the dashboard feed shows every stage of the finished mini run (V3-058: N stages)
    from v3.dashboard5 import feed5 as F
    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(F, "RUNS", env / "runs")
        mp.setattr(F, "CACHE", F.ScoreCache())
        p = F.build("mini")
    t = p["tiles"]
    assert t["order"] == ["stage1", "plus", "stage2", "s3_write", "s3_screen", "s3_confirm", "final2"]
    assert t["s3_screen"]["done"] == t["s3_screen"]["planned"] == 21 * 6 and t["s3_confirm"]["planned"] == 4 * 6
    assert t["s3_write"]["approved"] and t["position"] == len(t["order"])
    s = p["stages"]["3"]
    assert len(s["candidates"]) == 20 and s["screen"]["advancing"] == {"1": "s1a2", "5": "s5a2"} and s["confirm"]
    assert set(t["final2"]["verdicts"]) >= {"1_vs_plain_prose"} and p["sizes"]["stages"] == [3]
    json.dumps(p)                                                               # the payload is valid JSON


def test_infrastructure_failure_halts_then_resumes(env):
    mode = R.Mode("mini")
    a0p = R.founder_genomes(["A0P"])["A0P"]["id"]
    luna = FakeLuna(mode, infra_fail=("A0P", mode.gen[0]))
    st = R.loop(mode, runner=luna, plus_call=FakeOpus(PLUS_REPLY))
    assert "halted" in st["events"][-1]["msg"] and st["stage"] == "stage1"
    assert sum(1 for p in luna.pairs if p[:2] == (a0p, mode.gen[0])) == 1 + R.RETRIES
    luna.infra_fail = None                                # the service recovers; the human restarts
    st = R.load_state(mode)
    st.pop("halt_reason", None)
    st.pop("waiting_for", None)
    R.save_state(mode, st)
    st = R.loop(mode, runner=luna, plus_call=FakeOpus(PLUS_REPLY))
    assert st["stage"] == "done"
    assert R.score(mode, a0p, mode.gen[0]) is not None                         # ran again, not dropped


def test_save_state_retries_when_the_file_is_briefly_locked(env, monkeypatch):
    """V3-055: on Windows, replacing state.json fails with 'access denied' while another process (the dashboard,
    OneDrive) has it open for a moment; the save waits and tries again instead of stopping the loop."""
    import pathlib
    mode = R.Mode("mini")
    real = pathlib.Path.replace
    calls = {"n": 0}

    def flaky(self, target):
        if pathlib.Path(target).name == "state.json" and calls["n"] < 3:
            calls["n"] += 1
            raise PermissionError(5, "Access is denied")
        return real(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", flaky)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    R.save_state(mode, {"stage": "plus", "events": []})
    assert calls["n"] == 3 and R.load_state(mode)["stage"] == "plus"


def test_save_state_still_fails_on_a_lasting_lock(env, monkeypatch):
    import pathlib
    mode = R.Mode("mini")

    def locked(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(pathlib.Path, "replace", locked)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        R.save_state(mode, {"stage": "plus", "events": []})


def test_plus_rules(env):
    winner = R.founder_genomes(["FACTS"])["FACTS"]
    bad = json.loads(PLUS_REPLY)
    bad["C"]["answering_text"] = "Look in VALUES first, then answer."
    bad["E"] = {"max_turns": 24}
    assert any("VALUES" in p for p in P5.fit_check(winner, bad))
    long_b = json.loads(PLUS_REPLY)
    long_b["B"]["guidance_text"] = "x " * 1500
    long_b["E"] = {"max_turns": 24}
    assert any("length rule" in p for p in P5.fit_check(winner, long_b))
    good = json.loads(PLUS_REPLY)
    good["E"] = {"max_turns": 24}
    assert P5.fit_check(winner, good) == []


def test_plus_checks_from_the_code_review(env):
    winner = R.founder_genomes(["FACTS"])["FACTS"]
    base = json.loads(PLUS_REPLY)
    base["E"] = {"max_turns": 24}
    base["winner"] = winner["id"]
    b_names = json.loads(json.dumps(base))
    b_names["B"]["guidance_text"] = "Write each fact under the values section, one per bullet, with exact numbers and units."
    assert any("B names the layout header 'VALUES'" in p for p in P5.fit_check(winner, b_names))
    same_e = json.loads(json.dumps(base))
    same_e["E"] = {"max_turns": 12}
    assert any("E:" in p for p in P5.fit_check(winner, same_e))
    no_ex = json.loads(json.dumps(base))
    no_ex["D"] = {"examples": []}
    assert any("write one example" in p for p in P5.fit_check(winner, no_ex))
    # the example shown matches the layout of each design
    d_aplus_dplus = P5.build(winner, base, 1 + 8)          # A+ D+
    d_aminus_dplus = P5.build(winner, base, 8)             # A- D+
    assert "## PARTS" in d_aplus_dplus["genes"]["examples"]["items"][0]
    assert "## THINGS" in d_aminus_dplus["genes"]["examples"]["items"][0]
    assert P5.e_plus({"genes": {"numeric": {"max_turns": 24}}}) == 12 and P5.e_plus(winner) == 24


def test_stage_never_hands_over_with_missing_scores(env):
    mode = R.Mode("mini")
    st = R.load_state(mode)
    designs = R.founder_genomes(["A0P"])

    def silent(g, item, phase):                           # returns success but writes no record
        return {"grade": {"status": "ok", "score": 0.5}}
    with pytest.raises(R.Halted):
        R.run_jobs(mode, st, designs, mode.gen[:2], "test", silent)
    assert "no score" in st["halt_reason"]


def test_json_reader_ignores_stray_braces():
    from v3.jsonx import first_object, last_object
    t = 'Using {x} here. {"verdict": "keep", "reason": "ok"} and then {"other": 1}'
    assert first_object(t) == {"verdict": "keep", "reason": "ok"}
    assert last_object('log {step} 1\n{"H001-1": {"v": 2}}') == {"H001-1": {"v": 2}}


def test_plus_reject_only_for_a_written_reason(env):
    mode = R.Mode("real")
    st = {"stage": "plus", "winner": "FACTS", "events": []}
    calls = []

    def call(s, u):
        calls.append(1)
        return PLUS_REPLY
    assert R.plus(mode, st, call) is False and len(calls) == 1          # written, frozen, waiting for the human
    (mode.dir / "PLUS_REJECT").write_text("I do not like it")
    assert R.plus(mode, st, call) is False and "not accepted" in st["waiting_for"] and len(calls) == 1
    (mode.dir / "PLUS_REJECT").write_text("wrong factor\nA changes the answering")
    assert R.plus(mode, st, call) is False and len(calls) == 2           # sent back once, rewritten
    assert (mode.dir / "plus_rejected_1.json").exists()


def test_plus_failing_three_times_halts_cleanly(env):
    mode = R.Mode("real")
    st = {"stage": "plus", "winner": "FACTS", "events": []}
    seen = []

    def call(s, u):
        seen.append(u)
        return "not json at all"
    with pytest.raises(R.Halted):
        R.plus(mode, st, call)
    assert len(seen) == 3 and "failed these code checks" in seen[1]      # each retry is told what failed
    assert "waiting_for" in st and not (mode.dir / "plus.json").exists()
    lo, hi = P5.b_length_bounds(R.founder_genomes(["FACTS"])["FACTS"])
    assert f"{lo} to {hi} characters" in seen[0]                           # the length rule is in the brief


def _evidence_records(mode, winner):
    """Fake stage 1 records for the winner: generated items with checker problems that quote a name, and hand-written
    items with a very different score, plus fields the evidence must never read."""
    for n, iid in enumerate(list(mode.gen) + list(mode.hand)):
        hand = iid in mode.hand
        holes = [{"line": 3, "problem": "unknown section header 'SECRETHEADER'"},
                 {"line": 4, "problem": "more than one fact in one element"}] if n % 2 else []
        rec = {"genome": winner["id"], "item": iid, "grade": {"status": "ok", "score": 0.0 if hand else 0.5},
               "validity": {"usable": not holes, "executable": not holes, "holes": holes},
               "build": {"tool_calls": 1 + n % 3, "output_tokens": 1000 + n},
               "output_tokens": 1500 + n, "answer_output_tokens": 500,
               "tool_trace": [{"result": "SECRETNOTES"}], "final_raw": "SECRETANSWER", "family": "SECRETFAMILY"}
        d = R.RUNS / mode.phase / "codex" / winner["id"] / iid
        d.mkdir(parents=True, exist_ok=True)
        (d / "record.json").write_text(json.dumps(rec))


def test_plus_evidence_firewall(env):
    """V3-056: the stage 1 evidence for Opus uses the generated items only and the named safe fields only; quoted
    names in checker messages are dropped; no notes text, answer, family or hand-written result gets through."""
    from v3 import plus_evidence5 as E
    mode = R.Mode("mini")
    winner = R.founder_genomes(["FACTS"])["FACTS"]
    _evidence_records(mode, winner)
    read = []

    def record(m, gid, iid, phase=None):
        r = R.record(m, gid, iid, phase)
        assert iid in m.gen, "the evidence read a hand-written record"
        read.append(iid)

        class Guard(dict):
            def get(self, k, default=None):
                assert k in E.SAFE_FIELDS, f"the evidence read the field {k!r}"
                return super().get(k, default)

            def __getitem__(self, k):
                assert k in E.SAFE_FIELDS, f"the evidence read the field {k!r}"
                return super().__getitem__(k)
        return Guard(r) if r else r

    def score(m, gid, iid, phase=None):
        assert iid in m.gen, "the evidence read a hand-written score"
        return R.score(m, gid, iid, phase)
    ev = E.summary(mode, winner, record, score)
    text = E.dumps(ev) + E.render(ev)
    assert sorted(read) == sorted(mode.gen)
    assert ev["score_generated"] == 0.5 and ev["items"] == len(mode.gen)        # the hand-written 0.0 is not in it
    assert "SECRET" not in text and "famil" not in text.lower() and "hand" not in text.lower()
    fails = ev["builds"]["failed_check"]
    assert fails == len(mode.gen) // 2 and ev["builds"]["failed_builds_by_problem"] == {
        "unknown section header": fails, "more than one fact in one element": fails}
    assert set(ev) == {"winner", "items", "score_generated", "score_by_k", "builds", "build_tool_calls_by_k",
                       "output_tokens", "max_turns", "tools"}
    assert sum(int(v.split(" of ")[0]) for v in ev["builds"]["failed_by_k"].values()) == fails


def test_plus_evidence_tool_facts_match_what_the_builder_sees():
    """V3-057: V3-056's brief said the builder sees no checker problems; it does. The tool facts are now read from
    the rendered design, and this test checks them against a real save on the FACTS design."""
    from v3 import plus_evidence5 as E
    from v3.genome.render import render
    winner = R.founder_genomes(["FACTS"])["FACTS"]
    tf = E.tool_facts(winner)
    assert tf == {"save_tool": "write_model", "save_replaces": True, "check_feedback": "first_only"}
    arm = render(winner)
    st = arm.new_state()
    heads = winner["genes"]["core"]["params"]["headers"]
    first = arm.handle(st, "write_model", {"text": "\n".join(f"## {h}\n- P1: pump. It fills T1 and it drains T2."
                                                             for h in heads)})
    assert "CHECK: PROBLEM: line 2" in first and "more problem(s) not shown" in first     # first problem + count
    arm.handle(st, "write_model", {"text": f"## {heads[0]}\n- P1: pump"})
    assert "fills T1" not in arm.handle(st, "show_model", {})                             # a save replaces
    ev = {"items": 0, "score_generated": None, "score_by_k": {}, "max_turns": 12, "tools": tf,
          "builds": {"failed_check": 0, "checked": 0, "failed_share": None, "failed_builds_by_problem": {},
                     "failed_by_k": {}},
          "build_tool_calls_by_k": {}, "output_tokens": {"build": None, "answer": None}}
    text = E.render(ev)
    assert "Each save replaces the whole notes; there is no append." in text
    assert ("After each save, the builder is shown the first checker problem (with its line number) and the count of "
            "how many more problems remain.") in text
    assert "not told" not in text


def test_plus_evidence_categories():
    from v3 import plus_evidence5 as E
    assert E.category("element longer than 300 characters") == "element longer than N characters"
    assert E.category("section 'PUMP 7 SECRET' has no content") == "section has no content"
    assert E.category("missing section 'X'") == "missing section"
    assert E.category("a message nobody expected: 'quoted words'") == E.OTHER


def test_plus_brief_carries_the_evidence(env):
    """V3-056: Opus's brief includes the stage 1 evidence; it is saved beside plus.json and its hash is recorded."""
    mode = R.Mode("real")
    winner = R.founder_genomes(["FACTS"])["FACTS"]
    st = {"stage": "plus", "winner": "FACTS", "events": []}
    seen = []

    def call(s, u):
        seen.append(u)
        return PLUS_REPLY
    assert R.plus(mode, st, call) is False
    assert "WHAT STAGE 1 SHOWED ABOUT THIS DESIGN" in seen[0] and "k = 8" in seen[0]
    saved = json.loads((mode.dir / "plus_evidence.json").read_text())
    assert saved["text"] in seen[0]
    import hashlib
    p = json.loads((mode.dir / "plus.json").read_text())
    assert p["evidence_hash"] == hashlib.sha256(saved["text"].encode()).hexdigest()[:16]
    assert p["winner"] == winner["id"]


def test_blind_plus_texts_are_archived_and_rewritten(env):
    """V3-056: + texts frozen before the evidence brief are archived (not counted as a rejection) and written again;
    approved texts are never touched."""
    mode = R.Mode("real")
    winner = R.founder_genomes(["FACTS"])["FACTS"]
    st = {"stage": "plus", "winner": "FACTS", "events": []}
    blind = P5.freeze(dict(json.loads(PLUS_REPLY), E={"max_turns": 24}, winner=winner["id"]))
    mode.dir.mkdir(parents=True, exist_ok=True)
    (mode.dir / "plus.json").write_text(json.dumps(blind))
    calls = []

    def call(s, u):
        calls.append(u)
        return PLUS_REPLY
    assert R.plus(mode, st, call) is False and len(calls) == 1
    assert json.loads((mode.dir / "plus_blind_1.json").read_text())["hash"] == blind["hash"]
    assert "evidence_hash" in json.loads((mode.dir / "plus.json").read_text())
    assert not list(mode.dir.glob("plus_rejected_*.json"))
    assert R.plus(mode, st, call) is False and len(calls) == 1                 # the new texts are kept
    (mode.dir / "plus.json").write_text(json.dumps(blind))                     # approved texts stay as they are
    (mode.dir / "PLUS_APPROVED").write_text("ok")
    assert R.plus(mode, st, call) is True and len(calls) == 1
    assert not (mode.dir / "plus_blind_2.json").exists()


def test_plus_texts_from_an_earlier_evidence_brief_are_archived(env):
    """V3-057: unapproved texts written from an evidence brief other than the current one are archived as superseded
    (not a rejection) and written again; texts from the current brief are kept."""
    mode = R.Mode("real")
    winner = R.founder_genomes(["FACTS"])["FACTS"]
    st = {"stage": "plus", "winner": "FACTS", "events": []}
    old = P5.freeze(dict(json.loads(PLUS_REPLY), E={"max_turns": 24}, winner=winner["id"]))
    old["evidence_hash"] = "0123456789abcdef"
    mode.dir.mkdir(parents=True, exist_ok=True)
    (mode.dir / "plus.json").write_text(json.dumps(old))
    calls = []

    def call(s, u):
        calls.append(u)
        return PLUS_REPLY
    assert R.plus(mode, st, call) is False and len(calls) == 1
    assert json.loads((mode.dir / "plus_superseded_1.json").read_text())["evidence_hash"] == "0123456789abcdef"
    assert "Each save replaces the whole notes" in calls[0]
    assert not list(mode.dir.glob("plus_rejected_*.json")) and not list(mode.dir.glob("plus_blind_*.json"))
    assert R.plus(mode, st, call) is False and len(calls) == 1                 # the current brief's texts are kept


def test_factorial_table_is_balanced():
    lv = [P5.levels(n) for n in range(32)]
    for f in P5.FACTORS:
        assert sum(x[f] == "+" for x in lv) == 16
    for a in P5.FACTORS:
        for b in P5.FACTORS:
            if a < b:
                assert sum(x[a] == x[b] for x in lv) == 16                     # orthogonal columns


def test_real_run_refuses_without_a_verified_lock(monkeypatch):
    from v3 import prereg
    monkeypatch.setattr(prereg, "verify", lambda: ["no lock"])
    assert R.main(["loop"]) == 2


def test_failure_classes():
    assert R.classify_failure({"error": "INTEGRITY: shell call"}) == "design"
    assert R.classify_failure({"error": "build: rc=-9: timeout"}) == "design"
    assert R.classify_failure({"error": "rc=1: 502 bad gateway"}) == "infra"
    assert R.classify_failure({"error": "design tools outside the approved Codex allowlist"}) == "config"


def test_second_code_review_fixes():
    """Argon's second review (2026-10-29): each fix proven."""
    # F1: a network timeout is infrastructure; only the episode's own limit is the design's
    assert R.classify_failure({"error": "ReadTimeout: HTTPSConnectionPool timed out"}) == "infra"
    assert R.classify_failure({"error": "rc=1: 504 Gateway Timeout"}) == "infra"
    assert R.classify_failure({"error": "build: rc=-9: timeout"}) == "design"
    # F5: template-first designs keep their read tools when answering
    from v3.genome.render import render
    spec = R.founder_genomes(["SPEC"])["SPEC"]
    arm = render(spec)
    st = arm.load({"text": "## CONTEXT\n- c"}, given_gold=True)
    assert "show_model" in [t["function"]["name"] for t in arm.gold_tools(st)]
    # F6: a fenced ANSWERS block is read
    assert I.parse_answers('ANSWERS:\n```json\n{"q1": 2}\n```') == {"q1": 2}
    # F12: two comparisons on one line are not a placeholder
    from v3.cores.structured import PLACEHOLDER_RE
    assert not PLACEHOLDER_RE.search("if T1<T2 and P1>P2 then stop")
    assert PLACEHOLDER_RE.search("- <one fact>")
    # F3: B may exceed 1,200 characters when its length rule allows it
    from v3.genome import schema as SC
    g = SC.upgrade(R.founder_genomes(["FACTS"])["FACTS"]["genes"])
    g["guidance"]["text"] = "Write every fact as one short bullet with its exact value and unit. " * 7     # ~480 chars
    winner = SC.make(g, [], "seed", 0, 0, "LONGB", "test winner with a long instruction")
    lo, hi = P5.b_length_bounds(winner)
    assert hi > 1200
    plus = json.loads(PLUS_REPLY)
    plus["B"]["guidance_text"] = ("Write each requirement as one sentence with its trigger, response and exact values. "
                                  * 20)[: hi - 10]
    plus["E"] = {"max_turns": 24}
    assert not any("longer than 1,200" in x for x in P5.fit_check(winner, plus))


def test_leak_check_ties_and_corrected_keys():
    from v3.hand5 import factory as F
    item = {"answer_key": {"key_points": [{"id": "k1", "text": "a", "weight": 1.0},
                                          {"id": "k2", "text": "b", "weight": 1.0},
                                          {"id": "k3", "text": "c", "weight": 0.5}]}}
    def rec(verdicts, score, adj=None, fixed=None):
        r = {"leak": {"by": F.MODEL, "grade": {"status": "ok", "score": score, "grade": {
            "key_points": [{"id": k, "verdict": v} for k, v in verdicts.items()]}}},
             "adjudication": adj or {"verdict": "keep"}}
        if fixed is not None:
            r["leak_fixed"] = fixed
        return r
    # F7: a hit on any key point tied for the top weight is a leak
    assert not F.leak_ok(rec({"k1": "miss", "k2": "hit", "k3": "miss"}, 0.2), item)
    assert F.leak_ok(rec({"k1": "partial", "k2": "miss", "k3": "hit"}, 0.2), item)
    # F4: with a corrected key the leak answer is judged against that key, and must have been re-graded
    adj = {"verdict": "fix_key", "answer_key": {"summary": "s", "key_points": [{"id": "k1", "text": "a", "weight": 1}]}}
    assert not F.leak_ok(rec({"k1": "miss"}, 0.0, adj), item)                       # not re-graded yet
    fixed = {"status": "ok", "score": 1.0, "grade": {"key_points": [{"id": "k1", "verdict": "hit"}]}}
    assert not F.leak_ok(rec({"k1": "miss"}, 0.0, adj, fixed), item)                # leaks under the corrected key
    fixed_ok = {"status": "ok", "score": 0.0, "grade": {"key_points": [{"id": "k1", "verdict": "miss"}]}}
    assert F.leak_ok(rec({"k1": "miss"}, 0.0, adj, fixed_ok), item)


def test_third_code_review_fixes(tmp_path, monkeypatch):
    """Argon's third review (2026-10-29): each fix proven."""
    import numpy as np
    from v3 import analysis5 as A
    from v3 import sandbox
    from v3.builders import mcp_server
    from v3.cores.structured import PLACEHOLDER_RE, md_check
    from v3.formclass import classify
    # F2: a Docker outage inside an episode reaches the record as an infrastructure error
    monkeypatch.setattr(sandbox, "UNAVAILABLE", ["Docker image rpp-sandbox:run5 is not available"])

    class Arm:
        def new_state(self):
            return type("S", (), {})()
        def export(self, st):
            return None
        def validity(self, st):
            return {}
        def tools(self):
            return []
        def current_tools(self, st):
            return []
    out = tmp_path / "state.json"
    mcp_server.Session(Arm(), str(out)).flush()
    assert json.loads(out.read_text())["infra_error"].startswith("SANDBOX UNAVAILABLE")
    assert R.classify_failure({"error": "SANDBOX UNAVAILABLE: Docker image is not available"}) == "infra"
    # F4: founders are ranked on exact means, so a difference of 1/11520 is not a tie
    items = [f"K1x{100000 + 500 * b}" for b in range(4)]
    sc = {"X": {i: 0.5 for i in items}, "Y": {i: 0.5 for i in items}}
    sc["Y"][items[0]] = 0.5 + 1 / 2880                    # Y is higher by 1/11520 on the mean
    hand = {"X": 0.9, "Y": 0.1}
    hitems = ["H001-1"]
    for f in sc:
        sc[f]["H001-1"] = hand[f]
    res = A.stage1(sc, ["X", "Y"], items, hitems, control=None)
    assert res["winner"] == "Y"                            # not decided by the hand-written mean
    # F5: notes after FINAL ANSWER are what is classified
    assert classify('Thinking...\nFINAL ANSWER\n{"things": [], "rules": []}')["form"] == "json"
    # F6: comparisons separated by a comma are not placeholders; an unfilled table header row is caught
    assert not PLACEHOLDER_RE.search("if level<max_level, flow>min_flow")
    md = "## THINGS\n| <column> | b |\n|---|---|\n| P1 | pump |\n"
    assert any("placeholder" in pr["msg"] for pr in md_check(md, ["THINGS"], ["THINGS"]))
    # F9: a zero effect with zero spread is not called real
    Y = np.full((32, 8), 0.5)
    t = A.cluster_tests(Y, [f"K{k}x{100000 + 500 * b}" for b in range(2) for k in (1, 2, 4, 8)], boot=20)
    assert t["A"]["p"] == 1.0 and not t["A"]["holm"]["reject"]
    # F9: the mini run's stage 2 uses two base systems
    m = R.Mode("mini")
    assert len({i.split("x")[1] for i in m.stage2_items}) == 2


@pytest.mark.parametrize("founder", ["G1C3", "A5", "A2", "A3", "G1C1", "A0P", "OWN", "FACTS", "JSON", "MARKDOWN"])
def test_every_part_reaches_the_builder(founder):
    """Fourth review F1: whichever founder wins, the 32 stage 2 designs render to 32 different guides / tool sets."""
    winner = R.founder_genomes([founder])[founder]
    ak = P5.factor_a_kind(winner)
    g = winner["genes"]
    if ak == "core_params":
        params = dict(g["core"]["params"])
        if "headers" in params:
            params["headers"] = list(params["headers"]) + ["CHECKS"]
            new = "## CHECKS\n- c"
        else:
            params["keys"] = list(params["keys"]) + ["checks"]
            new = '{"checks": []}'
        a = {"core_params": params}
        ex, ex_new = ["an example of good notes"], ["an example of good notes " + new]
    else:
        a = {"layout_text": "Group the contents by subsystem, in the order things happen, with one name per thing."}
        ex, ex_new = ["an example of good notes"], ["an example of good notes, grouped by subsystem"]
    lo, hi = P5.b_length_bounds(winner)
    plus = {"A": a, "B": {"guidance_text": ("Write each fact once with exact values and units; mark unknowns. " * 20)[:max(lo, 300)]},
            "C": {"answering_text": "Find the relevant notes, reason step by step, answer with values and units."},
            "D": {"examples": ex, "examples_new_layout": ex_new}, "E": {"max_turns": P5.e_plus(winner)},
            "winner": winner["id"]}
    assert P5.fit_check(winner, plus) == [], founder


def test_docker_outage_paths_are_infrastructure(monkeypatch):
    """Fourth review F2: Docker down before an own-code episode is an outage (infra), not a config error."""
    from v3 import sandbox
    from v3 import episode as E
    from v3.genome.render import render
    monkeypatch.setattr(sandbox, "available", lambda: False)
    g = R.founder_genomes(["A3"])["A3"]
    rec = E._raw("codex", g, render(g), {"id": "x", "question": "q", "context": "c"}, None)
    assert rec["error"].startswith("SANDBOX UNAVAILABLE") and R.classify_failure(rec) == "infra"


def test_exact_ties_and_float_noise():
    """Fourth review F5: means equal up to float noise are ties (broken by the hand-written mean); a zero effect
    computed as 1e-17 is still zero."""
    import numpy as np
    from v3 import analysis5 as A
    items = [f"K1x{100000 + 500 * b}" for b in range(4)]
    vals = [1 / 3, 1 / 3, 1 / 24, 1.0]
    sc = {"X": dict(zip(items, vals)), "Y": dict(zip(items, vals[::-1])), }
    sc["X"]["H001-1"], sc["Y"]["H001-1"] = 0.2, 0.9
    res = A.stage1(sc, ["X", "Y"], items, ["H001-1"], control=None)
    assert res["winner"] == "Y"                            # an exact tie, broken by the hand-written mean
    Y = np.full((32, 8), 0.3)
    t = A.cluster_tests(Y, [f"K{k}x{100000 + 500 * b}" for b in range(2) for k in (1, 2, 4, 8)], boot=20)
    assert all(not t[n]["holm"]["reject"] for n in A.CONFIRMATORY)


def test_fifth_code_review_fixes(monkeypatch):
    """Argon's fifth review (2026-10-29): each fix proven."""
    import pathlib
    import numpy as np
    from v3 import analysis5 as A
    from v3 import episode as E
    from v3.builders import codex_builder as CB
    from v3.genome.render import render
    # F1: every design keeps its 95% range
    items = [f"K1x{100000 + 500 * b}" for b in range(6)]
    sc = {"X": {i: 0.4 + 0.01 * n for n, i in enumerate(items)}, "Y": {i: 0.5 for i in items}}
    res = A.compare(sc, ["X", "Y"], items, boot=200)
    assert all({"mean", "mean_exact", "range", "behind_best", "mcb_margin", "could_be_best"} <= set(d)
               for d in res["designs"].values())
    # F3: with no effects at all, the final design stays design 1 (float noise cannot pick another)
    Y = np.full((32, 8), 0.3)
    names = [f"D{n + 1:02d}" for n in range(32)]
    its = [f"K{k}x{100000 + 500 * b}" for b in range(2) for k in (1, 2, 4, 8)]
    s2 = A.stage2({nm: dict(zip(its, Y[j])) for j, nm in enumerate(names)}, names, its, boot=50)
    assert s2["final_design"] == "D01" and s2["observed_best"] == "D01"
    # F4: no winner example means no rewritten winner example
    winner = R.founder_genomes(["FACTS"])["FACTS"]
    plus = json.loads(PLUS_REPLY)
    plus["E"] = {"max_turns": 24}
    plus["D"]["winner_example_new_layout"] = ["## PARTS\n- x"]
    assert any("give no winner_example_new_layout" in x for x in P5.fit_check(winner, plus))
    # F5: a reconnect message before a completed turn is not an error
    lines = [json.dumps({"type": "error", "message": "Reconnecting... 1/5"}),
             json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "FINAL ANSWER: DONE"}}),
             json.dumps({"type": "turn.completed", "usage": {"output_tokens": 5}})]
    assert CB.parse_stream(lines)["error"] is None
    # F2: a prose design with the doubled budget gets one revision turn, announced in its guide
    a0p = R.founder_genomes(["A0P"])["A0P"]
    g = json.loads(json.dumps(a0p))
    from v3.genome import schema as SC
    genes = SC.upgrade(g["genes"])
    genes["numeric"]["max_turns"] = 24
    a0p_e2 = SC.make(genes, [a0p["id"]], "factorial", 0, 0, "A0P E+", "test")
    assert "asked once to check them" in render(a0p_e2).guide() and "asked once" not in render(a0p).guide()
    calls = []

    def fake_raw(builder, genome, arm, item, d, given_rep=None, client=None, cap_check=None):
        calls.append(item["question"][:40])
        return {"final_raw": "FINAL ANSWER\nnotes v%d" % len(calls), "output_tokens": 1}
    monkeypatch.setattr(E, "_raw", fake_raw)
    out = E._two_phase("codex", a0p_e2, render(a0p_e2), {"id": "x", "question": "q", "context": "c"},
                       pathlib.Path("."), None, None)
    assert len(calls) == 3 and calls[1].startswith("Here are the notes you wrote")
    assert out["notes"].endswith("notes v2")              # the answer phase used the revised notes
    calls.clear()
    E._two_phase("codex", a0p, render(a0p), {"id": "x", "question": "q", "context": "c"}, pathlib.Path("."), None,
                 None)
    assert len(calls) == 2                                 # the base budget has no revision turn


def test_sixth_code_review_fixes(tmp_path, monkeypatch):
    """Argon's sixth review (2026-10-29): each fix proven (written before the fixes and seen to fail)."""
    from v3.genome.render import render, REVISION_NOTE
    # F2: an Opus reply with the wrong shape is a failed attempt, never a crash
    mode = R.Mode("real")
    monkeypatch.setattr(R, "STATE_DIR", tmp_path / "state")
    mode = R.Mode("real")
    st = {"stage": "plus", "winner": "FACTS", "events": []}
    bad = json.dumps({"A": {"core_params": {"headers": None}}, "B": "not an object", "C": ["x"], "D": "y"})
    with pytest.raises(R.Halted):                          # three failed attempts, then a clean halt
        R.plus(mode, st, lambda s_, u: bad)
    # F3: a corrected key must have usable key points, and keeps the original wrong answers
    from v3.hand5 import factory as F
    item = {"answer_key": {"summary": "s", "key_points": [{"id": "k1", "text": "a"}], "wrong_answers": ["w"]}}
    assert F.usable_fixed_key({"summary": "s", "key_points": [{"text": "no id"}]}) is None
    k = F.usable_fixed_key({"summary": "s", "key_points": [{"id": "k1", "text": "a"}]}, item)
    assert k["wrong_answers"] == ["w"]
    # F4: the revision note speaks of the first draft only
    assert "first draft" in REVISION_NOTE
    # Q1: "built" means usable and executable, as everywhere else
    from v3 import episode as E
    rec_built = {"validity": {"usable": True, "executable": False}}
    assert not E.buildable(rec_built)
    assert R.counts_as_built(rec_built) is False
    assert R.counts_as_built({"validity": {"usable": True, "executable": True}}) is True


def test_hand5_final_item_labels_come_from_system_record():
    # the writer sometimes put the task type into an item's "kind"; the final item takes it from system.json
    from v3.hand5 import factory as F
    doc = {"files": {"system.json": json.dumps({"title": "T", "kind": "discrete", "domain": "telecom network",
                                                "messiness": "realistic"})}}
    item = {"id": "H1-1", "system_id": "H1", "kind": "diagnosis", "domain": "telecom network",
            "messiness": "realistic", "task_type": "diagnosis", "answer_key": {"summary": "s", "key_points": []}}
    it = F.final_item(doc, item, "desc", {"adjudication": {"verdict": "keep"}})
    assert (it["kind"], it["task_type"], it["system_title"], it["context"]) == ("discrete", "diagnosis", "T", "desc")
    assert item["kind"] == "diagnosis"                     # the written item itself is left as it was


def test_prereg_lock_leaves_out_every_dashboard():
    # V3-054: the lock skipped only a folder named "dashboard", so v3/dashboard5 was locked by accident
    from v3 import prereg
    fs = [p.as_posix() for p in prereg.files()]
    assert not [f for f in fs if "/v3/dashboard" in f]
    assert any(f.endswith("/v3/run5.py") for f in fs) and any(f.endswith("/v3/analysis5.py") for f in fs)


@pytest.mark.parametrize("founder", ["SEC_A1", "FACTS", "MARKDOWN"])
def test_plus_brief_states_the_name_rules(founder):
    # V3-051: the mini run's Opus wrote header 'OPEN_QUESTIONS' (underscore) and failed the fit check; the brief now
    # states the characters the schema allows for headers and JSON keys
    from v3.genome import schema as S
    w = R.founder_genomes([founder])[founder]
    if P5.factor_a_kind(w) != "core_params":
        pytest.skip("factor A is not core parameters for this founder")
    b = P5.brief(w)
    assert "no underscores" in b and "single spaces and & / -" in b and "lower-case letters, digits and _" in b
    assert not S.HEADER_RE.match("OPEN_QUESTIONS") and S.HEADER_RE.match("OPEN QUESTIONS")
    assert S.JSON_KEY_RE.match("open_questions") and not S.JSON_KEY_RE.match("Open")


def test_hand5_final_item_undoes_cp1252_mojibake():
    # the writer's tool server read UTF-8 through a cp1252 stdin; the final item restores the text exactly
    from v3.hand5 import factory as F
    clean = "Line 3 – pack ≈ 4 m³ at 90 °C; ‘quoted’ ✓"
    mangled = clean.encode("utf-8").decode("cp1252", "surrogateescape")
    assert mangled != clean and "â€" in mangled
    doc = {"files": {"system.json": json.dumps({"title": mangled, "kind": "discrete"})}}
    item = {"id": "H1-1", "system_id": "H1", "question": mangled, "task_type": "diagnosis",
            "answer_key": {"summary": mangled, "key_points": [{"id": "k1", "text": mangled}]}}
    it = F.final_item(doc, item, mangled, {"adjudication": {"verdict": "keep"}})
    assert it["context"] == it["question"] == it["system_title"] == clean
    assert it["answer_key"]["key_points"][0]["text"] == clean
    assert F.unmangle(clean) == clean and F.unmangle("plain") == "plain"   # clean text is left alone
