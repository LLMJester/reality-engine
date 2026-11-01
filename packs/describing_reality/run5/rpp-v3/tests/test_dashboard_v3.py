"""Live dashboard (spec section 12): server routes, feed contract, live updates, and mid-write resilience."""
import json
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DASH = ROOT / "v3" / "dashboard"
CONTRACT = {"time", "generation", "streak", "paused", "spend", "genomes", "controls", "front", "arena", "lineage",
            "habitat", "library", "operators", "generations", "workshop", "events", "limits", "progress", "single", "pilot", "calibration", "page_version", "codex",
            "alerts", "testing", "pipeline", "design", "claude_usage", "openrouter", "family_scores", "progress_small", "luna_picks",
            "champion", "seeds_info"}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port, path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return r.status, dict(r.headers), r.read()


def _status(port):
    code, headers, body = _get(port, "/status.json")
    assert code == 200
    return json.loads(body), headers


@pytest.fixture
def serve():
    procs = []

    def start(*args):
        port = _free_port()
        p = subprocess.Popen([sys.executable, "-m", "v3.dashboard.server", "--port", str(port), *args], cwd=ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        procs.append(p)
        for _ in range(100):
            try:
                _get(port, "/", timeout=1)
                return port
            except OSError:
                if p.poll() is not None:
                    raise RuntimeError(p.stderr.read().decode())
                time.sleep(0.1)
        raise RuntimeError("server did not start")

    yield start
    for p in procs:
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()


def _state(gen=1, streak=0, paused=False):
    return {"version": 1, "done": {}, "spend": {}, "generation": gen, "streak": streak, "paused": paused,
            "events": [{"t": "2026-10-21T10:00:00", "level": "info", "msg": f"generation {gen} complete"}]}


def test_sample_routes_and_contract(serve):
    port = serve("--sample")
    code, headers, body = _get(port, "/")
    assert code == 200 and headers["Cache-Control"] == "no-store"
    html = body.decode()
    assert "creatures3.js" in html and "status.json" in html
    code, headers, js = _get(port, "/creatures3.js")
    assert code == 200 and b"Creatures3" in js and "javascript" in headers["Content-Type"]
    data, headers = _status(port)
    assert headers["Cache-Control"] == "no-store"
    assert set(data) == CONTRACT
    assert data == json.loads((DASH / "sample_status.json").read_text(encoding="utf-8"))
    assert {"shares", "wins", "trials", "history"} <= set(data["operators"])
    assert {"cores", "protocols", "cells"} <= set(data["habitat"])
    row = data["arena"][0]
    assert {"id", "name", "e2e", "e2e_ci", "build", "tokens", "small_e2e", "is_control", "on_front"} <= set(row)
    g = next(iter(data["genomes"].values()))
    assert {"genes", "cond", "objectives", "stage", "niche", "on_front", "improved"} <= set(g)


def test_bad_requests_do_not_crash(serve):
    port = serve("--sample")
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(port, "/nope/../../etc")
    assert e.value.code == 404
    req = urllib.request.Request(f"http://127.0.0.1:{port}/status.json", data=b"x", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 405
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:   # garbage request line
        s.sendall(b"\x00\x01garbage\r\n\r\n")
        s.recv(1024)
    assert _status(port)[0]["generation"] == 3


def test_index_polls_and_uses_renderer():
    html = (DASH / "index.html").read_text(encoding="utf-8")
    assert '<script src="creatures3.js"></script>' in html
    assert "status.json" in html and "POLL_MS = 3000" in html and "STALE_MS = 15000" in html
    assert "requestAnimationFrame" in html and "imageSmoothingEnabled = false" in html
    for fn in ("C.draw(", "C.combine(", "C.combineStep(", "C.partOrigin(", "C.tintFor("):
        assert fn in html
    assert "http://" not in html.replace("http://127.0.0.1", "") and "https://" not in html   # no external resources


def test_live_update_and_spend(serve, tmp_path):
    state, runs = tmp_path / "state.json", tmp_path / "runs"
    for b, n in (("claude", 2), ("codex", 1)):
        for i in range(n):
            (runs / "gen0_full" / b / "g1" / f"it{i}").mkdir(parents=True)
            (runs / "gen0_full" / b / "g1" / f"it{i}" / "record.json").write_text("{}")
    (runs / "pilot_gen0_full" / "claude" / "g1" / "it0").mkdir(parents=True)               # pilot: not counted
    (runs / "pilot_gen0_full" / "claude" / "g1" / "it0" / "record.json").write_text("{}")
    (runs / "ledger.jsonl").write_text('{"cost": 0.25}\n{"cost": 0.5}\n{"cost": nul')   # partial last line
    state.write_text(json.dumps(_state(gen=1)), encoding="utf-8")
    port = serve("--state", str(state), "--runs", str(runs))
    data, _ = _status(port)
    assert set(data) == CONTRACT
    assert data["generation"] == 1 and data["paused"] is False
    assert data["spend"] == {"openrouter_usd": 0.75, "claude_runs": 2, "codex_runs": 1}
    state.write_text(json.dumps(_state(gen=2, streak=1, paused=True)), encoding="utf-8")
    data, headers = _status(port)
    assert data["generation"] == 2 and data["streak"] == 1 and data["paused"] is True
    assert headers["X-Status-Stale"] == "0"
    assert data["events"][-1]["msg"] == "generation 2 complete"


def test_missing_state_is_generation_zero(serve, tmp_path):
    port = serve("--state", str(tmp_path / "absent.json"), "--runs", str(tmp_path))
    data, _ = _status(port)
    assert set(data) == CONTRACT
    assert data["generation"] == 0 and data["genomes"] == {} and data["workshop"] == [] and data["library"] == {}


def test_truncated_state_returns_last_good(serve, tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps(_state(gen=5)), encoding="utf-8")
    port = serve("--state", str(state), "--runs", str(tmp_path))
    assert _status(port)[0]["generation"] == 5
    full = json.dumps(_state(gen=6))
    state.write_text(full[: len(full) // 2], encoding="utf-8")   # caught mid-write
    data, headers = _status(port)
    assert data["generation"] == 5 and headers["X-Status-Stale"] == "1"
    state.write_text(full, encoding="utf-8")   # writer finishes: next poll is live again
    data, headers = _status(port)
    assert data["generation"] == 6 and headers["X-Status-Stale"] == "0"


def test_slow_builds_answer_from_last_good_and_refresh_in_background(tmp_path):
    """Once builds are slow, requests return the last good payload at once while one background thread rebuilds."""
    from v3.dashboard.server import Feed
    f = Feed(tmp_path / "state.json", tmp_path)
    f._feed = object()            # mark warm, so builds are timed
    gen = [0]

    def build():
        gen[0] += 1
        time.sleep(0.3)
        return {"generation": gen[0]}
    f._build = build
    f.SLOW_SECS = 0.1
    assert json.loads(f.status()[0])["generation"] == 1       # first build: synchronous
    t0 = time.time()
    assert json.loads(f.status()[0])["generation"] == 1       # slow now: last good, at once
    assert json.loads(f.status()[0])["generation"] == 1       # one rebuild at a time
    assert time.time() - t0 < 0.2
    time.sleep(0.6)
    assert gen[0] == 2 and json.loads(f.status()[0])["generation"] == 2


def test_feed_builds_with_genomes_pipeline_and_design():
    """Regression: a helper name clash broke build() only when genomes exist (the sample state has none)."""
    from v3.dashboard import feed
    from v3.genome.seeds import load_all
    seeds = load_all()
    g = seeds["ONT_LOOSE"]
    full = {"claude": {"e2e": 0.8, "build": 1.0, "neg_tokens": -100, "scores": {"K1x1": 0.8}, "systems": {"K1x1": "s"}}}
    data = {"genomes": {g["id"]: g}, "evals": {g["id"]: {"full": full}}, "item_sets": {"full": ["K1x1"]},
            "lineage": [], "generations": [], "controls": []}
    out = feed.build(data)
    assert out["pipeline"]["stages"][0]["key"] == "baseline" and out["pipeline"]["generation"] == 0
    assert out["design"]["example"] == g["id"] and len(out["design"]["genes"]) == 9
    data["controls"] = [g["id"]]
    assert out["genomes"][g["id"]]["stage"]
    # run 3 (directed breeding) has no micro round: a round starts with breeding; runs 1-2 started with micro_a
    micro = feed.CFG["evolution"]["micro"].get("enabled", True) and not feed.run3()
    assert feed.build(data)["pipeline"]["stages"][0]["key"] == ("micro_a" if micro else "breed")


def test_stage_label_follows_quick_build_rule():
    from v3.dashboard.feed import _stage
    built = {"codex": {"build_items": {"a": 1, "b": 1, "c": 1, "d": 0, "e": 0}, "scores": {}}}
    few = {"codex": {"build_items": {"a": 1, "b": 1, "c": 0, "d": 0, "e": 0}, "scores": {}}}
    g = {"genes": {"core": {"kind": "sectioned"}}}
    assert _stage(g, {"quick": built}) == "quick_passed"      # passed, waiting for screening: not "failed"
    assert _stage(g, {"quick": few}) == "quick_failed"
    prose = {"genes": {"core": {"kind": "prose"}}}
    assert _stage(prose, {"quick": {"codex": {"build_items": {}, "scores": {"a": .5, "b": .2, "c": .1, "d": 0}}}}) == "quick_passed"


def test_acknowledged_alerts_are_flagged_not_dropped():
    from v3.dashboard import feed
    ack = json.loads((DASH / "acknowledged.json").read_text(encoding="utf-8"))
    ev = [{"t": ack[0]["t"], "level": "warn", "msg": "old"}, {"t": "2099-01-01T00:00:00", "level": "error", "msg": "new"},
          {"t": "2099-01-01T00:00:01", "level": "info", "msg": "fine"}]
    out = feed._alerts({"events": ev})
    assert [a["msg"] for a in out] == ["old", "new"]
    assert out[0]["resolved"] is True and out[0]["note"] and "resolved" not in out[1]


def test_builder_activity_states(tmp_path, monkeypatch):
    import os
    import time as _t
    from v3 import config
    from v3.dashboard import feed
    monkeypatch.setattr(config, "RUNS", tmp_path)
    monkeypatch.setitem(feed.CFG["budget"], "claude_runs_per_hour", 4)
    start = _t.time() - 1800
    (tmp_path / "loop.log").write_text(_t.strftime("[%Y-%m-%d %H:%M:%S.00] loop start \n", _t.localtime(start)))
    for i in range(3):                       # 3 finished episodes = 6 calls, quiet for 10 minutes: at the cap, waiting
        d = tmp_path / "gen0_screen30" / "claude" / "g1" / f"it{i}"
        (d / "build").mkdir(parents=True)
        (d / "record.json").write_text("{}")
        old = _t.time() - 600
        for x in (d / "build", d / "record.json", d, d.parent, d.parent.parent, d.parent.parent.parent):
            os.utime(x, (old, old))
    stages = [{"who": "Claude", "key": "screen2"}]
    later = _t.time() + 1200                 # folder creation times cannot be back-dated on Windows: look from later
    act = feed.builder_activity(stages, 0, True, False, now=later)
    assert act["claude"]["state"] == "waiting" and act["claude"]["next_slot"] and act["claude"]["calls_hour"] == 6
    assert act["codex"]["state"] == "idle" and not act["codex"]["used_now"]
    assert feed.builder_activity(stages, 0, True, True, now=later)["claude"]["state"] == "paused"
    assert feed.builder_activity(stages, 0, False, False, now=later)["claude"]["state"] == "stopped"


# ---------------------------------------------------------------- run 3 (V3-034): directed breeding
def _directed_state(tmp_path, monkeypatch, rounds=2):
    """A run 3 state from the stub runner/designer of tests/test_directed.py (no spending)."""
    import importlib.util
    from v3 import directed as DR
    from v3 import evolve as E
    from v3.state import Store
    spec = importlib.util.spec_from_file_location("_t_directed", ROOT / "tests" / "test_directed.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    monkeypatch.setitem(E.EV, "mode", "directed")
    monkeypatch.setitem(E.EV, "primary_builder", "codex")
    monkeypatch.setitem(E.EV, "small_builder", "codex")
    monkeypatch.setitem(E.EV, "children_per_generation", 20)
    monkeypatch.setitem(E.EV, "full_top_k", 3)
    monkeypatch.setitem(E.EV, "retry_wait_s", 0)
    monkeypatch.setitem(E.EV, "directed", dict(T.RUN3_DIRECTED, confirm_bases=4))   # run 3 settings (V3-037)
    monkeypatch.setitem(E.EV, "stop_no_improve_generations", 4)
    st = Store(tmp_path / "state.json")
    ev = DR.Directed(st, T.stub_runner, T.SETS, designer=T.stub_designer, log=lambda *a: None)
    ev.baseline(T.seeds3())
    st.data["item_sets"] = T.SETS
    for g in range(1, rounds + 1):
        ev.generation(g)
    st.save()
    return json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))


@pytest.fixture
def quiet_dirs(tmp_path, monkeypatch):
    """An empty runs/ and state/ so the feed reads nothing of the live experiment."""
    from v3 import config
    from v3 import items as I
    runs, state = tmp_path / "runs", tmp_path / "statedir"
    runs.mkdir()
    state.mkdir()
    monkeypatch.setattr(config, "RUNS", runs)
    monkeypatch.setattr(config, "STATE_DIR", state)
    monkeypatch.setattr(I, "FINAL_LOCK", state / "FINAL_LOCK.json")
    return runs, state


def test_seed_names_are_creature_names_with_the_code_beside_them():
    from v3.dashboard import feed
    from v3.genome.seeds import load_all
    seeds = {g["id"]: g for g in load_all().values()}
    names, codes = feed.pet_names(seeds, seeds_named=True), feed.code_names(seeds, seeds_named=True)
    assert not any(n.startswith("Seed ") for n in names.values()) and all(" " in n for n in names.values())
    assert len(set(names.values())) == len(names)                                          # unique
    assert names == feed.pet_names(dict(reversed(list(seeds.items()))), seeds_named=True)   # stable
    sec = next(gid for gid, g in seeds.items() if g["name"] == "SEC_A1")
    assert codes[sec] == "SEC_A1" and feed.labels(seeds, True)[sec] == f"{names[sec]} (SEC_A1)"
    assert feed.pet_names(seeds, seeds_named=False)[sec] == "SEC_A1"       # runs 1-2 keep the code name
    html = (DASH / "index.html").read_text(encoding="utf-8")
    assert "code(id).toLowerCase() === q" in html                          # the tree search finds the code name


def test_run3_feed_champion_pipeline_operators_and_seeds(tmp_path, monkeypatch, quiet_dirs):
    from v3.dashboard import feed
    data = _directed_state(tmp_path, monkeypatch)
    out = feed.build(data)
    assert set(out) == CONTRACT
    c = out["champion"]
    assert c["id"] == data["champion"] and c["chain"][0]["round"] == 0 and c["chain"][0]["code"]
    assert not c["chain"][0]["name"].startswith("Seed ") and c["stop_after"] == 4
    if len(data["champions"]) > 1:
        assert c["confirmation"]["passed"] and {v["role"] for v in c["confirmation"]["vs"]} == {"champion then", "control"}
    assert any(p["kind"] == "proof" and p["steps"] for p in c["proofs"])
    st = out["pipeline"]["stages"]
    assert [s["key"] for s in st] == ["breed", "quick", "screen1", "screen2", "full", "proof", "confirm", "crown"]
    for s in st:
        assert {"key", "status", "expected", "n"} <= set(s) and s["status"] in ("done", "active", "waiting", "paused")
    assert out["pipeline"]["run3"] and not out["pipeline"]["finals"]
    ops = out["operators"]
    assert set(ops["kinds"]) == {"fix", "combine", "explore"} and "transplant" in ops["kinds"]["combine"]
    assert set(ops["by_kind"]) == {"fix", "combine", "explore"} and ops["mix"]["fix"] == 0.5
    seed = next(g for g in out["genomes"].values() if g["operator"] == "seed")
    assert seed["code"] and not seed["name"].startswith("Seed ")
    gs = out["generations"][-1]
    assert set(gs["kinds"]) == {"fix", "combine", "explore"} and "reserve" in gs and gs["champion"]
    si = out["seeds_info"]
    assert si["research"] and all(r["url"].startswith("http") for r in si["research"])
    by = {s["code"]: s for s in si["seeds"]}
    assert by["SEC_A1"]["in_run"] and by["SEC_A1"]["name"] == out["genomes"][by["SEC_A1"]["id"]]["name"]
    assert by["ONT_BASIC"]["status"] == "dropped" and not by["ONT_BASIC"]["in_run"]


def test_run3_proof_and_confirmation_reviews_mid_round(tmp_path, monkeypatch, quiet_dirs):
    """A round whose proof and confirmation are recorded but not yet summarised shows them in the step reviews."""
    from v3.dashboard import feed
    data = _directed_state(tmp_path, monkeypatch, rounds=1)
    summ = data["generations"].pop()                     # as if round 1 were still in progress
    data["generation"], data["streak"] = 0, 0
    out = feed.build(data)
    st = {s["key"]: s for s in out["pipeline"]["stages"]}
    assert st["full"]["review"]["rows"] and st["proof"]["review"]["rows"]
    if summ["improved"]:
        assert st["confirm"]["review"]["rows"][0][-1] == "passed"
        assert st["proof"]["status"] == "done" and st["confirm"]["status"] == "done" and st["crown"]["status"] == "active"


def test_run3_finals_stages_and_results(tmp_path, monkeypatch, quiet_dirs):
    from v3.dashboard import feed
    from v3 import items as I
    runs, _ = quiet_dirs
    data = _directed_state(tmp_path, monkeypatch, rounds=1)
    data["streak"], data["finals_done"] = 4, ["A"]
    champ, ctrl = data["champion"], data["controls"][0]
    I.FINAL_LOCK.write_text(json.dumps({"t": "2026-11-01T10:00:00", "run": 3, "primary": champ, "control": ctrl,
                                        "transfer_models": {"or_qwen": "q"}}))
    for ph, b in (("final", "codex"), ("final_interaction", "codex"), ("final_transfer", "or_qwen")):
        f = runs / ph / b / champ / "K1x1"
        f.mkdir(parents=True)
        (f / "record.json").write_text("{}")
    out = feed.build(data)
    st = out["pipeline"]["stages"]
    assert [s["key"] for s in st] == ["final_lock", "final_a", "final_b", "final_c", "final_analysis"]
    assert out["pipeline"]["finals"] and out["pipeline"]["finished"]
    s = {x["key"]: x for x in st}
    assert s["final_lock"]["status"] == "done" and s["final_a"]["status"] == "done"
    assert s["final_b"]["status"] == "active" and s["final_b"]["n"] == 1 and s["final_c"]["n"] == 1
    assert out["champion"]["finals"]["lock"]["champion"] == champ and out["champion"]["finals"]["analysis"] is None
    ci = {"est": 0.05, "lo": 0.01, "hi": 0.09, "n_items": 100, "improved": True}
    (runs / "final" / "analysis3.json").write_text(json.dumps({
        "H1": ci, "H1_by_set": {"heldout": ci}, "H7": ci, "H5": {"not_tested": "x"},
        "H8": {"mean_over_turns": ci, "compounding_t4_minus_t1": ci, "per_turn": {"1": ci}},
        "H9": {"pooled": ci, "per_model": {"or_qwen": dict(ci, model="q")}, "models_ahead": 1, "n_models": 1,
               "passed": True}, "passed": {"H1": True, "H7": True, "H8": True, "H8_compounding": True, "H9": True}}))
    out = feed.build(data)
    an = out["champion"]["finals"]["analysis"]
    assert an["H1"]["est"] == 0.05 and an["H9"]["passed"] and an["H8"]["compounding"]["lo"] == 0.01
    assert all(x["status"] == "done" for x in out["pipeline"]["stages"])
    assert out["pipeline"]["stages"][1]["review"]["rows"]


def test_run3_feed_survives_run2_and_empty_states(quiet_dirs):
    from v3.dashboard import feed
    from v3.genome.seeds import load_all
    for data in ({}, {"version": 1, "done": {}, "events": [], "spend": {}, "generation": 0}):
        out = feed.build(data)
        assert set(out) == CONTRACT and not out["pipeline"].get("error")
    g = load_all()["A0P"]
    full = {"codex": {"e2e": 0.5, "build": 1.0, "neg_tokens": -10, "scores": {"K1x1": 0.5}, "systems": {"K1x1": "s"}}}
    run2 = {"genomes": {g["id"]: g}, "evals": {g["id"]: {"full": full}}, "controls": [g["id"]], "generation": 1,
            "streak": 1, "lineage": [{"gen": 1, "child": "gX", "parents": [g["id"]], "op": "point", "level": "macro"}],
            "generations": [{"gen": 1, "children": 1, "quick_passed": 0, "full": [], "improved": [], "streak": 1,
                             "micro": {"tried": 1, "winners": 0}}], "credit": {"shares": {"point": 1.0}}}
    out = feed.build(run2)
    assert not out["pipeline"].get("error") and not (out["champion"] or {}).get("error")
    assert (out["champion"] or {}).get("id") is None
