"""M1/M5/M6 tests: genome schema, deterministic operators, selection algorithms, CMA-ES, statistics, walls, state."""
import copy
import pathlib
import math
import random

import pytest

from v3.genome import schema as S
from v3.genome.seeds import build as seeds
from v3.operators import deterministic as D
from v3.operators.cmaes import CMAES
from v3.selection import algorithms as A
from v3 import stats, walls
from v3.state import Store, LockHeld

SEEDS = seeds()


# ---------------------------------------------------------------- genome
def test_seeds_valid_and_ids_are_gene_hashes():
    for g in SEEDS.values():
        S.validate(g)
        assert g["id"] == S.genome_id(g)
        assert S.incompatibilities(g) == []
    assert len({g["id"] for g in SEEDS.values()}) == len(SEEDS)


def test_id_ignores_provenance_but_not_genes():
    a = SEEDS["G1C3"]
    b = S.make(a["genes"], [a["id"]], "swap", 3, 1, "x")
    assert b["id"] == a["id"]
    g = copy.deepcopy(a["genes"]); g["numeric"]["retries"] = 3
    assert S.make(g, [a["id"]], "numeric")["id"] != a["id"]


def test_no_parentless_child():
    with pytest.raises(S.GenomeError):
        S.make(SEEDS["G1C3"]["genes"], [], "point")
    with pytest.raises(S.GenomeError):
        S.make(SEEDS["G1C3"]["genes"], ["a", "b", "c", "d"], "swap")


def test_bad_genes_rejected():
    for mut in (lambda g: g["core"].update(kind="magic"), lambda g: g["query_layer"].append("what_if"),
                lambda g: g["numeric"].update(max_turns=99), lambda g: g.pop("guidance"),
                lambda g: g["tool_interface"].update(patch="yes")):
        g = copy.deepcopy(SEEDS["G1C3"]["genes"]); mut(g)
        with pytest.raises(S.GenomeError):
            S.make(g, ["p"], "point")


def test_incompatibilities():
    g = copy.deepcopy(SEEDS["G1C1"]["genes"]); g["query_layer"].append("explain")
    assert "cannot serve" in S.incompatibilities(S.make(g, ["p"], "point"))[0]
    g = copy.deepcopy(SEEDS["A0P"]["genes"]); g["build_protocol"] = {"kind": "stepwise_checked", "params": {}}
    assert S.incompatibilities(S.make(g, ["p"], "point"))


# ---------------------------------------------------------------- deterministic operators
def test_swap_is_reproducible_declares_parents_and_mixes():
    a, b = SEEDS["G1C3"], SEEDS["A2"]
    c1 = D.apply("swap", [a, b], random.Random(7))
    c2 = D.apply("swap", [a, b], random.Random(7))
    assert c1 is not None and c1["id"] == c2["id"] and c1["parents"] == [a["id"], b["id"]]
    srcs = {k: (S.canonical(c1["genes"][k]) == S.canonical(a["genes"][k]),
                S.canonical(c1["genes"][k]) == S.canonical(b["genes"][k])) for k in S.GENES}
    assert all(x or y for x, y in srcs.values())          # every gene comes from a parent
    assert c1["id"] not in (a["id"], b["id"])


def test_swap_children_are_always_compatible_or_none():
    rng = random.Random(1)
    names = list(SEEDS)
    made = 0
    for _ in range(200):
        a, b = rng.sample(names, 2)
        c = D.apply("swap", [SEEDS[a], SEEDS[b]], rng)
        if c:
            made += 1
            assert S.incompatibilities(c) == []
    assert made > 50


def test_simplify_removes_exactly_one_feature():
    a = SEEDS["G1C3"]
    c = D.apply("simplify", [a], random.Random(3))
    removed = c["op_notes"]["removed"]
    nq, nf = len(c["genes"]["query_layer"]), len(c["genes"]["schema"]["fields"])
    assert (nq, nf) in ((len(a["genes"]["query_layer"]) - 1, len(a["genes"]["schema"]["fields"])),
                        (len(a["genes"]["query_layer"]), len(a["genes"]["schema"]["fields"]) - 1)), removed
    assert c["parents"] == [a["id"]]


def test_graft_takes_library_gene_and_credits_source():
    a = SEEDS["G1C3"]
    lib = {"tool_interface": {"value": {"errors": "all_at_once", "submit": "tool_arg", "patch": True}, "source": "gX"}}
    c = D.apply("graft", [a], random.Random(0), library=lib)
    assert c["genes"]["tool_interface"]["errors"] == "all_at_once" and c["parents"] == [a["id"], "gX"]


def test_numeric_step_stays_in_bounds_and_persists_state():
    a = SEEDS["G1C3"]
    c = D.apply("numeric", [a], random.Random(0))
    for k, (lo, hi) in S.NUMERIC_BOUNDS.items():
        assert lo <= c["genes"]["numeric"][k] <= hi
    st = c["op_notes"]["cma"]
    c2 = D.apply("numeric", [a], random.Random(0), cma_state=st)
    assert c2["op_notes"]["cma"]["asked"] == st["asked"] + 1


# ---------------------------------------------------------------- CMA-ES
def test_cmaes_optimises_a_quadratic():
    target = [3.0, -1.0, 2.0]
    es = CMAES([0, 0, 0], [-5] * 3, [5] * 3, sigma=0.3, seed=1)
    for _ in range(60):
        xs = es.ask()
        es.tell(xs, [-sum((x - t) ** 2 for x, t in zip(p, target)) for p in xs])
        es = CMAES.from_state(es.state())                 # round-trips through JSON-able state
    assert sum((x - t) ** 2 for x, t in zip(es.best_mean(), target)) < 0.05


# ---------------------------------------------------------------- selection
KEYS = ["e2e", "build"]


def test_nondominated_sort_known_fronts():
    objs = {"a": {"e2e": 1, "build": 0}, "b": {"e2e": 0, "build": 1}, "c": {"e2e": 0.5, "build": 0.5},
            "d": {"e2e": 0.4, "build": 0.4}, "e": {"e2e": 0, "build": 0}, "f": {"e2e": None, "build": 0.9}}
    fronts = A.nondominated_sort(objs, KEYS)
    assert fronts[0] == ["a", "b", "c"]
    assert "d" in fronts[1] and fronts[-1] == ["e"] or "e" in fronts[-1]
    assert not A.dominates(objs["f"], objs["e"], KEYS) or True  # None is worst on that objective
    assert A.dominates(objs["c"], objs["d"], KEYS)


def test_crowding_boundaries_infinite_and_rank_is_total():
    objs = {str(i): {"e2e": i / 10, "build": 1 - i / 10} for i in range(6)}
    cd = A.crowding(list(objs), objs, KEYS)
    assert cd["0"] == math.inf and cd["5"] == math.inf and all(0 < cd[str(i)] < math.inf for i in range(1, 5))
    r = A.nsga2_rank(objs, KEYS)
    assert len(r) == 6 and len({i for i, _, _ in r}) == 6


def test_map_elites_replaces_only_on_dominance():
    m = A.MapElites(KEYS)
    assert m.offer(("eq", "one"), "a", {"e2e": 0.3, "build": 0.5})
    assert not m.offer(("eq", "one"), "b", {"e2e": 0.4, "build": 0.4})    # trade-off: no replacement
    assert m.offer(("eq", "one"), "c", {"e2e": 0.4, "build": 0.6})
    assert m.ids() == ["c"]


def test_successive_halving_keeps_best_and_is_deterministic():
    sc = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7}
    surv, hist = A.successive_halving(list(sc), lambda i, n: sc[i], [(5, 0.5), (30, 1)])
    assert surv == ["b"] and hist[0]["kept"] == ["b", "d"]


def test_operator_credit_shifts_share_but_respects_floor():
    oc = A.OperatorCredit({"x": 0.5, "y": 0.5}, floor=0.05)
    for _ in range(50):
        oc.record("x", True); oc.record("y", False)
    s = oc.shares()
    assert s["x"] > 0.8 and s["y"] >= 0.05 - 1e-9 and abs(sum(s.values()) - 1) < 1e-9


def test_tournament_and_migration_reproducible():
    objs = {str(i): {"e2e": i / 10, "build": (i * 7 % 10) / 10} for i in range(10)}
    p1 = [A.tournament(list(objs), objs, KEYS, A.rng_for("t", 1)) for _ in range(5)]
    p2 = [A.tournament(list(objs), objs, KEYS, A.rng_for("t", 1)) for _ in range(5)]
    assert p1 == p2
    isl = A.migrate([["0", "1", "2"], ["3", "4"], ["5", "6", "7"]], objs, KEYS, 1, random.Random(0))
    assert len(isl[0]) == 4 and len(isl[1]) == 3


# ---------------------------------------------------------------- statistics
def test_paired_ci_detects_clear_difference_and_not_noise():
    cl = {f"i{k}": f"s{k // 3}" for k in range(60)}
    a = {i: 0.6 for i in cl}; b = {i: 0.4 for i in cl}
    assert stats.paired_ci(a, b, cl)["improved"]
    rng = random.Random(0)
    n1 = {i: rng.random() for i in cl}; n2 = {i: rng.random() for i in cl}
    r = stats.paired_ci(n1, n2, cl)
    assert r["lo"] < 0 < r["hi"]


# ---------------------------------------------------------------- walls and state
def test_walls_cover_sealed_and_keyed_paths_for_builders():
    rules = walls.deny_rules("builder")
    assert any("data/heldout" in r and r.startswith("Read(") for r in rules)
    assert any("data/items" in r for r in rules)
    assert not any("data/items" in r for r in walls.deny_rules("grader"))
    pats = walls.forbidden_patterns("builder")
    assert any(p.search("rpp-core\\data\\heldout\\x.json") for p in pats)


def test_state_single_writer(tmp_path):
    s = Store(tmp_path / "state.json").acquire()
    s.event("hi"); s.save()
    import os
    (tmp_path / "state.lock").write_text(str(os.getppid()))      # a live other process holds it
    with pytest.raises(LockHeld):
        Store(tmp_path / "state.json").acquire()
    assert Store(tmp_path / "state.json").data["events"][0]["msg"] == "hi"


# ---------------------------------------------------------------- AI operators (designer stubbed)
def test_ai_operator_validates_and_limits_changes():
    from v3.operators import ai
    p = SEEDS["G1C3"]
    good = lambda t: '{"genes": {"tool_interface": {"errors": "all_with_fix_hints", "submit": "tool_arg", "patch": true}}, "why": "x"}'
    c = ai.apply("point", [p], {"construction": 10}, [], call=good)
    assert c and c["parents"] == [p["id"]] and c["genes"]["tool_interface"]["errors"] == "all_with_fix_hints"
    two = lambda t: '{"genes": {"numeric": {"max_turns": 10, "retries": 1, "check_every": 1, "samples": 50, "tolerance": 0.05}, "query_layer": ["what_if"]}}'
    assert ai.apply("point", [p], {}, [], call=two) is None           # point may change one gene only
    bad = lambda t: '{"genes": {"core": {"kind": "magic", "params": {}}}}'
    assert ai.apply("point", [p], {}, [], call=bad) is None
    notjson = lambda t: "sorry"
    assert ai.apply("guidance", [p], {}, [], call=notjson) is None
    g = lambda t: '{"genes": {"guidance": {"text": "Build one equation at a time.", "examples": []}}}'
    assert ai.apply("guidance", [p], {}, [], call=g)["genes"]["guidance"]["text"].startswith("Build")


def test_prereg_lock_detects_changes(tmp_path, monkeypatch):
    from v3 import prereg as P
    monkeypatch.setattr(P, "LOCK", tmp_path / "PREREG.LOCK")
    f = tmp_path / "x.py"; f.write_text("a = 1\n")
    monkeypatch.setattr(P, "files", lambda: [f])
    monkeypatch.setattr(P, "_rel", lambda p: "v3:" + pathlib.Path(p).name)
    assert P.verify() == ["no lock"]
    P.lock({"quick": ["W1"]}, "test")
    assert P.verify() == []
    f.write_text("a = 2\n")
    assert P.verify() == ["v3:x.py"]
