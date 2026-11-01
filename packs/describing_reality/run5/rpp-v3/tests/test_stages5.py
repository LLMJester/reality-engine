"""V3-058: the gated retention rule (it fails on the old fitted-argmax rule that picked D18), the screen gate and cap,
the fixed-sequence final check, the slot checks and invariants, the retired sealed set and the stage item sets."""
import itertools
import json

import numpy as np
import pytest

from v3 import analysis5 as A
from v3 import run5 as R
from v3 import run5_sets as R5
from v3 import stages5 as G

from tests.stages5_fakes import EX_IDS, EX_SMALL, SLOT_REPLIES, TWEAK, FakeOpus


def _items(n_bases, start=2_600_000):
    return [f"K{k}x{start + 500 * b}" for b in range(n_bases) for k in (1, 2, 4, 8)]


def _factorial_scores(effects, factors, items, base=0.66, sd=0.05, seed=1):
    """Scores of the 2^m designs (standard order) from main and pair effects (mean(+) - mean(-)) plus item noise."""
    rng = np.random.default_rng(seed)
    item_eff = rng.normal(0, 0.15, len(items))
    names = G.confirm_labels(len(factors))
    out = {}
    for n, lab in enumerate(names):
        x = {f: (1 if (n >> b) & 1 else -1) for b, f in enumerate(factors)}
        mu = base + sum(effects.get(f, 0) / 2 * x[f] for f in factors)
        mu += sum(effects.get(f"{a}:{b}", 0) / 2 * x[a] * x[b] for a, b in itertools.combinations(factors, 2))
        out[lab] = {i: float(np.clip(mu + item_eff[j] + rng.normal(0, sd), 0, 1)) for j, i in enumerate(items)}
    return out, names


def test_retention_keeps_d01_where_the_old_rule_picked_an_a_plus_design():
    # the stage 2 pattern: A, B, C harmful, D and E unclear, A:B and A:C negative (A+ looks good at B-, C-)
    items = _items(120)
    sc, names = _factorial_scores({"A": -0.017, "B": -0.035, "C": -0.043, "D": -0.002, "E": 0.003,
                                   "A:B": -0.02, "A:C": -0.02}, A.FACTORS, items, sd=0.25)   # stage 2's noise
    old = A.stage2(sc, names, items)
    assert A.design_table()[names.index(old["final_design"])]["A"] == 1          # the old rule: A+ (as D18)
    new = A.retention(sc, names, items, A.FACTORS, family=A.CONFIRMATORY)
    assert new["final_design"] == "D01" and not new["replaces"] and new["kept_mains"] == []


def test_retention_takes_a_real_gain_and_checks_it_directly():
    items = _items(120)
    sc, names = _factorial_scores({"slot1": 0.05}, ["slot1", "slot5"], items)
    r = A.retention(sc, names, items, ["slot1", "slot5"])
    assert r["final_design"] == "D02" and r["replaces"] and r["kept_mains"] == ["slot1"]
    assert 0.03 < r["gain"] < 0.07 and r["direct"]["range"][0] > 0


def test_retention_marginality_drops_a_pair_without_its_mains():
    items = _items(120)
    sc, names = _factorial_scores({"slot1": 0.04, "slot2": -0.002, "slot1:slot2": 0.03}, ["slot1", "slot2"], items)
    r = A.retention(sc, names, items, ["slot1", "slot2"])
    assert "slot1:slot2" not in r["kept_pairs"] and r["candidate_levels"] == {"slot1": "+", "slot2": "-"}


def test_retention_needs_the_direct_contrast_above_zero():
    items = _items(120)
    sc, names = _factorial_scores({}, ["slot3"], items)
    r = A.retention(sc, names, items, ["slot3"])
    assert r["final_design"] == "D01" and r["gain"] == 0.0


def test_screen_gate_and_cap_of_four():
    items = _items(80)
    rng = np.random.default_rng(3)
    item_eff = rng.normal(0, 0.15, len(items))
    gains = {1: 0.05, 2: 0.04, 3: 0.03, 4: 0.02, 5: 0.06}
    sc = {"CUR": {i: 0.6 + item_eff[j] for j, i in enumerate(items)}}
    cands = {}
    for s in range(1, 6):
        for a in range(1, 5):
            lab = f"s{s}a{a}"
            g = gains[s] if a == 2 else -0.01
            sc[lab] = {i: 0.6 + g + item_eff[j] + rng.normal(0, 0.01) for j, i in enumerate(items)}
            cands[lab] = {"slot": s, "archetype": a, "length": 100}
    r = A.screen(sc, "CUR", cands, items)
    assert all(r["slots"][str(s)]["best"] == f"s{s}a2" for s in range(1, 6))
    assert set(r["advancing"]) == {"1", "2", "3", "5"} and r["capped"] == ["4"]      # 4 passes but is capped
    sc["s4a2"] = {i: v - 0.015 for i, v in sc["s4a2"].items()}                       # now +0.005: under the gate
    sc["s5a2"] = {i: v - 0.06 for i, v in sc["s5a2"].items()}
    r = A.screen(sc, "CUR", cands, items)
    assert not r["slots"]["4"]["passes_gate"] and set(r["advancing"]) == {"1", "2", "3"}
    assert r["candidates"]["s1a2"]["by_k"].keys() == {"1", "2", "4", "8"}


def test_final_check_sequence_stops_at_the_first_failure():
    items = _items(180)
    rng = np.random.default_rng(5)
    item_eff = rng.normal(0, 0.15, len(items))
    means = {"FINAL": 0.68, "FACTS": 0.66, "A0P": 0.50, "ONT_LOOSE": 0.68, "A3": 0.60}
    sc = {d: {i: float(np.clip(m + item_eff[j] + rng.normal(0, 0.05), 0, 1)) for j, i in enumerate(items)}
          for d, m in means.items()}
    r = A.final_stages(sc, "FINAL", "FACTS", "A0P", "ONT_LOOSE", "A3", items)
    q = r["sequence"]
    assert list(q) == ["1_vs_plain_prose", "2_vs_hand_top", "3_vs_runner_up", "4_vs_facts"]
    assert q["1_vs_plain_prose"]["counted"] and not q["2_vs_hand_top"]["passes"]
    assert q["3_vs_runner_up"]["passes"] and not q["3_vs_runner_up"]["counted"]     # after a failure: not counted
    same = A.final_stages(sc, "FACTS", "FACTS", "A0P", "ONT_LOOSE", "A3", items)
    assert "4_vs_facts" not in same["sequence"] and same["same_as_facts"]


# ---------------------------------------------------------------- slots, checks and invariants
@pytest.fixture
def facts():
    g = R.founder_genomes(["FACTS"])["FACTS"]
    return g, G.parts_of(g)


def test_facts_splits_into_slots_without_drifting(facts):
    g, cur = facts
    assert G.guidance_text(cur) == g["genes"]["guidance"]["text"] and cur["strategy"] == ""
    assert G.compose(g, cur, [g["id"]], "same")["id"] == g["id"]                     # D01 does not move a character


def _reply(slot, cur, **over):
    vals = list(SLOT_REPLIES[slot])
    if slot == 2:
        vals[3] = cur["rules"] + TWEAK
    for a, v in over.items():
        vals[int(a[1:]) - 1] = v
    return {"candidates": [{"archetype": a, "value": v, "rationale": "r"} for a, v in enumerate(vals, 1)]}


@pytest.mark.parametrize("slot", [1, 2, 3, 4, 5])
def test_good_candidates_pass_and_each_design_changes_only_its_slot(facts, slot):
    g, cur = facts
    cands, probs = G.check_slot(slot, _reply(slot, cur), cur, g, G.layout_names(cur))
    assert probs == [] and len(cands) == 4
    for lab, c in cands.items():
        d = G.compose(g, G.design_parts(cur, {slot: lab}, cands, G.initial_cube(cur, cands)), [g["id"]], lab)
        changed = {k for k in g["genes"] if d["genes"][k] != g["genes"][k]}
        assert changed == {{1: "core", 2: "guidance", 3: "guidance", 4: "answering", 5: "examples"}[slot]}
        if slot == 3:                                    # the build check is a separate block after the rules
            assert d["genes"]["guidance"]["text"].startswith(cur["rules"].rstrip() + "\n\n")


def test_slot_checks_reject_what_the_design_forbids(facts):
    g, cur = facts
    assert any("too alike" in p for p in G.check_slot(3, _reply(3, cur, a2=SLOT_REPLIES[3][0] + " Then stop."),
                                                      cur, g)[1])
    assert any("syntax" in p for p in G.check_slot(3, _reply(3, cur, a1="Write each link as 'a -> b: fills'."),
                                                   cur, g)[1])
    assert any("names the section header" in p for p in
               G.check_slot(4, _reply(4, cur, a1="Look in the TIMING section first, then answer."), cur, g)[1])
    assert any("fails the notes checker" in p for p in
               G.check_slot(5, _reply(5, cur, a1="## THINGS\n- pump P1\nnot a bullet"), cur, g)[1])
    assert any("not valid" in p for p in G.check_slot(1, _reply(1, cur, a1={"headers": ["x"], "tags": False,
                                                                            "granularity": "fine"}), cur, g)[1])
    assert any("the rules must be" in p for p in G.check_slot(2, _reply(2, cur, a1="Short."), cur, g)[1])
    assert any("characters; at most 2500" in p for p in G.check_slot(4, _reply(4, cur, a1="x " * 1300), cur, g)[1])
    assert any("adds what candidate" in p for p in
               G.check_slot(4, _reply(4, cur, a4=SLOT_REPLIES[4][2]), cur, g)[1])
    assert G.check_slot(2, {"candidates": [{"archetype": 1, "value": "x"}]}, cur, g)[1]


def test_example_cube_and_missing_examples(facts):
    g, cur = facts
    c1, _ = G.check_slot(1, _reply(1, cur), cur, g)
    c5, _ = G.check_slot(5, _reply(5, cur), cur, g)
    cands = {**c1, **c5}
    cube = G.initial_cube(cur, cands)
    assert G.needed_examples({1: "s1a2", 5: "s5a2"}, cands, cube) == ["s1a2|0|s5a2"]
    assert G.needed_examples({1: "s1a2"}, cands, cube) == []                         # FACTS has no example
    with pytest.raises(G.MissingExample):
        G.design_parts(cur, {1: "s1a2", 5: "s5a2"}, cands, cube)
    assert G.design_parts(cur, {5: "s5a2"}, cands, cube)["example"] == EX_IDS
    assert G.design_parts(cur, {1: "s1a2"}, cands, cube)["example"] == ""


def test_writer_retries_then_reports(facts):
    g, cur = facts
    opus = FakeOpus("{}", bad_slots={4})
    cands, probs = G.write_slot(4, cur, g, "EVIDENCE", call=opus)
    assert probs and len(opus.calls) == G.ATTEMPTS and "earlier reply failed" in opus.calls[-1]
    cands, probs = G.write_slot(5, cur, g, "EVIDENCE", call=FakeOpus("{}"))
    assert probs == [] and cands["s5a1"]["value"] == EX_SMALL


def test_brief_carries_only_the_genome_and_the_evidence(facts):
    g, cur = facts
    b = G.slot_brief(2, cur, g, "EVIDENCE BLOCK", {"THINGS", "EDGES"})
    assert "EVIDENCE BLOCK" in b and json.dumps(g["genes"], indent=1) in b and "EDGES" in b
    assert "K1x" not in b and "answer key" not in b.lower()


# ---------------------------------------------------------------- the retired sealed set
def test_retired_sealed_set_is_never_read(tmp_path, monkeypatch):
    item = sorted(R5.retired_items())[0]
    with pytest.raises(R5.Retired):
        A.matrix({"X": {item: 0.5}}, ["X"], [item])
    mode = R.Mode("real")
    with pytest.raises(R5.Retired):
        R.score(mode, "g000", item)
    from v3 import plus_evidence5 as E
    with pytest.raises(R5.Retired):
        E.summary(mode, R.founder_genomes(["FACTS"])["FACTS"], R.record, R.score, items=[item])
    with pytest.raises(ValueError):
        E.summary(mode, R.founder_genomes(["FACTS"])["FACTS"], R.record, R.score, items=["H1-1"])
    assert R.score(mode, "g000", item, phase="run5_s2") is None                   # other folders: no such item


# ---------------------------------------------------------------- the stage item sets
def test_allocation_is_disjoint_and_balanced(tmp_path, monkeypatch):
    (tmp_path / "runs" / "x" / "K1x1000000").mkdir(parents=True)                  # a used base
    (tmp_path / "runs" / "y" / "W1001700").mkdir(parents=True)                    # a used single seed
    doc = R5.allocate_stage_sets(root=tmp_path, family_of=lambda s: f"f{(s // 500) % 7}")
    sets = doc["sets"]
    assert {k: len(v) for k, v in sets.items()} == {**{f"s{n}_screen": 80 for n in (3, 4, 5)},
                                                   **{f"s{n}_confirm": 120 for n in (3, 4, 5)}, "sealed2": 180}
    allb = sorted(b for v in sets.values() for b in v)
    assert all(b2 - b1 >= R5.STRIDE for b1, b2 in zip(allb, allb[1:]))             # no two windows share a seed
    assert all(abs(b - 1_000_000) >= R5.STRIDE and not b <= 1_001_700 < b + R5.STRIDE for b in allb)
    with R5._choosing():
        old = {b for n in R5.N_BASES for b in R5._bases(n)}
    assert all(abs(b - o) >= R5.STRIDE for b in allb for o in old)
    assert all(not lo - R5.STRIDE < b < hi for b in allb for lo, hi in R5.EXCLUDED_RANGES)
    assert all(b < 1_000_000 for b in sets["sealed2"]) and all(1_000_000 <= b < 2_000_000 for k, v in sets.items()
                                                               if k != "sealed2" for b in v)
    fam = [f"f{(b // 500) % 7}" for b in sets["s3_screen"]]
    assert max(fam.count(f) for f in set(fam)) - min(fam.count(f) for f in set(fam)) <= 1


def test_the_frozen_stage_file_matches_its_hash():
    doc = R5._stage_file()
    if doc is None:
        pytest.skip("items/run5_stages.json not written yet")
    import hashlib
    assert hashlib.sha256(json.dumps(doc["sets"], sort_keys=True).encode()).hexdigest() == doc["sha256"]
    assert "sealed2" in R5.sealed_windows() and R5.is_sealed(doc["sets"]["sealed2"][0]) in (True, False)
