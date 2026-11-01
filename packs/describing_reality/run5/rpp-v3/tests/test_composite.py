"""Complexity ladder: composite items (v3.items.composite), ladder item sets and the calibration summary."""
import json

import pytest

from v3 import items as I
from v3.evolve import item_sets

BASE = 1_500_000


@pytest.fixture(scope="module")
def ladder():
    return {k: I.composite(k, BASE) for k in I.COMPOSITE_KS}


def perfect(item):
    return "ANSWERS: " + json.dumps(I.answer_key(item))


def test_deterministic_and_loadable(ladder):
    it = ladder[4]
    assert I.composite(4, BASE) == it
    assert I.load("K4x1500000") == it
    assert it["id"] == "K4x1500000" and it["k"] == 4 and it["grading"] == "script"


def test_members_distinct_prefix_and_never_test_split(ladder):
    from discovery.world.splits import split_of
    for k, it in ladder.items():
        assert len(it["member_seeds"]) == len(set(it["member_seeds"])) == k
        assert all(split_of(s) == "validation" for s in it["member_seeds"])
        assert it["families"] == [m["family"] for m in it["members"]]
    assert len(set(ladder[4]["families"])) == 4                       # distinct families while they last
    assert ladder[8]["member_seeds"][:4] == ladder[4]["member_seeds"]   # K4 ⊂ K8 (paired ladder)


def test_test_split_refused():
    with pytest.raises(I.Sealed):
        I.composite(2, 2_000_100)
    with pytest.raises(I.Sealed):
        I.load("K1x2000100")
    with pytest.raises(I.Sealed):
        I.ladder_seeds(2, 2_000_000)


def test_names_never_shared_and_every_paragraph_attributable(ladder):
    it = ladder[8]
    ms = [I._member(s, "realistic") for s in it["member_seeds"]]
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            assert not a["names"] & b["names"] and a["theme"] != b["theme"]
    paras = it["context"].split("\n\n")
    assert "separate systems" in paras[0]
    owner = []
    for p in paras[1:]:
        hits = [j for j, m in enumerate(ms) if I.names_in(p) & m["names"]]
        assert len(hits) == 1
        owner.append(hits[0])
    runs = 1 + sum(a != b for a, b in zip(owner, owner[1:]))
    assert runs > 2 * len(ms)                                           # interleaved, not one block per system


def test_question_ids_unique_and_prefixed(ladder):
    for k, it in ladder.items():
        ids = it["question_ids"]
        assert len(ids) == len(set(ids)) and 0 < len(ids) <= 3 * k
        assert all(q.split(".")[0] == f"s{j}" for j, m in enumerate(it["members"], 1)
                   for q in ids if q.startswith(f"s{j}."))
        assert all(f"[{q}]" in it["question"] for q in ids)
        assert set(I.answer_key(it)) == set(ids)


def test_score_generated_perfect_and_wrong(ladder):
    it = ladder[4]
    key = I.answer_key(it)
    assert I.score_generated(it, "model text\n" + perfect(it))["score"] == 1.0
    wrong = {q: (a + 1) % 2 if a in (0, 1) else 0 for q, a in key.items()}
    r = I.score_generated(it, "ANSWERS: " + json.dumps(wrong))
    assert r["score"] == 0.0 and r["n_probes"] == len(key) and r["answered"] == len(key)
    half = dict(list(key.items())[:len(key) // 2])
    assert I.score_generated(it, "ANSWERS: " + json.dumps(half))["score"] == pytest.approx(len(half) / len(key))


def test_description_grows_with_k(ladder):
    lens = [len(ladder[k]["context"]) for k in I.COMPOSITE_KS]
    assert lens == sorted(lens) and lens[-1] > 3 * lens[0]


def test_units_reproduce_v2_description():
    import discovery.world as W
    for s in (1_000_000, 1_000_002, 1_000_003):
        inst, fam, genome = I._light(s)
        for mess in ("clean", "realistic", "adversarial"):
            assert " ".join(I._units(inst, fam, genome, mess)) == " ".join(inst.description(mess).split("\n\n"))
    with pytest.raises(ValueError):
        I.composite(3, BASE)


def test_ladder_item_sets_deterministic_and_disjoint():
    lad = {"quick": {1: 1, 8: 1}, "screen": {1: 2, 2: 2, 4: 1}, "full": {1: 2, 2: 1, 4: 1, 8: 1}}
    a = item_sets(ladder=lad, dev=False)
    assert a == item_sets(ladder=lad, dev=False)
    assert [len(a[s]) for s in ("quick", "screen", "full")] == [2, 5, 5]
    ids = [i for s in a.values() for i in s]
    assert len(ids) == len(set(ids))
    bases = [int(i.split("x")[1]) for i in ids]
    assert len(set(bases)) == len(bases) and all(b % I.COMPOSITE_STRIDE == 0 for b in bases)
    assert [i.split("x")[0] for i in a["screen"]] == ["K1", "K1", "K2", "K2", "K4"]
    members = {s: {m for i in a[s][:2] for m in I.load(i)["member_seeds"]} for s in a}
    assert members["quick"].isdisjoint(members["screen"]) and members["screen"].isdisjoint(members["full"])
    full = item_sets(ladder=True, dev=False)
    assert [len(full[s]) for s in ("quick", "screen", "full")] == [5, 30, 90]


def test_calibration_summary_pure():
    from v3 import calibrate as C
    ids = C.calib_items([1, 2], 2)
    assert ids[1][0].split("x")[1] == ids[2][0].split("x")[1] and int(ids[1][0].split("x")[1]) >= I.CALIB_START
    items = {i: {"k": int(i[1]), "context": "x" * (100 * int(i[1]))} for per in ids.values() for i in per}
    recs = [{"builder": "qwen", "item": i, "grade": {"score": 0.5 + 0.25 * (n % 2)}, "output_tokens": 1000}
            for n, i in enumerate(sorted(items))]
    recs.append({"builder": "qwen", "item": ids[2][0], "error": "boom"})
    s = C.summarise(recs, items)
    assert set(s["qwen"]) == {"1", "2"}
    assert s["qwen"]["2"]["n_errors"] == 1 and s["qwen"]["1"]["description_chars"] == 100
    assert s["qwen"]["1"]["lo"] <= s["qwen"]["1"]["mean"] <= s["qwen"]["1"]["hi"]
    assert "qwen" in C.table(s)


def test_calibration_run_with_stub_runner(tmp_path):
    from v3 import calibrate as C
    seen = []

    def stub(genome, item, builder, phase):
        seen.append((builder, item["id"], phase))
        return {"builder": builder, "item": item["id"], "grade": I.score_generated(item, perfect(item)),
                "output_tokens": 10}
    s = C.run({"id": "g"}, [1, 2], 1, ["qwen"], runner=stub)
    assert {p for _, _, p in seen} == {"calib"} and len(seen) == 2
    assert s["qwen"]["1"]["mean"] == s["qwen"]["2"]["mean"] == 1.0
