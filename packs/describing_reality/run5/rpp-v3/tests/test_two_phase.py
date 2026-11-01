"""Two-phase episodes (V3-007): build without the question, answer without the description."""
import json

import pytest

from v3 import episode as E
from v3.genome.seeds import load_all
from tests.test_cores import FakeClient, ONT_PARTS, _tc, names

SEEDS = load_all()
DESC = "A reactor with a pump. UNIQUE-DESCRIPTION-MARKER."
QUESTION = "What happens to temperature? UNIQUE-QUESTION-MARKER"
ITEM = {"id": "T1", "system": "T", "system_id": "T", "bank": "synthetic", "question": QUESTION,
        "task_type": "prediction", "context": DESC, "answer_key": {"key_points": []}}


@pytest.fixture
def two_phase(monkeypatch, tmp_path):
    monkeypatch.setitem(E.CFG["evolution"], "episode_mode", "two_phase")
    monkeypatch.setattr(E, "RUNS", tmp_path)
    monkeypatch.setattr(E, "grade", lambda item, raw, **kw: {"status": "ok", "score": 1.0 if raw else 0.0})
    return tmp_path


def _text(call):
    return "\n".join(str(m.get("content") or "") for m in call["messages"])


def test_ontology_build_then_answer_from_representation_only(two_phase):
    build = [_tc("add_part", {"section": s, "text": t}, f"p{k}") for k, (s, t) in enumerate(ONT_PARTS)]
    script = [{"role": "assistant", "content": "", "tool_calls": build + [_tc("finish", {}, "f")]},
              {"role": "assistant", "content": "FINAL ANSWER: DONE"},
              {"role": "assistant", "content": "", "tool_calls": [_tc("path", {"source": "pump",
                                                                                "target": "temperature"}, "q")]},
              {"role": "assistant", "content": "FINAL ANSWER temperature falls"}]
    client = FakeClient(script)
    rec = E.run(SEEDS["ONT_BASIC"], ITEM, "qwen", "t", client=client)
    b0, a0 = client.calls[0], client.calls[2]
    assert "UNIQUE-DESCRIPTION-MARKER" in _text(b0) and "UNIQUE-QUESTION-MARKER" not in _text(b0)
    assert "UNIQUE-QUESTION-MARKER" in _text(a0) and "UNIQUE-DESCRIPTION-MARKER" not in _text(a0)
    assert "GIVEN REPRESENTATION" in _text(a0)
    assert "add_part" not in names(a0["tools"]) and {"lookup", "path"} <= set(names(a0["tools"]))
    assert rec["mode"] == "two_phase" and rec["rep"] == 0 and rec["representation"]["text"]
    assert E.buildable(rec) and rec["final_raw"].endswith("temperature falls")
    assert rec["output_tokens"] == rec["answer_output_tokens"] + rec["build"]["output_tokens"]
    saved = json.loads((two_phase / "t" / "qwen" / SEEDS["ONT_BASIC"]["id"] / "T1" / "record.json").read_text())
    assert saved["grade"]["score"] == 1.0


def test_prose_notes_are_the_only_context(two_phase):
    # A0+ is effort-matched (revision passes), so each phase may take several calls: split them by the question
    script = [{"role": "assistant", "content": "FINAL ANSWER\nPump cools the reactor. NOTES-MARKER"}] * 3 + \
             [{"role": "assistant", "content": "NO CHANGE"}] * 2
    client = FakeClient(script)
    rec = E.run(SEEDS["A0P"], ITEM, "qwen", "t", client=client)
    asked = [c for c in client.calls if "UNIQUE-QUESTION-MARKER" in _text(c)]
    built = client.calls[:client.calls.index(asked[0])]
    assert built and asked and all("UNIQUE-DESCRIPTION-MARKER" in _text(c) for c in built)
    assert all("UNIQUE-DESCRIPTION-MARKER" not in _text(c) and "NOTES-MARKER" in _text(c) for c in asked)
    assert E.buildable(rec) and "NOTES-MARKER" in rec["notes"]


def test_failed_build_answers_without_tools_and_is_not_buildable(two_phase):
    script = [{"role": "assistant", "content": "", "tool_calls": [_tc("add_part", {"section": "THINGS",
                                                                                    "text": "pump"}, "a"),
                                                                   _tc("finish", {}, "f")]},
              {"role": "assistant", "content": "FINAL ANSWER: DONE"},
              {"role": "assistant", "content": "FINAL ANSWER unsure"}]
    rec = E.run(SEEDS["ONT_BASIC"], ITEM, "qwen", "t", client=FakeClient(script))
    assert not E.buildable(rec) and rec["grade"]["status"] == "ok"


def test_single_mode_unchanged(monkeypatch, tmp_path):
    monkeypatch.setitem(E.CFG["evolution"], "episode_mode", "single")
    monkeypatch.setattr(E, "RUNS", tmp_path)
    monkeypatch.setattr(E, "grade", lambda item, raw, **kw: {"status": "ok", "score": 0.5})
    client = FakeClient([{"role": "assistant", "content": "FINAL ANSWER falls"}])
    rec = E.run(SEEDS["A0P"], ITEM, "qwen", "t", client=client)
    assert all("UNIQUE-QUESTION-MARKER" in _text(c) for c in client.calls) and rec.get("mode") != "two_phase"


def test_unfinished_draft_is_submitted_at_the_end(two_phase):
    """V3-009: a builder that writes every part but ends with DONE without finish() keeps its draft."""
    build = [_tc("add_part", {"section": s, "text": t}, f"p{k}") for k, (s, t) in enumerate(ONT_PARTS)]
    script = [{"role": "assistant", "content": "", "tool_calls": build},
              {"role": "assistant", "content": "FINAL ANSWER: DONE"},
              {"role": "assistant", "content": "FINAL ANSWER temperature falls"}]
    rec = E.run(SEEDS["ONT_BASIC"], ITEM, "qwen", "t", client=FakeClient(script))
    assert E.buildable(rec) and rec["representation"]["_v3"]["auto_finished"]
