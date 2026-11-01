"""Condition runners C1–C4. Each returns a record with raw messages, final
answer, artefacts (AST, verdict, tool trace) and cost. Prompts are the frozen
templates in prompts/ (hashes in PREREG.LOCK); C3/C4 also embed the language
guide from the language directory under test (that is the thing being refined).
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import time

from harness.llm import chat

HERE = pathlib.Path(__file__).resolve().parent
P = {p.stem: p.read_text(encoding="utf-8") for p in (HERE / "prompts").glob("*.txt")}
MAX_TOOL_CALLS = 10
GEN_MAX_TOKENS = 16000
SANDBOX = HERE.parent / "sandbox_runner.py"

PY_TOOL = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python 3 code in a fresh process. Returns stdout and stderr (truncated to 6000 characters). "
                       "clingo, z3-solver and numpy are installed.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
}]


def fill(t, **kw):
    for k, v in kw.items():
        t = t.replace("{" + k + "}", str(v))
    return t


def run_python(code, timeout=60):
    with tempfile.TemporaryDirectory(prefix="rpp_c2_") as d:
        f = pathlib.Path(d) / "main.py"
        f.write_text(code, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(SANDBOX), d, str(f)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, cwd=d, env={"PYTHONIOENCODING": "utf-8",
                                                            "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")})
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        except subprocess.TimeoutExpired:
            out = f"[timed out after {timeout}s]"
    return out[-6000:] if len(out) > 6000 else out


def _rec(cond, tier, model, seed):
    return {"condition": cond, "tier": tier, "model": model, "seed": seed, "calls": [], "cost": 0.0,
            "tokens": 0, "wall_s": 0.0, "final_raw": "", "error": None}


def _add(rec, usage, label):
    rec["calls"].append({"label": label, **{k: usage[k] for k in
                                            ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cost",
                                             "wall_s", "provider")}})
    rec["cost"] += usage["cost"]
    rec["tokens"] += usage["prompt_tokens"] + usage["completion_tokens"]
    rec["wall_s"] += usage["wall_s"]


def c1(item, ctx, tier, model, seed, purpose):
    rec = _rec("C1", tier, model, seed)
    msgs = [{"role": "system", "content": P["system_common"]},
            {"role": "user", "content": fill(P["c1_user"], context=ctx, question=item["question"])}]
    m, u = chat(model, msgs, purpose=purpose, seed=seed, max_tokens=GEN_MAX_TOKENS)
    _add(rec, u, "answer")
    rec["final_raw"] = m.get("content") or ""
    rec["messages"] = msgs + [{"role": "assistant", "content": rec["final_raw"]}]
    return rec


def c2(item, ctx, tier, model, seed, purpose):
    rec = _rec("C2", tier, model, seed)
    msgs = [{"role": "system", "content": P["system_common"]},
            {"role": "user", "content": fill(P["c2_user"], context=ctx, question=item["question"],
                                             max_tool_calls=MAX_TOOL_CALLS)}]
    trace = []
    n_calls = 0
    while True:
        tools = PY_TOOL if n_calls < MAX_TOOL_CALLS else None
        m, u = chat(model, msgs, purpose=purpose, seed=seed, max_tokens=GEN_MAX_TOKENS, tools=tools)
        _add(rec, u, f"turn{len(trace)}")
        tcs = m.get("tool_calls") or []
        am = {"role": "assistant", "content": m.get("content") or ""}
        if tcs:
            am["tool_calls"] = tcs
        msgs.append(am)
        if not tcs:
            rec["final_raw"] = m.get("content") or ""
            break
        for tc in tcs:
            n_calls += 1
            try:
                code = json.loads(tc["function"]["arguments"]).get("code", "")
            except Exception:
                code = tc["function"].get("arguments", "")
            t0 = time.time()
            out = run_python(code) if n_calls <= MAX_TOOL_CALLS else "[tool call limit reached]"
            rec["wall_s"] += time.time() - t0
            trace.append({"code": code, "output": out})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", f"call{n_calls}"), "content": out})
        if n_calls >= MAX_TOOL_CALLS + 2:
            msgs.append({"role": "user", "content": "Tool limit reached. Give your FINAL ANSWER now."})
    rec["tool_trace"] = trace
    rec["messages"] = msgs
    return rec


def parse_json(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                return None
    return None


def decoding_schema(schema):
    """Schema sent for constrained decoding: same structure as lang/schema.json, minus keywords that
    provider grammar engines reject (propertyNames, size bounds; oneOf->anyOf). Full validation is the checker's."""
    def walk(o):
        if isinstance(o, dict):
            o = {k: walk(v) for k, v in o.items() if k not in ("$schema", "$id", "propertyNames",
                                                                 "minLength", "minItems", "maxItems",
                                                                 "minimum", "maximum")}
            if "oneOf" in o:
                o["anyOf"] = o.pop("oneOf")
            if "const" in o:
                o["enum"] = [o.pop("const")]
            return o
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    return walk(schema)


def build_model(item, ctx, tier, model, seed, purpose, lang_dir, rec):
    guide = (pathlib.Path(lang_dir) / "GUIDE.md").read_text(encoding="utf-8")
    schema = json.loads((pathlib.Path(lang_dir) / "schema.json").read_text(encoding="utf-8"))
    schema = decoding_schema(schema)
    msgs = [{"role": "system", "content": fill(P["c34_model_system"], guide=guide)},
            {"role": "user", "content": fill(P["c34_model_user"], context=ctx, question=item["question"])}]
    fmt = {"type": "json_schema", "json_schema": {"name": "rpp_core_model", "strict": False, "schema": schema}}
    ast, raw = None, ""
    for attempt in range(2):  # one retry if the output is not parseable JSON
        m, u = chat(model, msgs, purpose=purpose, seed=seed + attempt, max_tokens=GEN_MAX_TOKENS,
                    response_format=fmt)
        _add(rec, u, f"model{attempt}")
        raw = m.get("content") or ""
        ast = parse_json(raw)
        if ast is not None:
            break
    rec["model_raw"] = raw
    rec["ast"] = ast
    rec["model_messages"] = msgs
    return ast, raw


_model_locks = {}
_model_locks_guard = __import__("threading").Lock()


def shared_model(item, ctx, tier, model, seed, purpose, lang_dir, cache_path):
    """C3 and C4 use the same world-model call per (item, tier, language): built once, cached."""
    with _model_locks_guard:
        lock = _model_locks.setdefault(str(cache_path), __import__("threading").Lock())
    with lock:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")), False
        rec = _rec("model", tier, model, seed)
        build_model(item, ctx, tier, model, seed, purpose, lang_dir, rec)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return rec, True


def c34(item, ctx, tier, model, seed, purpose, lang_dir, with_checker, check_fn, cache_path):
    rec = _rec("C4" if with_checker else "C3", tier, model, seed)
    mrec, built_here = shared_model(item, ctx, tier, model, seed, purpose, lang_dir, cache_path)
    ast, raw = mrec.get("ast"), mrec.get("model_raw", "")
    rec["model_call"] = {"cache": str(cache_path), "calls": mrec["calls"], "cost": mrec["cost"],
                         "tokens": mrec["tokens"], "wall_s": mrec["wall_s"]}
    # the shared model's cost is attributed in full to both C3 and C4 (each condition's true cost)
    rec["cost"] += mrec["cost"]
    rec["tokens"] += mrec["tokens"]
    rec["wall_s"] += mrec["wall_s"]
    rec["ast"] = ast
    model_txt = json.dumps(ast, indent=1) if ast is not None else (raw[:20000] or "(no valid JSON was produced)")
    t_check = time.time()
    # the checker always runs (for H4 post-hoc analysis); its verdict is shown only in C4
    try:
        verdict = check_fn(ast) if ast is not None else {
            "pass": False, "failures": [{"obligation": "schema", "claim_ids": [],
                                         "explanation": "output was not parseable JSON"}], "claims": {},
            "sensitivity": []}
    except Exception as e:  # checker crash is recorded, never hidden
        verdict = {"pass": False, "failures": [{"obligation": "well_formed", "claim_ids": [],
                                                "explanation": f"checker error: {str(e)[:300]}"}],
                   "claims": {}, "sensitivity": [], "crash": True}
    rec["verdict"] = verdict
    rec["wall_s"] += time.time() - t_check
    if with_checker:
        vshow = {"pass": verdict["pass"], "failures": verdict["failures"][:25],
                 "claims": verdict.get("claims", {}), "sensitivity": verdict.get("sensitivity", [])}
        user = fill(P["c4_answer_user"], context=ctx, question=item["question"], model=model_txt,
                    verdict=json.dumps(vshow, indent=1))
    else:
        user = fill(P["c3_answer_user"], context=ctx, question=item["question"], model=model_txt)
    msgs = [{"role": "system", "content": P["system_common"]}, {"role": "user", "content": user}]
    m, u = chat(model, msgs, purpose=purpose, seed=seed, max_tokens=GEN_MAX_TOKENS)
    _add(rec, u, "answer")
    rec["final_raw"] = m.get("content") or ""
    rec["messages"] = msgs + [{"role": "assistant", "content": rec["final_raw"]}]
    return rec
