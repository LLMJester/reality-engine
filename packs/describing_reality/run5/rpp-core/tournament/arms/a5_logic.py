"""A5: logic / rules world model in R++ Core (gen-1 candidate 2: candidates/gen1/2).

The compiler and checker of that language run in a subprocess with cwd=candidates/gen1/2 (as
harness.run.make_check_fn does), through the helper tournament/arms/_rpp_exec.py. Checker errors
are returned with a location and a hint. Graceful degradation: elements named in blocking errors
are removed and reported as holes; the rest is evaluated (see _rpp_exec.py)."""
import copy
import hashlib
import json
import pathlib
import re
import subprocess
import sys

from tournament.arms.base import Arm, ArmState, ERROR_PREFIX

ROOT = pathlib.Path(__file__).resolve().parents[2]
LANG_ROOT = ROOT / "candidates" / "gen1" / "2"
HELPER = pathlib.Path(__file__).resolve().parent / "_rpp_exec.py"
# Runtime dependencies for the final lock (harness.orchestrator.lock_files; recheck item 1): files this module
# executes in a subprocess or reads at run time, as repo-relative globs. Kept in sync by
# tests/test_recheck_lock.py (every subprocess launch target / cwd must be covered).
# _rpp_exec.py runs with cwd=LANG_ROOT and imports its checker (which starts checker.worker) and compiler, and
# the checker and schema_errors read lang/schema.json.
RUNTIME_DEPS = ["tournament/arms/_rpp_exec.py", "candidates/gen1/2/checker/*.py", "candidates/gen1/2/compiler/*.py",
                "candidates/gen1/2/lang/schema.json"]
TIMEOUT_S = 240
LIST_SECTIONS = ("sorts", "statics", "fluents", "initial", "actions", "laws", "derived", "constraints", "scenarios",
                 "claims")
ID_KEY = {"sorts": "name", "statics": "name", "fluents": "name", "actions": "name", "laws": "id", "derived": "id",
          "constraints": "id", "scenarios": "id", "claims": "id"}

HINTS = {
    "schema": "This field does not match the language structure; see 'Structure' and 'Vocabulary' in the guide.",
    "well_formed": "Fix the named element: every name must be declared, lowercase snake_case; variables Capitalised.",
    "determinate": "The rules allow more than one trajectory: make conditions mutually exclusive, or use one effect "
                   "with if-then-else.",
    "integrity": "A constraint is violated in this scenario: the dynamics, the constraint, or both may be wrong — "
                 "or the violation is a real finding.",
    "state_consistency": "Two effects give this fluent different values at the same step: merge them into one effect "
                         "or make their conditions exclusive.",
    "preconditions": "The event occurs while its precondition is false: change its time or its precondition.",
    "derivation": "An inferred claim needs a derivation naming an existing law/derived/action/constraint/scenario id "
                  "and existing premise claim ids.",
    "prediction": "The claim does not hold when the model is run (see the value the model computes): correct the "
                  "claim or the mechanism.",
    "intervention": "The claim does not hold in the intervention scenario (see the value the model computes): "
                    "correct the claim or the mechanism.",
    "assumptions_listed": "List every assumed element id in answer.assumptions.",
    "observations_explained": "Fix the model so it reproduces the observation, or list the claim in "
                              "answer.unexplained.",
    "conflicts_surfaced": "List the contradicting claims together in answer.conflicts.",
}
LOC_RE = re.compile(r"\b(scenario|law|claim|derived|constraint|action|fluent|static|sort|event)\s+'?([a-z][a-z0-9_]*)")


def _tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


SCEN_SCHEMA = {"type": "object", "description": "An R++ Core scenario object (id, horizon, init, events, "
                                                "interventions), or {\"id\": \"<existing scenario id>\"}."}
TOOLS = [
    _tool("submit_model", "Submit a complete R++ Core world model (JSON object, or a JSON string). Replaces the "
                          "current model. Returns structural errors at once; call check() to run it.",
          {"model": {"type": ["object", "string"]}}, ["model"]),
    _tool("patch_model", "Change part of the current model. Either a JSON Patch list "
                         "[{\"op\": \"add\"|\"replace\"|\"remove\", \"path\": \"/laws/-\", \"value\": ...}], or a "
                         "replacement fragment {\"laws\": [...], ...}: list elements whose id/name matches an existing "
                         "element replace it, others are appended; {\"id\": x, \"delete\": true} removes x; "
                         "other fields are replaced.",
          {"patch": {"type": ["array", "object"]}}, ["patch"]),
    _tool("check", "Compile and run the current model with the checker: every scenario, claim and constraint. "
                   "Returns failures with location and hint, claim results, sensitivity, and which parts are usable."),
    _tool("run_scenario", "Run one scenario on the current model and return the state trajectory. Optional "
                          "'claims': [{\"id\", \"time\": step|\"any\"|\"all\", \"expr\"}] are evaluated on it; "
                          "optional 'show': fluent names to include.",
          {"scenario": SCEN_SCHEMA, "claims": {"type": "array", "items": {"type": "object"}},
           "show": {"type": "array", "items": {"type": "string"}}}, ["scenario"]),
]


def _parse_json(v):
    if isinstance(v, (dict, list)):
        return v
    from harness.conditions.runners import parse_json
    return parse_json(v if isinstance(v, str) else "")


def _pointer(path):
    if path in ("", "/"):
        return []
    if not path.startswith("/"):
        raise ValueError(f"path {path!r} must start with '/'")
    return [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]


def apply_json_patch(doc, ops):
    doc = copy.deepcopy(doc)
    for k, op in enumerate(ops):
        if not isinstance(op, dict) or "op" not in op or "path" not in op:
            raise ValueError(f"operation {k}: needs 'op' and 'path'")
        parts = _pointer(op["path"])
        if not parts:
            if op["op"] in ("add", "replace"):
                doc = copy.deepcopy(op["value"])
                continue
            raise ValueError(f"operation {k}: cannot remove the whole document")
        parent = doc
        for p in parts[:-1]:
            parent = parent[int(p)] if isinstance(parent, list) else parent[p]
        last = parts[-1]
        if isinstance(parent, list):
            if op["op"] == "add":
                if last == "-":
                    parent.append(op["value"])
                else:
                    parent.insert(int(last), op["value"])
            elif op["op"] == "replace":
                parent[int(last)] = op["value"]
            elif op["op"] == "remove":
                parent.pop(int(last))
            else:
                raise ValueError(f"operation {k}: unsupported op {op['op']!r}")
        else:
            if op["op"] in ("add", "replace"):
                if op["op"] == "replace" and last not in parent:
                    raise ValueError(f"operation {k}: {op['path']} does not exist")
                parent[last] = op["value"]
            elif op["op"] == "remove":
                del parent[last]
            else:
                raise ValueError(f"operation {k}: unsupported op {op['op']!r}")
    return doc


def apply_fragment(doc, frag):
    doc = copy.deepcopy(doc)
    changed = []
    for key, val in frag.items():
        if key in LIST_SECTIONS and isinstance(val, list) and key in ID_KEY:
            cur = doc.setdefault(key, [])
            idk = ID_KEY[key]
            for el in val:
                if not isinstance(el, dict):
                    raise ValueError(f"{key}: elements must be objects")
                ident = el.get(idk)
                pos = next((i for i, e in enumerate(cur) if isinstance(e, dict) and ident is not None
                            and e.get(idk) == ident), None)
                if el.get("delete"):
                    if pos is None:
                        raise ValueError(f"{key}: no element {ident!r} to delete")
                    cur.pop(pos)
                    changed.append(f"deleted {key[:-1] if key.endswith('s') else key} {ident}")
                elif pos is not None:
                    cur[pos] = el
                    changed.append(f"replaced {ident}")
                else:
                    cur.append(el)
                    changed.append(f"added {ident or key}")
        else:
            doc[key] = val
            changed.append(f"set {key}")
    return doc, changed


def location(expl):
    if ":" in expl and "/" in expl.split(":", 1)[0] and " " not in expl.split(":", 1)[0]:
        return expl.split(":", 1)[0]
    locs = [f"{k} {v}" for k, v in LOC_RE.findall(expl)[:2]]
    return ", ".join(locs) or "model"


def format_failures(fails, limit=25):
    out = []
    for f in fails[:limit]:
        t = f" at t={f['timestep']}" if "timestep" in f else ""
        out.append(f"- [{f['obligation']}] {location(f['explanation'])}{t}: {f['explanation']}\n"
                   f"  hint: {HINTS.get(f['obligation'], '')}")
    if len(fails) > limit:
        out.append(f"... {len(fails) - limit} more failures")
    return "\n".join(out)


class A5State(ArmState):
    pass


class A5(Arm):
    name = "A5"
    buildable = True
    executable = True
    guide_file = "a5_guide.md"
    mutating_tools = ("submit_model", "patch_model")

    def __init__(self, lang_root=LANG_ROOT, timeout=TIMEOUT_S):
        self.lang_root = pathlib.Path(lang_root)
        self.timeout = timeout
        self._schema = None
        self._cache = {}

    # ---------- subprocess ----------
    def _exec(self, job):
        try:
            r = subprocess.run([sys.executable, str(HELPER)], input=json.dumps(job), cwd=str(self.lang_root),
                               capture_output=True, text=True, encoding="utf-8", timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return {"error": f"the checker timed out after {self.timeout}s (model too large to evaluate; reduce the "
                             f"horizon or value ranges)"}
        if r.returncode != 0:
            return {"error": f"checker crashed: {(r.stderr or '')[-600:]}"}
        try:
            return json.loads(r.stdout)
        except ValueError:
            return {"error": f"checker returned no result: {(r.stderr or r.stdout or '')[-400:]}"}

    def check_rep(self, rep, sensitivity=True):
        key = hashlib.sha256(json.dumps([rep, sensitivity], sort_keys=True, default=str).encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._exec({"mode": "check", "ast": rep, "sensitivity": sensitivity})
        return self._cache[key]

    def schema_errors(self, rep):
        try:
            import jsonschema
            if self._schema is None:
                self._schema = json.loads((self.lang_root / "lang" / "schema.json").read_text(encoding="utf-8"))
            full = dict(rep)
            for k, v in (("rpp_core", "0.1"), ("system", "model"), ("sorts", []), ("fluents", []), ("claims", []),
                         ("answer", {})):
                full.setdefault(k, v)
            errs = sorted(jsonschema.Draft202012Validator(self._schema).iter_errors(full), key=lambda e: list(e.path))
            return [f"{'/'.join(str(p) for p in e.path) or '(top level)'}: {e.message[:200]}" for e in errs[:10]]
        except Exception as e:
            return [f"(structure check unavailable: {e})"]

    # ---------- interface ----------
    def tools(self):
        return TOOLS

    def probe_tools(self):
        return [t for t in TOOLS if t["function"]["name"] == "run_scenario"]

    def new_state(self):
        return A5State()

    def _summary(self, rep):
        parts = [f"{len(rep.get(k, []))} {k}" for k in LIST_SECTIONS if rep.get(k)]
        return ", ".join(parts) or "empty model"

    def _stored(self, state, rep, what):
        state.rep = rep
        errs = self.schema_errors(rep)
        msg = f"Model {what}: {self._summary(rep)}."
        if errs:
            msg += ("\nStructural errors (location: message); these elements will be holes until fixed:\n"
                    + "\n".join(f"- {e}\n  hint: {HINTS['schema']}" for e in errs))
        else:
            msg += " Structure OK. Call check() to compile and run it."
        return msg

    def handle(self, state, name, args):
        try:
            if name == "submit_model":
                rep = _parse_json(args.get("model"))
                if not isinstance(rep, dict):
                    return f"{ERROR_PREFIX}: 'model' must be a JSON object (the whole world model)."
                state.log.append(("submit_model", {"keys": sorted(rep)}, "stored"))
                return self._stored(state, rep, "stored")
            if name == "patch_model":
                patch = args.get("patch")
                if isinstance(patch, str):
                    patch = _parse_json(patch)
                if not state.rep:
                    return f"{ERROR_PREFIX}: there is no model to patch yet; use submit_model first."
                if isinstance(patch, list):
                    try:
                        rep = apply_json_patch(state.rep, patch)
                        changed = [f"{p.get('op')} {p.get('path')}" for p in patch if isinstance(p, dict)]
                    except (ValueError, KeyError, IndexError, TypeError) as e:
                        return f"{ERROR_PREFIX}: patch not applied (nothing changed): {e}"
                elif isinstance(patch, dict):
                    try:
                        rep, changed = apply_fragment(state.rep, patch)
                    except ValueError as e:
                        return f"{ERROR_PREFIX}: patch not applied (nothing changed): {e}"
                else:
                    return f"{ERROR_PREFIX}: 'patch' must be a JSON Patch list or a fragment object."
                state.log.append(("patch_model", {"changes": changed[:20]}, "applied"))
                return self._stored(state, rep, "patched (" + "; ".join(changed[:10]) + ")")
            if name == "check":
                if not state.rep:
                    return f"{ERROR_PREFIX}: no model yet; use submit_model first."
                return self._check_text(state.rep)
            if name == "run_scenario":
                if not state.rep:
                    return f"{ERROR_PREFIX}: no model yet; use submit_model first."
                return self._run_text(state.rep, args)
        except Exception as e:
            return f"{ERROR_PREFIX}: {name} failed: {type(e).__name__}: {str(e)[:300]}"
        return f"{ERROR_PREFIX}: unknown tool {name!r}."

    def _check_text(self, rep):
        res = self.check_rep(rep)
        if "error" in res:
            return f"{ERROR_PREFIX}: {res['error']}"
        v = res["verdict"]
        out = [f"Checker: {'PASS' if v.get('pass') else 'FAIL'} ({len(v.get('failures', []))} failures)."]
        if res.get("filled"):
            out.append("Missing top-level fields were filled with empty defaults: " + ", ".join(res["filled"]))
        if v.get("failures"):
            out.append(format_failures(v["failures"]))
        if res.get("holes"):
            rv = res.get("reduced_verdict") or {}
            out.append("\nGraceful degradation: these broken parts were set aside as holes, and the rest was run:")
            out += [f"- hole: {h['element']} - {h['problem']}" for h in res["holes"][:20]]
            if rv.get("failures"):
                out.append("Remaining failures in the usable part:\n" + format_failures(rv["failures"], 15))
            v = rv or v
        claims = v.get("claims", {})
        if claims:
            out.append("Claims (holds, model value):")
            for cid, e in list(claims.items())[:40]:
                mv = f", model computes {e['model_value']}" if e.get("model_value") is not None else ""
                out.append(f"- {cid} [{e.get('scenario')}]: holds={e.get('holds')}{mv}")
        if v.get("sensitivity"):
            out.append("Sensitivity (conclusions that change across assumed ranges): "
                       + json.dumps(v["sensitivity"])[:1500])
        if v.get("info", {}).get("diagnosis_candidates"):
            out.append("Diagnosis candidates: " + json.dumps(v["info"]["diagnosis_candidates"])[:1500])
        out.append(f"Usable: {res.get('usable')}; executable: {res.get('executable')}.")
        return "\n".join(out)

    def _run_text(self, rep, args):
        sc = args.get("scenario")
        if isinstance(sc, str):
            sc = _parse_json(sc) if sc.strip().startswith("{") else {"id": sc}
        if not isinstance(sc, dict):
            return f"{ERROR_PREFIX}: 'scenario' must be a scenario object."
        if set(sc) <= {"id"} and sc.get("id"):
            known = {s.get("id"): s for s in rep.get("scenarios", []) if isinstance(s, dict)}
            if sc["id"] in known:
                sc = known[sc["id"]]
            elif sc["id"] != "base":
                return f"{ERROR_PREFIX}: no scenario {sc['id']!r} in the model (known: {', '.join(known) or 'base'})."
        claims = args.get("claims") or []
        if not isinstance(claims, list):
            return f"{ERROR_PREFIX}: 'claims' must be a list."
        res = self._exec({"mode": "run", "ast": rep, "scenario": sc, "claims": claims, "show": args.get("show")})
        if "error" in res:
            extra = ""
            if res.get("holes"):
                extra = "\nholes: " + "; ".join(h["element"] for h in res["holes"])
            return f"{ERROR_PREFIX}: {res['error']}{extra}\nhint: {HINTS['well_formed']}"
        lines = [f"Scenario {sc.get('id', 'probe_run')}: horizon {res['horizon']}"
                 + ("" if res.get("determinate") else " (WARNING: more than one trajectory; first shown)")]
        if res.get("holes"):
            lines.append("Run on the usable part; holes: " + "; ".join(h["element"] for h in res["holes"]))
        for t, row in res["trajectory"].items():
            lines.append(f"t={t}: " + ", ".join(f"{k}={v}" for k, v in row.items()))
        if res.get("violations"):
            lines.append("Constraint violations: " + json.dumps(res["violations"]))
        if res.get("claims"):
            lines.append("Claims: " + ", ".join(f"{k}={v}" for k, v in res["claims"].items()))
        text = "\n".join(lines)
        return text if len(text) < 7000 else text[:3000] + "\n[...]\n" + text[-3500:]

    def load(self, rep, given_gold=False):
        st = self.new_state()
        st.rep = copy.deepcopy(rep) if isinstance(rep, dict) else {}
        st.given_gold = given_gold
        return st

    def export(self, state):
        return state.rep

    def validity(self, state):
        if not state.rep:
            return {"usable": False, "executable": False, "holes": [], "errors": ["no model"]}
        res = self.check_rep(state.rep)
        if "error" in res:
            return {"usable": False, "executable": False, "holes": [], "errors": [res["error"]]}
        v = res.get("reduced_verdict") or res["verdict"]
        return {"usable": bool(res.get("usable")), "executable": bool(res.get("executable")),
                "holes": res.get("holes", []), "pass": bool(res["verdict"].get("pass")),
                "errors": [f"[{f['obligation']}] {f['explanation']}" for f in v.get("failures", [])][:30]}

    def render(self, state):
        return json.dumps(state.rep, indent=1)[:60000]

    def strip_answers(self, rep):
        """Claims and conclusions are answer-bearing: drop them, keep the assumption list (non-claim ids)."""
        rep = copy.deepcopy(rep)
        claim_ids = {c.get("id") for c in rep.get("claims", []) if isinstance(c, dict)}
        rep["claims"] = []
        ans = rep.get("answer") or {}
        rep["answer"] = {"assumptions": [i for i in ans.get("assumptions", []) or [] if i not in claim_ids]}
        return rep
