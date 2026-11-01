"""Genome renderer (spec §3, milestone M2): genome -> runnable arm with the v2 Arm interface.

`render(genome)` returns a `RenderedArm`, a subclass of the frozen v2 `tournament.arms.base.Arm`, so the v2 episode
machinery (`tournament.run.run_episode`, `tournament.episode.tool_loop`) can drive it unchanged. The arm is composed
from the genes:

  core            an adapter around a v2 arm (looked up through the v2 registry): representation storage, analysis,
                  execution and queries. `core.params.v2_arm` names it; otherwise core.kind maps to DEFAULT_V2_ARM.
  build_protocol  which construction tools the builder sees (one_shot / stepwise_checked / template_first /
                  sketch_refine / decompose); the extra tools are implemented here, on top of the core's own
                  checker and submit tool.
  tool_interface  error-report style, patch tools on/off, line-statement submission for JSON cores.
  query_layer     which of the core's query tools are offered (QUERY_MAP).
  guidance        the gene's text, then the examples gene, then the answering gene (V3-038; nothing when empty),
                  followed by protocol notes and the core arm's own guide.
  numeric         `limits()` for the episode runner; `check_every` drives stepwise checking.

The v2 code is imported read-only (rpp-core is appended to sys.path; nothing in it is modified). The episode state
is the core's own state object with one extra attribute, `v3` (draft parts, sketch, subsystems), so the core's
custom episode loops (G1C3, G1C1, A0+) run unchanged with this arm in place of the core.

Behavioural fidelity: a seed genome (one_shot, tool_arg submit, patch on, first_only errors, the core's queries) offers
exactly the core's tools and returns the core's results unchanged; only the guide gains the gene text in front.
"""
import copy
import json
import re
import sys

from v3.config import V2
from v3.genome import schema as S

if str(V2) not in sys.path:
    sys.path.append(str(V2))     # appended, so v3's own packages (v3, tests) are never shadowed by rpp-core's

from tournament.arms import ArmUnavailable, get_arm   # noqa: E402  (v2, read-only)
from tournament.arms.base import ERROR_PREFIX, Arm, ArmState   # noqa: E402

from v3.cores import CORE_ARMS, make_core   # noqa: E402  (v3-native plain-language cores: sectioned, ontology)

DEFAULT_V2_ARM = {"prose": "A0+", "equations": "G1C3", "probabilistic": "G1C1", "simulation": "A3", "rules": "A5",
                  "graph": "A2"}
V3_CORES = set(CORE_ARMS.values())   # cores built by v3.cores.make_core from genes.core.params (not v2 arms)

# v2 query tool -> the query kinds (schema.QUERIES) it serves. A query tool is offered when any of its kinds is in
# the genome's query_layer. Construction, patch and inspection tools (show, show_model, A5 check) are not queries
# and are governed by build_protocol / tool_interface instead.
QUERY_MAP = {
    "evaluate": {"trajectory", "explain", "sensitivity"},        # A2/G1C3: values static, over time, Monte Carlo
    "intervene": {"what_if", "compare", "sensitivity"},          # A2/G1C3: do() vs baseline, batched what-ifs
    "run_scenario": {"what_if", "compare", "sensitivity", "trajectory"},   # A3/G1C1: runs with overrides, series
}
# per-arm overrides of QUERY_MAP
ARM_QUERY_MAP = {
    "A5": {"run_scenario": {"what_if", "compare", "trajectory", "explain"}},   # trajectory + claim evaluation
    "ONT": {"lookup": {"explain"}, "path": {"what_if"}, "hierarchy": {"compare"}},   # one tool per query kind
    "JSN": {"lookup": {"explain"}, "search": {"compare"}},                           # run 5 (V3-042)
    "MD": {"lookup": {"explain"}, "search": {"compare"}},
}

# How each v2 core is driven. submit: the whole-model tool and its argument; text: the argument is program text
# (else a JSON object); patch: edit tools removed by patch=false; inspect: always offered; fence: the label of the
# core's fenced-block channel (custom episode); comment: line-comment marker of the text syntax; heading_parts: a
# stepwise part whose text does not start with a header gets its `section` as a header line (sectioned cores).
CORE_SPECS = {
    "G1C3": {"submit": "submit_graph", "arg": "source", "text": True, "patch": ("patch_graph",),
             "inspect": ("show",), "fence": "graph", "comment": "#"},
    "G1C1": {"submit": "submit_model", "arg": "source", "text": True, "patch": ("patch_model",),
             "inspect": ("show_model",), "fence": "model", "comment": "#"},
    "A3": {"submit": "write_model", "arg": "code", "text": True, "patch": ("patch_model",),
           "inspect": ("show_model",), "fence": None, "comment": "#"},
    "A5": {"submit": "submit_model", "arg": "model", "text": False, "patch": ("patch_model",),
           "inspect": ("check",), "fence": None, "comment": None},
    "A2": {"submit": "set_graph", "arg": "graph", "text": False,
           "patch": ("add_variable", "add_edge", "set_equation"), "inspect": ("show",), "fence": None,
           "comment": None},
    "A0+": {"submit": None, "arg": None, "text": True, "patch": (), "inspect": (), "fence": None, "comment": None},
    "SEC": {"submit": "write_model", "arg": "text", "text": True, "patch": (),
            "inspect": ("show_model",), "fence": None, "comment": None, "heading_parts": True},
    "ONT": {"submit": "write_model", "arg": "text", "text": True, "patch": (), "inspect": ("show_model",),
            "fence": None, "comment": None, "heading_parts": True},
    "JSN": {"submit": "write_model", "arg": "text", "text": True, "patch": (), "inspect": ("show_model",),
            "fence": None, "comment": None},
    "MD": {"submit": "write_model", "arg": "text", "text": True, "patch": (), "inspect": ("show_model",),
           "fence": None, "comment": None, "heading_parts": True},
}

V3_TOOLS = ("get_template", "sketch", "add_part", "add_subsystem", "check", "finish")
# Run 5 factor E for prose designs (no tools, so a turn budget would change nothing): a budget above the base gives one
# revision turn after the notes, announced in the guide (v3.episode._two_phase runs it)
PROSE_BASE_TURNS = 12
REVISION_NOTE = ("When you build notes from the description, after your first draft you will be asked once to check "
                 "them against the description and give a corrected, complete version.")
BUILD_ONLY = ("get_template", "sketch", "add_part", "add_subsystem", "finish")   # refused / hidden in given-gold mode
MAX_PARTS = 60
MAX_DRAFT = 60_000
SECTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_ .:-]{0,39}$")
LINE_RE = re.compile(r"\bline (\d+)")

GENERIC_HINTS = [
    (re.compile(r"undeclared|undefined|unknown variable", re.I), "declare the name as a variable, or fix its spelling"),
    (re.compile(r"syntax|unexpected|parse|unrecognised", re.I), "rewrite that line in the documented syntax"),
    (re.compile(r"no class Model|Model\.\w+ missing", re.I), "define class Model with init_state, step and observe"),
    (re.compile(r"not declared|declare", re.I), "add the missing declaration at the top of the model"),
    (re.compile(r"json", re.I), "give valid JSON for this section"),
    (re.compile(r"required|missing", re.I), "add the missing field or element"),
]


def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


def _tname(t):
    return t["function"]["name"]


def _as_args(args):
    if args is None:
        return {}
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return {}
        try:
            args = json.loads(s)
        except ValueError:
            return None
    return args if isinstance(args, dict) else None


# ============================================================================ problems (the v3 checker layer)
def _problem(msg, line=None, where=None, hint=None):
    return {"msg": str(msg), "line": line, "where": where, "hint": hint}


def _hint_for(p):
    if p.get("hint"):
        return p["hint"]
    for rx, h in GENERIC_HINTS:
        if rx.search(p["msg"]):
            return h
    return "fix this element, then check again"


def check_core(core_name, core, rep):
    """Run the core's own static checker on a representation (text or JSON object). Returns [problem]. Never
    executes builder code: static analysis only (A5: structural schema check; the full A5 checker stays the core's
    `check` tool). v3 plain-language cores: their own checker (the build criterion)."""
    try:
        if core_name in V3_CORES:
            return core.problems(rep if isinstance(rep, str) else "")
        if core_name == "G1C3" or (core_name not in CORE_SPECS and isinstance(rep, str) and hasattr(core, "_doc")):
            from tournament.arms.g1c3 import Doc
            doc = Doc(source=rep) if isinstance(rep, str) else Doc(graph=rep)
            out, seen_eq = [], set()
            for e in doc.errors:
                out.append(_problem(e["error"], e.get("line"), None, e.get("hint")))
                m = re.match(r"equation for '(\w+)'", e["error"])
                if m:
                    seen_eq.add(m.group(1))
            for n, w in doc.undefined():
                m = LINE_RE.search(w)
                out.append(_problem(f"undefined name '{n}' in {w}", int(m.group(1)) if m else None, w,
                                    "declare it as a variable (name : type = value or := expression) or fix the "
                                    "spelling"))
            for h in doc.cg.holes:
                m = re.match(r"equation for '(\w+)'", h["where"])
                if m and m.group(1) in seen_eq:
                    continue                     # already reported with its line by the document parser
                if "references undeclared" in h["error"] or "unknown variable(s)" in h["error"]:
                    continue                     # reported above as undefined names
                line = doc.eq_line.get(m.group(1)) if m else None
                out.append(_problem(f"{h['where']}: {h['error']}", line, h["where"], h.get("hint")))
            return out
        if core_name == "G1C1":
            from tournament.arms.g1c1 import analyse
            info = analyse(rep if isinstance(rep, str) else "")
            return [_problem(x, _line_of(x)) for x in info["errors"] + info["holes"]]
        if core_name == "A3":
            from tournament.arms.a3_sim import static_check
            holes, errors = static_check(rep if isinstance(rep, str) else "")
            return [_problem(x, _line_of(x)) for x in errors + holes]
        if core_name == "A5":
            if not isinstance(rep, dict) or not rep:
                return [_problem("the model is empty", hint="add sections such as fluents, actions, laws")]
            errs = [e for e in core.schema_errors(rep) if not e.startswith("(structure check unavailable")]
            return [_problem(e, None, e.split(":", 1)[0], None) for e in errs]
        if core_name == "A2":
            from tournament.arms.a2_causal import Compiled
            cg = Compiled(rep if isinstance(rep, dict) else {})
            return [_problem(f"{h['where']}: {h['error']}", None, h["where"], h.get("hint")) for h in cg.holes]
    except Exception as e:   # a checker failure is itself a problem, never an exception
        return [_problem(f"checker failed: {type(e).__name__}: {str(e)[:200]}")]
    return []


def _line_of(text):
    m = LINE_RE.search(text)
    return int(m.group(1)) if m else None


def format_problems(problems, style, title="CHECK", locate=None):
    """Render problems in the tool_interface.errors style. locate(line) -> extra location text (draft parts)."""
    if not problems:
        return f"{title}: no problems found."

    def loc(p):
        bits = []
        if p.get("line") is not None:
            bits.append(f"line {p['line']}" + (f" ({locate(p['line'])})" if locate and locate(p["line"]) else ""))
        elif p.get("where") and p["where"] not in p["msg"]:
            bits.append(p["where"])
        return (", ".join(bits) + ": ") if bits else ""
    if style == "first_only":
        more = len(problems) - 1
        return (f"{title}: PROBLEM: {loc(problems[0])}{problems[0]['msg']}"
                + (f"\n({more} more problem(s) not shown; fix this one and check again)" if more else ""))
    lines = [f"{title}: {len(problems)} PROBLEM(S):"]
    for k, p in enumerate(problems, 1):
        lines.append(f"  {k}. {loc(p)}{p['msg']}")
        if style == "all_with_fix_hints":
            lines.append(f"     hint: {_hint_for(p)}")
    return "\n".join(lines)


# ============================================================================ templates
TEXT_FIELD_LINES = {
    "G1C3": {
        "time_scale": "time steps=<N> dt=<D>                    # unit: <time unit per step>",
        "variables": "targets: <every quantity the question asks for>\n"
                     "<name> : <real|int|bool|category> [<unit>] := <expression>   # derived from ...",
        "units": "# units: give every variable its unit after the type, e.g. stock : real units = 500",
        "parameters": "<parameter> : real [<unit>] = <value>          # stated \"<quote>\"",
        "assumptions": "<assumed_input> : real ~ normal(<mean>, <sd>)    # assumed: <why> (<plausible range>)",
        "uncertainty": "<uncertain_input> : real ~ uniform(<lo>, <hi>)  # assumed range",
        "interventions": "whatif <name>: do <variable> = <value> ; targets <a, b>",
        "observations": "# observations: state each observed value as a parameter and compare with the baseline",
    },
    "G1C1": {
        "parameters": "PARAMS = {\n    \"<name>\": <value>,   # stated / derived / assumed: why\n}",
        "assumptions": "RULES = {\n    \"<switch_or_policy>\": <value>,   # stated / assumed\n}",
        "time_scale": "HORIZON = <steps>   # one step = <time unit>",
        "variables": "class Model:\n    def init_state(self, scenario, rng):\n        return {\"<variable>\": 0}\n\n"
                     "    def step(self, state, t, rng):\n        # update every variable for one step\n"
                     "        return state\n\n    def observe(self, state):\n        return dict(state)",
        "uncertainty": "# uncertainty: draw every uncertain quantity from rng inside step()/init_state()",
        "units": "# units: name keys with their unit, e.g. \"wait_min\"",
        "interventions": "# interventions: every what-if must be an override of a PARAMS/RULES name",
        "observations": "# observations: observe() returns every quantity the question asks about",
    },
    "A3": {
        "parameters": "P = {\"<name>\": <value>}   # every stated number",
        "time_scale": "# one step = <time unit>",
        "variables": "class Model:\n    def init_state(self, scenario):\n        return {\"<variable>\": 0}\n\n"
                     "    def step(self, state, t):\n        # advance from t to t+1\n        return state\n\n"
                     "    def observe(self, state):\n        return dict(state)",
        "assumptions": "# assumptions: one comment line per assumed value",
        "units": "# units: name keys with their unit",
        "uncertainty": "# uncertainty: use scenario-provided seeds",
        "interventions": "# interventions: read every what-if quantity from scenario[...]",
        "observations": "# observations: observe() returns every quantity the question asks about",
    },
}
JSON_FIELD_SECTIONS = {
    "A5": {"variables": ["sorts", "fluents", "statics"], "parameters": ["statics", "initial"],
           "assumptions": ["laws", "constraints"], "interventions": ["actions", "scenarios"],
           "observations": ["claims"], "time_scale": ["scenarios"], "units": [], "uncertainty": []},
    "A2": {"variables": ["variables"], "parameters": ["variables"], "assumptions": ["edges"],
           "interventions": ["equations"], "time_scale": ["time"], "units": [], "uncertainty": [],
           "observations": []},
}
JSON_SECTION_SKELETON = {
    "sorts": [{"name": "<sort>", "instances": ["<a>", "<b>"]}],
    "statics": [{"name": "<constant>", "type": "int", "value": 0}],
    "fluents": [{"name": "<state_variable>", "args": [], "type": "int", "range": [0, 100]}],
    "initial": [{"fluent": "<state_variable>", "args": [], "value": 0}],
    "actions": [{"name": "<event>", "params": {}, "effects": [{"fluent": "<state_variable>", "args": [], "add": "1"}]}],
    "laws": [{"id": "<law>", "params": {}, "when": "<condition>",
              "effects": [{"fluent": "<state_variable>", "args": [], "add": "1"}]}],
    "constraints": [{"id": "<constraint>", "expr": "<condition>"}],
    "scenarios": [{"id": "s1", "horizon": 10, "events": []}],
    "claims": [],
    "variables": [{"name": "<name>", "type": "real", "unit": "<unit>", "exogenous": True, "value": 0}],
    "edges": [{"from": "<cause>", "to": "<effect>", "sign": "+"}],
    "equations": {"<effect>": "<expression of its causes>"},
    "time": {"steps": 10, "dt": 1},
}


def build_template(core_name, spec, fields):
    fields = list(fields) or ["variables"]
    if core_name in TEXT_FIELD_LINES:
        table = TEXT_FIELD_LINES[core_name]
        order = ["time_scale", "parameters", "assumptions", "uncertainty", "variables", "units", "interventions",
                 "observations"] if core_name != "G1C3" else \
            ["time_scale", "variables", "parameters", "units", "assumptions", "uncertainty", "interventions",
             "observations"]
        body = [f"{spec.get('comment') or '#'} --- {f} ---\n{table[f]}" for f in order if f in fields and f in table]
        lab = spec.get("fence") or ""
        return ("TEMPLATE (fill every <...>; one section per schema field: " + ", ".join(fields) + ")\n"
                + f"```{lab}\n" + "\n\n".join(body) + "\n```")
    if core_name in JSON_FIELD_SECTIONS:
        secs = []
        for f in fields:
            for s in JSON_FIELD_SECTIONS[core_name].get(f, []):
                if s not in secs:
                    secs.append(s)
        skel = {s: JSON_SECTION_SKELETON[s] for s in secs if s in JSON_SECTION_SKELETON}
        return ("TEMPLATE (JSON; replace every <...>; sections chosen from the schema fields: " + ", ".join(fields)
                + ")\n" + json.dumps(skel, indent=1))
    return ("TEMPLATE: write one section per schema field: " + ", ".join(fields)
            + (f"; submit with {spec['submit']}" if spec.get("submit") else ""))


# ============================================================================ JSON drafts and line statements
def parse_statements(text):
    """Line statements for JSON cores: `section: <json element>` appends to a list section; `field = <json>` sets a
    field. Blank lines and lines starting with # are ignored. Returns (dict, [problem])."""
    out, probs = {}, []
    for no, raw in enumerate((text or "").replace("\r\n", "\n").split("\n"), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(:|=)\s*(.+)$", s)
        if not m:
            probs.append(_problem("not a statement", no, None, "write `section: {json element}` or "
                                                               "`field = json value`"))
            continue
        try:
            v = json.loads(m.group(3))
        except ValueError as e:
            probs.append(_problem(f"bad JSON after {m.group(1)}{m.group(2)}: {str(e)[:120]}", no))
            continue
        if m.group(2) == ":":
            cur = out.setdefault(m.group(1), [])
            if not isinstance(cur, list):
                probs.append(_problem(f"{m.group(1)} was set with '=' and cannot also take elements", no))
                continue
            cur.append(v)
        else:
            out[m.group(1)] = v
    return out, probs


def merge_json(base, frag):
    """Lists extend, dicts merge recursively, anything else replaces."""
    out = copy.deepcopy(base)
    for k, v in frag.items():
        if isinstance(out.get(k), list) and isinstance(v, list):
            out[k] = out[k] + copy.deepcopy(v)
        elif isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = merge_json(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ============================================================================ the rendered arm
class RenderedArm(Arm):
    """A genome rendered into a v2-compatible arm. Unknown attributes are forwarded to the core arm, so the core's
    own episode loop can run with this arm as `self`."""

    def __init__(self, genome, core_name, core):
        self.genome = copy.deepcopy(genome)
        self.genome["genes"] = S.upgrade(self.genome["genes"])      # pre-split genomes render unchanged (V3-038)
        self.genes = self.genome["genes"]
        self.core_name = core_name
        self._core = core
        self.name = f"v3:{genome.get('name') or genome.get('id') or S.genome_id(genome)}"
        self.genome_id = genome.get("id") or S.genome_id(genome)
        self.buildable = getattr(core, "buildable", False)
        self.executable = getattr(core, "executable", False)
        self.custom_episode = bool(getattr(core, "custom_episode", False))
        self.protocol = self.genes["build_protocol"]["kind"]
        self.ti = self.genes["tool_interface"]
        self.queries = set(self.genes["query_layer"])
        self.numeric = dict(self.genes["numeric"])
        self.prose = self.genes["core"]["kind"] == "prose"
        self.spec = CORE_SPECS.get(core_name) or self._infer_spec(core)
        self.v3_core = core_name in V3_CORES      # checks are the build criterion: always reported
        sub = self.spec.get("submit")
        self.line_statements = (self.ti["submit"] == "line_statements" and sub is not None
                                and not self.spec["text"])
        self.has_draft = self.protocol in ("stepwise_checked", "decompose") and sub is not None
        self.mutating_tools = tuple(dict.fromkeys(tuple(getattr(core, "mutating_tools", ())) + BUILD_ONLY))
        qmap = dict(QUERY_MAP)
        qmap.update(ARM_QUERY_MAP.get(core_name, {}))
        self.query_map = qmap

    def __getattr__(self, item):
        if item.startswith("__") or item in ("_core", "genome", "spec"):
            raise AttributeError(item)
        return getattr(self.__dict__["_core"], item)

    @staticmethod
    def _infer_spec(core):
        """Spec for a v2 arm not listed in CORE_SPECS: the first non-patch mutating tool is the whole submit."""
        mut = list(getattr(core, "mutating_tools", ()) or ())
        tools = {_tname(t): t for t in core.tools()}
        patch = tuple(m for m in mut if "patch" in m or m.startswith("add_") or m.startswith("set_eq"))
        sub = next((m for m in mut if m not in patch and m in tools), None)
        arg, text = None, True
        if sub:
            props = tools[sub]["function"].get("parameters", {}).get("properties", {})
            arg = next((k for k in ("source", "code", "text", "program") if k in props), None)
            if arg is None:
                arg, text = next(iter(props), None), False
        probe = {_tname(t) for t in core.probe_tools()}
        inspect = tuple(n for n in tools if n not in mut and n not in probe)
        return {"submit": sub, "arg": arg, "text": text, "patch": patch, "inspect": inspect, "fence": None,
                "comment": "#" if text else None}

    # ---------------------------------------------------------------- numeric
    def limits(self):
        out = {}
        for k, v in self.numeric.items():
            out[k] = int(round(v)) if k in ("max_turns", "retries", "check_every", "samples") else float(v)
        return out

    # ---------------------------------------------------------------- guidance
    def guide(self):
        g = self.genes["guidance"]
        parts = [g["text"].strip()]
        ex = [str(e).strip() for e in list(g.get("examples", [])) + list(self.genes["examples"]["items"])
              if str(e).strip()]
        if ex:
            parts.append("Examples:\n" + "\n\n".join(f"Example {k}:\n{e}" for k, e in enumerate(ex, 1)))
        ans = self.genes["answering"]["text"].strip()
        if ans:                                  # V3-038: how to answer questions from the representation
            parts.append("## When you answer questions from the representation\n" + ans)
        notes = self._protocol_notes()
        if notes:
            parts.append("## How this design builds the model (overrides the guide below where they differ)\n"
                         + "\n".join(f"- {n}" for n in notes))
        core_guide = (self._core.guide() or "").strip()
        if core_guide:
            parts.append(core_guide)
        return "\n\n".join(p for p in parts if p)

    def _protocol_notes(self):
        n, sub = [], self.spec.get("submit")
        if self.prose:
            if self.protocol == "sketch_refine":
                n.append("First write a short outline of the system, then refine it into your answer.")
            elif self.protocol == "decompose":
                n.append("Reason about each subsystem separately, then combine them into your answer.")
            if self.limits()["max_turns"] > PROSE_BASE_TURNS:     # run 5 factor E for prose (code review)
                n.append(REVISION_NOTE)
            return n
        n.append(f"Tool budget: at most {self.limits()['max_turns']} tool calls in this phase.")  # factor E, visible
        if self.protocol == "stepwise_checked":
            n.append(f"Do not submit the whole model at once ({sub} is not available). Build it in parts: "
                     f"add_part(section, text) adds one section to a draft; every {self.limits()['check_every']} "
                     f"part(s) the draft is checked and problems come back at once. check() checks the draft now; "
                     f"finish() submits the assembled draft.")
        elif self.protocol == "decompose":
            n.append(f"Do not submit the whole model at once ({sub} is not available). Write it subsystem by "
                     f"subsystem: add_subsystem(name, text) per subsystem (the same name replaces it); check() "
                     f"checks the merged draft; finish() merges the subsystems and submits them.")
        elif self.protocol == "template_first":
            n.append("Call get_template() first: it returns a skeleton of the model with one section per required "
                     "field. Fill it in, then build as described below.")
        elif self.protocol == "sketch_refine":
            n.append("Call sketch(outline) first with a free-text outline of the mechanisms and quantities; it is "
                     "shown back with every check result, then build as described below.")
        if self.protocol == "stepwise_checked" and self.has_draft and self.spec.get("heading_parts"):
            n.append("For this design `section` is the section header (e.g. " + ", ".join(
                (self.genes["core"].get("params") or {}).get("headers", [])[:2] or ["ENTITIES"]) + ") and `text` "
                "is the lines under it; the header line is added for you.")
        if self.protocol == "decompose" and self.has_draft and self.spec.get("heading_parts"):
            n.append("Each subsystem's text is a piece of the document with its own `## HEADER` lines; the same "
                     "header in several subsystems is merged.")
        if self.has_draft and self.spec.get("fence"):
            n.append(f"A ```{self.spec['fence']} block in a response without a tool call is added to the draft as "
                     f"a part, not submitted.")
        if self.has_draft and not self.spec["text"]:
            n.append(("Each part's" if self.protocol == "stepwise_checked" else "Each subsystem's") + " text is "
                     + ("line statements (`section: {json element}` per line, or `field = json`)"
                        if self.line_statements else
                        "the JSON value of the top-level section it names" if self.protocol == "stepwise_checked"
                        else "a JSON object of top-level sections (lists are concatenated on finish)") + ".")
        elif self.line_statements:
            n.append(f"{sub} takes `statements`: plain text, one statement per line, `section: {{json element}}` "
                     f"to add an element to a list section, `field = json value` to set a field.")
        if not self.ti["patch"] and self.spec.get("patch"):
            n.append("Patch tools (" + ", ".join(self.spec["patch"]) + ") are not available: fix a model by "
                     "submitting it again" + (" (finish again after changing parts)" if self.has_draft else "") + ".")
        if self.ti["errors"] == "first_only" and self.protocol != "one_shot":
            n.append("Checks report the first problem only; fix it and check again.")
        elif self.ti["errors"] in ("all_at_once", "all_with_fix_hints"):
            n.append("Checks report every problem at once, with line references"
                     + (" and a one-line fix hint" if self.ti["errors"] == "all_with_fix_hints" else "") + ".")
        return n

    # ---------------------------------------------------------------- tools
    def _v3_tool_defs(self):
        out = []
        txt = ("line statements (`section: {json element}` per line)" if self.line_statements else
               ("program text" if self.spec["text"] else "JSON value of a top-level section"))
        if self.protocol == "template_first" and not self.prose:
            out.append(_tool("get_template", "Return a skeleton of the model for this design (one section per "
                                             "required field), in the core's syntax.", {}))
        if self.protocol == "sketch_refine" and not self.prose:
            out.append(_tool("sketch", "Store a free-text outline of the model (mechanisms, quantities, what-ifs). "
                                       "It is shown back with every check result. Calling it again replaces it.",
                             {"outline": {"type": "string"}}, ["outline"]))
        if self.protocol == "stepwise_checked" and self.has_draft:
            out.append(_tool("add_part", f"Add one part to the draft model: `section` names it, `text` is its "
                                         f"{txt}. replace=true replaces every earlier part with the same section. "
                                         f"The draft is checked every {self.limits()['check_every']} part(s).",
                             {"section": {"type": "string"}, "text": {"type": "string"},
                              "replace": {"type": "boolean"}}, ["section", "text"]))
        if self.protocol == "decompose" and self.has_draft:
            out.append(_tool("add_subsystem", f"Add (or replace, by name) one subsystem of the model: `text` is "
                                              f"its {txt}. Subsystems are merged in order on finish.",
                             {"name": {"type": "string"}, "text": {"type": "string"}}, ["name", "text"]))
        if self.protocol in ("stepwise_checked", "decompose", "sketch_refine") and not self.prose:
            out.append(_tool("check", "Check the " + ("draft" if self.has_draft else "current model")
                             + " with the core's checker and list the problems.", {}))
        if self.has_draft:
            out.append(_tool("finish", "Assemble the draft and submit it as the model (through "
                                       f"{self.spec['submit']}); returns the submission result.", {}))
        return out

    def _submit_def(self, t):
        if not self.line_statements:
            return t
        t = copy.deepcopy(t)
        f = t["function"]
        f["description"] = ("Submit the whole model as plain line statements: `section: {json element}` adds an "
                            "element to a list section, `field = json value` sets a field; one per line. "
                            + f.get("description", ""))[:900]
        f["parameters"] = {"type": "object", "properties": {"statements": {"type": "string"}},
                           "required": ["statements"]}
        return t

    def _keep_core_tool(self, name):
        sp = self.spec
        if name == sp.get("submit"):
            return not self.has_draft
        if name in sp.get("patch", ()):
            return bool(self.ti["patch"])
        if name in sp.get("inspect", ()):
            return True
        kinds = self.query_map.get(name)
        if kinds is not None:
            return bool(kinds & self.queries)
        mut = getattr(self._core, "mutating_tools", ())
        if name in mut:               # other construction tools of an unlisted core
            return not self.has_draft
        return True                   # unclassified non-query tool: pass through

    def _filter(self, core_tools):
        out = []
        for t in core_tools:
            n = _tname(t)
            if n == "check" and n in self.spec.get("inspect", ()) and any(_tname(x) == "check"
                                                                            for x in self._v3_tool_defs()):
                continue              # the v3 check replaces a same-named core check in the tool list (A5)
            if self._keep_core_tool(n):
                out.append(self._submit_def(t) if n == self.spec.get("submit") else t)
        return out

    def tools(self):
        if self.prose:
            return []
        return self._v3_tool_defs() + self._filter(self._core.tools())

    def current_tools(self, state):
        if self.prose:
            return []
        try:
            core_tools = self._core.current_tools(state)
        except Exception:
            core_tools = self._core.tools()
        v3 = self._v3_tool_defs()
        # the template-first gate applies to building only: in the answer phase (given_gold) get_template is a
        # build-only tool, so gating there left template-first designs with no read tools (code review 2026-10-29)
        if self.protocol == "template_first" and not self._v3(state).get("template_seen") \
                and not getattr(state, "given_gold", False):
            return [t for t in v3 if _tname(t) == "get_template"]
        return v3 + self._filter(core_tools)

    def probe_tools(self):
        return self._core.probe_tools()

    def current_probe_tools(self, state):
        return self._core.current_probe_tools(state)

    # ---------------------------------------------------------------- state
    @staticmethod
    def _v3(state):
        d = getattr(state, "v3", None)
        if not isinstance(d, dict):
            d = {"parts": [], "subsystems": [], "sketch": None, "template_seen": False, "n_since_check": 0,
                 "finished": 0}
            try:
                state.v3 = d
            except Exception:
                pass
        return d

    def new_state(self):
        st = self._core.new_state()
        self._v3(st)
        return st

    def _final(self, state):
        """The state as it stands at the end of an episode: a draft that was never (re)submitted with finish() is
        submitted as it is, on a copy, so the checker judges what was written (V3-009: builders often ended with
        DONE without calling finish, and the whole draft was lost). The live state is never changed."""
        d = self._v3(state)
        if getattr(state, "given_gold", False) or not self.has_draft or not self._draft_items(state) \
                or (d.get("finished") and not d.get("dirty")):
            return state, False
        st = copy.deepcopy(state)
        out = self._t_finish(st, {})
        return (st, True) if not str(out).startswith(ERROR_PREFIX) else (state, False)

    def export(self, state):
        state, auto = self._final(state)
        rep = self._core.export(state)
        extra = {k: copy.deepcopy(v) for k, v in self._v3(state).items()
                 if k in ("parts", "subsystems", "sketch") and v}
        if auto:
            extra["auto_finished"] = True
        if extra and isinstance(rep, dict):
            rep = dict(rep)
            rep["_v3"] = extra
        return rep

    def load(self, rep, given_gold=False):
        extra = {}
        if isinstance(rep, dict) and "_v3" in rep:
            rep = dict(rep)
            extra = rep.pop("_v3") or {}
        st = self._core.load(rep, given_gold=given_gold)
        d = self._v3(st)
        for k in ("parts", "subsystems", "sketch"):
            if k in extra:
                d[k] = copy.deepcopy(extra[k])
        return st

    def validity(self, state):
        return self._core.validity(self._final(state)[0])

    def render(self, state):
        return self._core.render(state)

    def strip_answers(self, rep):
        rep = dict(rep) if isinstance(rep, dict) else rep
        if isinstance(rep, dict):
            rep.pop("_v3", None)
        return self._core.strip_answers(rep)

    # ---------------------------------------------------------------- episode
    def run_episode(self, ep, state, messages):
        """The core's own episode loop (A0+ reviews, G1C3/G1C1 fenced-block channels) with this arm as `self`, so
        it sees the rendered tools, handle and guide."""
        return type(self._core).run_episode(self, ep, state, messages)

    # ---------------------------------------------------------------- dispatch
    def handle(self, state, name, args):
        try:
            out = self._handle(state, name, args)
        except Exception as e:   # never raise
            out = f"{ERROR_PREFIX}: internal {type(e).__name__}: {str(e)[:300]} (hint: check the arguments)"
        if not isinstance(out, str):
            out = json.dumps(out, default=str)
        return out

    def _handle(self, state, name, args):
        if self.prose:
            return f"{ERROR_PREFIX}: this arm has no tools."
        args = _as_args(args)
        if args is None:
            return f"{ERROR_PREFIX}: arguments must be a JSON object (hint: see the {name} schema)"
        if not isinstance(name, str) or not name:
            return f"{ERROR_PREFIX}: missing tool name"
        offered = {_tname(t) for t in self.tools()}
        gold = bool(getattr(state, "given_gold", False))
        if name in {_tname(t) for t in self._v3_tool_defs()}:
            if gold and name in BUILD_ONLY:
                return f"{ERROR_PREFIX}: the representation is given in this mode; it cannot be rebuilt."
            return getattr(self, "_t_" + name)(state, args)
        core_names = {_tname(t) for t in self._core.tools()}
        if name in core_names:
            if name == self.spec.get("submit") and self.has_draft and not gold:
                return self._implicit_part(state, args)     # fenced block channel, or a stray whole submit
            if name not in offered:
                return (f"{ERROR_PREFIX}: tool '{name}' is not available in this design (hint: available: "
                        + ", ".join(sorted(offered)) + ")")
            return self._core_call(state, name, args)
        return (f"{ERROR_PREFIX}: unknown tool '{name}' (hint: available: " + ", ".join(sorted(offered)) + ")")

    def _core_call(self, state, name, args):
        if name == self.spec.get("submit") and self.line_statements:
            st = args.get("statements")
            if not isinstance(st, str) or not st.strip():
                return f"{ERROR_PREFIX}: {name} needs `statements` (plain text, one statement per line)"
            model, probs = parse_statements(st)
            if probs:
                return (f"{ERROR_PREFIX}: the statements were not submitted\n"
                        + format_problems(probs, self.ti["errors"], "STATEMENTS"))
            args = {self.spec["arg"]: model}
        out = self._core.handle(state, name, args)
        construct = name == self.spec.get("submit") or name in self.spec.get("patch", ()) or name == "check"
        if construct and not out.startswith(ERROR_PREFIX) and self.v3_core:
            out += "\n" + format_problems(check_core(self.core_name, self._core, self._stored_rep(state)),
                                          self.ti["errors"], "CHECK")
        elif construct and not out.startswith(ERROR_PREFIX):
            if self.ti["errors"] != "first_only" and name != "check":
                probs = check_core(self.core_name, self._core, self._stored_rep(state))
                if probs:
                    out += "\n" + format_problems(probs, self.ti["errors"], "PROBLEMS")
        if construct:
            out = self._with_sketch(state, out)
        return out

    def _stored_rep(self, state):
        if self.spec["text"]:
            if self.core_name == "G1C3" or hasattr(getattr(state, "doc", None), "source"):
                doc = getattr(state, "doc", None)
                if doc is not None and doc.source is not None:
                    return doc.source
                return getattr(doc, "graph", {}) if doc is not None else ""
            r = state.rep if isinstance(state.rep, dict) else {}
            return r.get(self.spec["arg"]) if isinstance(r.get(self.spec["arg"]), str) else ""
        return state.rep

    def _with_sketch(self, state, out):
        sk = self._v3(state).get("sketch")
        if self.protocol == "sketch_refine" and sk:
            out += "\nYOUR SKETCH (outline):\n" + sk
        return out

    def _implicit_part(self, state, args):
        """A whole-model submit while the protocol builds a draft (e.g. a fenced block without a tool call): the
        text becomes one more part / subsystem."""
        text = args.get(self.spec["arg"]) if self.spec.get("arg") else None
        if not isinstance(text, str) or not text.strip():
            text = getattr(state, "pending_block", None)
        if isinstance(text, dict):
            text = json.dumps(text)
        if not isinstance(text, str) or not text.strip():
            return (f"{ERROR_PREFIX}: {self.spec['submit']} is not available in this design (hint: use "
                    + ("add_part" if self.protocol == "stepwise_checked" else "add_subsystem") + " and finish)")
        d = self._v3(state)
        if self.protocol == "stepwise_checked":
            return self._t_add_part(state, {"section": f"block{len(d['parts']) + 1}", "text": text})
        return self._t_add_subsystem(state, {"name": f"block{len(d['subsystems']) + 1}", "text": text})

    # ---------------------------------------------------------------- v3 tools
    def _t_get_template(self, state, args):
        self._v3(state)["template_seen"] = True
        if self.v3_core:
            return self._core.template()
        return build_template(self.core_name, self.spec, self.genes["schema"].get("fields", []))

    def _t_sketch(self, state, args):
        o = args.get("outline")
        if not isinstance(o, str) or not o.strip():
            return f"{ERROR_PREFIX}: sketch needs a non-empty `outline` (free text)"
        self._v3(state)["sketch"] = o.strip()[:4000]
        return f"sketch stored ({len(o.strip().splitlines())} lines). It will be shown with every check result."

    def _part_text_ok(self, text, label):
        if not isinstance(text, str) or not text.strip():
            return f"{ERROR_PREFIX}: {label} needs a non-empty `text`"
        if len(text) > MAX_DRAFT:
            return f"{ERROR_PREFIX}: text too long ({len(text)} chars, max {MAX_DRAFT})"
        if not self.spec["text"]:
            _, probs = self._json_part(text, "section")
            if probs:
                return (f"{ERROR_PREFIX}: the part was not added\n"
                        + format_problems(probs, self.ti["errors"], "PART"))
        return None

    def _json_part(self, text, section):
        if self.line_statements:
            return parse_statements(text)
        try:
            v = json.loads(text)
        except ValueError as e:
            return None, [_problem(f"bad JSON: {str(e)[:150]}", hint="give the JSON value of the section")]
        return v, []

    def _t_add_part(self, state, args):
        sec, text = args.get("section"), args.get("text")
        if not isinstance(sec, str) or not SECTION_RE.match(sec.strip() or "-"):
            return f"{ERROR_PREFIX}: add_part needs a short `section` name (letters, digits, _ . : -)"
        sec = sec.strip()
        bad = self._part_text_ok(text, "add_part")
        if bad:
            return bad
        d = self._v3(state)
        if args.get("replace"):
            d["parts"] = [p for p in d["parts"] if p["section"] != sec]
        if len(d["parts"]) >= MAX_PARTS:
            return f"{ERROR_PREFIX}: too many parts ({MAX_PARTS}); replace parts or finish"
        d["parts"].append({"section": sec, "text": text})
        d["dirty"] = True
        d["n_since_check"] = d.get("n_since_check", 0) + 1
        head = f"part {len(d['parts'])} added (section {sec!r}, {len(text.splitlines())} lines)."
        every = max(1, self.limits()["check_every"])
        if d["n_since_check"] >= every:
            d["n_since_check"] = 0
            return head + "\n" + self._check_draft(state)
        return head + f" Next automatic check after {every - d['n_since_check']} more part(s)."

    def _t_add_subsystem(self, state, args):
        nm, text = args.get("name"), args.get("text")
        if not isinstance(nm, str) or not SECTION_RE.match(nm.strip() or "-"):
            return f"{ERROR_PREFIX}: add_subsystem needs a short `name` (letters, digits, _ . : -)"
        nm = nm.strip()
        bad = self._part_text_ok(text, "add_subsystem")
        if bad:
            return bad
        d = self._v3(state)
        replaced = any(s["section"] == nm for s in d["subsystems"])
        if replaced:
            d["subsystems"] = [s if s["section"] != nm else {"section": nm, "text": text} for s in d["subsystems"]]
        else:
            if len(d["subsystems"]) >= MAX_PARTS:
                return f"{ERROR_PREFIX}: too many subsystems ({MAX_PARTS})"
            d["subsystems"].append({"section": nm, "text": text})
        d["dirty"] = True
        return (f"subsystem {nm!r} {'replaced' if replaced else 'added'} ({len(d['subsystems'])} subsystem(s): "
                + ", ".join(s["section"] for s in d["subsystems"]) + "). Call finish() to merge and submit.")

    def _draft_items(self, state):
        d = self._v3(state)
        return d["parts"] if self.protocol == "stepwise_checked" else d["subsystems"]

    def _assemble(self, state):
        """Returns (draft, locate, problems): draft text (text cores) or dict (JSON cores); locate maps a draft line
        to its part."""
        items = self._draft_items(state)
        if self.spec["text"]:
            lines, where = [], []
            for k, it in enumerate(items, 1):
                if self.protocol == "decompose" and self.spec.get("comment"):
                    lines.append(f"{self.spec['comment']} subsystem: {it['section']}")
                    where.append(None)
                if self.protocol == "stepwise_checked" and self.spec.get("heading_parts"):
                    first = next((ln for ln in it["text"].splitlines() if ln.strip()), "")
                    if self._core.heading_of(first) is None:
                        lines.append(f"## {it['section']}")
                        where.append((k, it["section"], 0))
                for j, ln in enumerate(it["text"].replace("\r\n", "\n").split("\n"), 1):
                    lines.append(ln)
                    where.append((k, it["section"], j))
            text = "\n".join(lines)

            def locate(no):
                if isinstance(no, int) and 1 <= no <= len(where) and where[no - 1]:
                    k, s, j = where[no - 1]
                    if j == 0:
                        return f"part {k} '{s}' header"
                    return f"part {k} '{s}' line {j}" if self.protocol == "stepwise_checked" \
                        else f"subsystem '{s}' line {j}"
                return ""
            return text, locate, []
        model, probs = {}, []
        for k, it in enumerate(items, 1):
            v, pp = self._json_part(it["text"], it["section"])
            probs += [dict(p, where=f"{it['section']}" + (f" line {p['line']}" if p.get("line") else ""),
                           line=None) for p in pp]
            if v is None:
                continue
            if self.line_statements or self.protocol == "decompose":
                if isinstance(v, dict):
                    model = merge_json(model, v)
                else:
                    probs.append(_problem(f"{it['section']}: expected a JSON object of sections", None,
                                          it["section"]))
            else:
                model = merge_json(model, {it["section"]: v})
        return model, None, probs

    def _check_draft(self, state):
        if not self._draft_items(state):
            return "CHECK: the draft is empty."
        draft, locate, probs = self._assemble(state)
        probs = probs + check_core(self.core_name, self._core, draft)
        return self._with_sketch(state, format_problems(probs, self.ti["errors"], "CHECK", locate))

    def _t_check(self, state, args):
        if self.has_draft and self._draft_items(state) and self._v3(state).get("dirty", True):
            return self._check_draft(state)
        rep = self._stored_rep(state)
        if not rep:
            return self._with_sketch(state, "CHECK: nothing built yet.")
        if self.core_name == "A5":   # a stored A5 model gets the core's own full checker (compile and run)
            return self._with_sketch(state, self._core.handle(state, "check", {}))
        probs = check_core(self.core_name, self._core, rep)
        return self._with_sketch(state, format_problems(probs, self.ti["errors"], "CHECK"))

    def _t_finish(self, state, args):
        if getattr(state, "given_gold", False):
            return f"{ERROR_PREFIX}: the representation is given in this mode; it cannot be rebuilt."
        if not self._draft_items(state):
            return (f"{ERROR_PREFIX}: the draft is empty (hint: "
                    + ("add_part" if self.protocol == "stepwise_checked" else "add_subsystem") + " first)")
        draft, locate, probs = self._assemble(state)
        if probs:   # JSON that cannot be assembled is not submitted
            return (f"{ERROR_PREFIX}: the draft cannot be assembled\n"
                    + format_problems(probs, self.ti["errors"], "DRAFT", locate))
        out = self._core.handle(state, self.spec["submit"], {self.spec["arg"]: draft})
        d = self._v3(state)
        d["finished"] = d.get("finished", 0) + 1
        d["dirty"] = False
        n = len(self._draft_items(state))
        res = f"draft of {n} {'part' if self.protocol == 'stepwise_checked' else 'subsystem'}(s) submitted.\n" + out
        if self.v3_core and not out.startswith(ERROR_PREFIX):
            res += "\n" + format_problems(check_core(self.core_name, self._core, draft), self.ti["errors"], "CHECK",
                                          locate)
        elif self.ti["errors"] != "first_only" and not out.startswith(ERROR_PREFIX):
            pp = check_core(self.core_name, self._core, draft)
            if pp:
                res += "\n" + format_problems(pp, self.ti["errors"], "PROBLEMS", locate)
        return self._with_sketch(state, res) if not res.startswith(ERROR_PREFIX) else res


# ============================================================================ entry point
def core_arm_name(genome):
    core = genome["genes"]["core"]
    if core["kind"] in CORE_ARMS:
        return CORE_ARMS[core["kind"]]
    return (core.get("params") or {}).get("v2_arm") or DEFAULT_V2_ARM[core["kind"]]


def render(genome):
    """Genome -> RenderedArm. Raises GenomeError for a malformed or incompatible genome, or an unknown v2 arm."""
    S.validate(genome)
    bad = S.incompatibilities(genome)
    if bad:
        raise S.GenomeError("incompatible genome: " + "; ".join(bad))
    name = core_arm_name(genome)
    try:
        core = make_core(name, genome["genes"]["core"]["params"]) if name in V3_CORES else get_arm(name)
    except ArmUnavailable as e:
        raise S.GenomeError(f"core v2 arm {name!r} is not available: {e}") from e
    if genome["genes"]["core"]["kind"] in ("probabilistic", "simulation"):
        from v3 import sandbox
        sandbox.install()           # V3-043: model-written code runs in Docker, never in a host subprocess
    arm = RenderedArm(genome, name, core)
    proto = genome["genes"]["build_protocol"]["kind"]
    if proto in ("stepwise_checked", "decompose") and not arm.prose and not arm.spec.get("submit"):
        raise S.GenomeError(f"protocol {proto} needs a whole-model submit tool; v2 arm {name} has none")
    return arm
