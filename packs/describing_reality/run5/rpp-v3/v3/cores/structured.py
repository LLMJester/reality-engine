"""Cores "json" and "markdown" (run 5 founders JSON and MARKDOWN, V3-042): checked structured notes.

Like the sectioned and ontology cores, the notes are text the builder writes with write_model; they count as built
only when the core's checker reports no problems (the builder sees the problems with line numbers and fixes them
within its turn budget, as for every checked core). In the answer phase the notes are shown in full and two read
tools search them: lookup (one name or path) and search (words anywhere).

json  genes.core.params: keys (3-10 required top-level keys, lower case), max_depth (2-8 levels of nesting)
  J1 the notes are not empty
  J2 they parse as one JSON object (a single surrounding ``` fence is ignored); the error gives line and column
  J3 every required key is present and not empty; no other top-level key
  J4 nesting is at most max_depth levels
  J5 no unfilled template placeholder (`<...>`)

markdown  genes.core.params: headers (3-10 required section headers), tables (the headers whose content must be
  Markdown tables)
  M1 the notes are not empty
  M2 a header line (`## NAME`) names one of the genome's headers; no text before the first header
  M3 every other line is a table row (`| a | b |`) or a bullet (`- ...`)
  M4 a table starts with a header row and a separator row (`|---|---|`); every row has as many cells as its header
  M5 every header is present with at least one element; a table header holds at least one table with a data row
  M6 no unfilled template placeholder (`<...>`)
"""
import json
import re

from v3.cores.text import heading_of, numbered, problem, section_problems, sort_problems
from v3.genome import schema as S

from tournament.arms.a1_prose import A1                     # noqa: E402  (v2, read-only)
from tournament.arms.base import ERROR_PREFIX, Arm, ArmState   # noqa: E402

MAX_TEXT = 40_000
NAME_FIELDS = ("name", "id", "label", "thing", "entity", "part", "from", "to", "what")   # JSON lookup by name
MAX_OUT = 4_000
MAX_HITS = 40
# a template placeholder: <...> or <word ...> (starts and ends with a letter), so comparisons such as
# "T1 < 10% and P1 > 50" are not placeholders (code review, 2026-10-29)
PLACEHOLDER_RE = re.compile(r"<(?:\.\.\.|(?![^<>\n]*\b(?:and|or)\b)[A-Za-z_][A-Za-z_ /-]{0,38}[A-Za-z_])>")
FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*)\n\s*```\s*$", re.S)
WORD_RE = re.compile(r"[a-z0-9]+")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEP_CELL_RE = re.compile(r"^\s*:?-{3,}:?\s*$")


def _tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


def _words(s):
    return WORD_RE.findall(str(s).lower())


def _matches(hay, words):
    h = set(_words(hay))
    return bool(words) and all(w in h for w in words)


def _clip(s, n=MAX_OUT):
    return s if len(s) <= n else s[:n] + f"\n(... {len(s) - n} more characters; narrow the query)"


class _NotesCore(Arm):
    """write_model / show_model / lookup / search over checked text notes."""
    buildable = True
    executable = False
    mutating_tools = ("write_model",)
    QUERY_TOOLS = ("lookup", "search")
    KIND = "notes"

    def tools(self):
        return [
            _tool("write_model", f"Write the whole {self.KIND} (replaces any earlier version). The result lists the "
                                 "check problems with line numbers.", {"text": {"type": "string"}}, ["text"]),
            _tool("show_model", f"Show the current {self.KIND} with line numbers."),
            _tool("lookup", self.LOOKUP_DESC, {"term": {"type": "string"}}, ["term"]),
            _tool("search", "Every line or entry that contains all of the given words.",
                  {"text": {"type": "string"}}, ["text"]),
        ]

    def new_state(self):
        st = ArmState()
        st.rep = {"text": ""}
        st.versions = 0
        return st

    def heading_of(self, line):
        return None

    def handle(self, state, name, args):
        text = state.rep.get("text", "") if isinstance(state.rep, dict) else ""
        args = args or {}
        if name == "write_model":
            t = args.get("text")
            if not isinstance(t, str) or not t.strip():
                return f"{ERROR_PREFIX}: write_model needs a non-empty 'text' string (the whole {self.KIND})."
            state.rep = {"text": t[:MAX_TEXT]}
            state.versions = getattr(state, "versions", 0) + 1
            state.log.append((name, {"chars": len(t)}, "saved"))
            n = len(self.problems(state.rep["text"]))
            return (f"{self.KIND.capitalize()} saved (version {state.versions}): {self.summary(state.rep['text'])}; "
                    f"{n} check problem(s)." + ("\n(text truncated to 40000 characters)" if len(t) > MAX_TEXT else ""))
        if name == "show_model":
            return numbered(text) if text else f"{ERROR_PREFIX}: nothing has been written yet; use write_model."
        if name in self.QUERY_TOOLS:
            if not text.strip():
                return f"{ERROR_PREFIX}: nothing has been written yet; use write_model."
            q = args.get("term") if name == "lookup" else args.get("text")
            if not isinstance(q, str) or not q.strip():
                return f"{ERROR_PREFIX}: {name} needs a non-empty string."
            out = self.lookup(text, q) if name == "lookup" else self.search(text, q)
            return _clip(out)
        return f"{ERROR_PREFIX}: unknown tool {name!r}."

    def load(self, rep, given_gold=False):
        st = self.new_state()
        if isinstance(rep, str):
            rep = {"text": rep}
        st.rep = {"text": (rep or {}).get("text", "") if isinstance(rep, dict) else ""}
        st.versions = 1 if st.rep["text"] else 0
        st.given_gold = given_gold
        return st

    def export(self, state):
        return {"text": state.rep.get("text", "")}

    def validity(self, state):
        text = state.rep.get("text", "")
        probs = self.problems(text) if text.strip() else [problem(f"no {self.KIND} written")]
        ok = bool(text.strip()) and not probs
        return {"usable": ok, "executable": ok, "checked": ok,
                "holes": [{"line": p["line"], "problem": p["msg"]} for p in probs],
                "errors": [] if ok else [f"{len(probs)} checker problem(s)"],
                "stats": {"problems": len(probs), "chars": len(text)}}

    def render(self, state):
        return state.rep.get("text", "")

    def strip_answers(self, rep):
        return A1.strip_answers(self, rep)


# ============================================================================ JSON
def json_text(text):
    """The text to parse: a single surrounding ``` fence is ignored."""
    m = FENCE_RE.match(text or "")
    return m.group(1) if m else (text or "")


def _depth(x):
    if isinstance(x, dict):
        return 1 + max((_depth(v) for v in x.values()), default=0)
    if isinstance(x, list):
        return 1 + max((_depth(v) for v in x), default=0)
    return 0


def _leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def _nodes(x, path=""):
    yield path, x
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _nodes(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _nodes(v, f"{path}[{i}]")


def _resolve(obj, path):
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    if not parts:
        return None, False
    cur = obj
    for part in parts:
        if part.startswith("["):
            i = int(part[1:-1])
            if not isinstance(cur, list) or i >= len(cur):
                return None, False
            cur = cur[i]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None, False
    return cur, True


def json_check(text, keys, max_depth):
    if not isinstance(text, str) or not text.strip():
        return [problem("the notes are empty", None, None, "write one JSON object with the keys: " + ", ".join(keys))]
    src = json_text(text)
    try:
        obj = json.loads(src)
    except json.JSONDecodeError as e:
        return [problem(f"not valid JSON: {e.msg} (line {e.lineno}, column {e.colno})", e.lineno, None,
                        "fix the JSON syntax at that point: quotes, commas, brackets")]
    if not isinstance(obj, dict):
        return [problem("the notes must be one JSON object ({...}), not a " + type(obj).__name__, 1, None,
                        "wrap everything in one object with the keys: " + ", ".join(keys))]
    probs = []
    for k in keys:
        if k not in obj:
            probs.append(problem(f"missing key '{k}'", None, k, f"add \"{k}\" at the top level"))
        elif obj[k] in (None, "", [], {}):
            probs.append(problem(f"key '{k}' is empty", None, k, "fill it from the description, or say what is "
                                                                  "unknown there"))
    for k in obj:
        if k not in keys:
            probs.append(problem(f"unknown top-level key '{k}'", None, k,
                                 "use only these top-level keys: " + ", ".join(keys)))
    d = _depth(obj)
    if d > max_depth:
        probs.append(problem(f"nesting is {d} levels deep (at most {max_depth})", None, None,
                             "flatten the deepest parts"))
    for path, v in _leaves(obj):
        if isinstance(v, str) and PLACEHOLDER_RE.search(v):
            probs.append(problem(f"unfilled template placeholder at {path}", None, path,
                                 "replace every <...> with content from the description"))
            break
    return sort_problems(probs)


class JsonCore(_NotesCore):
    name = "JSN"
    KIND = "JSON notes"
    LOOKUP_DESC = ("What the notes hold under a key path (for example 'parts.pump' or 'links[2]'), or every object "
                   "whose key, name or id is the given term, with its path.")

    def __init__(self, params=None):
        p = dict(S.JSON_DEFAULT)
        p.update(params or {})
        self.keys = list(p["keys"])
        self.max_depth = int(p["max_depth"])

    def problems(self, text):
        return json_check(text, self.keys, self.max_depth)

    @staticmethod
    def _obj(text):
        try:
            return json.loads(json_text(text))
        except ValueError:
            return None

    def summary(self, text):
        o = self._obj(text)
        if not isinstance(o, dict):
            return "not yet valid JSON"
        return f"{len(o)} top-level keys, {sum(1 for _ in _leaves(o))} values"

    def lookup(self, text, term):
        o = self._obj(text)
        if o is None:
            return f"{ERROR_PREFIX}: the notes are not valid JSON yet; use search."
        v, ok = _resolve(o, term.strip())
        if ok:
            return f"{term.strip()}:\n" + json.dumps(v, indent=1, ensure_ascii=False)
        words = _words(term)
        hits = []
        for path, node in _nodes(o):
            last = re.split(r"[.\[]", path)[-1].rstrip("]") if path else ""
            named = isinstance(node, dict) and any(isinstance(v, (str, int, float)) and _matches(v, words)
                                                   for k, v in node.items() if k in NAME_FIELDS)
            if path and (_matches(last, words) or named):
                hits.append(f"{path}:\n" + json.dumps(node, indent=1, ensure_ascii=False))
            if len(hits) >= 12:
                break
        return "\n\n".join(hits) if hits else f"nothing named '{term}'. Try search with a word from it."

    def search(self, text, query):
        words = _words(query)
        o = self._obj(text)
        if o is None:
            lines = [f"{i}: {ln}" for i, ln in enumerate(text.splitlines(), 1) if _matches(ln, words)]
        else:
            lines = [f"{p}: {json.dumps(v, ensure_ascii=False)}" for p, v in _leaves(o) if _matches(f"{p} {v}", words)]
        if not lines:
            return f"no entry contains all of: {' '.join(words)}"
        more = len(lines) - MAX_HITS
        return "\n".join(lines[:MAX_HITS]) + (f"\n(... {more} more; narrow the words)" if more > 0 else "")

    def template(self):
        body = ",\n".join(f'  "{k}": <...>' for k in self.keys)
        return "TEMPLATE (one JSON object; replace every <...>; keep every key; no other top-level keys)\n{\n" + \
            body + "\n}"

    def guide(self):
        return "\n".join([
            "# JSON notes - guide", "",
            "Describe the system as one JSON object. The notes are checked; they count as built only when the check "
            "reports no problems.", "",
            "## Keys", "", "The object has exactly these top-level keys, each filled:", ""]
            + [f"- `{k}`" for k in self.keys]
            + ["", f"Nesting at most {self.max_depth} levels. Valid JSON: double quotes, no comments, no trailing "
               "commas.", "",
               "## Tools", "",
               "- `write_model(text)` - write the whole JSON object (replaces any earlier version).",
               "- `show_model()` - show it with line numbers.",
               "- `lookup(term)` - a key path or a name; `search(text)` - entries containing words."])


# ============================================================================ Markdown
def _cells(row):
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def md_parse(text, headers):
    """-> (problems, sections) with sections = {header: {"tables": [{"line", "cols", "rows": [(line, cells)]}],
    "bullets": [(line, text)]}}."""
    probs, seen, counts, secs = [], {}, {}, {}
    current, table = None, None
    lines = (text or "").replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        no, raw = i + 1, lines[i]
        i += 1
        if not raw.strip():
            table = None
            continue
        if ROW_RE.match(raw):
            if current is None:
                probs.append(problem("table before the first header", no, None,
                                     "put a header line such as `## " + headers[0] + "` above it"))
                continue
            cells = _cells(raw)
            if table is None:
                if PLACEHOLDER_RE.search(raw):         # an unfilled header row from the template (code review)
                    probs.append(problem("unfilled template placeholder", no, current,
                                         "replace every <...> with content from the description"))
                nxt = lines[i] if i < len(lines) else ""
                if not (ROW_RE.match(nxt) and all(SEP_CELL_RE.match(c) for c in _cells(nxt))):
                    probs.append(problem("table has no separator row under its header row", no + 1, current,
                                         "add a line like `|---|---|` (one --- per column) under the header row"))
                    table = {"line": no, "cols": cells, "rows": []}
                    secs[current]["tables"].append(table)
                    continue
                if len(_cells(nxt)) != len(cells):
                    probs.append(problem("separator row has a different number of columns from the header row",
                                         no + 1, current, f"use {len(cells)} --- cells"))
                table = {"line": no, "cols": cells, "rows": []}
                secs[current]["tables"].append(table)
                i += 1                       # the separator row
                continue
            if len(cells) != len(table["cols"]):
                probs.append(problem(f"row has {len(cells)} cells; its table has {len(table['cols'])} columns", no,
                                     current, "give every row exactly one cell per column (leave a cell empty "
                                              "rather than dropping it)"))
            table["rows"].append((no, cells))
            counts[current] = counts.get(current, 0) + 1
            if PLACEHOLDER_RE.search(raw):
                probs.append(problem("unfilled template placeholder", no, current,
                                     "replace every <...> with content from the description"))
            continue
        table = None
        h = heading_of(raw, headers)
        if h is not None:
            if h not in headers:
                probs.append(problem(f"unknown section header '{h}'", no, h,
                                     "use only these headers: " + ", ".join(headers)))
            seen.setdefault(h, no)
            current = h
            secs.setdefault(h, {"tables": [], "bullets": []})
            continue
        if BULLET_RE.match(raw):
            if current is None:
                probs.append(problem("bullet before the first header", no, None,
                                     "put a header line such as `## " + headers[0] + "` above it"))
                continue
            secs[current]["bullets"].append((no, raw.strip()))
            counts[current] = counts.get(current, 0) + 1
            if PLACEHOLDER_RE.search(raw):
                probs.append(problem("unfilled template placeholder", no, current,
                                     "replace every <...> with content from the description"))
            continue
        probs.append(problem("line is not a table row or a bullet" if current else "text before the first header",
                             no, current, "write it as a table row `| ... |` or a bullet `- ...`" if current else
                             "start with a header line such as `## " + headers[0] + "`"))
    return probs, seen, counts, secs


def md_check(text, headers, tables):
    if not isinstance(text, str) or not text.strip():
        return [problem("the notes are empty", None, None, "write the sections: " + ", ".join(headers))]
    probs, seen, counts, secs = md_parse(text, headers)
    probs += section_problems(headers, seen, counts)
    for h in tables:
        if h in secs and not any(t["rows"] for t in secs[h]["tables"]):
            probs.append(problem(f"section '{h}' needs a table with at least one data row", seen.get(h), h,
                                 "write it as a Markdown table: header row, separator row, one row per entry"))
    return sort_problems(probs)


class MarkdownCore(_NotesCore):
    name = "MD"
    KIND = "Markdown notes"
    LOOKUP_DESC = ("Every table row (with its column names) and every bullet that mentions the given name, grouped "
                   "by section.")

    def __init__(self, params=None):
        p = dict(S.MARKDOWN_DEFAULT)
        p.update(params or {})
        self.headers = list(p["headers"])
        self.tables = list(p["tables"])

    def problems(self, text):
        return md_check(text, self.headers, self.tables)

    def heading_of(self, line):
        return heading_of(line, self.headers)

    def summary(self, text):
        _, _, _, secs = md_parse(text, self.headers)
        nt = sum(len(s["tables"]) for s in secs.values())
        nr = sum(len(t["rows"]) for s in secs.values() for t in s["tables"])
        nb = sum(len(s["bullets"]) for s in secs.values())
        return f"{len(secs)} sections, {nt} tables with {nr} rows, {nb} bullets"

    def _hits(self, text, pred):
        _, _, _, secs = md_parse(text, self.headers)
        out = []
        for h, s in secs.items():
            got = []
            for t in s["tables"]:
                rows = [(no, c) for no, c in t["rows"] if pred(" ".join(c))]
                if rows:
                    got.append("| " + " | ".join(t["cols"]) + " |")
                    got += [f"| {' | '.join(c)} |   (line {no})" for no, c in rows]
            got += [f"{b}   (line {no})" for no, b in s["bullets"] if pred(b)]
            if got:
                out.append(f"## {h}\n" + "\n".join(got))
        return out

    def lookup(self, text, term):
        words = _words(term)
        out = self._hits(text, lambda s: _matches(s, words))
        return "\n\n".join(out) if out else f"nothing mentions '{term}'. Try search with one word from it."

    def search(self, text, query):
        words = _words(query)
        lines = [f"{i}: {ln.strip()}" for i, ln in enumerate(text.splitlines(), 1) if _matches(ln, words)]
        if not lines:
            return f"no line contains all of: {' '.join(words)}"
        more = len(lines) - MAX_HITS
        return "\n".join(lines[:MAX_HITS]) + (f"\n(... {more} more; narrow the words)" if more > 0 else "")

    def template(self):
        parts = []
        for h in self.headers:
            if h in self.tables:
                parts.append(f"## {h}\n| <column> | <column> |\n|---|---|\n| <...> | <...> |")
            else:
                parts.append(f"## {h}\n- <...>")
        return ("TEMPLATE (replace every <...>; keep every header; every line is a table row or a bullet)\n"
                + "\n\n".join(parts))

    def guide(self):
        lines = ["# Markdown notes - guide", "",
                 "Describe the system as Markdown under fixed section headers. The notes are checked; they count as "
                 "built only when the check reports no problems.", "", "## Sections", ""]
        lines += [f"- `## {h}`" + (" (a table)" if h in self.tables else "") for h in self.headers]
        lines += ["", "Every line under a header is a table row (`| a | b |`) or a bullet (`- ...`). A table starts "
                  "with a header row and a separator row (`|---|---|`); every row has one cell per column.", "",
                  "## Tools", "",
                  "- `write_model(text)` - write the whole notes (replaces any earlier version).",
                  "- `show_model()` - show them with line numbers.",
                  "- `lookup(term)` - rows and bullets about a name; `search(text)` - lines containing words."]
        return "\n".join(lines)
