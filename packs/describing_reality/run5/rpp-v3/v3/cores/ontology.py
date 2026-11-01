"""Core "ontology": a plain-text ontology in a small controlled natural language.

The model is plain English sentences, one per line, under section headers. genes.core.params:
  headers    required section headers (e.g. THINGS, HIERARCHY, RELATIONSHIPS, RULES, PROPERTIES)
  relations  the relation verb vocabulary (e.g. causes, increases, depends on)
  hierarchy  which hierarchy sentences are allowed: subset of ["is_a", "part_of"]
  max_depth  the deepest allowed is-a / part-of chain (edges from a term up to a root)

Grammar (one statement per line; a leading bullet `- ` / `1.` and a final period are optional; case-insensitive):
  declaration   TERM                                  only under a declaring header (THINGS, ENTITIES, PARTS,
                TERM: free description                ACTORS); declares TERM
  is-a          TERM is a kind of TERM.               (needs "is_a" in hierarchy)
  part-of       TERM is part of TERM.                 (needs "part_of" in hierarchy)
  relation      TERM REL TERM[, after TIME][, when CONDITION].       REL from the relation vocabulary
  rule          If CONDITION then CONSEQUENCE.
  property      TERM has PROPERTY VALUE [UNIT].       VALUE = the first number after "has" (UNIT = up to 3 words
                                                      after it); without a number, VALUE is the last word
  CONDITION / CONSEQUENCE = CLAUSE (and|or CLAUSE)* (consequence: and only), where
  CLAUSE        TERM REL TERM | TERM has <free text> | TERM is [not] <free text> (not "a kind of" / "part of")
  TERM          1-6 words of letters, digits, _ ' - /; a leading "the", "a" or "an" is dropped; terms are
                compared in lower case; a term cannot contain the words "and" / "or" or start with a grammar word
  TIME          free text (e.g. "2 days")

Checker rules (problems carry line numbers and fix hints):
  O1 the document is not empty; O2 header lines name the genome's headers; O3 no statement before the first header;
  O4 every line parses in the grammar (one statement per line; an unknown relation verb is named); O5 is-a / part-of
  sentences only when the genome allows them; O6 every term used in a relation, rule, condition or property is
  declared (a declaring-header line or either side of an is-a / part-of sentence); O7 no term is its own parent and
  there are no cycles in is-a or part-of; O8 no is-a / part-of chain is deeper than max_depth; O9 every header is
  present and non-empty.

Query tools (deterministic, pure Python): lookup(term), path(source, target), hierarchy(term).
"""
import difflib
import re
from collections import deque

from v3.cores.text import heading_of, numbered, problem, section_problems, sort_problems
from v3.genome import schema as S

from tournament.arms.a1_prose import A1                       # v2, read-only (strip_answers is reused)
from tournament.arms.base import ERROR_PREFIX, Arm, ArmState

WORD = r"[A-Za-z0-9][A-Za-z0-9_'’/-]*"
TERM_RE = re.compile(rf"^{WORD}(?: {WORD}){{0,5}}$")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
NUMBER_RE = re.compile(r"^[-+]?(?:\d+(?:[.,]\d+)*|\.\d+)%?$")
ARTICLES = ("the", "a", "an")
BAD_TERM_WORDS = {"and", "or"}
LEAD_WORDS = {"if", "then", "when", "after", "is", "are", "has", "have", "not"}
MAX_TEXT = 40000
MAX_OUT = 6000
FORMS = ("`TERM is a kind of TERM.`, `TERM is part of TERM.`, `TERM <relation> TERM[, after <time>][, when "
         "<condition>].`, `If <condition> then <consequence>.`, `TERM has <property> <value> [unit].`")


def norm_term(s):
    """Normalised term (lower case, single spaces, leading article dropped) or None if it is not a valid term."""
    s = " ".join(str(s).strip().strip(".").split())
    if not s or not TERM_RE.match(s):
        return None
    words = s.lower().split()
    if len(words) > 1 and words[0] in ARTICLES:
        words = words[1:]
    if set(words) & BAD_TERM_WORDS or words[0] in LEAD_WORDS:
        return None
    return " ".join(words)


class Ontology:
    """Parsed ontology: statements, declarations, hierarchy edges, relations, rules, properties and problems."""

    def __init__(self, text, params):
        self.params = params
        self.headers = list(params["headers"])
        self.relations = sorted(params["relations"], key=lambda r: (-len(r), r))
        self.hier = set(params["hierarchy"])
        self.max_depth = params["max_depth"]
        self.strict = params.get("strict", True)   # loose (V3-009): free-text lines become notes, terms self-declare
        self.notes = []           # (line, text) kept as plain-language notes in loose mode
        self.text = text or ""
        self.lines = self.text.replace("\r\n", "\n").split("\n")
        self.statements = []      # {"line", "kind", "text", "terms"}
        self.decl = {}            # term -> first declaring line
        self.isa, self.partof = [], []          # (child, parent, line)
        self.rels = []            # (a, rel, b, line, after, when_text)
        self.rules = []           # (cond_terms, cons_terms, line)
        self.props = []           # (term, property, value, unit, line)
        self.uses = []            # (term, line): every term that must be declared
        self.problems = []
        self._parse()
        self._check_terms()
        for kind, edges in (("is-a", self.isa), ("part-of", self.partof)):
            self._check_graph(kind, edges)
        self.problems = sort_problems(self.problems)

    # ---------------------------------------------------------------- parsing
    def _parse(self):
        if not self.text.strip():
            self.problems.append(problem("the document is empty", None, None,
                                         "write statements under the headers: " + ", ".join(self.headers)))
            return
        seen, counts, current = {}, {}, None
        for no, raw in enumerate(self.lines, 1):
            if not raw.strip():
                continue
            h = heading_of(raw, self.headers)
            if h is not None:
                if h not in self.headers:
                    self.problems.append(problem(f"unknown section header '{h}'", no, h,
                                                 "use only these headers: " + ", ".join(self.headers)))
                seen.setdefault(h, no)
                current = h
                continue
            if current is None:
                self.problems.append(problem("statement before the first header", no, None,
                                             f"put a header line such as `## {self.headers[0]}` above it"))
                continue
            counts[current] = counts.get(current, 0) + 1
            self._statement(no, raw, current)
        self.problems += section_problems(self.headers, seen, counts)

    def _statement(self, no, raw, header):
        body = " ".join(BULLET_RE.sub("", raw).split())
        if body.endswith("."):
            body = body[:-1].rstrip()
        if re.search(r"[.!?]\s+[A-Z]", body):
            self.problems.append(problem("more than one statement on one line", no, header,
                                         "write one statement per line"))
            return
        low = body.lower()
        # rule
        m = re.match(r"^if (.+?),? then (.+)$", body, re.I)
        if m:
            ct, cp = self._clauses(m.group(1), no, ("and", "or"))
            kt, kp = self._clauses(m.group(2), no, ("and",))
            if cp or kp:
                self.problems += cp + kp
                return
            self.rules.append((ct, kt, no))
            return self._add(no, "rule", raw, ct + kt)
        # hierarchy
        for kind, phrase in (("is_a", " is a kind of "), ("part_of", " is part of ")):
            k = low.find(phrase)
            if k > 0:
                a, b = norm_term(body[:k]), norm_term(body[k + len(phrase):])
                if a is None or b is None:
                    bad = body[:k] if a is None else body[k + len(phrase):]
                    self.problems.append(problem(f"'{bad.strip()}' is not a valid term", no, header,
                                                 "a term is 1-6 plain words (letters, digits, - ' /), without "
                                                 "'and' / 'or'"))
                    return
                if kind not in self.hier:
                    self.problems.append(problem(f"'{phrase.strip()}' sentences are not used in this design", no,
                                                 header, "declare the terms under a declaring header and relate "
                                                         "them with a relation verb instead"))
                    return
                if a == b:
                    self.problems.append(problem(f"'{a}' {phrase.strip()} itself", no, header,
                                                 "a term cannot be its own parent; remove the line"))
                    return
                (self.isa if kind == "is_a" else self.partof).append((a, b, no))
                for t in (a, b):
                    self.decl.setdefault(t, no)
                return self._add(no, kind, raw, [a, b])
        if re.match(r"^.+? is (?:a )?(?:kind|type|sort|part) of ", low) or " is a part of " in low:
            self.problems.append(problem("hierarchy sentence not in the grammar", no, header,
                                         "write exactly `X is a kind of Y.` or `X is part of Y.`"))
            return
        # relation (with optional qualifiers)
        rel = self._relation_statement(body, no)
        if rel is not None:
            if isinstance(rel, list):        # problems
                self.problems += rel
                return
            a, r, b, after, when_terms, when_text = rel
            self.rels.append((a, r, b, no, after, when_text))
            return self._add(no, "relation", raw, [a, b] + when_terms)
        # property
        m = re.match(r"^(.+?) has (.+)$", body, re.I)
        if m and norm_term(m.group(1)):
            t = norm_term(m.group(1))
            words = m.group(2).split()
            k = next((i for i, w in enumerate(words) if NUMBER_RE.match(w)), None)
            if k is not None and k >= 1 and len(words) - k - 1 <= 3:
                prop, val, unit = " ".join(words[:k]), words[k], " ".join(words[k + 1:])
            elif k is None and len(words) >= 2:
                prop, val, unit = " ".join(words[:-1]), words[-1], ""
            else:
                self.problems.append(problem("property sentence not in the grammar", no, header,
                                             "write `X has <property> <value> [unit]`, e.g. `tank has capacity "
                                             "500 litres` (unit at most 3 words)"))
                return
            self.props.append((t, prop.lower(), val, unit, no))
            self.uses.append((t, no))
            return self._add(no, "property", raw, [t])
        # declaration
        if header in S.DECLARE_HEADERS:
            m = re.match(r"^(.+?)(?:\s*:\s+.+|\s+[-–—]\s+.+|\s*\(.+\))?$", body)
            t = norm_term(m.group(1)) if m else None
            if t is not None:
                self.decl.setdefault(t, no)
                return self._add(no, "declaration", raw, [t], declares=True)
        if not self.strict:
            self.notes.append((no, raw.strip()))
            self.statements.append({"line": no, "kind": "note", "text": raw.strip(), "terms": []})
            return
        self.problems.append(self._unparsed(body, no, header))

    def _add(self, no, kind, raw, terms, declares=False):
        self.statements.append({"line": no, "kind": kind, "text": raw.strip(), "terms": list(dict.fromkeys(terms))})
        if kind in ("relation", "rule"):
            self.uses += [(t, no) for t in terms]

    def _unparsed(self, body, no, header):
        words = body.split()
        for i in range(1, len(words) - 1):          # TERM <unknown verb> TERM ?
            w = words[i].lower()
            if re.match(r"^[a-z]+s$", w) and w not in S.GRAMMAR_WORDS and norm_term(" ".join(words[:i])) \
                    and norm_term(" ".join(words[i + 1:]).split(",")[0]):
                return problem(f"relation '{w}' is not in the vocabulary", no, header,
                               "use one of: " + ", ".join(self.params["relations"]))
        decl = (" Under " + "/".join(h for h in self.headers if h in S.DECLARE_HEADERS)
                + " a line may also be just a term, or `term: description`.") \
            if set(self.headers) & set(S.DECLARE_HEADERS) else ""
        return problem("not a statement in the grammar", no, header, "use one of the forms " + FORMS + decl)

    def _split_relation(self, s):
        """'A rel B' -> (a, rel, b) with valid terms, leftmost-longest relation; None if no reading."""
        low = s.lower()
        cands = []
        for r in self.relations:
            for m in re.finditer(rf"(?<=\s){re.escape(r)}(?=\s)", low):
                cands.append((m.start(), -len(r), r))
        for start, _, r in sorted(cands):
            a, b = norm_term(s[:start]), norm_term(s[start + len(r):])
            if a and b:
                return a, r, b
        return None

    def _clauses(self, text, no, joins):
        """Condition / consequence clauses -> (terms, problems)."""
        parts = re.split(r"\s+(?:" + "|".join(joins) + r")\s+", text.strip(), flags=re.I)
        terms, probs = [], []
        for c in parts:
            c = c.strip().strip(",")
            rel = self._split_relation(c)
            if rel:
                terms += [rel[0], rel[2]]
                continue
            m = re.match(r"^(.+?) (?:has|is|are) (.+)$", c, re.I)
            if m and norm_term(m.group(1)) and not re.match(r"^(?:a kind of|part of)\b", m.group(2), re.I):
                terms.append(norm_term(m.group(1)))
                continue
            probs.append(problem(f"clause '{c}' is not in the grammar", no, None,
                                 "a clause is `X <relation> Y`, `X has <...>` or `X is <state>`, joined by "
                                 + " / ".join(joins) + " (relations: " + ", ".join(self.params["relations"]) + ")"))
        return terms, probs

    def _relation_statement(self, body, no):
        quals = list(re.finditer(r",\s*(after|when)\s+", body, re.I))
        core = body[:quals[0].start()] if quals else body
        rel = self._split_relation(core)
        if rel is None:
            return None
        after, when_text, when_terms, probs = None, None, [], []
        for k, q in enumerate(quals):
            seg = body[q.end():quals[k + 1].start() if k + 1 < len(quals) else len(body)].strip()
            kind = q.group(1).lower()
            if not seg or (kind == "after" and after is not None) or (kind == "when" and when_text is not None):
                probs.append(problem(f"bad or repeated ', {kind} ...' qualifier", no, None,
                                     "add at most one `, after <time>` and one `, when <condition>`"))
                continue
            if kind == "after":
                after = seg
            else:
                when_text = seg
                t, p = self._clauses(seg, no, ("and", "or"))
                when_terms += t
                probs += p
        if probs:
            return probs
        return rel[0], rel[1], rel[2], after, when_terms, when_text

    # ---------------------------------------------------------------- checks
    def _check_terms(self):
        known = sorted(self.decl)
        done = set()
        if not self.strict:     # loose: a term used in a parsed statement declares itself
            for t, no in self.uses:
                self.decl.setdefault(t, no)
            return
        for t, no in self.uses:
            if t in self.decl or (t, no) in done:
                continue
            done.add((t, no))
            close = difflib.get_close_matches(t, known, n=1, cutoff=0.75)
            where = [h for h in self.headers if h in S.DECLARE_HEADERS]
            hint = (f"declare it (a line `{t}: what it is` under {where[0]})" if where
                    else f"declare it in a hierarchy line such as `{t} is a kind of <category>`")
            if close:
                hint += f", or did you mean '{close[0]}'?"
            self.problems.append(problem(f"undeclared term '{t}'", no, None, hint))

    def _check_graph(self, kind, edges):
        parents = {}
        line_of = {}
        for c, p, no in edges:
            parents.setdefault(c, [])
            if p not in parents[c]:
                parents[c].append(p)
                line_of[(c, p)] = no
        color, cycles = {}, []

        def dfs(n, stack):
            color[n] = 1
            stack.append(n)
            for p in parents.get(n, []):
                if color.get(p) == 1:      # reported at the cycle's last line in document order
                    cyc = stack[stack.index(p):] + [p]
                    cycles.append((cyc, max(line_of[(cyc[i], cyc[i + 1])] for i in range(len(cyc) - 1))))
                elif not color.get(p):
                    dfs(p, stack)
            stack.pop()
            color[n] = 2
        for n in sorted(parents):
            if not color.get(n):
                dfs(n, [])
        for cyc, no in cycles:
            self.problems.append(problem(f"{kind} cycle: " + " -> ".join(cyc), no, None,
                                         f"a {kind} chain must not loop back; remove or reverse one of these lines"))
        if cycles:
            return
        depth = {}

        def d(n):
            if n not in depth:
                depth[n] = 0 if not parents.get(n) else 1 + max(d(p) for p in parents[n])
            return depth[n]
        for c in sorted(parents):
            if d(c) == self.max_depth + 1:
                p = max(parents[c], key=d)
                self.problems.append(problem(f"{kind} chain from '{c}' is {d(c)} levels deep (max "
                                             f"{self.max_depth})", line_of[(c, p)], None,
                                             "flatten the hierarchy: attach the term to a higher category"))

    # ---------------------------------------------------------------- queries
    def known(self):
        out = set(self.decl)
        for s in self.statements:
            out |= set(s["terms"])
        return out

    def _resolve(self, term):
        t = norm_term(term) if isinstance(term, str) else None
        if t is None:
            return None, f"{ERROR_PREFIX}: give a term (1-6 plain words)"
        if t in self.known():
            return t, None
        close = difflib.get_close_matches(t, sorted(self.known()), n=3, cutoff=0.6)
        return None, (f"{ERROR_PREFIX}: '{t}' is not in the model" + (f" (similar: {', '.join(close)})" if close
                                                                      else ""))

    @staticmethod
    def _up(edges, t):
        """Transitive parents of t, nearest first (breadth-first), with cycle guard."""
        par = {}
        for c, p, no in edges:
            par.setdefault(c, []).append((p, no))
        out, seen, q = [], {t}, deque([t])
        while q:
            n = q.popleft()
            for p, no in par.get(n, []):
                if p not in seen:
                    seen.add(p)
                    out.append((p, no))
                    q.append(p)
        return out

    @staticmethod
    def _down(edges, t):
        ch = {}
        for c, p, no in edges:
            ch.setdefault(p, []).append((c, no))
        out, seen, q = [], {t}, deque([t])
        while q:
            n = q.popleft()
            for c, no in ch.get(n, []):
                if c not in seen:
                    seen.add(c)
                    out.append((c, no))
                    q.append(c)
        return out

    def lookup(self, term):
        t, err = self._resolve(term)
        if err:
            return err
        out = [f"LOOKUP {t}"]
        if t in self.decl:
            out.append(f"declared: line {self.decl[t]}: {self.lines[self.decl[t] - 1].strip()}")
        fmt = lambda xs: ", ".join(f"{n} (line {no})" for n, no in xs)
        anc, kinds = self._up(self.isa, t), self._down(self.isa, t)
        whole, parts = self._up(self.partof, t), self._down(self.partof, t)
        if anc:
            out.append("is a kind of (nearest first): " + fmt(anc))
        if kinds:
            out.append(f"kinds of {t}: " + fmt(kinds))
        if whole:
            out.append("is part of (nearest first): " + fmt(whole))
        if parts:
            out.append(f"parts of {t}: " + fmt(parts))
        own = [s for s in self.statements if t in s["terms"] and s["kind"] not in ("declaration",)]
        if own:
            out.append("statements:")
            out += [f"  line {s['line']}: {s['text']}" for s in own]
        inh = [s for a, _ in anc for s in self.statements
               if a in s["terms"] and s["kind"] in ("relation", "rule", "property") and s not in own]
        if inh:
            out.append("inherited from its categories:")
            out += [f"  line {s['line']}: {s['text']}" for s in dict((s["line"], s) for s in inh).values()]
        if len(out) == 1:
            out.append("(no statements)")
        return self._clip("\n".join(out))

    def _edges(self, with_hierarchy):
        out = []
        for a, r, b, no, after, when in self.rels:
            lab = r + (f", after {after}" if after else "") + (f", when {when}" if when else "")
            out.append((a, b, lab, no))
        for ct, kt, no in self.rules:
            for a in ct:
                for b in kt:
                    if a != b:
                        out.append((a, b, "rule: " + self.lines[no - 1].strip(), no))
        if with_hierarchy:
            out += [(c, p, "is a kind of", no) for c, p, no in self.isa]
            out += [(c, p, "is part of", no) for c, p, no in self.partof]
        return sorted(out, key=lambda e: (e[3], e[0], e[1]))

    def _bfs(self, a, b, with_hierarchy):
        adj = {}
        for x, y, lab, no in self._edges(with_hierarchy):
            adj.setdefault(x, []).append((y, lab, no))
        prev, q = {a: None}, deque([a])
        while q:
            n = q.popleft()
            if n == b:
                break
            for y, lab, no in adj.get(n, []):
                if y not in prev:
                    prev[y] = (n, lab, no)
                    q.append(y)
        if b not in prev:
            return None
        steps, n = [], b
        while prev[n] is not None:
            p, lab, no = prev[n]
            steps.append((p, lab, n, no))
            n = p
        return steps[::-1]

    def path(self, source, target):
        a, err = self._resolve(source)
        if err:
            return err
        b, err = self._resolve(target)
        if err:
            return err
        if a == b:
            return f"PATH {a} -> {b}: the same term."
        for hier in (False, True):
            steps = self._bfs(a, b, hier)
            if steps:
                out = [f"PATH {a} -> {b} ({len(steps)} step(s)" + (", using hierarchy links" if hier else "") + "):"]
                out += [f"  {x} --[{lab}]--> {y}   (line {no})" for x, lab, y, no in steps]
                return self._clip("\n".join(out))
        back = self._bfs(b, a, False)
        msg = f"PATH {a} -> {b}: no directed chain of relations or rules from {a} to {b}."
        if back:
            msg += f"\nThere is one in the other direction ({b} -> {a}):\n" + "\n".join(
                f"  {x} --[{lab}]--> {y}   (line {no})" for x, lab, y, no in back)
        return self._clip(msg)

    def hierarchy(self, term=None):
        kids = {}
        for c, p, no in self.isa:
            kids.setdefault(p, []).append((c, "kind", no))
        for c, p, no in self.partof:
            kids.setdefault(p, []).append((c, "part", no))
        lines = []

        def walk(n, depth, seen):
            for c, rel, no in sorted(kids.get(n, []), key=lambda x: (x[1], x[0])):
                loop = c in seen
                lines.append("  " * depth + f"- {c} ({rel}, line {no})" + (" [cycle]" if loop else ""))
                if not loop:
                    walk(c, depth + 1, seen | {c})
        if term is None or (isinstance(term, str) and not term.strip()):
            children = {c for c, _, _ in self.isa + self.partof}
            roots = sorted({p for _, p, _ in self.isa + self.partof} - children)
            if not roots:
                return "HIERARCHY: no is-a or part-of statements" + (" (or only cycles)" if self.isa or self.partof
                                                                      else "") + "."
            out = ["HIERARCHY (roots and their kinds / parts):"]
            for r in roots:
                lines.clear()
                walk(r, 1, {r})
                out += [f"{r}"] + lines
            return self._clip("\n".join(out))
        t, err = self._resolve(term)
        if err:
            return err
        walk(t, 1, {t})
        up = [f"{p} (is-a, line {no})" for p, no in self._up(self.isa, t)] + \
             [f"{p} (part-of, line {no})" for p, no in self._up(self.partof, t)]
        out = [f"HIERARCHY {t}"] + (["above: " + ", ".join(up)] if up else []) + [t] + (lines or ["  (no kinds or "
                                                                                                  "parts below)"])
        return self._clip("\n".join(out))

    @staticmethod
    def _clip(s):
        return s if len(s) <= MAX_OUT else s[:MAX_OUT] + "\n[... truncated]"


def check(text, params):
    return Ontology(text, params).problems


def _tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


class OntologyCore(Arm):
    name = "ONT"
    buildable = True
    executable = False
    mutating_tools = ("write_model",)
    QUERY_TOOLS = ("lookup", "path", "hierarchy")

    def __init__(self, params=None):
        p = dict(S.ONTOLOGY_DEFAULT)
        p.update(params or {})
        self.params = {"headers": list(p["headers"]), "relations": list(p["relations"]),
                       "hierarchy": list(p["hierarchy"]), "max_depth": int(p["max_depth"]),
                       "strict": bool(p.get("strict", True))}
        self._cache = (None, None)

    def tools(self):
        return [
            _tool("write_model", "Write the whole ontology (plain sentences under the section headers). Replaces "
                                 "any earlier version.", {"text": {"type": "string"}}, ["text"]),
            _tool("show_model", "Show the current ontology with line numbers."),
            _tool("lookup", "Everything the ontology says about a term: its declaration, categories (is-a, "
                            "transitively), wholes and parts (part-of), its statements and those inherited from "
                            "its categories.", {"term": {"type": "string"}}, ["term"]),
            _tool("path", "The shortest directed chain of relations (and rules) from source to target, with the "
                          "relation on every step.", {"source": {"type": "string"}, "target": {"type": "string"}},
                  ["source", "target"]),
            _tool("hierarchy", "The kinds and parts below a term (its subtree) and what is above it; with no term, "
                               "the whole hierarchy.", {"term": {"type": "string"}}),
        ]

    def new_state(self):
        st = ArmState()
        st.rep = {"text": ""}
        st.versions = 0
        return st

    def onto(self, text):
        if self._cache[0] != text:
            self._cache = (text, Ontology(text, self.params))
        return self._cache[1]

    def problems(self, text):
        return self.onto(text if isinstance(text, str) else "").problems

    def heading_of(self, line):
        return heading_of(line, self.params["headers"])

    def handle(self, state, name, args):
        text = state.rep.get("text", "") if isinstance(state.rep, dict) else ""
        if name == "write_model":
            t = args.get("text")
            if not isinstance(t, str) or not t.strip():
                return f"{ERROR_PREFIX}: write_model needs a non-empty 'text' string (the whole ontology)."
            state.rep = {"text": t[:MAX_TEXT]}
            state.versions = getattr(state, "versions", 0) + 1
            state.log.append((name, {"chars": len(t)}, "saved"))
            o = self.onto(state.rep["text"])
            kinds = {}
            for s in o.statements:
                kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
            return (f"Ontology saved (version {state.versions}): {len(o.statements)} statements ("
                    + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + f"), {len(o.decl)} declared terms."
                    + ("\n(text truncated to 40000 characters)" if len(t) > MAX_TEXT else ""))
        if name == "show_model":
            return numbered(text) if text else f"{ERROR_PREFIX}: no ontology has been written yet; use write_model."
        if name in self.QUERY_TOOLS:
            if not text.strip():
                return f"{ERROR_PREFIX}: no ontology has been written yet; use write_model."
            o = self.onto(text)
            if name == "lookup":
                out = o.lookup(args.get("term"))
            elif name == "path":
                out = o.path(args.get("source"), args.get("target"))
            else:
                out = o.hierarchy(args.get("term"))
            if o.problems and not out.startswith(ERROR_PREFIX):
                out += f"\n(note: the ontology has {len(o.problems)} checker problem(s); unparsed lines are ignored)"
            return out
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
        o = self.onto(text)
        probs = o.problems if text.strip() else [problem("no ontology written")]
        ok = bool(text.strip()) and not probs
        return {"usable": ok, "executable": ok, "checked": ok,
                "holes": [{"line": p["line"], "problem": p["msg"]} for p in probs],
                "errors": [] if ok else [f"{len(probs)} checker problem(s)"],
                "stats": {"statements": len(o.statements), "terms": len(o.decl), "problems": len(probs)}}

    def render(self, state):
        return state.rep.get("text", "")

    def strip_answers(self, rep):
        return A1.strip_answers(self, rep)

    # ---------------------------------------------------------------- template and guide
    def _examples(self, header):
        hier = self.params["hierarchy"]
        if header in S.DECLARE_HEADERS:
            return ["<thing>: <what it is>"]
        if header == "HIERARCHY":
            return (["<thing> is a kind of <category>."] if "is_a" in hier else []) + \
                   (["<part> is part of <whole>."] if "part_of" in hier else []) or \
                ["<thing> <relation> <thing>."]
        if header in ("RULES", "CONSTRAINTS"):
            return ["If <condition> then <consequence>."]
        if header == "PROPERTIES":
            return ["<thing> has <property> <value> [unit]."]
        return ["<thing> <relation> <thing>[, after <time>][, when <condition>]."]

    def template(self):
        body = "\n\n".join(f"## {h}\n" + "\n".join(self._examples(h)) for h in self.params["headers"])
        return ("TEMPLATE (replace every <...>; one statement per line; keep every header)\n" + body
                + "\nRelations: " + ", ".join(self.params["relations"]))

    def guide(self):
        p = self.params
        hier = p["hierarchy"]
        decl = [h for h in p["headers"] if h in S.DECLARE_HEADERS]
        lines = ["# Plain-text ontology - guide", "",
                 "Describe the system as a plain-text ontology: short English sentences in a fixed grammar, one per "
                 "line, under section headers. The ontology is checked; it counts as built only when the check "
                 "reports no problems. Once built, query it with the tools.", "",
                 "## Sections", "", "Write every one of these headers as a `## NAME` line, each followed by at least "
                 "one statement: " + ", ".join(p["headers"]) + ".", "", "## Sentence forms", ""]
        if decl:
            lines.append(f"- Under {'/'.join(decl)}: `term` or `term: free description` declares a term.")
        if "is_a" in hier:
            lines.append("- `X is a kind of Y.` (declares X and Y)")
        if "part_of" in hier:
            lines.append("- `X is part of Y.` (declares X and Y)")
        lines += ["- `X <relation> Y.`, optionally `, after <time>` and/or `, when <condition>` before the period.",
                  "- `If <condition> then <consequence>.`",
                  "- `X has <property> <value> [unit].` (e.g. `tank has capacity 500 litres.`)", "",
                  "Relations (the only verbs allowed): " + ", ".join(p["relations"]) + ".",
                  "A condition is clauses `X <relation> Y`, `X has ...` or `X is <state>` joined by `and` / `or`.",
                  "Terms are 1-6 plain words without `and` / `or`; `the` / `a` in front is ignored. Every term used "
                  "in a relation, rule or property must be declared"
                  + (" (under " + "/".join(decl) + " or in a hierarchy sentence)." if decl else
                     " in a hierarchy sentence."),
                  f"No cycles in the hierarchy; at most {p['max_depth']} levels from a term to the top.", "",
                  *([] if p.get("strict", True) else [
                      "LOOSE MODE: a line that does not fit these forms is kept as a plain-language note (it is not "
                      "an error), and a term used in a sentence counts as declared. Use the forms for facts you want "
                      "the tools to find; write anything else as notes.", ""]),
                  "## Tools", "",
                  "- `write_model(text)` - write the whole ontology; the result lists check problems with line "
                  "numbers.", "- `show_model()` - show it with line numbers.",
                  "- `lookup(term)`, `path(source, target)`, `hierarchy(term)` - query the built ontology "
                  "(offered when this design includes them)."]
        return "\n".join(lines)
