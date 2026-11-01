"""Run 5 stage 2: the five "+" versions of the stage 1 winner's parts, and the 32 factorial designs built from them
(V3-042; docs/RUN5_DESIGN.md section 4).

The "+" texts are written by Claude Opus in one headless session with no tools: it sees the winner's genome, the task,
the factor definitions, the fit rule and (V3-056) a short summary of the winner's stage 1 evidence on the generated
items (v3.plus_evidence5: score overall and by k, notes-check failures by checker category, tool calls, tokens); never
anything from the hand-written items, scores by family, run 4, an item, question or answer key. The texts are frozen
(hashed) before stage 2 and shown to the human.

Factors (level "-" = the winner exactly as it is):
  A notes layout     core.params for a checked notes core (sectioned, ontology, json, markdown); schema.fields for a
                     structured core (equations, rules, graph, simulation, probabilistic); for prose, a layout
                     paragraph added to the writing guidance
  B fact writing     guidance.text (the rules for writing the notes)
  C answering        answering.text (the procedure for answering from the notes)
  D worked example   examples.items: one short example if the winner has none, none if it has some
  E thinking budget  numeric.max_turns: the winner's value, or double it (at most 24)
"""
import copy
import hashlib
import json
import re

from v3.genome import schema as S

FACTORS = ("A", "B", "C", "D", "E")
PARAM_CORES = tuple(S.CORE_PARAM_BOUNDS)              # sectioned, ontology, json, markdown

BRIEF = """You design part of an experiment on how a small AI model should write down a system it reads about.

The task the design serves: a small model reads a long, messy description of a system (a plant, a network, a
service, a mix of several systems) WITHOUT seeing any questions, and writes notes in the form the design prescribes.
Later the description is taken away and the model must answer questions about the system using only its notes (and
the read tools of its core, if any). A design scores well only if its notes really capture the system: parts,
quantities, causes, rules, timing and what is unknown.

Below is the winning design of a first stage, as a genome (JSON). Your job: for each of five parts of it, write ONE
alternative version - what you judge, from first principles and the stage 1 evidence below, most likely to make a
small model's notes and answers more accurate. Do not write a variation for its own sake, and do not copy the existing text with small edits.

The five parts:
- A notes layout: {a_def}
- B fact writing: the rules for writing each fact down (the genome's guidance text). Write a complete replacement
  of {b_min} to {b_max} characters (the current text has {b_now}).
- C answering: a step-by-step procedure for answering a question from the notes (the genome's answering text,
  currently "{c_now}"). It must work with BOTH layouts (the current one and your A version): do not name specific
  sections, headers, keys or columns. The same holds for B: the fact-writing rules must not name them either.{keep_rule}
- D worked example: {d_def}
- E thinking budget: fixed by code (no text needed).

Fit rule (checked by code): every combination of old and new parts must form a working design; neither the
answering procedure nor the fact-writing rules may name a section, header, key or column of either layout; B must be {b_min} to {b_max} characters;
the other texts under 1,200 characters (an example under 2,500).

THE WINNING DESIGN
{genome}
{evidence}

Reply with ONE JSON object and nothing else:
{{"A": {a_shape}, "B": {{"guidance_text": "..."}}, "C": {{"answering_text": "..."}}, "D": {d_shape},
  "rationale": {{"A": "one sentence", "B": "one sentence", "C": "one sentence", "D": "one sentence"}}}}"""


# V3-044: the only reasons the human may send the + texts back (state/run5/PLUS_REJECT, first line = the reason);
# each sends them back to a fresh Opus session; at most two rejections
REJECT_REASONS = ("wrong factor", "refers to tools the design lacks", "mentions items, answers or results",
                  "breaks the length rule")


def factor_a_kind(winner):
    """How factor A changes the design. Code review (2026-10-29): the schema gene reaches the builder only through
    get_template (template-first designs), so for the one-shot calculator and own-code founders a schema change would
    be invisible; for every core without layout parameters, A is a layout paragraph added to the writing guidance,
    which the builder always sees."""
    kind = winner["genes"]["core"]["kind"]
    return "core_params" if kind in PARAM_CORES else "layout_text"


EVIDENCE = """
WHAT STAGE 1 SHOWED ABOUT THIS DESIGN (generated items only; the builder is a small model)
{lines}

Use this evidence. Write each alternative so that it is more likely to fix what the evidence shows: heavier syntax
or more rules risk more notes failing the checker, and the score falls as systems get larger, so each part should be
easy for a small model to follow without tripping the checker and should help the notes stay accurate at k = 8. Each
part must still work on its own and with every combination of the other parts."""


def brief(winner, evidence=None):
    g = winner["genes"]
    ak = factor_a_kind(winner)
    kind = g["core"]["kind"]
    a_def = {
        "core_params": f"the organisation of the notes, i.e. the core parameters of the {kind} core "
                       f"(currently {json.dumps(g['core']['params'])}). Give a complete replacement with exactly the "
                       f"same keys and types; header names 2-30 characters, starting with a letter, using only upper-case "
                       f"letters, digits, single spaces and & / - (no underscores); JSON keys 2-30 characters, "
                       f"starting with a letter, using only lower-case letters, digits and _.",
        "layout_text": ("the organisation of the notes. This design prescribes no layout; write one paragraph that "
                        "tells the model how to organise its notes (it is added to the writing guidance)."
                        if kind in S.PLAIN_CORES else
                        f"how the model is organised (the {kind} core fixes its syntax, not its organisation): write "
                        "one paragraph on how to group, order and name its contents, for example by subsystem, in "
                        "the order things happen, or with a naming scheme (it is added to the writing guidance)."),
        "schema_fields": f"which kinds of content the model must declare (the schema fields, currently "
                         f"{g['schema']['fields']}; choose from {list(S.SCHEMA_FIELDS)}, keep 'variables').",
    }[ak]
    a_shape = {"core_params": "{\"core_params\": {...}}", "layout_text": "{\"layout_text\": \"...\"}",
               "schema_fields": "{\"schema_fields\": [...]}"}[ak]
    has_ex = bool(g["examples"]["items"])
    # code review: an example must fit the layout it is shown with, so it is written in both layouts
    d_def = ("the design has a worked example; the alternative is none (code removes it). Because the winner's "
             "example is shown in designs that use your new layout A, also give the winner's example rewritten in "
             "your new layout with the same content (\"winner_example_new_layout\")."
             if has_ex else "the design has no worked example; write one short example of good notes for a small "
                            "invented system (not about any real incident), TWICE with identical content: in the "
                            "current layout (\"examples\") and in your new layout A (\"examples_new_layout\").")
    d_shape = ("{\"examples\": [], \"winner_example_new_layout\": [\"...\"]}" if has_ex else
               "{\"examples\": [\"...\"], \"examples_new_layout\": [\"...\"]}")
    named = named_in_guidance(winner)
    keep_rule = (f"\n  Your layout A must keep these section names, because the current fact-writing rules (shown in "
                 f"half the designs) name them: {', '.join(sorted(named))}." if named and ak == "core_params" else "")
    b_min, b_max = b_length_bounds(winner)
    return BRIEF.format(a_def=a_def, c_now=g["answering"]["text"] or "(empty)", d_def=d_def,
                        genome=json.dumps(g, indent=1), a_shape=a_shape, d_shape=d_shape, b_min=b_min, b_max=b_max,
                        b_now=len(g["guidance"]["text"]), keep_rule=keep_rule,
                        evidence=EVIDENCE.format(lines=evidence) if evidence else "")


def named_in_guidance(winner):
    """Layout names (headers or keys) that the winner's own fact-writing rules mention (upper-case headers as written;
    keys as whole words). They must survive in A+, because B- is shown with A+ in eight designs."""
    g = S.upgrade(winner["genes"])
    params = g["core"].get("params") or {}
    text = g["guidance"]["text"] + "\n" + g["answering"]["text"]      # B- and C- are both shown with A+
    out = {h for h in params.get("headers", []) if re.search(r"(?<![A-Z])" + re.escape(h) + r"(?![A-Z])", text)}
    out |= {k for k in params.get("keys", []) if re.search(r"\b" + re.escape(k) + r"\b", text)}
    return out


def b_length_bounds(winner):
    """V3-044 length rule for B: 0.5 to 3 times the winner's guidance, at least 600 characters allowed."""
    n = len(S.upgrade(winner["genes"])["guidance"]["text"])
    return int(0.5 * n), max(3 * n, 600)


def parse_reply(text):
    from v3.jsonx import first_object
    return first_object(text)


def e_plus(winner):
    """Factor E's alternative: double the winner's turn budget, or half of it if doubling passes the maximum (code
    review: E+ must differ from E-)."""
    lo, hi = S.NUMERIC_BOUNDS["max_turns"]
    mt = winner["genes"]["numeric"]["max_turns"]
    return 2 * mt if 2 * mt <= hi else max(lo, mt // 2)


def write_plus(winner, call=None, problems=None, evidence=None):
    """Ask Opus for the "+" texts. call(system, user) -> text (default: headless Opus on the Claude plan). problems:
    the fit-check problems of the previous attempt, shown so a retry can fix them (each attempt is a fresh session).
    evidence: the stage 1 evidence text (v3.plus_evidence5.render), V3-056."""
    user = brief(winner, evidence)
    if problems:
        user += ("\n\nAn earlier reply failed these code checks; make sure yours passes them:\n- "
                 + "\n- ".join(problems))
    if call is None:
        from v3.hand5.factory import claude_text
        text = claude_text("You are an expert in knowledge representation for language models.", user,
                           timeout=1800, model="opus")["final_raw"]
    else:
        text = call("You are an expert in knowledge representation for language models.", user)
    plus = parse_reply(text)
    plus["E"] = {"max_turns": e_plus(winner)}
    plus["winner"] = winner["id"]
    return plus


def freeze(plus):
    body = dict({k: plus[k] for k in FACTORS}, winner=plus.get("winner"))   # the winner is covered by the hash
    return dict(plus, hash=hashlib.sha256(S.canonical(body).encode()).hexdigest()[:16])


def levels(n):
    """Design n (0-31, standard order: A changes fastest) -> {factor: '+'/'-'}."""
    return {f: "+" if (n >> b) & 1 else "-" for b, f in enumerate(FACTORS)}


def build(winner, plus, n):
    lv = levels(n)
    g = copy.deepcopy(S.upgrade(winner["genes"]))
    ak = factor_a_kind(winner)
    if lv["B"] == "+":
        g["guidance"]["text"] = plus["B"]["guidance_text"]
    if lv["A"] == "+":
        if ak == "core_params":
            g["core"]["params"] = copy.deepcopy(plus["A"]["core_params"])
        elif ak == "schema_fields":
            g["schema"]["fields"] = list(plus["A"]["schema_fields"])
        else:
            g["guidance"]["text"] = g["guidance"]["text"].rstrip() + "\n\nLayout: " + plus["A"]["layout_text"]
    if lv["C"] == "+":
        g["answering"]["text"] = plus["C"]["answering_text"]
    d = plus.get("D") or {}
    if lv["D"] == "+":
        g["examples"]["items"] = list(d.get("examples_new_layout") if lv["A"] == "+" and d.get("examples_new_layout")
                                      else d.get("examples", []))
    elif lv["A"] == "+" and g["examples"]["items"] and d.get("winner_example_new_layout"):
        g["examples"]["items"] = list(d["winner_example_new_layout"])      # the winner's example, in layout A+
    if lv["E"] == "+":
        g["numeric"]["max_turns"] = plus["E"]["max_turns"]
    if n == 0:
        return winner                                   # design 1 is the winner itself (same genes, same id)
    name = f"S2-{n + 1:02d} " + "".join(f"{f}{lv[f]}" for f in FACTORS)
    return S.make(g, [winner["id"]], "factorial", 0, 0, name, f"run 5 stage 2 design {n + 1}")


def _layout_names(winner, plus):
    names = set()
    for params in (winner["genes"]["core"].get("params") or {}, (plus.get("A") or {}).get("core_params") or {}):
        names |= set(params.get("headers", [])) | set(params.get("keys", []))
    return names


# words that make a name a reference to the layout; "key" and "part" are not among them before a name, because
# "key events" or "each part links" are ordinary English (code review)
SECTION_WORDS = r"(?:section|sections|heading|headings|header|headers|table|tables|column|columns|field|fields|block)"
AFTER_WORDS = r"(?:section|sections|heading|headings|header|headers|table|tables|column|columns|key|keys|field|fields|block)"


def names_layout(text, nm):
    """True if the text names a layout header or key: a header in upper case anywhere, or in any case next to a word
    like 'section' or 'table'; a key quoted, after a dot or lookup(, or next to 'key' / 'field' (code review)."""
    e = re.escape(nm)
    if nm.isupper():
        if re.search(r"(?<![A-Za-z])" + e + r"(?![A-Za-z])", text):
            return True
        return bool(re.search(r"\b" + e + r"\b\W{0,3}" + AFTER_WORDS + r"\b|\b" + SECTION_WORDS + r"\W{1,3}\W?" + e
                              + r"\b", text, re.I))
    return bool(re.search(r"(?:[\"'`.]|lookup\s*\(\s*)" + e + r"\b|\b" + e + r"\b\W{0,3}" + AFTER_WORDS + r"\b",
                          text, re.I))


def fit_check(winner, plus):
    """Problems (empty = fits): all 32 designs validate, render and are distinct; neither the answering procedure nor
    the fact-writing rules name a header or key of either layout; A+ keeps the names B- uses; each level differs from
    the winner's; length limits; examples match their layout (code review, 2026-10-29)."""
    from v3.genome.render import render
    probs = []
    g0 = S.upgrade(winner["genes"])
    ak = factor_a_kind(winner)
    a = plus.get("A") or {}
    if ak == "layout_text":
        lt = a.get("layout_text")
        if not isinstance(lt, str) or not lt.strip():
            probs.append("A: missing layout_text")
        elif len(lt) > 1200:
            probs.append("A: layout_text longer than 1,200 characters")
    elif ak == "core_params" and a.get("core_params") == g0["core"].get("params"):
        probs.append("A: the new layout is the same as the winner's")
    elif ak == "schema_fields" and sorted(a.get("schema_fields") or []) == sorted(g0["schema"].get("fields") or []):
        probs.append("A: the new schema fields are the same as the winner's")
    if ak == "core_params":
        new = a.get("core_params") or {}
        missing = named_in_guidance(winner) - set(new.get("headers", [])) - set(new.get("keys", []))
        if missing:
            probs.append(f"A: the new layout must keep {sorted(missing)} (the winner's fact-writing rules name them)")
    if (plus.get("B") or {}).get("guidance_text", "").strip() == g0["guidance"]["text"].strip():
        probs.append("B: the new fact-writing rules are the same as the winner's")
    if (plus.get("C") or {}).get("answering_text", "").strip() == g0["answering"]["text"].strip():
        probs.append("C: the new answering procedure is the same as the winner's")
    # V3-044 length rule: the new fact-writing rules are between half and three times the winner's (at least 600
    # characters allowed), so B changes the rules, not just the amount of instruction
    lo, hi = b_length_bounds(winner)
    n_new = len((plus.get("B") or {}).get("guidance_text") or "")
    if n_new and not (lo <= n_new <= hi):
        probs.append(f"B: {n_new} characters; the length rule allows {lo} to {hi}")
    for f in ("B", "C"):
        key = "guidance_text" if f == "B" else "answering_text"
        t = (plus.get(f) or {}).get(key)
        if not isinstance(t, str) or not t.strip():
            probs.append(f"{f}: missing {key}")
        elif f == "C" and len(t) > 1200:                 # B is bounded by its length rule alone (code review)
            probs.append("C: longer than 1,200 characters")
    d = plus.get("D") or {}
    has_ex = bool(g0["examples"]["items"])
    for key in ("examples", "examples_new_layout", "winner_example_new_layout"):
        ex = d.get(key, [])
        if not isinstance(ex, list) or any(not isinstance(e, str) or len(e) > 2500 for e in ex):
            probs.append(f"D: {key} must be a list of strings under 2,500 characters")
    if bool(d.get("examples")) == has_ex:
        probs.append("D: " + ("the alternative to the winner's example is none (examples: [])" if has_ex else
                              "write one example (examples), since the winner has none"))
    if not has_ex and d.get("examples") and not d.get("examples_new_layout"):
        probs.append("D: also give the same example in your new layout (examples_new_layout)")
    if not has_ex and d.get("winner_example_new_layout"):
        probs.append("D: the winner has no example, so give no winner_example_new_layout")
    if has_ex and not d.get("winner_example_new_layout"):
        probs.append("D: give the winner's example rewritten in your new layout (winner_example_new_layout)")
    # the new-layout example must really be in the new layout (code review): whichever example it rewrites, for any
    # core, it differs from the old one; for a notes core it uses a name that is new in layout A+
    for key in ("examples", "examples_new_layout", "winner_example_new_layout"):
        if any(isinstance(e, str) and not e.strip() for e in (d.get(key) or [])):
            probs.append(f"D: {key} has an empty example")
    old_p, new_p = g0["core"].get("params") or {}, (a.get("core_params") or {}) if ak == "core_params" else {}
    old_names = set(old_p.get("headers", [])) | set(old_p.get("keys", []))
    new_names = set(new_p.get("headers", [])) | set(new_p.get("keys", []))
    must_use = (new_names - old_names) or new_names
    pairs = [(g0["examples"]["items"], d.get("winner_example_new_layout") or [], "winner_example_new_layout")] \
        if has_ex else [(d.get("examples") or [], d.get("examples_new_layout") or [], "examples_new_layout")]
    for old_ex, new_ex, new_key in pairs:
        if new_ex and old_ex and [e.strip() for e in new_ex] == [e.strip() for e in old_ex]:
            probs.append(f"D: {new_key} is identical to the old-layout example; write it in the new layout")
        if ak == "core_params" and new_ex and must_use and not any(nm in " ".join(new_ex) for nm in must_use):
            probs.append(f"D: {new_key} uses none of the new layout's own section names or keys")
    for f, key in (("B", "guidance_text"), ("C", "answering_text")):
        t = (plus.get(f) or {}).get(key) or ""
        for nm in sorted(_layout_names(winner, plus)):
            if names_layout(t, nm):
                probs.append(f"{f} names the layout {'header' if nm.isupper() else 'key'} {nm!r}")
    if plus.get("E", {}).get("max_turns") == g0["numeric"]["max_turns"]:
        probs.append("E: the alternative turn budget equals the winner's")
    if probs:
        return probs
    ids = set()
    rendered = set()
    for n in range(32):
        try:
            g = build(winner, plus, n)
            arm = render(g)
            ids.add(g["id"])
            rendered.add((arm.guide(), json.dumps(arm.tools(), sort_keys=True)))   # only what the builder sees
        except Exception as e:
            probs.append(f"design {n + 1} ({''.join(f + v for f, v in levels(n).items())}) does not work: "
                         f"{type(e).__name__}: {str(e)[:200]}")
    if not probs and len(ids) != 32:
        probs.append(f"the 32 designs have only {len(ids)} distinct genomes (a part's two versions are identical)")
    if not probs and len(rendered) != 32:            # code review: a part that never reaches the builder
        probs.append(f"the 32 designs render to only {len(rendered)} distinct guides and tool sets: a part does not "
                     "reach the builder")
    return probs
