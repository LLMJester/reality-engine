"""Run 5 stages 3 to N (V3-058, docs/RUN5_STAGES_DESIGN.md sections 4, 5, 7, 10): the five slots of the current best
design, the 20 candidates Opus writes for a stage, their mechanical checks, the designs built from them and the
worked-example cube.

The five slots (each edits a separate part of the genome, so any mix of slot levels builds):
  1 layout     core.params (headers, tags, granularity) of the sectioned notes core
  2 rules      the fact-writing rules: the first block of guidance.text
  3 strategy   how the builder checks its notes before saving: a second, separate block of guidance.text (process
               and verification only, never syntax); empty in FACTS, so FACTS's guidance.text is its rules exactly
  4 answering  answering.text
  5 example    the worked example (examples.items): one notes document, always written in the design's own layout
               and fact rules. Examples are indexed by (slot 1 level, slot 2 level, slot 5 level); a missing one is
               written by Opus (examples step) and must pass that design's own notes checker with no problem.

Opus writes 4 candidates per slot, one per archetype (1 subtract - for a slot that is empty in the current best, the
smallest useful addition; 2 explicit structure; 3 verify; 4 targeted tweak), in one headless session per slot with no
tools. It sees the current best's genome and the firewalled evidence of v3.plus_evidence5.stage_brief, never an item,
question, answer key, hand-written result, family score or sealed record. Every candidate passes the mechanical
checks below (failures go back to Opus with the reason, up to 3 attempts per slot) before the human gate.
"""
import copy
import hashlib
import json
import re

from v3.genome import schema as S

SLOTS = {1: "layout", 2: "rules", 3: "strategy", 4: "answering", 5: "example"}
ARCHETYPES = {
    1: "Subtract: loosen or remove rules that cause verbosity or needless constraints. If this part is empty in the "
       "current design, write the smallest useful addition instead (a few short sentences, or a small example).",
    2: "Explicit structure: entity and edge identifiers, input-output tables, aimed at large systems (k = 8).",
    3: "Verify: explicit trace or completeness checks before output.",
    4: "Targeted tweak: keep the current structure and make one surgical change aimed at k = 4 and k = 8.",
}
MAX_CHARS = {2: 2500, 3: 2500, 4: 2500, 5: 4000}       # raised by the human 2026-10-30 (V3-058); lengths are reported
MIN_RULES_SHARE = 0.5                                   # the fact rules keep at least half the current length
DIVERSE = 0.25                                          # archetypes 1-3: token edit distance between each pair
TWEAK_MIN = 0.10                                        # archetype 4: changes at least 10% of the slot
ATTEMPTS = 3
# the only reasons the human may send candidates back (state/run5/stage<n>/REJECT: lines "slot <k>: <reason>")
REJECT_REASONS = ("refers to tools the design lacks", "mentions items, answers or results", "breaks the length rule",
                  "changes another slot", "breaks the firewall")
SYNTAX_RE = re.compile(r"->|=>|##|`|['\"][^'\"\n]*[:=][^'\"\n]*['\"]")   # slot 3 may not define or quote syntax

SYSTEM = "You are an expert in knowledge representation for language models."
TASK = """You design part of an experiment on how a small AI model should write down a system it reads about.

The task the design serves: a small model reads a long, messy description of a system (a plant, a network, a
service, or a mix of several separate systems: k = 1, 2, 4 or 8 of them in one description) WITHOUT seeing any
questions, and writes notes in the form the design prescribes, using the design's tools. The notes are checked by a
fixed checker. Later the description is taken away and the model must answer questions about the system using only
its notes. A design scores well only if its notes really capture the system: parts, quantities, causes, rules,
timing and what is unknown."""

SLOT_DEF = {
    1: "the LAYOUT of the notes: the sectioned core's parameters (currently {layout}). Give a complete replacement "
       "with exactly the keys headers, tags, granularity: 3 to 10 headers, each 2-30 characters, starting with a "
       "letter, using only upper-case letters, digits, single spaces and & / - (no underscores); tags true or false "
       "(true makes every bullet carry a provenance tag); granularity \"fine\" (one fact per one-line bullet) or "
       "\"coarse\" (an element may continue on indented lines).{keep}",
    2: "the FACT-WRITING RULES: how each fact is written down in the notes (currently: \"{rules}\"). Give a complete "
       "replacement, {lo} to {hi} characters. Do not name any section header (the layout may change separately).",
    3: "the BUILD CHECK: how the builder checks its notes before and after saving - a strategy of steps and "
       "self-checks (currently {strategy}). It is added after the fact-writing rules as a separate paragraph. "
       "Process and verification only: do not define or quote any bullet punctuation, fact syntax or checker syntax "
       "(no quoted examples of notes, no '->', no '##', no backticks) and do not name any section header. At most "
       "{hi} characters.",
    4: "the ANSWERING PROCEDURE: how the answerer works from the notes to an answer (currently {answering}). It must "
       "work with any layout: do not name any section header. At most {hi} characters.",
    5: "the WORKED EXAMPLE shown to the builder: one notes document for a small invented system (not a real "
       "incident), exactly as the builder would save it, in the CURRENT layout and following the CURRENT fact-writing "
       "rules (currently {example}). It must pass the notes checker with no problem: {checker} At most {hi} "
       "characters.",
}


def _q(text):
    return f'"{text}"' if text.strip() else "empty"


# ---------------------------------------------------------------- parts and designs
def parts_of(genome):
    """The five slot values of a genome whose guidance has no separate strategy block (FACTS at stage 3)."""
    g = S.upgrade(genome["genes"])
    if g["core"]["kind"] != "sectioned":
        raise ValueError("the stages are defined for a sectioned notes core (FACTS)")
    ex = [e for e in g["examples"]["items"] if str(e).strip()]
    if len(ex) > 1:
        raise ValueError("the stages expect at most one worked example")
    return {"layout": copy.deepcopy(g["core"]["params"]), "rules": g["guidance"]["text"], "strategy": "",
            "answering": g["answering"]["text"], "example": ex[0] if ex else ""}


def guidance_text(parts):
    """The fact rules, then the build strategy as its own paragraph; with no strategy, exactly the rules."""
    return parts["rules"] if not parts["strategy"].strip() else parts["rules"].rstrip() + "\n\n" + parts["strategy"]


def compose(base_genome, parts, parents, name, notes=""):
    """A genome: the base genome's genes with the five slots set from parts. Parts equal to the base genome's give the
    base genome's id (the same genes)."""
    g = copy.deepcopy(S.upgrade(base_genome["genes"]))
    g["core"]["params"] = copy.deepcopy(parts["layout"])
    g["guidance"]["text"] = guidance_text(parts)
    g["answering"]["text"] = parts["answering"]
    g["examples"]["items"] = [parts["example"]] if parts["example"].strip() else []
    return S.make(g, parents, "stage", 0, 0, name, notes)


def checker_problems(parts, text):
    """The design's own notes checker on an example (sectioned core)."""
    from v3.cores.sectioned import check
    p = parts["layout"]
    return check(text, list(p["headers"]), bool(p["tags"]), p["granularity"])


def example_key(l1, l2, l5):
    return f"{l1}|{l2}|{l5}"


class MissingExample(KeyError):
    pass


def design_parts(current, levels, cands, cube):
    """Parts of the design with slot levels {slot: "0" (current) or a candidate label}. The example is the cube's
    rendering for (slot 1, slot 2, slot 5) levels; none if the slot 5 level has no example."""
    parts = copy.deepcopy(current)
    for slot, lab in levels.items():
        if lab != "0" and slot != 5:
            parts[SLOTS[slot]] = copy.deepcopy(cands[lab]["value"])
    l1, l2, l5 = (levels.get(s, "0") for s in (1, 2, 5))
    base = cube.get(example_key("0", "0", l5), "")
    if not base.strip():
        parts["example"] = ""
    else:
        key = example_key(l1, l2, l5)
        if key not in cube:
            raise MissingExample(key)
        parts["example"] = cube[key]
    return parts


def initial_cube(current, cands):
    """The examples known after the candidates are written: the current one, each slot 5 candidate (in the current
    layout and rules) and, when the current design has an example, each slot 1 and slot 2 candidate's rewrite of it."""
    cube = {example_key("0", "0", "0"): current["example"]}
    for lab, c in cands.items():
        if c["slot"] == 5:
            cube[example_key("0", "0", lab)] = c["value"]
        elif c["slot"] in (1, 2) and c.get("example_rewrite"):
            cube[example_key(lab, "0", "0") if c["slot"] == 1 else example_key("0", lab, "0")] = c["example_rewrite"]
    return cube


def needed_examples(advancing, cands, cube):
    """Cube keys the confirmation needs that do not exist yet: every (slot 1, slot 2, slot 5) level combination over
    the advancing slots whose slot 5 level has an example."""
    lv = {s: ["0"] + ([advancing[s]] if s in advancing else []) for s in (1, 2, 5)}
    out = []
    for l1 in lv[1]:
        for l2 in lv[2]:
            for l5 in lv[5]:
                if cube.get(example_key("0", "0", l5), "").strip() and example_key(l1, l2, l5) not in cube:
                    out.append(example_key(l1, l2, l5))
    return out


# ---------------------------------------------------------------- what the writer sees
def layout_names(current, cands=None):
    """Every header of the current layout and of any slot 1 candidate."""
    names = set(current["layout"]["headers"])
    for c in (cands or {}).values():
        if c["slot"] == 1 and isinstance(c.get("value"), dict):
            names |= set(c["value"].get("headers") or [])
    return names


def naming_clashes(current, cands):
    """Slots 2-4 whose candidates name a header of the current layout or of any slot 1 candidate."""
    from v3.plus5 import names_layout
    names = layout_names(current, cands)
    return sorted({c["slot"] for c in cands.values() if c["slot"] in (2, 3, 4)
                   and any(names_layout(c["value"], nm) for nm in names)})


def named_in_texts(current):
    """Headers the current rules, strategy or answering name: a new layout must keep them."""
    from v3.plus5 import names_layout
    text = "\n".join(current[k] for k in ("rules", "strategy", "answering"))
    return {h for h in current["layout"]["headers"] if names_layout(text, h)}


def rules_bounds(current):
    return int(MIN_RULES_SHARE * len(current["rules"])), MAX_CHARS[2]


def slot_brief(slot, current, genome, evidence, forbidden=()):
    from v3.cores.sectioned import SectionedCore
    lo, hi = rules_bounds(current)
    keep = named_in_texts(current)
    checker = SectionedCore(current["layout"]).guide().split("## Tools")[0].strip().replace("\n", " ")
    sdef = SLOT_DEF[slot].format(layout=json.dumps(current["layout"]), rules=current["rules"],
                                 strategy=_q(current["strategy"]), answering=_q(current["answering"]),
                                 example=_q(current["example"]), lo=lo, hi=MAX_CHARS.get(slot, hi), checker=checker,
                                 keep=(f" Keep these headers, because the current texts name them: "
                                       f"{', '.join(sorted(keep))}." if keep else ""))
    rewrite = ""
    if slot in (1, 2) and current["example"].strip():
        rewrite = ("\nThe current design has a worked example. For each candidate also give the same example "
                   "rewritten in your candidate's " + ("layout" if slot == 1 else "fact-writing rules")
                   + " (same content), as \"example_rewrite\"; it must pass the notes checker of the candidate.")
    forb = (f"\nNever name any of these section headers (the layout can change separately): "
            f"{', '.join(sorted(forbidden))}." if forbidden and slot in (2, 3, 4) else "")
    value = {1: '{"headers": [...], "tags": false, "granularity": "fine"}', 5: '"## HEADER\\n- ..."'}.get(slot, '"..."')
    arche = "\n".join(f"  {k}. {v}" for k, v in ARCHETYPES.items())
    return f"""{TASK}

Below is the current best design, as a genome (JSON), and what earlier stages measured. Your job: write FOUR
alternative versions of ONE part of it, {sdef}

The four versions must follow four different approaches, one each:
{arche}
Versions 1-3 must differ clearly from each other (at least a quarter of their words); version 4 makes one small
change to the current part that none of the others makes. Each must be what you judge most likely to make a small
model's notes and answers more accurate, from first principles and the evidence below - not a variation for its own
sake. Each version must work with every combination of the other parts' versions.{rewrite}{forb}

THE CURRENT BEST DESIGN
{json.dumps(genome["genes"], indent=1)}

{evidence}

Reply with ONE JSON object and nothing else:
{{"candidates": [{{"archetype": 1, "value": {value}, "rationale": "one sentence"{', "example_rewrite": "..."' if rewrite else ''}}},
  ... one for each archetype 1, 2, 3, 4]}}"""


def example_brief(target, source_example):
    from v3.cores.sectioned import SectionedCore
    checker = SectionedCore(target["layout"]).guide().split("## Tools")[0].strip()
    return f"""{TASK}

Rewrite the worked example of notes below so that it follows this design's layout and fact-writing rules, with
exactly the same content (the same invented system, parts, values, rules and timing).

THE DESIGN'S LAYOUT AND CHECKER
{checker}

THE DESIGN'S FACT-WRITING RULES
{target['rules']}

THE EXAMPLE TO REWRITE
{source_example}

It must pass the checker above with no problem, at most {MAX_CHARS[5]} characters. Reply with ONE JSON object and
nothing else: {{"example": "## HEADER\\n- ..."}}"""


def call_opus(user, call=None):
    if call is not None:
        return call(SYSTEM, user)
    from v3.hand5.factory import claude_text
    return claude_text(SYSTEM, user, timeout=1800, model="opus")["final_raw"]


def parse(text):
    from v3.jsonx import first_object
    return first_object(text)


# ---------------------------------------------------------------- mechanical checks
def tokens(v):
    if isinstance(v, dict):
        v = " ".join(list(v.get("headers") or []) + [f"tags {v.get('tags')}", f"granularity {v.get('granularity')}"])
    return re.findall(r"\w+|[^\w\s]", str(v or "").lower())


def units(v):
    if isinstance(v, dict):
        return {f"header {h}" for h in v.get("headers") or []} | {f"tags {v.get('tags')}",
                                                                   f"granularity {v.get('granularity')}"}
    out = set()
    for line in str(v or "").splitlines():
        if line.lstrip().startswith("#") or S.HEADER_RE.match(line.strip()):
            continue                                      # a header line is layout, not content
        for s in re.split(r"(?<=[.!?;])\s+", line):
            s = " ".join(s.lower().split())
            if s:
                out.add(s)
    return out


def distance(a, b):
    """Token edit distance, normalised by the longer token list (0 = same, 1 = nothing in common)."""
    x, y = tokens(a), tokens(b)
    if not x and not y:
        return 0.0
    prev = list(range(len(y) + 1))
    for i, tx in enumerate(x, 1):
        cur = [i] + [0] * len(y)
        for j, ty in enumerate(y, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (tx != ty))
        prev = cur
    return prev[-1] / max(len(x), len(y))


def length(v):
    return len(json.dumps(v, sort_keys=True)) if isinstance(v, dict) else len(str(v or ""))


def check_slot(slot, raw, current, genome, forbidden=()):
    """(candidates {label: {...}}, problems) for one slot's reply. Labels: s<slot>a<archetype>."""
    from v3.genome.render import render
    from v3.plus5 import names_layout
    probs, out = [], {}
    cs = raw.get("candidates") if isinstance(raw, dict) else None
    if not isinstance(cs, list) or sorted(c.get("archetype") for c in cs if isinstance(c, dict)) != [1, 2, 3, 4]:
        return {}, ["give exactly four candidates, archetypes 1, 2, 3 and 4, one each"]
    cur = current[SLOTS[slot]]
    for c in sorted(cs, key=lambda c: c["archetype"]):
        a, v = c["archetype"], c.get("value")
        lab = f"s{slot}a{a}"
        p = []
        if slot == 1:
            try:
                S.validate_core_params("sectioned", v)
            except (S.GenomeError, TypeError, KeyError, AttributeError) as e:
                p.append(f"the layout is not valid: {e}")
            else:
                missing = named_in_texts(current) - set(v["headers"])
                if missing:
                    p.append(f"the layout must keep {sorted(missing)} (the current texts name them)")
        elif not isinstance(v, str) or not v.strip():
            p.append("the value must be a non-empty text")
        if not p:
            n = length(v)
            if slot == 2:
                lo, hi = rules_bounds(current)
                if not lo <= n <= hi:
                    p.append(f"{n} characters; the rules must be {lo} to {hi}")
            elif slot in MAX_CHARS and n > MAX_CHARS[slot]:
                p.append(f"{n} characters; at most {MAX_CHARS[slot]}")
            if (json.dumps(v, sort_keys=True) if slot == 1 else str(v).strip()) == \
                    (json.dumps(cur, sort_keys=True) if slot == 1 else str(cur).strip()):
                p.append("the same as the current part")
            if slot in (2, 3, 4):
                for nm in sorted(set(forbidden) | layout_names(current)):
                    if names_layout(v, nm):
                        p.append(f"names the section header {nm!r}")
            if slot == 3 and SYNTAX_RE.search(v):
                p.append("the build check quotes or defines syntax ('->', '##', backticks or a quoted 'name: "
                         "value'); describe process and checks only")
            if slot == 5:
                bad = checker_problems(current, v)
                if bad:
                    p.append(f"the example fails the notes checker: {bad[0]['msg']}"
                             + (f" (line {bad[0]['line']})" if bad[0].get("line") else "")
                             + (f", and {len(bad) - 1} more" if len(bad) > 1 else ""))
        rw = c.get("example_rewrite")
        if not p and slot in (1, 2) and current["example"].strip():
            target = dict(current, **{SLOTS[slot]: v})
            if not isinstance(rw, str) or not rw.strip():
                p.append("give example_rewrite: the current example in this candidate's "
                         + ("layout" if slot == 1 else "rules"))
            elif rw.strip() == current["example"].strip():
                p.append("example_rewrite is identical to the current example")
            elif checker_problems(target, rw):
                p.append(f"example_rewrite fails the candidate's notes checker: {checker_problems(target, rw)[0]['msg']}")
            elif len(rw) > MAX_CHARS[5]:
                p.append(f"example_rewrite is longer than {MAX_CHARS[5]} characters")
        if not p:
            try:
                levels = {slot: lab}
                cands = {lab: {"slot": slot, "value": v, "example_rewrite": rw}}
                parts = design_parts(current, levels, cands, initial_cube(current, cands))
                render(compose(genome, parts, [genome["id"]], f"check {lab}"))
            except Exception as e:
                p.append(f"the design does not build: {type(e).__name__}: {str(e)[:160]}")
        probs += [f"candidate {a}: {x}" for x in p]
        out[lab] = {"slot": slot, "archetype": a, "value": v, "rationale": str(c.get("rationale") or "")[:300],
                    "length": length(v), **({"example_rewrite": rw} if rw else {})}
    if not probs:
        probs += diversity(slot, out, cur)
    return out, probs


def diversity(slot, cands, cur):
    """Archetypes 1-3 differ pairwise by at least DIVERSE; archetype 4 changes at least TWEAK_MIN of the current part
    and adds nothing that one of archetypes 1-3 also adds."""
    probs = []
    v = {c["archetype"]: c["value"] for c in cands.values()}
    for a, b in ((1, 2), (1, 3), (2, 3)):
        d = distance(v[a], v[b])
        if d < DIVERSE:
            probs.append(f"candidates {a} and {b} are too alike ({d:.0%} of their words differ; at least "
                         f"{DIVERSE:.0%})")
    if distance(v[4], cur) < TWEAK_MIN:
        probs.append(f"candidate 4 changes less than {TWEAK_MIN:.0%} of the current part")
    added = units(v[4]) - units(cur)
    for a in (1, 2, 3):
        same = added & units(v[a])
        if same:
            probs.append(f"candidate 4 adds what candidate {a} also has: {sorted(same)[0][:80]!r}")
    return probs


def check_screen(current, genome, cands):
    """All 21 screen designs build, render and reach the builder differently."""
    from v3.genome.render import render
    probs, seen = [], {}
    cube = initial_cube(current, cands)
    for lab in ["0"] + list(cands):
        levels = {} if lab == "0" else {cands[lab]["slot"]: lab}
        try:
            g = genome if lab == "0" else compose(genome, design_parts(current, levels, cands, cube),
                                                  [genome["id"]], lab)
            arm = render(g)
            key = (arm.guide(), json.dumps(arm.tools(), sort_keys=True))
        except Exception as e:
            probs.append(f"{lab}: does not build: {type(e).__name__}: {str(e)[:160]}")
            continue
        if key in seen:
            probs.append(f"{lab} renders the same as {seen[key]}")
        seen[key] = lab
    return probs


def write_slot(slot, current, genome, evidence, forbidden=(), call=None, log=None):
    """Up to ATTEMPTS fresh Opus sessions for one slot; each retry is told the previous attempt's problems.
    Returns (candidates, problems of the last attempt; empty when it passed)."""
    probs = None
    for attempt in range(ATTEMPTS):
        user = slot_brief(slot, current, genome, evidence, forbidden)
        if probs:
            user += "\n\nAn earlier reply failed these code checks; make sure yours passes them:\n- " + "\n- ".join(probs)
        try:
            cands, probs = check_slot(slot, parse(call_opus(user, call)), current, genome, forbidden)
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError,
                __import__("subprocess").TimeoutExpired) as e:
            cands, probs = {}, [f"the reply was not the JSON object asked for ({type(e).__name__}: {str(e)[:120]})"]
        if not probs:
            return cands, []
        if log:
            log(f"slot {slot} ({SLOTS[slot]}) failed the checks (attempt {attempt + 1}): {probs[:3]}")
    return cands, probs


def write_example(target, source, call=None, log=None):
    probs = None
    for attempt in range(ATTEMPTS):
        user = example_brief(target, source)
        if probs:
            user += "\n\nAn earlier reply failed these code checks; make sure yours passes them:\n- " + "\n- ".join(probs)
        try:
            ex = parse(call_opus(user, call)).get("example")
            probs = []
            if not isinstance(ex, str) or not ex.strip():
                probs = ["give the rewritten example as \"example\""]
            elif len(ex) > MAX_CHARS[5]:
                probs = [f"longer than {MAX_CHARS[5]} characters"]
            else:
                bad = checker_problems(target, ex)
                if bad:
                    probs = [f"fails the notes checker: {bad[0]['msg']}"
                             + (f" (line {bad[0]['line']})" if bad[0].get("line") else "")]
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError,
                __import__("subprocess").TimeoutExpired) as e:
            ex, probs = None, [f"the reply was not the JSON object asked for ({type(e).__name__}: {str(e)[:120]})"]
        if not probs:
            return ex, []
        if log:
            log(f"example failed the checks (attempt {attempt + 1}): {probs[:2]}")
    return None, probs


def freeze(doc):
    body = {k: doc[k] for k in ("stage", "current", "candidates")}
    return dict(doc, hash=hashlib.sha256(S.canonical(body).encode()).hexdigest()[:16])


def confirm_labels(m):
    return [f"D{n + 1:02d}" for n in range(2 ** m)]
