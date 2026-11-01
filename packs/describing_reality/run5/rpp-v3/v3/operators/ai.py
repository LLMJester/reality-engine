"""AI variation operators (spec §4.1): a Claude designer edits genes; code validates the result.

The designer runs headless with no tools at all (`--tools ""`): everything it may use is in the prompt (parent
genomes, failure targets, short builder transcript excerpts). It never sees answer keys or items' expected answers.
It must return JSON {"genes": {gene: new_value, ...}, "why": "..."}; only the genes the operator allows may change.
"""
import json
import re
import shutil
import subprocess
import tempfile

from v3.config import CFG
from v3.genome import schema as S
from v3.builders.claude_builder import claude_exe

ALLOWED = {"point": 1, "repair": 1, "guidance": 2, "big_jump": 3}
GENE_HELP = f"""Genes and their allowed values:
- core: {{"kind": one of {list(S.CORES)}, "params": {{}}}}  (v2-backed cores keep their params as given)
  - "sectioned" (structured plain language) params, exactly: {{"headers": 3-10 distinct section headers in the order
    the builder writes them, from {list(S.HEADER_VOCAB)} or new ones (UPPER CASE, letters/digits/space & / -,
    2-30 chars); "tags": bool (every element carries a [stated|derived|assumed] tag); "granularity": "coarse" |
    "fine" (fine = one fact per bullet)}}. Built = the document passes its checker (every header present and
    non-empty, only bullet elements, tags when required). Queries: none.
  - "ontology" (plain-text ontology; sentences `X is a kind of Y.`, `X is part of Y.`, `X <relation> Y[, after T]
    [, when C].`, `If C then K.`, `X has <property> <value> [unit].`) params, exactly: {{"headers": 2-10 headers (e.g.
    THINGS, HIERARCHY, RELATIONSHIPS, RULES, PROPERTIES; lines under THINGS/ENTITIES/PARTS/ACTORS declare terms);
    "relations": 2-24 distinct relation verbs of 1-3 lower-case words, from {list(S.RELATION_POOL)} or new ones;
    "hierarchy": subset of {list(S.HIERARCHY_KINDS)}; "max_depth": 1-8; optional "strict": true|false}}. Strict
    (default): built = every sentence parses, every term is declared, no hierarchy cycles. Loose (strict false):
    lines outside the grammar are kept as plain-language notes and used terms self-declare. Queries: explain = lookup(term), what_if = path(a, b), compare =
    hierarchy(term).
- schema: {{"fields": subset of {list(S.SCHEMA_FIELDS)}}}  (structured cores need "variables"; the plain-language
  cores prose / sectioned / ontology ignore it)
- build_protocol: {{"kind": one of {list(S.PROTOCOLS)}, "params": {{}}}}
- tool_interface: {{"errors": one of {list(S.ERROR_STYLES)}, "submit": one of {list(S.SUBMIT_STYLES)}, "patch": bool}}
- query_layer: list of distinct items from {list(S.QUERIES)} (the core must support them)
- guidance: {{"text": str (how the builder writes its representation), "examples": []}} (examples go in "examples")
- answering: {{"text": str (how the builder answers questions from its representation; "" = no extra advice)}}
- examples: {{"items": [str, ...] (worked examples shown to the builder)}}
- numeric: {{"max_turns": 4-24, "retries": 0-4, "check_every": 1-4, "samples": 1-200, "tolerance": 0-0.5}}"""

TASK = {
    "point": "Change exactly ONE gene to fix the most important problem in the failure targets. Keep the change small.",
    "repair": "Change exactly ONE gene so the builder stops making the most frequent construction failure shown.",
    "guidance": "Rewrite ONLY the guidance gene and/or the examples gene so the builder constructs valid models more often. "
                "Be concrete and short; teach the construction steps, not the answers.",
    "big_jump": "Redesign up to THREE genes to escape a dead end, keeping what the evidence says works.",
}


# run 4 (V3-037): the designer works from a report card aimed at answering about large systems, not at builds
ALLOWED_GENES = {"answer_method": {"answering", "query_layer"},      # V3-038: answering is its own gene
                 "guidance": {"guidance", "examples"}}
ALLOWED["answer_method"] = 2
TASK4 = {
    "point": "Change exactly ONE gene to fix the most important cause of WRONG ANSWERS on large systems (k = 4 and 8) "
             "shown in the report card. Keep the change small.",
    "repair": "Change exactly ONE gene so the builder stops making the most frequent construction failure shown.",
    "guidance": "Rewrite ONLY the guidance gene and/or the examples gene (how the builder writes its "
                "representation) so the builder answers questions about large systems (k = 4 and 8) correctly more "
                "often. Be concrete and short; teach how to record, not the answers.",
    "big_jump": "Redesign up to THREE genes with a genuinely different idea for describing and reasoning about "
                "large systems, keeping what the evidence says works.",
    "answer_method": "Change how the builder USES its representation to answer: the answering gene (and, if the "
                     "core supports queries, the query layer), e.g. an explicit step-by-step procedure: find which "
                     "subsystems a question touches, collect their facts, follow the links and timing, then answer. "
                     "Do not change how the notes are written (the guidance gene).",
}
GOAL4 = ("Goal: the design with which a small AI builder (GPT-6 Luna) describes complex systems most accurately. "
         "The builder first writes its representation from the description; questions are then answered from that "
         "representation alone. Systems have k = 1, 2, 4 or 8 subsystems whose paragraphs are mixed together. "
         "Builds already pass almost always: the score is lost on WRONG ANSWERS about large systems (k = 4 and 8), "
         "where every design so far falls from about 0.8 at k = 1 to about 0.53 at k = 8. The REPORT CARD below is "
         "PARENT 1's results: its score by system size, its build rate and, from a few PRACTICE items (never used to "
         "score designs), questions it answered wrongly with its own representation, so you can see what went wrong. ")


def prompt(op, parents, targets, excerpts):
    ps = "\n\n".join(f"PARENT {i + 1} (id {p['id']}):\n{json.dumps(p['genes'], indent=1)}" for i, p in enumerate(parents))
    ex = "\n\n".join(excerpts[:3]) or "(none)"
    if isinstance(targets, dict) and "report_card" in targets:
        return (f"You design representations that an AI builder uses to construct a model of a described system, "
                f"then answer questions from it. {GOAL4}Your job: {TASK4[op]}\n\n{GENE_HELP}\n\n{ps}\n\n"
                f"REPORT CARD (PARENT 1):\n{json.dumps(targets['report_card'], indent=1)}\n\n"
                f"Reply with ONLY a JSON object: {{\"genes\": {{<gene>: <new value>, ...}}, \"why\": \"<one sentence>\"}}. "
                f"Include only the genes you change.")
    # run 3 (V3-034): targets are parent 1's own failure profile, and the designer is told the goal
    goal = ("" if not isinstance(targets, dict) or "score_by_k" not in targets else
            "Goal: the design with which a small AI builder describes complex systems most accurately. The builder "
            "first writes its representation from the description; questions are then answered from that "
            "representation alone. Systems have k = 1, 2, 4 or 8 independent subsystems whose paragraphs are mixed "
            "together. The failure targets below are PARENT 1's results on its latest evaluation (construction = "
            "the representation failed its checks; reasoning = built, but scored under 0.5). ")
    return (f"You design representations that an AI builder uses to construct and query a model of a described "
            f"system, then answer questions. {goal or 'The builder fails mostly at CONSTRUCTION. '}Your job: "
            f"{TASK[op]}\n\n"
            f"{GENE_HELP}\n\n{ps}\n\nFAILURE TARGETS (counts from the latest evaluation):\n{json.dumps(targets, indent=1)}"
            f"\n\nBUILDER TRANSCRIPT EXCERPTS:\n{ex}\n\n"
            f"Reply with ONLY a JSON object: {{\"genes\": {{<gene>: <new value>, ...}}, \"why\": \"<one sentence>\"}}. "
            f"Include only the genes you change.")


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if isinstance(d.get("genes"), dict) else None


def designer_call(text, timeout=300, model=None):
    work = tempfile.mkdtemp(prefix="v3d_")
    try:
        p = subprocess.run([claude_exe(), "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
                            "--setting-sources", "project", "--no-session-persistence",
                            "--model", model or CFG["models"]["designer_claude"],
                            "--system-prompt", "You are a precise designer. Output only JSON."],
                           input=text, cwd=work, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        try:
            return json.loads(p.stdout).get("result", "")
        except ValueError:
            return p.stdout
    finally:
        shutil.rmtree(work, ignore_errors=True)


def apply(op, parents, targets, excerpts, *, call=designer_call, generation=0, island=0, tries=2):
    """Returns a validated, compatible child genome or None. `call(text) -> str` is injectable for tests."""
    base = parents[0]
    for _ in range(tries):
        d = parse(call(prompt(op, parents, targets, excerpts)))
        if not d:
            continue
        changed = [k for k in d["genes"] if k in S.GENES]
        if not changed or len(changed) > ALLOWED[op] or \
                (op in ALLOWED_GENES and not set(changed) <= ALLOWED_GENES[op]):
            continue
        genes = json.loads(json.dumps(base["genes"]))
        for k in changed:
            genes[k] = d["genes"][k]
        try:
            child = S.make(genes, [p["id"] for p in parents], op, generation, island, notes=d.get("why", ""))
        except S.GenomeError:
            continue
        if child["id"] in {p["id"] for p in parents} or S.incompatibilities(child):
            continue
        child["op_notes"] = {"changed": changed, "why": d.get("why", "")}
        return child
    return None


# ---------------------------------------------------------------- run 4 (V3-037): smart combining
def combine_prompt(a, b, card_a, card_b, gene_stats, bad_mixes):
    return ("You combine two parent designs into one child for a small AI builder (GPT-6 Luna) that writes a "
            "representation of a described system and then answers questions from it alone. Systems have k = 1, 2, "
            f"4 or 8 subsystems; the score is lost mostly on large systems. The child takes EACH of the {len(S.GENES)} genes whole "
            "from parent A or parent B: no new values. Take each gene from the parent whose STRENGTHS it carries, "
            "never a weak part: use the strengths cards (score by system size, build rate, practice accuracy) and "
            "the gene evidence (how designs carrying each gene variant scored against the average). Genes must fit "
            "together (e.g. a guidance text written for one core rarely suits another core).\n\n"
            f"{GENE_HELP}\n\nPARENT A (id {a['id']}):\n{json.dumps(a['genes'], indent=1)}\n\n"
            f"STRENGTHS CARD A:\n{json.dumps(card_a, indent=1)}\n\nPARENT B (id {b['id']}):\n"
            f"{json.dumps(b['genes'], indent=1)}\n\nSTRENGTHS CARD B:\n{json.dumps(card_b, indent=1)}\n\n"
            f"GENE EVIDENCE (per gene: A's and B's variant, mean score difference of designs carrying it, designs):\n"
            f"{json.dumps(gene_stats, indent=1)}\n\nRECENT BAD MIXES (children that scored below both parents; "
            f"avoid repeating these picks):\n{json.dumps(bad_mixes, indent=1) if bad_mixes else '(none yet)'}\n\n"
            "Reply with ONLY a JSON object: {\"pick\": {<gene>: \"A\" or \"B\", ... all 7 genes}, \"why\": {<gene>: "
            "\"<the strength this gene carries, one short sentence>\", ...}, \"summary\": \"<one sentence>\"}. Take at "
            "least one gene from each parent. Guidance (how notes are written), answering (how questions are answered "
            "from them) and examples are separate genes, so one parent's way of writing can be joined with another's "
            "way of answering.")


def smart_combine(a, b, card_a, card_b, gene_stats, bad_mixes, *, call=designer_call, generation=0, tries=2):
    """A child taking each gene whole from the stronger parent for it, chosen by the designer with a reason per gene.
    Returns a validated, compatible child or None."""
    for _ in range(tries):
        m = re.search(r"\{.*\}", call(combine_prompt(a, b, card_a, card_b, gene_stats, bad_mixes)) or "", re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except ValueError:
            d = None
        pick = (d or {}).get("pick") or {}
        if set(pick) != set(S.GENES) or set(pick.values()) != {"A", "B"}:
            continue
        live = S.live_genes(a["genes"] if pick["core"] == "A" else b["genes"])
        if {pick[k] for k in live} != {"A", "B"}:      # a parent giving only an ignored gene is no real mix
            continue
        genes = {k: json.loads(json.dumps((a if pick[k] == "A" else b)["genes"][k])) for k in S.GENES}
        try:
            child = S.make(genes, [a["id"], b["id"]], "smart_combine", generation, 0, notes=d.get("summary", ""))
        except S.GenomeError:
            continue
        if child["id"] in (a["id"], b["id"]) or S.incompatibilities(child):
            continue
        why = d.get("why") if isinstance(d.get("why"), dict) else {}
        child["op_notes"] = {"pick": pick, "why": {k: str(why.get(k, ""))[:200] for k in S.GENES},
                             "summary": str(d.get("summary", ""))[:300]}
        return child
    return None


def combine3_prompt(parents, cards, gene_stats, bad_mixes):
    """The three-parent version of combine_prompt (V3-038): parents A, B, C from three different families."""
    L = "ABC"[:len(parents)]
    blocks = "".join(f"PARENT {x} (id {p['id']}):\n{json.dumps(p['genes'], indent=1)}\n\nSTRENGTHS CARD {x}:\n"
                     f"{json.dumps(c, indent=1)}\n\n" for x, p, c in zip(L, parents, cards))
    return ("You combine THREE parent designs, each from a different family, into one child for a small AI builder "
            "(GPT-6 Luna) that writes a representation of a described system and then answers questions from it "
            "alone. Systems have k = 1, 2, 4 or 8 subsystems; the score is lost mostly on large systems. The child "
            f"takes EACH of the {len(S.GENES)} genes whole from parent A, B or C: no new values. Take each gene from the parent "
            "whose STRENGTHS it carries, never a weak part: use the strengths cards (score by system size, build "
            "rate, practice accuracy) and the gene evidence (how designs carrying each gene variant scored against "
            "the average). Genes must fit together (e.g. a guidance text written for one core rarely suits another "
            "core); with three parents a clash is more likely, so check every pick against the core you take.\n\n"
            f"{GENE_HELP}\n\n{blocks}"
            f"GENE EVIDENCE (per gene: each parent's variant, mean score difference of designs carrying it, designs):\n"
            f"{json.dumps(gene_stats, indent=1)}\n\nRECENT BAD MIXES (children that scored below all their parents; "
            f"avoid repeating these picks):\n{json.dumps(bad_mixes, indent=1) if bad_mixes else '(none yet)'}\n\n"
            "Reply with ONLY a JSON object: {\"pick\": {<gene>: \"A\", \"B\" or \"C\", ... all 7 genes}, \"why\": "
            "{<gene>: \"<the strength this gene carries, one short sentence>\", ...}, \"summary\": \"<one sentence>\"}. "
            "Each of the three parents must give at least one gene that matters AND whose value differs from the "
            "other two parents' (a gene all parents share, or the schema gene that plain-language cores ignore, does "
            "not count); if no such mix fits together, reply with {\"pick\": null}. Guidance (how notes are written), answering (how questions are answered from them) "
            "and examples are separate genes, so the three parents can each give a different part.")


def smart_combine3(parents, cards, gene_stats, bad_mixes, *, call=designer_call, generation=0, tries=2):
    """A child taking each gene whole from one of three parents (V3-038), every parent used, a reason per gene.
    Returns a validated, compatible child or None."""
    L = "ABC"
    if len(parents) != 3 or len({p["id"] for p in parents}) != 3:
        return None
    by = dict(zip(L, parents))
    for _ in range(tries):
        m = re.search(r"\{.*\}", call(combine3_prompt(parents, cards, gene_stats, bad_mixes)) or "", re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except ValueError:
            d = None
        pick = (d or {}).get("pick") or {}
        if set(pick) != set(S.GENES) or set(pick.values()) != set(L):
            continue
        live = S.live_genes(by[pick["core"]]["genes"])
        own = {x for k in live for x in [pick[k]]       # a gene that matters, and that only this parent has
               if all(S.canonical(by[x]["genes"][k]) != S.canonical(by[y]["genes"][k]) for y in L if y != x)}
        if own != set(L):                               # every parent must really contribute (V3-038)
            continue
        genes = {k: json.loads(json.dumps(by[pick[k]]["genes"][k])) for k in S.GENES}
        try:
            child = S.make(genes, [p["id"] for p in parents], "smart_combine3", generation, 0,
                           notes=d.get("summary", ""))
        except S.GenomeError:
            continue
        if child["id"] in {p["id"] for p in parents} or S.incompatibilities(child):
            continue
        why = d.get("why") if isinstance(d.get("why"), dict) else {}
        child["op_notes"] = {"pick": pick, "why": {k: str(why.get(k, ""))[:200] for k in S.GENES},
                             "summary": str(d.get("summary", ""))[:300]}
        return child
    return None
