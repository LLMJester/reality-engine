"""Seed genomes (spec M2): the starting population, ported from v2 designs. Each seed's core is implemented by the
named v2 arm (read-only, via v3.genome.render), or by a v3 plain-language core (v3.cores) for V3_SEEDS; seeds are the
only genomes without parents."""
import json
import pathlib

from v3.genome.schema import gene_hash, genome_id, make, upgrade

SEED_DIR = pathlib.Path(__file__).resolve().parent / "seeds"

_NUM = {"max_turns": 12, "retries": 2, "check_every": 1, "samples": 50, "tolerance": 0.05}
_TI = {"errors": "first_only", "submit": "tool_arg", "patch": True}

SEEDS = {
    # name: (v2 arm, core, schema fields, protocol, queries, guidance)
    "A0P": ("A0+", "prose", [], "one_shot", [],
            "Reason step by step in plain language, then answer."),
    "G1C3": ("G1C3", "equations", ["variables", "units", "parameters", "assumptions"], "one_shot",
             ["what_if", "explain", "trajectory"],
             "Write the whole system as named variables with explicit equations, then query the model."),
    "G1C1": ("G1C1", "probabilistic", ["variables", "parameters", "assumptions", "uncertainty"], "one_shot",
             ["what_if", "sensitivity"],
             "Write a parameterised probabilistic program; a what-if is a re-run with overridden parameters."),
    "A3": ("A3", "simulation", ["variables", "parameters", "time_scale"], "one_shot", ["what_if", "trajectory"],
           "Write a step-by-step simulation of the system and run it."),
    "A5": ("A5", "rules", ["variables", "assumptions"], "one_shot", ["what_if", "explain"],
           "Write the system as logical rules and facts, then query them."),
    "A2": ("A2", "graph", ["variables", "assumptions", "interventions"], "one_shot", ["what_if", "explain"],
           "Write the system as a causal graph with signed edges, then trace interventions."),
    # run 5 (V3-042): the baseline "Luna's own choice" - plain notes core, no format suggested, no format check
    # V3-044: the no-notes baseline; run 5 skips its build phase (v3.run5.blind_episode)
    "BLIND": ("A0+", "prose", [], "one_shot", [],
              "BLIND baseline: no notes are written; the question is answered from the question alone."),
    "OWN": ("A0+", "prose", [], "one_shot", [],
            "Write notes about this system in whatever form you judge best for answering questions about it "
            "later."),
}

# v3-native plain-language cores (v3.cores): name: (core, core params, protocol, queries, guidance, notes)
V3_SEEDS = {
    "SEC_A1": ("sectioned", {"headers": ["ENTITIES", "STATE", "MECHANISMS", "CAUSAL LINKS", "TIMING", "UNCERTAINTY"],
                             "tags": True, "granularity": "coarse"}, "stepwise_checked", [],
               "Describe the system in plain language under the section headers, one element per bullet.",
               "structured prose: v2 arm A1 sections and provenance tags"),
    "SEC_FLOW": ("sectioned", {"headers": ["PURPOSE", "PARTS", "FLOWS", "RULES", "TIMELINE", "UNKNOWNS"],
                               "tags": False, "granularity": "coarse"}, "template_first", [],
                 "Describe the system in plain language under the section headers, one element per bullet.",
                 "structured prose: purpose, parts, flows, rules, timeline, unknowns"),
    "ONT_BASIC": ("ontology", {"headers": ["THINGS", "HIERARCHY", "RELATIONSHIPS", "RULES", "PROPERTIES"],
                               "relations": ["causes", "increases", "decreases", "drives", "requires", "prevents",
                                             "precedes", "contains", "produces", "consumes", "depends on"],
                               "hierarchy": ["is_a", "part_of"], "max_depth": 4}, "stepwise_checked",
                  ["explain", "what_if"],
                  "Describe the system as plain sentences in the ontology grammar, then query it.",
                  "plain-text ontology with lookup and path queries"),
    # run 3 seeds from the format research (V3-034, docs/SEEDS_RUN3.md): plain-language formats the literature rates
    # and runs 1-2 did not cover, written with the sectioned core (no new code)
    "FACTS": ("sectioned", {"headers": ["THINGS", "VALUES", "RULES", "LINKS", "TIMING"], "tags": False,
                            "granularity": "fine"}, "one_shot", [],
              "Write every fact as one short bullet in key-value form: 'name: value' (for example 'T1.level: 40% at "
              "start') or 'a -> b: relation' (for example 'P1 -> T1: fills'). One fact per bullet, no prose, no "
              "repetition. Put each fact under the section it belongs to.",
              "key-value fact lines, one fact per bullet (research: compact facts and key-value lists aid lookup)"),
    "STATES": ("sectioned", {"headers": ["PARTS", "STATES", "EVENTS", "TRANSITIONS", "TIMING", "UNKNOWNS"],
                             "tags": False, "granularity": "coarse"}, "one_shot", [],
               "Describe the system as a state machine. STATES: each part's possible states or value range and its "
               "starting state. EVENTS: what can happen and when it counts as happening. TRANSITIONS: one bullet per "
               "rule, written 'WHEN <event> [IF <condition>] THEN <part> -> <new state or change>'. TIMING: delays "
               "and durations. UNKNOWNS: what the description leaves open.",
               "state machine with WHEN/IF/THEN transitions (research: explicit state tracking)"),
    "SPEC": ("sectioned", {"headers": ["CONTEXT", "COMPONENTS", "INTERFACES", "REQUIREMENTS", "CONSTRAINTS",
                                       "OPEN QUESTIONS"], "tags": False, "granularity": "coarse"}, "template_first",
             [],
             "Write the system as a specification for someone who will build it. REQUIREMENTS: one bullet per "
             "behaviour in EARS form: 'WHEN <trigger> THE SYSTEM SHALL <response>', 'WHILE <state> THE SYSTEM SHALL "
             "<response>', 'IF <condition> THEN <part> SHALL <response>', with exact values and timings. COMPONENTS "
             "and INTERFACES: the parts and what passes between them, with quantities.",
             "EARS-style build specification (research: spec-driven development for AI builders)"),
    "ONT_LOOSE": ("ontology", {"headers": ["THINGS", "HIERARCHY", "RELATIONSHIPS", "RULES", "NOTES"],
                               "relations": ["causes", "increases", "decreases", "drives", "requires", "prevents",
                                             "precedes", "contains", "produces", "consumes", "depends on"],
                               "hierarchy": ["is_a", "part_of"], "max_depth": 4, "strict": False}, "one_shot",
                  ["explain", "what_if"],
                  "Describe the system in plain sentences; use the ontology forms for key facts and plain notes "
                  "for the rest, then query it.",
                  "loose plain-text ontology: grammar for key facts, free notes for the rest (V3-009)"),
    # run 5 founders (V3-042, docs/RUN5_DESIGN.md): JSON and Markdown as real checked cores (v3/cores/structured.py)
    "JSON": ("json", {"keys": ["things", "quantities", "links", "events", "rules", "unknowns"], "max_depth": 4},
             "one_shot", ["explain", "compare"],
             "Write the system as one JSON object a program could read. things: one object per part or actor, "
             "with name, kind and what it does. quantities: one object per measured value, with thing, name, "
             "value, unit and when it holds. links: one object per connection or cause, with from, to, relation "
             "and condition. events: one object per thing that happens, with when, what and effect. rules: one "
             "string per rule, written 'if <condition> then <consequence>'. unknowns: one string per thing the "
             "description leaves open. Use the exact names, numbers and units from the description; never invent "
             "a value.",
             "one checked JSON object: things, quantities, links, events, rules, unknowns (human, run 5)"),
    "MARKDOWN": ("markdown", {"headers": ["THINGS", "VALUES", "LINKS", "EVENTS", "RULES", "UNKNOWNS"],
                              "tables": ["THINGS", "VALUES", "LINKS", "EVENTS"]}, "one_shot", ["explain", "compare"],
                 "Write the system as Markdown tables a person could scan. THINGS: columns name | kind | what it "
                 "does. VALUES: columns thing | quantity | value | unit | when. LINKS: columns from | relation | to "
                 "| condition. EVENTS: columns when | what happens | effect. RULES: one bullet per rule, written "
                 "'if <condition> then <consequence>'. UNKNOWNS: one bullet per thing the description leaves open. "
                 "Use the exact names, numbers and units from the description; never invent a value.",
                 "checked Markdown: tables for things, values, links, events; rule bullets (human, run 5)"),
}


def build():
    out = {}
    for name, (arm, core, fields, proto, queries, text) in SEEDS.items():
        genes = {"core": {"kind": core, "params": {"v2_arm": arm}}, "schema": {"fields": fields},
                 "build_protocol": {"kind": proto, "params": {}}, "tool_interface": dict(_TI),
                 "query_layer": queries, "guidance": {"text": text, "examples": []}, "numeric": dict(_NUM)}
        out[name] = make(genes, [], "seed", 0, 0, name, f"ported from v2 arm {arm}")
    for name, (core, params, proto, queries, text, notes) in V3_SEEDS.items():
        genes = {"core": {"kind": core, "params": params}, "schema": {"fields": []},
                 "build_protocol": {"kind": proto, "params": {}}, "tool_interface": dict(_TI),
                 "query_layer": queries, "guidance": {"text": text, "examples": []}, "numeric": dict(_NUM)}
        out[name] = make(genes, [], "seed", 0, 0, name, notes)
    return out


def write():
    SEED_DIR.mkdir(exist_ok=True)
    for name, g in build().items():
        (SEED_DIR / f"{name}.json").write_text(json.dumps(g, indent=1), encoding="utf-8")


def load_all():
    """The seed genomes in the split gene form (V3-038); the files keep the pre-split form, and the ids are the same."""
    out = {}
    for p in sorted(SEED_DIR.glob("*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        g["genes"] = upgrade(g["genes"])
        g["gene_hashes"] = {k: gene_hash(k, v) for k, v in g["genes"].items()}
        assert g["id"] == genome_id(g), p.name
        out[p.stem] = g
    return out


if __name__ == "__main__":
    write()
