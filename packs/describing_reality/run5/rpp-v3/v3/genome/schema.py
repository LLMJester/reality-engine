"""The genome (spec §3): a design as named, typed, swappable genes, with provenance.

A genome is plain JSON:
    {"genes": {gene: value, ...}, "parents": [genome ids], "operator": str, "generation": int, "island": int,
     "name": str, "notes": str}
Its identity is the hash of its genes only: two genomes with equal genes render identical arms. Provenance (parents,
operator) is recorded beside the genes and never changes behaviour.

V3-038 (run 4): the old guidance gene is split in three, so combining can take one parent's way of writing the notes
and another's way of answering from them: guidance (how to write the representation), answering (how to answer
questions from it) and examples. A genome written before the split (7 genes, examples inside guidance) is upgraded on
make(); its identity is unchanged, because the id is the hash of the pre-split form (`legacy`), which renders the same
builder prompt: an empty answering gene adds nothing and the examples are shown exactly as before.
"""
import copy
import hashlib
import json
import re

GENES = ("core", "schema", "build_protocol", "tool_interface", "query_layer", "guidance", "answering", "examples",
         "numeric")
LEGACY_GENES = ("core", "schema", "build_protocol", "tool_interface", "query_layer", "guidance", "numeric")

CORES = ("prose", "equations", "probabilistic", "simulation", "rules", "graph", "sectioned", "ontology", "json",
         "markdown")
PLAIN_CORES = ("prose", "sectioned", "ontology", "json", "markdown")   # notes cores (no variables schema needed)
PROTOCOLS = ("one_shot", "stepwise_checked", "template_first", "sketch_refine", "decompose")
ERROR_STYLES = ("first_only", "all_at_once", "all_with_fix_hints")
SUBMIT_STYLES = ("tool_arg", "line_statements")
QUERIES = ("what_if", "explain", "compare", "sensitivity", "trajectory")
SCHEMA_FIELDS = ("variables", "units", "parameters", "assumptions", "interventions", "uncertainty", "time_scale",
                 "observations")
NUMERIC_BOUNDS = {"max_turns": (4, 24), "retries": (0, 4), "check_every": (1, 4), "samples": (1, 200),
                  "tolerance": (0.0, 0.5)}

# Which cores support which query capabilities (a query the core cannot serve makes the genome incompatible)
CORE_QUERIES = {
    "prose": set(),
    "equations": {"what_if", "explain", "compare", "sensitivity", "trajectory"},
    "probabilistic": {"what_if", "compare", "sensitivity", "trajectory"},
    "simulation": {"what_if", "compare", "sensitivity", "trajectory"},
    "rules": {"what_if", "explain", "compare"},
    "graph": {"what_if", "explain", "compare"},
    "sectioned": set(),
    "ontology": {"explain", "what_if", "compare"},    # lookup / path / hierarchy (render.ARM_QUERY_MAP["ONT"])
    "json": {"explain", "compare"},                   # lookup / search (run 5, V3-042)
    "markdown": {"explain", "compare"},
}

# ---------------------------------------------------------------- plain-language core params (genes.core.params)
# Section headers: a starter vocabulary; designers may invent others that match HEADER_RE (upper case, short).
HEADER_VOCAB = ("ENTITIES", "STATE", "MECHANISMS", "CAUSAL LINKS", "TIMING", "UNCERTAINTY", "PURPOSE", "PARTS",
                "FLOWS", "RULES", "TIMELINE", "UNKNOWNS", "HIERARCHY", "INTERFACES", "THINGS", "RELATIONSHIPS",
                "PROPERTIES", "CONSTRAINTS", "ACTORS", "GOALS")
HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 &/-]{1,29}$")
GRANULARITY = ("coarse", "fine")
# Ontology relation verbs: a starter set plus a pool the deterministic param operator draws from; designers may add
# verbs matching RELATION_RE (1-3 lower-case words, none of them a grammar word).
RELATION_VOCAB = ("causes", "increases", "decreases", "drives", "requires", "prevents", "precedes", "contains",
                  "produces", "consumes", "depends on")
RELATION_POOL = RELATION_VOCAB + ("enables", "limits", "triggers", "delays", "stores", "transfers", "controls",
                                  "replaces", "signals", "feeds", "follows", "blocks")
RELATION_RE = re.compile(r"^[a-z]+(?: [a-z]+){0,2}$")
GRAMMAR_WORDS = {"is", "are", "has", "have", "if", "then", "after", "when", "and", "or", "not", "a", "an", "the",
                 "kind", "part"}
HIERARCHY_KINDS = ("is_a", "part_of")
DECLARE_HEADERS = ("THINGS", "ENTITIES", "PARTS", "ACTORS")   # headers whose lines declare terms (ontology)
CORE_PARAM_BOUNDS = {
    "sectioned": {"headers": (3, 10)},
    "ontology": {"headers": (2, 10), "relations": (2, 24), "max_depth": (1, 8)},
    "json": {"keys": (3, 10), "max_depth": (2, 8)},
    "markdown": {"headers": (3, 10)},
}
JSON_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,29}$")
SECTIONED_DEFAULT = {"headers": ["ENTITIES", "STATE", "MECHANISMS", "CAUSAL LINKS", "TIMING", "UNCERTAINTY"],
                     "tags": True, "granularity": "coarse"}
JSON_DEFAULT = {"keys": ["things", "quantities", "links", "events", "rules", "unknowns"], "max_depth": 4}
MARKDOWN_DEFAULT = {"headers": ["THINGS", "VALUES", "LINKS", "EVENTS", "RULES", "UNKNOWNS"],
                    "tables": ["THINGS", "VALUES", "LINKS", "EVENTS"]}
ONTOLOGY_DEFAULT = {"headers": ["THINGS", "HIERARCHY", "RELATIONSHIPS", "RULES", "PROPERTIES"],
                    "relations": list(RELATION_VOCAB), "hierarchy": list(HIERARCHY_KINDS), "max_depth": 4}


class GenomeError(ValueError):
    pass


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def gene_hash(name, value):
    return hashlib.sha256(canonical([name, value]).encode("utf-8")).hexdigest()[:12]


def upgrade(genes):
    """A copy of the genes in the split form (V3-038): pre-split genes gain an empty answering gene and their examples
    move to the examples gene. Genes already split are returned as a copy."""
    g = copy.deepcopy(genes)
    if isinstance(g, dict) and set(g) == set(LEGACY_GENES) and isinstance(g.get("guidance"), dict):
        gd = g["guidance"]
        ex = gd.get("examples", [])
        g["examples"] = {"items": list(ex) if isinstance(ex, list) else []}
        g["guidance"] = {"text": gd.get("text"), "examples": []}
        g["answering"] = {"text": ""}
    return g


def legacy(genes):
    """The pre-split form of split genes: examples back inside guidance, an empty answering gene dropped. A design with
    a non-empty answering gene keeps it (no design had one before the split)."""
    if set(genes) != set(GENES):
        return genes
    g = {k: genes[k] for k in LEGACY_GENES}
    g["guidance"] = {"text": genes["guidance"]["text"],
                     "examples": list(genes["guidance"].get("examples", [])) + list(genes["examples"]["items"])}
    if genes["answering"]["text"].strip():
        g["answering"] = genes["answering"]
    return g


def live_genes(genes):
    """Genes that change what the builder does: the plain-language cores ignore the schema gene."""
    return [k for k in GENES if not (k == "schema" and genes["core"]["kind"] in PLAIN_CORES)]


def genome_id(genome):
    """Content identity: the hash of the genes only (of the pre-split form, so ids survive the V3-038 split)."""
    return "g" + hashlib.sha256(canonical(legacy(genome["genes"])).encode("utf-8")).hexdigest()[:11]


def validate(genome):
    """Raise GenomeError if a genome is malformed. Returns the genome for chaining."""
    g = genome.get("genes")
    if isinstance(g, dict) and set(g) == set(LEGACY_GENES):
        g = upgrade(g)                       # pre-split genomes (V3-038) are checked in their split form
    if not isinstance(g, dict) or set(g) != set(GENES):
        raise GenomeError(f"genes must be exactly {GENES}, got {sorted(g or {})}")
    core = g["core"]
    if not isinstance(core, dict) or core.get("kind") not in CORES:
        raise GenomeError(f"core.kind must be one of {CORES}")
    validate_core_params(core["kind"], core.get("params"))
    if not isinstance(g["schema"], dict) or not set(g["schema"].get("fields", [])) <= set(SCHEMA_FIELDS):
        raise GenomeError(f"schema.fields must be a subset of {SCHEMA_FIELDS}")
    bp = g["build_protocol"]
    if not isinstance(bp, dict) or bp.get("kind") not in PROTOCOLS:
        raise GenomeError(f"build_protocol.kind must be one of {PROTOCOLS}")
    ti = g["tool_interface"]
    if not isinstance(ti, dict) or ti.get("errors") not in ERROR_STYLES or ti.get("submit") not in SUBMIT_STYLES \
            or not isinstance(ti.get("patch"), bool):
        raise GenomeError("tool_interface needs errors, submit and boolean patch")
    if not isinstance(g["query_layer"], list) or not set(g["query_layer"]) <= set(QUERIES) \
            or len(set(g["query_layer"])) != len(g["query_layer"]):
        raise GenomeError(f"query_layer must be a list of distinct items from {QUERIES}")
    gd = g["guidance"]
    if not isinstance(gd, dict) or not isinstance(gd.get("text"), str) or not isinstance(gd.get("examples", []), list):
        raise GenomeError("guidance needs text (str) and examples (list)")
    if gd.get("examples"):
        raise GenomeError("guidance examples belong in the examples gene (V3-038)")
    an = g["answering"]
    if not isinstance(an, dict) or set(an) != {"text"} or not isinstance(an["text"], str):
        raise GenomeError("answering needs exactly text (str)")
    exs = g["examples"]
    if not isinstance(exs, dict) or set(exs) != {"items"} or not isinstance(exs["items"], list) \
            or not all(isinstance(e, str) for e in exs["items"]):
        raise GenomeError("examples needs exactly items (a list of strings)")
    num = g["numeric"]
    if not isinstance(num, dict) or set(num) != set(NUMERIC_BOUNDS):
        raise GenomeError(f"numeric must have exactly {sorted(NUMERIC_BOUNDS)}")
    for k, (lo, hi) in NUMERIC_BOUNDS.items():
        if not isinstance(num[k], (int, float)) or not lo <= num[k] <= hi:
            raise GenomeError(f"numeric.{k}={num[k]} outside [{lo}, {hi}]")
    parents = genome.get("parents")
    if not isinstance(parents, list) or not (genome.get("operator") == "seed" or 1 <= len(parents) <= 3):
        raise GenomeError("every non-seed genome declares 1-3 parents (D-065: no parentless child)")
    return genome


def _headers_ok(headers, lo, hi):
    if not isinstance(headers, list) or not lo <= len(headers) <= hi or len(set(headers)) != len(headers):
        raise GenomeError(f"core.params.headers must be a list of {lo}-{hi} distinct headers")
    for h in headers:
        if not isinstance(h, str) or not HEADER_RE.match(h) or h != " ".join(h.split()):
            raise GenomeError(f"header {h!r} must be upper case: letters, digits, space & / - (2-30 chars)")


def validate_core_params(kind, params):
    """Params of the plain-language cores (sectioned, ontology) are typed genes: exactly the documented keys."""
    if kind not in CORE_PARAM_BOUNDS:
        if params is not None and not isinstance(params, dict):
            raise GenomeError("core.params must be an object")
        return
    if not isinstance(params, dict):
        raise GenomeError(f"core {kind} needs params")
    b = CORE_PARAM_BOUNDS[kind]
    if kind == "json":
        if set(params) != {"keys", "max_depth"}:
            raise GenomeError("json core params must be exactly keys, max_depth")
        keys, (lo, hi) = params["keys"], b["keys"]
        if not isinstance(keys, list) or not lo <= len(keys) <= hi or len(set(keys)) != len(keys) \
                or not all(isinstance(k, str) and JSON_KEY_RE.match(k) for k in keys):
            raise GenomeError(f"json core params.keys must be {lo}-{hi} distinct lower-case keys (a-z, 0-9, _)")
        md = params["max_depth"]
        if not isinstance(md, int) or isinstance(md, bool) or not b["max_depth"][0] <= md <= b["max_depth"][1]:
            raise GenomeError(f"json core params.max_depth must be an integer in {list(b['max_depth'])}")
        return
    if kind == "markdown":
        if set(params) != {"headers", "tables"}:
            raise GenomeError("markdown core params must be exactly headers, tables")
        _headers_ok(params["headers"], *b["headers"])
        t = params["tables"]
        if not isinstance(t, list) or len(set(t)) != len(t) or not set(t) <= set(params["headers"]):
            raise GenomeError("markdown core params.tables must be distinct headers from params.headers")
        return
    if kind == "sectioned":
        if set(params) != {"headers", "tags", "granularity"}:
            raise GenomeError("sectioned core params must be exactly headers, tags, granularity")
        _headers_ok(params["headers"], *b["headers"])
        if not isinstance(params["tags"], bool):
            raise GenomeError("sectioned core params.tags must be a boolean")
        if params["granularity"] not in GRANULARITY:
            raise GenomeError(f"sectioned core params.granularity must be one of {GRANULARITY}")
        return
    if not {"headers", "relations", "hierarchy", "max_depth"} <= set(params) <=             {"headers", "relations", "hierarchy", "max_depth", "strict"}:
        raise GenomeError("ontology core params must be headers, relations, hierarchy, max_depth (+ optional strict)")
    if not isinstance(params.get("strict", True), bool):
        raise GenomeError("ontology core params.strict must be a boolean")
    _headers_ok(params["headers"], *b["headers"])
    rel = params["relations"]
    lo, hi = b["relations"]
    if not isinstance(rel, list) or not lo <= len(rel) <= hi or len(set(rel)) != len(rel):
        raise GenomeError(f"ontology core params.relations must be a list of {lo}-{hi} distinct verbs")
    for r in rel:
        if not isinstance(r, str) or not RELATION_RE.match(r) or set(r.split()) & GRAMMAR_WORDS:
            raise GenomeError(f"relation {r!r} must be 1-3 lower-case words, none of {sorted(GRAMMAR_WORDS)}")
    hier = params["hierarchy"]
    if not isinstance(hier, list) or not set(hier) <= set(HIERARCHY_KINDS) or len(set(hier)) != len(hier):
        raise GenomeError(f"ontology core params.hierarchy must be a list of distinct items from {HIERARCHY_KINDS}")
    md = params["max_depth"]
    if not isinstance(md, int) or isinstance(md, bool) or not b["max_depth"][0] <= md <= b["max_depth"][1]:
        raise GenomeError(f"ontology core params.max_depth must be an integer in {list(b['max_depth'])}")
    if not hier and not set(params["headers"]) & set(DECLARE_HEADERS):
        raise GenomeError(f"an ontology needs a hierarchy kind or a declaring header {DECLARE_HEADERS}")


def incompatibilities(genome):
    """Reasons a well-formed genome cannot be rendered into a working arm (checked before any paid run)."""
    g = genome["genes"]
    out = []
    core = g["core"]["kind"]
    missing = set(g["query_layer"]) - CORE_QUERIES[core]
    if missing:
        out.append(f"core {core} cannot serve queries {sorted(missing)}")
    if core == "prose" and g["build_protocol"]["kind"] in ("stepwise_checked", "template_first"):
        out.append("prose has no checkable structure for a stepwise or template protocol")
    if core == "json" and g["build_protocol"]["kind"] not in ("one_shot", "template_first"):
        out.append("json notes are one object: only one_shot or template_first")
    if g["tool_interface"]["submit"] == "line_statements" and core not in ("equations", "rules", "probabilistic"):
        out.append(f"line-statement submission is not defined for core {core}")
    if core not in PLAIN_CORES and "variables" not in g["schema"].get("fields", []):
        out.append("a structured core needs schema field 'variables'")
    return out


def make(genes, parents, operator, generation=0, island=0, name="", notes=""):
    genome = {"genes": upgrade(genes), "parents": list(parents), "operator": operator,
              "generation": generation, "island": island, "name": name, "notes": notes}
    validate(genome)
    genome["id"] = genome_id(genome)
    genome["gene_hashes"] = {k: gene_hash(k, v) for k, v in genome["genes"].items()}
    return genome
