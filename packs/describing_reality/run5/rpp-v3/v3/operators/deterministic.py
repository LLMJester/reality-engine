"""Deterministic variation operators (spec §4): no AI, reproducible from a seed.

Each operator returns (genes, parents, operator_name, notes) or None if it cannot apply; v3.genome.schema.make then
validates the child and records provenance. Every child declares its parents (D-065).
"""
import copy

from v3.genome import schema as S
from v3.operators.cmaes import CMAES


def module_swap(a, b, rng, mask=None):
    """Gene-level crossover: each gene from parent a or b. mask = {gene: 0|1}; uniform random if omitted.
    Guarantees the child differs from both parents when they differ in at least two genes."""
    diff = [k for k in S.GENES if S.canonical(a["genes"][k]) != S.canonical(b["genes"][k])]
    if not diff:
        return None
    if mask is None:
        mask = {k: rng.randint(0, 1) for k in S.GENES}
        if len(diff) >= 2 and len({mask[k] for k in diff}) == 1:   # all differing genes from one parent: flip one
            mask[rng.choice(diff)] ^= 1
    genes = {k: copy.deepcopy((b if mask[k] else a)["genes"][k]) for k in S.GENES}
    return genes, [a["id"], b["id"]], "swap", {"mask": mask}


def one_point(a, b, rng):
    cut = rng.randint(1, len(S.GENES) - 1)
    mask = {k: int(i >= cut) for i, k in enumerate(S.GENES)}
    out = module_swap(a, b, rng, mask)
    return out and (out[0], out[1], "one_point", {"mask": mask})


def transplant(a, b, rng, b_parent=None):
    """Combine two leaders (run 3, V3-034): a's genes plus the change that made b better than its own parent (the
    genes where b differs from b_parent, given only when b beat it), limited to genes where b also differs from a.
    Without such a change (b is a seed, or b did not beat its parent) one gene where b differs from a, drawn at
    random. The child declares both parents."""
    diff = [k for k in S.GENES if S.canonical(a["genes"][k]) != S.canonical(b["genes"][k])]
    if not diff:
        return None
    key = [k for k in diff if b_parent and S.canonical(b["genes"][k]) != S.canonical(b_parent["genes"][k])]
    keyed = bool(key)
    if not key:
        key = [diff[rng.randrange(len(diff))]]
    genes = copy.deepcopy(a["genes"])
    for k in key:
        genes[k] = copy.deepcopy(b["genes"][k])
    return genes, [a["id"], b["id"]], "transplant", {"genes": key, "from": b["id"], "key_change": keyed}


def simplify(a, rng):
    """Remove one feature: a query, a schema field, a guidance example, a section header or relation verb of a
    plain-language core (down to its minimum), or downgrade the protocol to one_shot."""
    g = copy.deepcopy(a["genes"])
    options = [("query", q) for q in g["query_layer"]] + \
              [("field", f) for f in g["schema"]["fields"] if f != "variables"] + \
              [("example", i) for i in range(len(g["examples"]["items"]))]
    params = g["core"].get("params") or {}
    bounds = S.CORE_PARAM_BOUNDS.get(g["core"]["kind"], {}) if g["core"]["kind"] in ("sectioned", "ontology") else {}
    for key in ("headers", "relations"):
        if key in bounds and len(params.get(key, [])) > bounds[key][0]:
            options += [(key, x) for x in params[key]]
    if g["build_protocol"]["kind"] != "one_shot":
        options.append(("protocol", g["build_protocol"]["kind"]))
    if not options:
        return None
    kind, what = options[rng.randrange(len(options))]
    if kind == "query":
        g["query_layer"].remove(what)
    elif kind == "field":
        g["schema"]["fields"].remove(what)
    elif kind == "example":
        g["examples"]["items"].pop(what)
    elif kind in ("headers", "relations"):
        g["core"]["params"][kind].remove(what)
    else:
        g["build_protocol"] = {"kind": "one_shot", "params": {}}
    return g, [a["id"]], "simplify", {"removed": [kind, what]}


def param_step(a, rng):
    """Point mutation inside genes.core.params of a plain-language core (sectioned / ontology): add, remove or move
    a section header, toggle tags or granularity, add or remove a relation verb, toggle a hierarchy kind, or step
    max_depth. Children stay within schema.CORE_PARAM_BOUNDS (an invalid draw returns None; apply() retries)."""
    kind = a["genes"]["core"]["kind"]
    if kind not in ("sectioned", "ontology"):        # run 4 operators; the run 5 cores (json, markdown) do not breed
        return None
    g = copy.deepcopy(a["genes"])
    p = g["core"]["params"]
    b = S.CORE_PARAM_BOUNDS[kind]
    hs = p["headers"]
    new_h = [h for h in S.HEADER_VOCAB if h not in hs]
    new_r = [r for r in S.RELATION_POOL if r not in p.get("relations", [])]
    opts = (["add_header"] if len(hs) < b["headers"][1] and new_h else []) + \
           (["remove_header"] if len(hs) > b["headers"][0] else []) + (["move_header"] if len(hs) >= 2 else [])
    if kind == "sectioned":
        opts += ["tags", "granularity"]
    else:
        opts += (["add_relation"] if len(p["relations"]) < b["relations"][1] and new_r else []) + \
                (["remove_relation"] if len(p["relations"]) > b["relations"][0] else []) + ["hierarchy", "depth",
                                                                                              "strict"]
    op = opts[rng.randrange(len(opts))]
    if op == "add_header":
        what = new_h[rng.randrange(len(new_h))]
        hs.insert(rng.randint(0, len(hs)), what)
    elif op == "remove_header":
        what = hs.pop(rng.randrange(len(hs)))
    elif op == "move_header":
        i = rng.randrange(len(hs) - 1)
        hs[i], hs[i + 1] = hs[i + 1], hs[i]
        what = hs[i + 1]
    elif op == "tags":
        p["tags"] = what = not p["tags"]
    elif op == "granularity":
        p["granularity"] = what = "fine" if p["granularity"] == "coarse" else "coarse"
    elif op == "add_relation":
        what = new_r[rng.randrange(len(new_r))]
        p["relations"].append(what)
    elif op == "remove_relation":
        what = p["relations"].pop(rng.randrange(len(p["relations"])))
    elif op == "strict":
        p["strict"] = what = not p.get("strict", True)
    elif op == "hierarchy":
        what = S.HIERARCHY_KINDS[rng.randrange(len(S.HIERARCHY_KINDS))]
        p["hierarchy"] = [k for k in S.HIERARCHY_KINDS if (k in p["hierarchy"]) != (k == what)]
    else:
        lo, hi = b["max_depth"]
        p["max_depth"] = what = min(hi, max(lo, p["max_depth"] + rng.choice((-1, 1))))
    try:
        S.validate_core_params(kind, p)
    except S.GenomeError:
        return None
    return g, [a["id"]], "param", {"param": op, "value": what}


def graft(a, library, rng, gene=None):
    """Replace one gene with the gene library's best variant for this niche (spec §4.2)."""
    cands = [k for k in S.GENES if k in library and S.canonical(library[k]["value"]) != S.canonical(a["genes"][k])]
    if gene:
        cands = [gene] if gene in cands else []
    if not cands:
        return None
    k = cands[rng.randrange(len(cands))]
    g = copy.deepcopy(a["genes"])
    g[k] = copy.deepcopy(library[k]["value"])
    return g, [a["id"], library[k]["source"]] if library[k].get("source") not in (None, a["id"]) else [a["id"]], \
        "graft", {"gene": k, "from": library[k].get("source")}


def numeric_step(a, cma_state, rng):
    """Numeric tuning by CMA-ES over the `numeric` gene (spec §4.1). cma_state persists between generations."""
    keys = sorted(S.NUMERIC_BOUNDS)
    lo = [S.NUMERIC_BOUNDS[k][0] for k in keys]
    hi = [S.NUMERIC_BOUNDS[k][1] for k in keys]
    es = CMAES.from_state(cma_state) if cma_state else CMAES([a["genes"]["numeric"][k] for k in keys], lo, hi,
                                                          seed=rng.randrange(2 ** 31))
    x = es.ask_one()
    g = copy.deepcopy(a["genes"])
    for k, v in zip(keys, x):
        g["numeric"][k] = int(round(v)) if isinstance(a["genes"]["numeric"][k], int) else round(float(v), 4)
    return g, [a["id"]], "numeric", {"cma": es.state()}


def apply(op, parents, rng, library=None, cma_state=None, tries=12, key_parent=None):
    """Dispatch a deterministic operator; returns a validated, compatible child genome, or None if `tries` draws
    (each a fresh mask or choice from the same seeded rng, so still reproducible) give no compatible new child."""
    for _ in range(tries):
        child = _apply_once(op, parents, rng, library, cma_state, key_parent)
        if child is not None:
            return child
    return None


def _apply_once(op, parents, rng, library, cma_state, key_parent=None):
    if op == "transplant":
        out = transplant(parents[0], parents[1], rng, key_parent) if len(parents) >= 2 else None
    elif op == "swap":
        out = module_swap(parents[0], parents[1], rng) if len(parents) >= 2 else None
    elif op == "one_point":
        out = one_point(parents[0], parents[1], rng) if len(parents) >= 2 else None
    elif op == "simplify":
        out = simplify(parents[0], rng)
    elif op == "param":
        out = param_step(parents[0], rng)
    elif op == "graft":
        out = graft(parents[0], library or {}, rng)
    elif op == "numeric":
        out = numeric_step(parents[0], cma_state, rng)
    else:
        raise ValueError(op)
    if out is None:
        return None
    genes, pids, name, notes = out
    gen = max(p.get("generation", 0) for p in parents) + 1
    child = S.make(genes, pids, name, gen, parents[0].get("island", 0), notes=str(notes))
    if any(child["id"] == p["id"] for p in parents) or S.incompatibilities(child):
        return None
    child["op_notes"] = notes
    return child
