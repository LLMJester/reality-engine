"""Items (spec §8): v2 real dev items (prose answer keys, AI-graded) and generated items from the v2 world generator
(multiple-choice probes with exact answers, script-graded).

Harness code only: builders never see answer keys (v3.walls). The v2 held-out and generalisation sets are not
reachable from here; `heldout` is refused until the final hash lock exists (spec §13).
"""
import functools
import json
import random
import re
import sys

from v3.config import V2, ROOT

if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

ANSWER_FORMAT = ("Answer every question. End your reply with a line\n"
                 "ANSWERS: {\"<question id>\": <option number>, ...}\n"
                 "using the option numbers shown (0-based).")
FINAL_LOCK = ROOT / "state" / "FINAL_LOCK.json"


class Sealed(PermissionError):
    pass


# ---------------------------------------------------------------- generated items (script-graded)
def generated(seed, messiness="realistic", family=None):
    """One generated system as an item. Validation seeds (1e6-2e6) for screening and full evaluation; train seeds
    (<1e6) for the micro level; test seeds only for the final generalisation report (never during evolution)."""
    import discovery.world as W
    _check_seed(seed)
    inst = W.make_instance(seed, family=family)
    qs = inst.questions()
    qtext = "\n\n".join(f"[{q['id']}] {q['text']}\n" + "\n".join(f"  {i}. {o}" for i, o in enumerate(q["options"]))
                        for q in qs)
    return {"id": f"W{seed}", "system": f"W{seed}", "bank": "generated", "grading": "script", "seed": seed,
            "family": getattr(inst, "family", family), "context": inst.description(messiness, seed=0),
            "question": qtext + "\n\n" + ANSWER_FORMAT, "task_type": "multiple_choice",
            "question_ids": [q["id"] for q in qs]}


def _check_seed(seed):
    from discovery.world.splits import split_of
    if split_of(seed) == "test" and not FINAL_LOCK.exists():
        raise Sealed("generated test-split seeds are reserved for the final evaluation")
    from v3 import run5_sets as R5
    if R5.is_sealed(seed):
        raise Sealed("run 5 sealed set: opened only once, at the final check (v3.run5_sets.open_sealed)")


def parse_answers(text, lenient=False):
    """The last `ANSWERS: {...}` JSON object in the reply; tolerant of code fences and trailing text.
    lenient (final C only, V3-035): also `FINAL ANSWER {...}` (the A0P prompt asks for a FINAL ANSWER line) and, failing
    any JSON object, explicit `s<j>.p<n>: <option>` pairs after the last ANSWERS / FINAL ANSWER marker."""
    if not text:
        return {}
    pat = (r"(?:ANSWERS|FINAL ANSWER)\s*:?\s*(?:```(?:json)?\s*)?(\{.*?\})" if lenient
           else r"ANSWERS\s*:\s*(?:```(?:json)?\s*)?(\{.*?\})")     # code review: a fenced block after ANSWERS: counts
    for m in reversed(list(re.finditer(pat, text, re.S))):
        try:
            d = json.loads(m.group(1))
            return {str(k): int(v) for k, v in d.items() if str(v).lstrip("-").isdigit()}
        except (ValueError, TypeError):
            continue
    if lenient and re.search(r"ANSWERS|FINAL ANSWER", text):
        tail = re.split(r"ANSWERS|FINAL ANSWER", text)[-1]
        return {k: int(v) for k, v in re.findall(r"\b(s\d+\.p\d+)\s*[:=]\s*(-?\d+)", tail)}
    return {}


def score_generated(item, final_raw, lenient=False):
    """Fraction of probes answered correctly (unanswered = wrong). Exact, deterministic, free."""
    key = answer_key(item)
    got = parse_answers(final_raw, lenient)
    correct = sum(1 for q, a in key.items() if got.get(q) == a)
    return {"score": correct / len(key) if key else None, "status": "ok", "grader": "script",
            "answered": len(set(got) & set(key)), "n_probes": len(key), "cost": 0.0}


def answer_key(item):
    """{question id: correct option} for a generated or composite item (harness only)."""
    import discovery.world as W
    if item.get("members"):
        out = {}
        for j, m in enumerate(item["members"], 1):
            _check_seed(m["seed"])
            key = W.make_instance(m["seed"], family=m["family"], size=m["size"]).answer_key()
            out.update({f"s{j}.{pid}": key[pid] for pid in m["questions"]})
        return out
    return W.make_instance(item["seed"], family=item.get("family")).answer_key()


# ---------------------------------------------------------------- composite items: the complexity ladder
COMPOSITE_KS = (1, 2, 4, 8)
COMPOSITE_SIZE = 15             # the generator's maximum size knob
COMPOSITE_QUESTIONS = 3         # per subsystem
COMPOSITE_STRIDE = 500          # members of K<k>x<seed> come from [seed, seed + stride), so stride-spaced bases share none
LADDER_START = 1_500_000        # evaluation ladder bases (the default W items use the start of the validation split)
CALIB_START = 1_900_000         # calibration bases, disjoint from the evaluation ladder
# theme terms that some family templates also use as plain words (found by scanning train/validation seeds): not names
_TEMPLATE_WORDS = frozenset({"requests", "returns", "controller", "rejects"})
_ROLES = ("items", "places", "resources", "indicators", "alarms", "events", "actors", "schedules", "rules", "people")
_GENERIC = [("sir", r"\b(?:unaffected|affected|recovered) (member|unit|site|account)s?\b"),
            ("sir", r"(?i)\b(member)s?\b"),
            ("rule#", r"\bRule \d+\b"), ("cap", r"(?<=[a-z0-9,;(] )([A-Z][a-z]+)\b")]


@functools.lru_cache(maxsize=1)
def _name_patterns():
    """(key, regex) for every surface name the world generator can use: theme vocabularies (all themes), numbered
    unit labels and generic family names. Lower-case terms match case-insensitively, without their article."""
    from discovery.world.themes import THEMES
    pats = {}
    for th in THEMES.values():
        for role in _ROLES:
            for term in th[role]:
                t = re.sub(r"^the ", "", term)
                pats.setdefault(t.lower(), re.compile(rf"\b{re.escape(t)}\b", 0 if t[0].isupper() else re.I))
        for stem in th["units"]:
            pats.setdefault(f"{stem} #", re.compile(rf"\b{re.escape(stem)} \d+\b"))
    return list(pats.items()) + [(key, re.compile(rx)) for key, rx in _GENERIC]


def names_in(text):
    """Surface names used in a text, as comparable keys (generic matches keyed by the matched word)."""
    found = set()
    for key, rx in _name_patterns():
        if key in ("cap", "sir"):
            found |= {f"{key}:{w.lower() if key == 'sir' else w}" for w in rx.findall(text)}
        elif rx.search(text):
            found.add(key)
    return found


def _units(inst, fam, genome, messiness):
    """The member's description as render units (intro, body sentence groups, timing), exactly as
    `Instance.description(messiness, seed=0)` renders them (common.render_with/_assemble), before joining."""
    from discovery.world import common as C
    from discovery.world.themes import THEMES
    if messiness not in C.MESSINESS:
        raise ValueError(messiness)
    theme = THEMES[genome["theme"]]
    omit = C.choose_omission(fam, genome, inst._probe_list(), 0)[0] if messiness == "adversarial" else None
    rng = C.rng_for("render", genome["family"], genome["seed"], messiness, 0)
    F = C.Formatter(theme["time"], messiness, rng, omit=omit)
    fam.sentences(genome, F)
    intro, body, timing = ([t for g, _, t in F.items if g == grp] for grp in ("intro", "body", "timing"))
    if messiness != "clean":
        rng.shuffle(body)
        pool = list(theme["distractors"])
        rng.shuffle(pool)
        for d in pool[: rng.randint(2, 3)]:
            d = d.format(n=rng.randint(3, 60), place=rng.choice(theme["places"]))
            body.insert(rng.randrange(len(body) + 1), d[0].upper() + d[1:])
    return [u for u in [" ".join(intro)] + body + [" ".join(timing)] if u]


@functools.lru_cache(maxsize=4096)
def _light(seed):
    """(instance, family module, genome) at the maximum size, probes not yet built; None if the generator fails."""
    import discovery.world as W
    _check_seed(seed)
    try:
        inst = W.make_instance(seed, size=COMPOSITE_SIZE)
        fam = W.get_family(inst.family)
        return inst, fam, fam.generate(seed, COMPOSITE_SIZE, None)
    except Exception:
        return None


@functools.lru_cache(maxsize=1024)
def _member(seed, messiness):
    """One subsystem with its render units, questions and surface names, or None when the generator cannot build it."""
    light = _light(seed)
    if light is None:
        return None
    inst, fam, genome = light
    try:
        units = _units(inst, fam, genome, messiness)
        qs = inst.questions()
    except Exception:
        return None
    text = "\n".join([" ".join(units), inst.description("clean")] +
                     [q["text"] + " " + " ".join(q["options"]) for q in qs])
    return {"seed": seed, "family": inst.family, "theme": genome["theme"], "units": units, "questions": qs,
            "names": frozenset(names_in(text) - _TEMPLATE_WORDS)}


def _chunks(units, anchors, target=300):
    """Consecutive render units grouped into paragraphs of about `target` chars, each containing at least one name
    that only this member uses (so it stays attributable); sentences are never split."""
    has = lambda t: bool(names_in(t) & anchors)
    out, cur = [], []
    for u in units:
        cur.append(u)
        if len(" ".join(cur)) >= target and has(" ".join(cur)):
            out.append(" ".join(cur))
            cur = []
    if cur:
        if out and not has(" ".join(cur)):
            out[-1] += " " + " ".join(cur)
        else:
            out.append(" ".join(cur))
    return out


def composite_members(k, seed, messiness="realistic"):
    """k subsystems scanned deterministically from `seed` within [seed, seed + COMPOSITE_STRIDE) of the same split: a
    new family each time until every family is used (repeats are filled in seed order only if the window runs out),
    never a theme or a surface name already taken (renaming would break the answer keys, so colliding seeds are
    skipped). The members of K<k>x<s> are a prefix of those of K<2k>x<s>."""
    import discovery.world as W
    from discovery.world.splits import split_of
    _check_seed(seed)
    split = split_of(seed)
    fams = set(W.families(include_test_only=split == "test"))
    window = [s for s in range(seed, seed + COMPOSITE_STRIDE) if split_of(s) == split]
    chosen, deferred = [], []

    def fits(s):
        light = _light(s)
        if light is None or any(light[2]["theme"] == c["theme"] for c in chosen):
            return None
        m = _member(s, messiness)
        return m if m is not None and not any(m["names"] & c["names"] for c in chosen) else None

    for s in window:
        if len(chosen) == k:
            break
        light = _light(s)
        used = {c["family"] for c in chosen}
        if light is None:
            continue
        if light[0].family in used and not fams <= used:
            deferred.append(s)
            continue
        m = fits(s)
        if m:
            chosen.append(m)
    for s in deferred:
        if len(chosen) == k:
            break
        m = fits(s)
        if m:
            chosen.append(m)
    if len(chosen) < k:
        raise ValueError(f"only {len(chosen)} compatible subsystems in [{seed}, {seed + COMPOSITE_STRIDE})")
    return chosen


def composite(k, seed, messiness="realistic"):
    """One item made of k independent generated subsystems at maximum size. Their paragraphs are interleaved (seeded,
    each subsystem keeping its own order) behind a neutral intro that does not say which paragraph belongs to which
    subsystem. Up to 3 questions per subsystem, ids prefixed `s<j>.`; script-graded by `score_generated`."""
    if k not in COMPOSITE_KS:
        raise ValueError(f"k must be one of {COMPOSITE_KS}")
    ms = composite_members(k, seed, messiness)
    paras, members, qblocks = [], [], []
    for j, m in enumerate(ms, 1):
        anchors = m["names"].difference(*[o["names"] for o in ms if o is not m])
        paras.append(_chunks(m["units"], anchors))
        rng = random.Random(f"composite-q|{m['seed']}")
        named = [q for q in m["questions"] if names_in(q["text"]) & anchors]
        rest = [q for q in m["questions"] if q not in named]
        pick = (rng.sample(named, len(named)) + rng.sample(rest, len(rest)))[:COMPOSITE_QUESTIONS]
        pick = [q for q in m["questions"] if q in pick]
        members.append({"seed": m["seed"], "family": m["family"], "size": COMPOSITE_SIZE,
                        "questions": [q["id"] for q in pick]})
        qblocks += [(f"s{j}.{q['id']}", q) for q in pick]
    order = [j for j, ps in enumerate(paras) for _ in ps]
    random.Random(f"composite|{k}|{seed}|{messiness}").shuffle(order)
    pos, text = [0] * len(paras), []
    for j in order:
        text.append(paras[j][pos[j]])
        pos[j] += 1
    intro = ("This text describes one system." if k == 1 else
             f"This text describes {k} separate systems that run side by side and do not affect one another. Their "
             "descriptions are mixed together paragraph by paragraph, in no particular order; each paragraph is about "
             "exactly one of the systems.")
    head = "" if k == 1 else ("Questions are numbered by system (s1, s2, ...); working out which paragraphs describe "
                              "which system is part of the task.\n\n")
    qtext = "\n\n".join(f"[{qid}] {q['text']}\n" + "\n".join(f"  {i}. {o}" for i, o in enumerate(q["options"]))
                        for qid, q in qblocks)
    iid = f"K{k}x{seed}"
    return {"id": iid, "system": iid, "bank": "composite", "grading": "script", "seed": seed, "k": k,
            "family": "composite", "families": [m["family"] for m in members],
            "member_seeds": [m["seed"] for m in members], "members": members, "messiness": messiness,
            "context": "\n\n".join([intro] + text), "question": head + qtext + "\n\n" + ANSWER_FORMAT,
            "task_type": "multiple_choice", "question_ids": [qid for qid, _ in qblocks]}


GROWTH_KS = (1, 2, 4, 8)         # run 3 interaction final (V3-034): the system grows over four turns


def growth_turns(seed, messiness="realistic"):
    """Run 3's interaction final (V3-034): one composite system built up over four turns. Turn t's full system is
    K<k_t>x<seed> (k = 1, 2, 4, 8; members are prefixes, so each turn adds subsystems to the last). Each turn gives
    `item` (that turn's composite item: every question so far, script-graded) and `new_text` (only the paragraphs of
    the subsystems added this turn, mixed as in a composite item, behind an intro that says they are additions)."""
    out, prev = [], 0
    for t, k in enumerate(GROWTH_KS, 1):
        item = composite(k, seed, messiness)
        if t == 1:
            new_text = item["context"]
        else:
            ms = composite_members(k, seed, messiness)
            paras = []
            for m in ms[prev:]:
                anchors = m["names"].difference(*[o["names"] for o in ms if o is not m])
                paras.append(_chunks(m["units"], anchors))
            order = [j for j, ps in enumerate(paras) for _ in ps]
            random.Random(f"growth|{k}|{seed}|{messiness}").shuffle(order)
            pos, text = [0] * len(paras), []
            for j in order:
                text.append(paras[j][pos[j]])
                pos[j] += 1
            n = k - prev
            intro = (f"NEW INFORMATION: {n} more separate system{'s' if n > 1 else ''} now run{'' if n > 1 else 's'} "
                     f"side by side with the {prev} already described. They do not affect the earlier systems or one "
                     "another." + (" Their descriptions are mixed together paragraph by paragraph, in no particular "
                                   "order; each paragraph is about exactly one of the new systems." if n > 1 else ""))
            new_text = "\n\n".join([intro] + text)
        out.append({"turn": t, "k": k, "item": item, "new_text": new_text})
        prev = k
    return out


def ladder_seeds(n, start=LADDER_START):
    """n composite base seeds, COMPOSITE_STRIDE apart from `start`, round-robin over the base seed's family."""
    import discovery.world as W
    from discovery.world.splits import split_of
    _check_seed(start)
    fams = {}
    for j in range(n * 4):
        sd = start + j * COMPOSITE_STRIDE
        if split_of(sd + COMPOSITE_STRIDE - 1) != split_of(start):
            break
        fams.setdefault(W.make_instance(sd).family, []).append(sd)
    order, i = [], 0
    while len(order) < n:
        if all(i >= len(v) for v in fams.values()):
            raise ValueError("not enough ladder seeds")
        for f in sorted(fams):
            if i < len(fams[f]) and len(order) < n:
                order.append(fams[f][i])
        i += 1
    return order


# ---------------------------------------------------------------- v2 real dev items (AI-graded)
def v2_dev_ids():
    from harness.run import split_ids
    return list(split_ids("dev"))


def v2_item(iid):
    from harness.run import load_item
    from harness.run import split_ids
    if iid not in set(split_ids("dev")):
        raise Sealed(f"{iid} is not a v2 dev item")
    item, ctx = load_item(iid)
    return dict(item, context=ctx, system=item.get("system_id") or item["id"], grading="ai")


@functools.lru_cache(maxsize=None)
def _hand5():
    """Run 5 hand-written-style items (v3.hand5; the final 500 and the 6 pilot items), by id."""
    out = {}
    for name in ("items.json", "pilot_items.json"):
        p = ROOT / "items" / "hand5" / name
        if p.exists():
            out.update({it["id"]: it for it in json.loads(p.read_text(encoding="utf-8"))})
    return out


def hand5_ids(pilot=False):
    return [i for i in _hand5() if i.startswith("HP") == pilot]


def load(item_id):
    if re.fullmatch(r"HP?\d+-\d", item_id):
        if item_id not in _hand5():
            raise KeyError(f"{item_id} is not a selected run 5 hand-written item")
        return dict(_hand5()[item_id])
    m = re.fullmatch(r"K(\d+)x(\d+)", item_id)
    if m:
        return composite(int(m.group(1)), int(m.group(2)))
    if item_id.startswith("W"):
        return generated(int(item_id[1:]))
    return v2_item(item_id)
