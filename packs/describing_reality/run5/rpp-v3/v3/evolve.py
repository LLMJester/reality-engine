"""The generation loop (spec §4-§6, §8): variation → quick build test → successive-halving screening → full evaluation
→ NSGA-II front, MAP-Elites archive, improvement rule, operator credit, gene library, failure targets.

Everything here is deterministic given the episode results. Episodes come from `runner(genome, item, builder, phase)`
(v3.episode.run in production, a stub in dry runs), so the loop can be tested without spending anything.
"""
import json
import statistics
import threading
import time

from v3 import stats
from v3.config import CFG
from v3.genome import schema as S
from v3.operators import ai as AI
from v3.operators import deterministic as D
from v3.selection import algorithms as A

EV = CFG["evolution"]
KEYS = ["e2e", "build", "neg_tokens", "e2e_small"]   # NSGA-II objectives; e2e_small = small builder (V3-023)
DETERMINISTIC = {"simplify", "numeric", "swap", "one_point", "graft", "param"}
OP_PARENTS = {"swap": 2, "one_point": 2}


# ---------------------------------------------------------------- item sets
LADDER = {"quick": {1: 1, 2: 1, 4: 2, 8: 1}, "screen": {1: 6, 2: 8, 4: 8, 8: 8},
          "full": {1: 20, 2: 20, 4: 25, 8: 25}}


def small_builder():
    """The second, smaller builder (V3-013: GPT-6 Luna through Codex on the human's ChatGPT plan; Qwen retired)."""
    return EV.get("small_builder", "codex")


def small_ok(g):
    """Codex never builds designs whose tools run model-written code (V3-011): those are Claude-only."""
    from v3.builders.codex_builder import SANDBOXED_CORES   # runs 1-4 (V3-011): never given to Codex
    return small_builder() != "codex" or g["genes"]["core"]["kind"] not in SANDBOXED_CORES


def primary():
    """The builder selection runs on: "claude" (run 1) or the small builder alone (V3-030, run 2: GPT-6 Luna only)."""
    return EV.get("primary_builder", "claude")


def single():
    """V3-030: one builder only; Claude builds nothing and designs the small builder cannot build are dropped."""
    return primary() == small_builder()


def keys():
    """NSGA-II objectives: on the primary builder, plus the small builder's e2e when there are two builders."""
    return KEYS[:3] if single() else KEYS


def quick_builder(g):
    """The quick test's builder; None when the only builder cannot build the design (it then fails the test)."""
    if single():
        return primary() if small_ok(g) else None
    return small_builder() if small_ok(g) else "claude"


def full_builders(g):
    if single():
        return [primary()] if small_ok(g) else []
    return ["claude"] + ([small_builder()] if small_ok(g) else [])


def dev_subset(dev):
    """v2 dev items for the full set: True = all 30, False = none, an int n = every third item from the start, then
    the rest in order, so the n items spread over the dev systems (fixed, deterministic)."""
    from v3 import items as I
    ids = I.v2_dev_ids()
    if dev is True or dev is False:
        return ids if dev else []
    spread = ids[::3] + [i for i in ids if i not in ids[::3]]
    return spread[:int(dev)]


def run_sets():
    """The sets the real run and the pre-registration lock use (V3-006): the complexity ladder, with or without the
    v2 dev items per config evolution.dev_items."""
    return item_sets(ladder=EV.get("ladder") or True, dev=EV.get("dev_items", True))


def item_sets(n_screen=None, n_full_generated=None, ladder=None, dev=True):
    """Fixed, pre-registered item ids. quick: 5 generated; screen: generated (script-graded, free);
    full: v2 dev (30, AI-graded) + generated, total >= full_items_min. Generated items are family-stratified.

    `ladder` (True for LADDER, or {set: {k: n}}) replaces the generated items by composite items K<k>x<seed>
    (v3.items.composite) stratified over k; bases are family-stratified and never shared between items or sets.
    `dev=False` leaves the v2 dev items out of full."""
    from v3 import items as I
    from discovery.world.splits import seeds
    import discovery.world as W
    if ladder:
        ladder = LADDER if ladder is True else ladder
        bases = iter(I.ladder_seeds(sum(n for per_k in ladder.values() for n in per_k.values())))
        out = {s: [f"K{k}x{next(bases)}" for k in sorted(ladder.get(s, {})) for _ in range(ladder[s][k])]
               for s in ("quick", "screen", "full", "micro") if ladder.get(s)}
        out["full"] = dev_subset(dev) + out["full"]
        return out
    n_screen = n_screen or EV["screen_items"]
    want = (n_full_generated or max(0, EV["full_items_min"] - 30)) + n_screen + EV["quick_build_items"]
    fams = {}
    for sd in seeds("validation", want * 4):
        fams.setdefault(W.make_instance(sd).family if hasattr(W.make_instance(sd), "family") else "x", []).append(sd)
    order, i = [], 0
    while len(order) < want:                       # round-robin over families
        for f in sorted(fams):
            if i < len(fams[f]) and len(order) < want:
                order.append(f"W{fams[f][i]}")
        i += 1
    q = EV["quick_build_items"]
    return {"quick": order[:q], "screen": order[q:q + n_screen],
            "full": (I.v2_dev_ids() if dev else []) + order[q + n_screen:]}


def quick_ok(g, r):
    """The quick build test's pass rule on one builder's result."""
    if g["genes"]["core"]["kind"] == "prose":   # nothing to build: pass on answering (scored above 0), V3-006
        return sum(1 for v in r["scores"].values() if v > 0) >= EV["quick_build_pass"]
    return sum(r["build_items"].values()) >= EV["quick_build_pass"]


# ---------------------------------------------------------------- objectives
def transient(rec):
    """V3-031: an episode that failed for a reason worth retrying (a provider or connection error, e.g. ChatGPT's 403
    burst at the start of run 2). Integrity breaches, allowlist refusals and content flags are never retried."""
    e = str(rec.get("error") or "")
    return bool(e) and not (e.startswith("INTEGRITY") or "allowlist" in e or "flagged" in e)


def voided(r):
    """Items of a stored stage result whose episodes failed (summarise() leaves them out of build_items)."""
    return (r or {}).get("n", 0) - len((r or {}).get("build_items") or {})


def summarise(recs):
    from v3.episode import buildable
    scores = {r["item"]: r["grade"]["score"] for r in recs if r.get("grade", {}).get("score") is not None}
    build = {r["item"]: float(buildable(r)) for r in recs if not r.get("error")}
    toks = [r.get("output_tokens") or 0 for r in recs if not r.get("error")]
    return {"e2e": statistics.mean(scores.values()) if scores else None,
            "build": statistics.mean(build.values()) if build else None,
            "neg_tokens": -statistics.mean(toks) if toks else None,
            "scores": scores, "build_items": build, "n": len(recs), "build_problems": build_problems(recs)}


def build_problems(recs, top=5):
    """V3-034: the most frequent checker problems of failed builds, for the designer's targeted fixes. Quoted text
    (terms taken from the description) is blanked, so no item content reaches the designer; the design's own section
    and header names are kept."""
    import collections
    import re
    from v3.episode import buildable
    c = collections.Counter()
    for r in recs:
        if r.get("error") or buildable(r):
            continue
        for h in ((r.get("validity") or {}).get("holes") or [])[:3]:
            c[re.sub(r"(?<!section )(?<!header )('[^']*'|\"[^\"]*\")", "'...'", str(h.get("problem", "")))[:140]] += 1
    return [[m, n] for m, n in c.most_common(top)]


def systems(recs):
    return {r["item"]: r["system"] for r in recs}


# ---------------------------------------------------------------- the loop
class Evolution:
    def __init__(self, store, runner, sets, *, designer=AI.designer_call, log=print):
        self.st, self.runner, self.sets, self.designer, self.log = store, runner, sets, designer, log
        d = store.data
        d.setdefault("genomes", {}); d.setdefault("evals", {}); d.setdefault("archive", {})
        d.setdefault("library", {}); d.setdefault("credit", None); d.setdefault("cma", None)
        d.setdefault("streak", 0); d.setdefault("front", []); d.setdefault("islands", [[] for _ in range(EV["islands"])])
        d.setdefault("lineage", [])
        self._lock = threading.Lock()
        # V3-026: a design is listed once per island (generation 0's setup used to re-add the seeds on every rerun)
        dedup = [list(dict.fromkeys(isl)) for isl in d["islands"]]
        if dedup != d["islands"]:
            n = sum(map(len, d["islands"])) - sum(map(len, dedup))
            d["islands"] = dedup
            store.event(f"V3-026: removed {n} duplicate island entries (seeds re-added by reruns of generation 0)")
        self._credit_full_evaluation_wins()

    def _credit_full_evaluation_wins(self):
        """V3-027, once: finished generations were credited before reaching full evaluation counted as a win. Each
        bred (non-micro) child of a finished generation that was fully evaluated, including catch-up picks, gets the
        win now. Generations 0-1 recorded no wins at all, so nothing is counted twice."""
        d = self.st.data
        if not d.get("credit") or d.get("credit_v3_027"):
            return
        done = {s["gen"] for s in d.get("generations", [])}
        fulls = [gid for s in d.get("generations", []) for gid in s.get("full", [])]
        fulls += [gid for gen, c in (d.get("catchup") or {}).items() if int(gen) in done for gid in c.get("picks", [])]
        added = []
        for gid in dict.fromkeys(fulls):
            g = d["genomes"].get(gid, {})
            op = g.get("operator")
            if g.get("level") == "micro" or op not in d["credit"]["wins"]:
                continue
            if d["credit"]["wins"][op] < d["credit"]["trials"][op]:
                d["credit"]["wins"][op] += 1
                added.append(f"{op} ({gid})")
        c = A.OperatorCredit(EV["operator_shares"], EV["operator_floor"])
        c.wins, c.trials = d["credit"]["wins"], d["credit"]["trials"]
        d["credit"]["shares"] = c.shares()
        d["credit_v3_027"] = True
        self.st.event(f"V3-027: operator credit for fully evaluated children of finished generations: {added}")

    # -- evaluation stages
    def evaluate(self, g, stage, items, builders, workers=None):
        """Every builder on every item; the builders run at the same time (each with its own worker pool), and each
        builder's results keep item order (deterministic)."""
        from concurrent.futures import ThreadPoolExecutor
        ev = self.st.data["evals"].setdefault(g["id"], {})

        def one_builder(b):
            n = (workers or {}).get(b) or {"claude": CFG["budget"]["max_parallel_agents"],
                                           "codex": CFG["budget"].get("codex_parallel_agents", 4)}.get(b, 8)
            with ThreadPoolExecutor(max_workers=n) as pool:
                recs = list(pool.map(lambda item: self.runner(g, item, b, stage), items))
                # V3-031: transient failures are retried up to `retries` more times after a short wait; a failure
                # that survives the retries stays voided (left out of the scores) and is reported
                for _ in range(EV.get("retries", 2)):
                    again = [i for i, r in enumerate(recs) if transient(r)]
                    if not again:
                        break
                    time.sleep(EV.get("retry_wait_s", 30))
                    for i, r in zip(again, pool.map(lambda i: self.runner(g, items[i], b, stage), again)):
                        recs[i] = r
            r = summarise(recs)
            r["systems"] = systems(recs)
            return b, r

        with ThreadPoolExecutor(max_workers=max(1, len(builders))) as outer:
            out = dict(outer.map(one_builder, builders))
        with self._lock:            # the two screening tracks (V3-023) evaluate at the same time
            ev.setdefault(stage, {}).update(out)
            self.st.save()
        return out

    def quick_pass(self, g):
        b = quick_builder(g)
        if b is None:
            return False
        return quick_ok(g, self.evaluate(g, "quick", self.sets["quick"], [b])[b])

    def baseline(self, seeds, controls=("A0P",)):
        """Generation 0: every seed on the full set with both builders; seeds populate the archive and islands."""
        d = self.st.data
        for i, (name, g) in enumerate(sorted(seeds.items())):
            d["genomes"][g["id"]] = g
            done = d["evals"].get(g["id"], {}).get("full")
            # V3-031: a seed whose full evaluation has voided items is evaluated again: finished episodes come back
            # from their records, only the failed ones run again
            if not done or any(voided(done.get(b)) for b in full_builders(g)):
                self.evaluate(g, "full", self.sets["full"], full_builders(g))
            self.offer(g)
            if not any(g["id"] in isl for isl in d["islands"]):     # reruns must not re-add a seed (V3-026)
                d["islands"][i % len(d["islands"])].append(g["id"])
        d["controls"] = [seeds[c]["id"] for c in controls if c in seeds]
        d["generation"] = 0
        self.refresh_front()
        self.st.save()

    def objectives(self, gid):
        """The primary builder's e2e, build and token cost plus, with two builders, the small builder's e2e (V3-023;
        None if it cannot build the design, which counts as worst on that objective only)."""
        full = self.st.data["evals"].get(gid, {}).get("full", {})
        f = full.get(primary())
        if not f:
            return None
        o = {k: f[k] for k in keys() if k in f}
        if not single():
            o["e2e_small"] = (full.get(small_builder()) or {}).get("e2e")
        return o

    def offer(self, g):
        o = self.objectives(g["id"])
        if o is None:
            return False
        cells = {tuple(k.split("|")): v for k, v in self.st.data["archive"].items()}
        for c in cells.values():    # occupants are compared on current objectives (V3-023 added e2e_small)
            c["obj"] = self.objectives(c["id"]) or c["obj"]
        m = A.MapElites(keys(), cells)
        ok = m.offer(A.niche_of(g), g["id"], o)
        self.st.data["archive"] = m.to_json()
        return ok

    def refresh_front(self):
        objs = {gid: o for gid in self.st.data["genomes"] if (o := self.objectives(gid))}
        self.st.data["front"] = A.nondominated_sort(objs, keys())[0] if objs else []

    # -- improvement rule (spec §5.6): paired CI vs each declared parent and the best control, on e2e or build;
    # V3-023: judged per builder, improved if it holds on either builder
    def improvement(self, g):
        ev = self.st.data["evals"]
        comps, by_builder = [], {}
        for b in full_builders(g):
            mine = ev[g["id"]]["full"].get(b)
            if not mine:
                continue
            refs = [p for p in g["parents"] if b in ev.get(p, {}).get("full", {})]
            ctrls = [c for c in self.st.data.get("controls", []) if b in ev.get(c, {}).get("full", {})]
            best_ctrl = max(ctrls, key=lambda c: ev[c]["full"][b]["e2e"] or 0) if ctrls else None
            mine_c = []
            for ref, kind in [(p, "parent") for p in refs] + ([(best_ctrl, "best_control")] if best_ctrl else []):
                other = ev[ref]["full"][b]
                for metric, key in (("e2e", "scores"), ("build", "build_items")):
                    ci = stats.paired_ci(mine[key], other[key], {**other["systems"], **mine["systems"]}, n_boot=2000)
                    mine_c.append({"ref": ref, "kind": kind, "metric": metric, "builder": b, "ci": ci})
            by_builder[b] = bool(mine_c) and all(any(c["ci"]["improved"] for c in mine_c if c["ref"] == r)
                                                 for r in {c["ref"] for c in mine_c})
            comps += mine_c
        nd = g["id"] in self.st.data["front"]
        return {"improved": nd and any(by_builder.values()), "non_dominated": nd, "by_builder": by_builder,
                "comparisons": comps}

    # -- variation
    def make_child(self, op, rng, gen, island, targets, excerpts):
        d = self.st.data
        pool = [gid for gid in d["islands"][island] if self.objectives(gid)] or \
               [gid for gid in d["genomes"] if self.objectives(gid)]
        objs = {gid: self.objectives(gid) for gid in pool}
        n = OP_PARENTS.get(op, 1)
        pids = []
        def score(gid):     # tie-break (V3-026): end-to-end score on both builders
            return (objs[gid]["e2e"] or 0) + (objs[gid].get("e2e_small") or 0)
        for _ in range(n):
            pids.append(A.tournament([p for p in pool if p not in pids] or pool, objs, keys(), rng, score=score))
        parents = [d["genomes"][p] for p in pids]
        if op in DETERMINISTIC:
            lib = {k: v for k, v in d["library"].items()}
            child = D.apply(op, parents, rng, library=lib, cma_state=d["cma"])
            if child is not None and op == "numeric":
                d["cma"] = child["op_notes"]["cma"]
        else:
            child = AI.apply(op, parents, targets, excerpts, call=self.designer, generation=gen, island=island)
        if child is None or child["id"] in d["genomes"]:
            return None
        child["generation"], child["island"] = gen, island
        return child

    def failure_targets(self, gids):
        """Counts of failure classes on the latest full evaluations (construction / reasoning)."""
        t = {"construction": 0, "reasoning": 0, "answered_ok": 0}
        for gid in gids:
            f = self.st.data["evals"].get(gid, {}).get("full", {}).get(primary())
            if not f:
                continue
            for item, b in f["build_items"].items():
                s = f["scores"].get(item)
                if not b:
                    t["construction"] += 1
                elif s is not None and s < 0.5:
                    t["reasoning"] += 1
                else:
                    t["answered_ok"] += 1
        return t

    def micro(self, gen, rng):
        """Micro level (V3-015, V3-017): fast code-only tweaks of the best designs, judged by the small builder.
        Items come from a separate micro pool (not used by the quick test, screening or full sets); each generation
        draws its own two disjoint batches of 5. A tweak is a candidate if, on batch A, it builds on >= quick_build_pass
        items and its paired mean beats the parent's; candidates (best first, up to `confirm_tries`) are re-tested on
        fresh batch B against the parent, and the first that beats the parent again is that base's winner. Winners
        join this generation's children (level "micro"). Bases: top designs by Claude full e2e that the small builder
        can build (controls included: a tweak of the control is a structured design)."""
        d, M = self.st.data, EV.get("micro") or {}
        pool = list(self.sets.get("micro") or [])
        n = M.get("batch", 5)
        if not M.get("enabled", True) or not M.get("bases") or len(pool) < 2 * n:
            return [], {"tried": 0, "winners": 0, "skipped": "no micro pool" if len(pool) < 2 * n else "disabled"}
        batch = rng.sample(pool, 2 * n)
        A_items, B_items = batch[:n], batch[n:]
        pool_ids = [gid for gid in d["genomes"] if self.objectives(gid) and small_ok(d["genomes"][gid])]
        ranked = sorted(pool_ids, key=lambda gid: (-(self.objectives(gid)["e2e"] or 0), gid))
        # V3-023: the last `bases_small` bases are the best designs on the small builder not already chosen
        n_small = min(M["bases"], M.get("bases_small", 0))
        chosen = ranked[:M["bases"] - n_small]
        by_small = sorted((gid for gid in pool_ids if self.objectives(gid).get("e2e_small") is not None),
                          key=lambda gid: (-self.objectives(gid)["e2e_small"], gid))
        chosen += [gid for gid in by_small if gid not in chosen][:n_small]
        chosen += [gid for gid in ranked if gid not in chosen][:M["bases"] - len(chosen)]
        b, need, ops = small_builder(), EV["quick_build_pass"], M.get("ops", ["param", "simplify", "numeric"])

        def paired(r, pr):
            common = set(r["scores"]) & set(pr["scores"])
            return (sum(r["scores"][i] - pr["scores"][i] for i in common) / len(common)) if common else None

        def builds(g, r):
            return sum(1 for v in r["scores"].values() if v > 0) if g["genes"]["core"]["kind"] == "prose"                 else sum(r["build_items"].values())

        out, tried, confirmed_tries = [], 0, 0
        for base_id in chosen:
            base = d["genomes"][base_id]
            pa = self.evaluate(base, f"micro{gen}a", A_items, [b])[b]
            # resume (V3-022): tweaks of this base made before a pause are reused, not replaced by new ones
            tweaks = [g for g in d["genomes"].values() if g.get("level") == "micro" and g.get("generation") == gen
                      and (g.get("micro") or {}).get("base") == base_id]
            attempts = 0
            while len(tweaks) < M.get("tweaks_per_base", 6) and attempts < M.get("tweaks_per_base", 6) * 4:
                attempts += 1
                op = ops[rng.randrange(len(ops))]
                c = D.apply(op, [base], rng, library=dict(d["library"]), cma_state=d["cma"])
                if c is None or c["id"] in d["genomes"] or c["id"] in {t["id"] for t in tweaks}:
                    continue
                if op == "numeric":
                    d["cma"] = c["op_notes"]["cma"]
                c.update(generation=gen, island=base.get("island", 0), level="micro", micro={"base": base_id})
                tweaks.append(c)
            cands = []
            for c in tweaks:
                tried += 1
                d["genomes"][c["id"]] = c
                r = self.evaluate(c, f"micro{gen}a", A_items, [b])[b]
                c["micro"] = {"base": base_id, "gain_a": paired(r, pa), "builds_a": builds(c, r),
                              "items_a": A_items, "items_b": B_items}
                if c["micro"]["gain_a"] is not None and c["micro"]["gain_a"] > 0 and c["micro"]["builds_a"] >= need:
                    cands.append(c)
            cands.sort(key=lambda c: -c["micro"]["gain_a"])
            winner, pb = None, None
            for c in cands[:M.get("confirm_tries", 2)]:
                pb = pb or self.evaluate(base, f"micro{gen}b", B_items, [b])[b]
                confirmed_tries += 1
                r = self.evaluate(c, f"micro{gen}b", B_items, [b])[b]
                c["micro"].update(gain_b=paired(r, pb), builds_b=builds(c, r))
                if c["micro"]["gain_b"] is not None and c["micro"]["gain_b"] > 0 and c["micro"]["builds_b"] >= need:
                    winner = c
                    break
            for c in tweaks:            # losing tweaks stay recorded (genome + evals) but leave no lineage edge
                if c is not winner:
                    d["genomes"][c["id"]]["retired_micro"] = True
            if winner is not None:
                out.append(winner)
        self.log(f"gen {gen}: micro tried {tried} tweaks of {len(chosen)} designs, "
                 f"{confirmed_tries} confirmation tests, {len(out)} confirmed winners")
        return out, {"tried": tried, "confirm_tests": confirmed_tries, "winners": len(out),
                     "winner_ids": [c["id"] for c in out], "items_a": A_items, "items_b": B_items}

    def screening(self, passed):
        """Successive halving on two tracks at once (V3-023): every child is screened on Claude and, if it can build
        the design, on the small builder; each track keeps the better half after round 1 and ranks its survivors on
        the full screening set. Full evaluation takes the top `full_top_k - full_top_k_small` by Claude, then the top
        `full_top_k_small` by the small builder that Claude's track did not already pick.
        Returns (survivors, {builder: halving history}, round-1 size)."""
        from concurrent.futures import ThreadPoolExecutor
        d = self.st.data
        n1 = max(1, EV["screen_items"] // 3)
        k_all = EV["full_top_k"]
        k_small = 0 if single() else min(k_all, EV.get("full_top_k_small", 0))
        main = primary()
        tracks = {main: [c["id"] for c in passed if not single() or small_ok(c)]}
        if k_small:
            tracks[small_builder()] = [c["id"] for c in passed if small_ok(c)]

        def track(b):
            def score(gid, n):
                return self.evaluate(d["genomes"][gid], f"screen{n}", self.sets["screen"][:n], [b])[b]["e2e"]
            # the small track ranks enough survivors to fill its picks even if Claude's track took some of them
            keep = k_all - k_small if b == main else k_all
            return b, A.successive_halving(tracks[b], score, [(n1, 0.5), (EV["screen_items"], keep)])

        with ThreadPoolExecutor(max_workers=len(tracks)) as pool:
            res = dict(pool.map(track, list(tracks)))
        surv = list(res[main][0])
        if k_small:
            surv += [gid for gid in res[small_builder()][0] if gid not in surv][:k_small]
        return surv, {b: h for b, (_, h) in res.items()}, n1

    def catch_up_small(self):
        """V3-023: a generation screened without a small-builder track (its halving history has none) gets
        its small-builder track afterwards, by the same rule: its quick-test survivors are halved on the small builder
        (better half after round 1, ranked on the full screening set), and the top `full_top_k_small` not already
        fully evaluated get the full evaluation on both builders, then join the front, archive, islands and the
        improvement test. Runs once per such generation, before the next generation breeds; resumes from records."""
        d = self.st.data
        k_small = min(EV["full_top_k"], EV.get("full_top_k_small", 0))
        if not k_small or single():
            return
        done = d.setdefault("catchup", {})
        sb, n1 = small_builder(), max(1, EV["screen_items"] // 3)
        todo = [x for x in d.get("generations", []) if str(x["gen"]) not in done and sb not in
                (x.get("halving") if isinstance(x.get("halving"), dict) else {})]
        if not todo:
            return
        # V3-025: the catch-up runs the small builder flat out (its own parallelism and hourly cap); normal limits
        # come back when it ends. The weekly-allowance guard still applies to every call.
        B = CFG["budget"]
        fast = {sb: B.get("catchup_codex_parallel", B.get("codex_parallel_agents", 4))}
        gate = None
        if sb == "codex" and self.runner is not None and B.get("catchup_codex_runs_per_hour"):
            from v3 import episode as EP
            gate = EP.codex_gate_throttle()
            gate.per_hour = B["catchup_codex_runs_per_hour"]
        try:
            self._catch_up(todo, fast)
        finally:
            if gate is not None:
                gate.per_hour = B.get("codex_runs_per_hour", 60)

    def _catch_up(self, todo, fast):
        d = self.st.data
        done, k_small = d["catchup"], min(EV["full_top_k"], EV.get("full_top_k_small", 0))
        sb, n1 = small_builder(), max(1, EV["screen_items"] // 3)
        for summ in todo:
            g_old = summ["gen"]
            kids = [d["genomes"][x["child"]] for x in d["lineage"] if x["gen"] == g_old]
            passed = [c["id"] for c in kids if small_ok(c) and d["evals"].get(c["id"], {}).get("quick")
                      and quick_ok(c, next(iter(d["evals"][c["id"]]["quick"].values())))]

            def score(gid, n):
                return self.evaluate(d["genomes"][gid], f"screen{n}", self.sets["screen"][:n], [sb], workers=fast)[sb]["e2e"]
            surv, hist = A.successive_halving(passed, score, [(n1, 0.5), (EV["screen_items"], EV["full_top_k"])])
            picks = [gid for gid in surv if "full" not in d["evals"].get(gid, {})][:k_small]
            improved = []
            for gid in picks:
                g = d["genomes"][gid]
                self.evaluate(g, "full", self.sets["full"], full_builders(g), workers=fast)
                self.refresh_front()
                self.offer(g)
                imp = self.improvement(g)
                d["evals"][gid]["improvement"] = imp
                if imp["improved"]:
                    improved.append(gid)
                    for k in S.GENES:
                        d["library"][k] = {"value": g["genes"][k], "source": gid, "gen": g_old}
                d["islands"][g["island"]].append(gid)
            self.refresh_front()
            if improved:
                d["streak"] = 0
            done[str(g_old)] = {"halving": hist, "picks": picks, "improved": improved}
            self.st.event(f"V3-023 catch-up for generation {g_old}: Luna track screened {len(passed)} children, "
                          f"fully evaluated {picks}, improved {improved}")
            self.st.save()

    def generation(self, gen, excerpts=()):
        d = self.st.data
        self.catch_up_small()
        rng = A.rng_for("gen", gen)
        # resume (V3-022): the micro round and breeding are saved in d["pending"] as soon as they are done, so a
        # paused generation reuses its tweaks and children; everything after breeding resumes from stored records
        pend = d.get("pending") if (d.get("pending") or {}).get("gen") == gen else None
        if pend and "micro" in pend:
            micro_summary = pend["micro"]
            micro_children = [d["genomes"][i] for i in micro_summary.get("winner_ids", [])]
        else:
            micro_children, micro_summary = self.micro(gen, A.rng_for("micro", gen))
            pend = d["pending"] = {"gen": gen, "micro": micro_summary}
            self.st.save()
        credit = A.OperatorCredit(EV["operator_shares"], EV["operator_floor"])
        if d["credit"]:
            credit.wins, credit.trials = d["credit"]["wins"], d["credit"]["trials"]
        if "children" in pend:
            children = [d["genomes"][i] for i in pend["children"]]
            targets = pend.get("targets") or self.failure_targets(d["front"])
        else:
            targets = self.failure_targets(d["front"])
            children, attempts = [], 0
            while len(children) < EV["children_per_generation"] and attempts < EV["children_per_generation"] * 4:
                attempts += 1
                op = credit.pick(rng)
                island = len(children) % len(d["islands"])
                c = self.make_child(op, rng, gen, island, targets, list(excerpts))
                if c and c["id"] not in {x["id"] for x in children}:
                    children.append(c)
            children += [c for c in micro_children if c["id"] not in {x["id"] for x in children}]
            for c in children:
                d["genomes"][c["id"]] = c
                d["lineage"].append({"gen": gen, "child": c["id"], "parents": c["parents"], "op": c["operator"],
                                     "level": c.get("level", "macro")})
            pend.update(children=[c["id"] for c in children], targets=targets)
            self.st.save()
            self.log(f"gen {gen}: {len(children)} children")
        # quick build test → screening (successive halving on two tracks) → full evaluation
        passed = [c for c in children if self.quick_pass(c)]
        surv, hist, n1 = self.screening(passed)
        for gid in surv:
            self.evaluate(d["genomes"][gid], "full", self.sets["full"], full_builders(g=d["genomes"][gid]))
        # credit: a child "wins" if its screening score beats its first parent's on the same items (paired mean),
        # on either builder (V3-023), or if it reaches full evaluation (V3-027)
        for c in passed:
            if c.get("level") == "micro":       # pre-selected on the small builder: no operator credit
                continue
            won = c["id"] in surv
            for b in dict.fromkeys((primary(), small_builder())):
                s = d["evals"][c["id"]].get(f"screen{n1}", {}).get(b, {}).get("scores", {})
                ps = (d["evals"].get(c["parents"][0], {}).get("full") or {}).get(b, {}).get("scores", {})
                common = set(s) & set(ps)
                won = won or (bool(common) and sum(s[i] for i in common) > sum(ps[i] for i in common))
            credit.record(c["operator"], won)
        for c in children:
            if c not in passed and c.get("level") != "micro":
                credit.record(c["operator"], False)
        d["credit"] = {"wins": credit.wins, "trials": credit.trials, "shares": credit.shares()}
        d["credit_v3_027"] = True           # credited under the V3-027 rule: no catch-up of wins needed on load
        # front, archive, improvement, library, islands
        self.refresh_front()
        improved = []
        for gid in surv:
            g = d["genomes"][gid]
            self.offer(g)
            imp = self.improvement(g)
            d["evals"][gid]["improvement"] = imp
            if imp["improved"]:
                improved.append(gid)
                for k in S.GENES:                    # genes of improving designs enter the library
                    d["library"][k] = {"value": g["genes"][k], "source": gid, "gen": gen}
            d["islands"][g["island"]].append(gid)
        if gen % EV["migrate_every"] == 0:
            objs = {gid: o for gid in d["genomes"] if (o := self.objectives(gid))}
            d["islands"] = A.migrate([[i for i in isl if i in objs] for isl in d["islands"]], objs, keys(),
                                     EV["migrants"], rng)
        d["streak"] = 0 if improved else d["streak"] + 1
        d["generation"] = gen
        summary = {"gen": gen, "children": len(children), "quick_passed": len(passed), "full": surv,
                   "micro": micro_summary,
                   "improved": improved, "front": d["front"], "streak": d["streak"], "halving": hist,
                   "operator_shares": d["credit"]["shares"], "targets": targets}
        d.setdefault("generations", []).append(summary)
        d.pop("pending", None)
        self.st.event(f"generation {gen} complete: {len(children)} children, {len(passed)} passed quick build, "
                      f"{len(surv)} fully evaluated, improved {improved}")
        self.st.save()
        return summary

    def should_stop(self):
        d = self.st.data
        return d["streak"] >= EV["stop_no_improve_generations"] or d["generation"] >= EV["max_generations"]
