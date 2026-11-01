"""Run 4 (V3-037): directed breeding with honest screening. Same goal as run 3 (v3.directed): the design with which
GPT-6 Luna describes complex systems most accurately. Run 3's code is unchanged; this class changes how children are
screened and bred, from what run 3 showed (archive/run3_2026-10-25, paper/side/replay_screening.txt).

Each round:
  1. children (evolution.children_per_generation, 20): fix ~30% / combine ~35% / big ideas ~35%;
       fix      AI changes aimed at wrong answers on large systems (point, guidance; repair only for a parent whose
                builds still fail),
       combine  smart_combine: each gene taken whole from the parent that carries the strength, chosen by the designer
                from both parents' strengths cards and the gene evidence, a reason per gene; or transplant / swap;
                some combine slots pair two near-winners (the shelf) or a near-winner with the champion, and
                `three_parent` (2) take each gene from one of three parents of three different families (V3-038),
       big      big_jump, answer_method (how Luna uses its notes to answer), and a few code-only changes;
     about 1 in 5 children are reserve children of the best design of another FAMILY (starting design) than the
     champion's. AI designers get a report card per parent: score by system size, build rate, and wrong answers on
     PRACTICE items (a separate item set, never used to score designs), with the parent's own representation.
  2. one ordered main set of 120 items, every prefix spread over k = 1, 2, 4, 8; scores add up:
       quick    first 5 items: pass/fail build check (>= 3 builds), not a ranking; they count toward the score,
       screen35 the next 30: cut only clear losers (paired cluster-bootstrap 95% CI vs the champion, same items,
                entirely below 0),
       screen65 the next 30 for the rest: clear losers cut again; the top `full_top_k` (6) by the paired estimate
                go on, one slot held for the best child of a family other than the champion's if it is not a clear
                loser,
       full     topped up to all 120 items.
  3. growing proof and fresh confirmation as run 3 (V3-036), on run 4's own fresh extension and confirmation sets.
     A child whose proof ends within `shelf_margin` of the champion joins the near-winners shelf.
  4. stop after 8 rounds without a new champion, or 20 rounds.
"""
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

from v3 import stats
from v3.directed import Directed, base_of, cfg, shared_base_set
from v3.evolve import EV, primary, quick_ok
from v3.genome import schema as S
from v3.operators import ai as AI

# item ranges for run 4, disjoint from every item used before (scan of all run records, 2026-10-25): runs 1-3 used
# validation 1.0-1.3M (run 3 main sets 1.20-1.29M), 1.5-1.83M, 1.9-1.98M and test 2.0M, 2.5M, 2.7M
MAIN_START = 1_300_000          # validation: the main set (quick, screens, full)
EXT_START = 100_000             # train: extension sets ext1-ext3 for the growing proof
PRACTICE_START = 300_000        # train: practice items the designer may see in full; never used to score
CONFIRM_START = 500_000         # train: fresh confirmation sets
# the main set's composite items per k in each segment; every segment is interleaved over k
SEGMENTS = (("quick", {1: 1, 2: 1, 4: 2, 8: 1}), ("screen35", {1: 7, 2: 7, 4: 8, 8: 8}),
            ("screen65", {1: 7, 2: 7, 4: 8, 8: 8}), ("full", {1: 12, 2: 12, 4: 10, 8: 11}))
STAGE_N = {"quick": 5, "screen35": 35, "screen65": 65}
PILOT_START = 1_450_000         # validation: the run 4 pilot's items (unused, disjoint from the main set)
PILOT_STAGE_N = {"quick": 5, "screen35": 9, "screen65": 13}


def stage_n(label):
    return (d4().get("stage_n") or STAGE_N)[label]


def d4():
    return cfg()


def interleave(per_k, bases):
    """Composite ids for one segment, ordered so that every prefix is spread over k (largest first on ties)."""
    slots = sorted(((j + 0.5) / n, -k) for k, n in per_k.items() for j in range(n))
    return [f"K{-nk}x{next(bases)}" for _, nk in slots]


def run4_sets():
    from v3 import items as I
    from v3.evolve import dev_subset
    n_main = sum(sum(v.values()) for _, v in SEGMENTS)
    bases = iter(I.ladder_seeds(n_main, start=d4().get("main_start", MAIN_START)))
    main = [i for _, per_k in SEGMENTS for i in interleave(per_k, bases)]
    main += dev_subset(EV.get("dev_items", True))                    # the 10 v2 dev items close the full set
    n_ext = d4().get("ext_bases", 30)
    ext = I.ladder_seeds(3 * n_ext, start=d4().get("ext_start", EXT_START))
    prac = I.ladder_seeds(d4().get("practice_bases", 4), start=d4().get("practice_start", PRACTICE_START))
    return {"main": main, "quick": main[:5], "screen": main[5:65], "full": main,
            "ext1": shared_base_set(ext[:n_ext]), "ext2": shared_base_set(ext[n_ext:2 * n_ext]),
            "ext3": shared_base_set(ext[2 * n_ext:]),
            "practice": [f"K{k}x{b}" for b in prac for k in (4, 8)]}


def pilot4_sets():
    """A shake-down slice of run 4 (stages of 5 / 9 / 13 items, a 16-item main set, 2 practice items, extension sets
    of 2 bases x 4 k); pair with evolution.directed.stage_n = PILOT_STAGE_N."""
    from v3 import items as I
    bases = iter(I.ladder_seeds(16 + 1 + 6, start=PILOT_START))
    main = [i for per_k in ({1: 1, 2: 1, 4: 2, 8: 1}, {1: 1, 2: 1, 4: 1, 8: 1}, {1: 1, 2: 1, 4: 1, 8: 1},
                            {1: 1, 2: 1, 4: 0, 8: 1}) for i in interleave({k: n for k, n in per_k.items() if n}, bases)]
    pb = next(bases)
    rest = [next(bases) for _ in range(6)]
    return {"main": main, "quick": main[:5], "screen": main[5:13], "full": main,
            "ext1": shared_base_set(rest[:2]), "ext2": shared_base_set(rest[2:4]), "ext3": shared_base_set(rest[4:]),
            "practice": [f"K4x{pb}", f"K8x{pb}"]}


def k_of(iid):
    m = re.match(r"^K(\d+)x", iid)
    return int(m.group(1)) if m else None


class Directed4(Directed):
    # -- families: the starting design a design descends from (carried-over run 3 designs keep their run 3 family)
    def family(self, gid):
        g = self.st.data["genomes"]
        r = self.root(gid)
        return g[r].get("family") or r

    def reserve_parent(self, rng):
        d = self.st.data
        cf = self.family(d["champion"])
        best = {}
        for gid in d["genomes"]:
            e, f = self.e2e(gid), self.family(gid)
            if e is not None and f != cf and (f not in best or (e, gid) > best[f]):
                best[f] = (e, gid)
        if not best:
            return None
        fams = sorted(best, key=lambda f: -best[f][0])[:3]          # the three strongest other families
        return best[fams[rng.randrange(len(fams))]][1]

    # -- scores on the main set
    def scores(self, gid, stage="main"):
        return ((self.st.data["evals"].get(gid, {}).get(stage) or {}).get(primary()) or {}).get("scores") or {}

    def mean(self, gid):
        s = self.scores(gid)
        return statistics.mean(s.values()) if s else None

    def vs_champion(self, gid):
        """Paired 95% CI of gid minus the champion on the items gid has run (cluster = base system)."""
        champ = self.st.data["champion"]
        a, b = self.scores(gid), self.scores(champ, "full")
        return stats.paired_ci(a, b, {i: base_of(i) for i in set(a) | set(b)}, n_boot=2000)

    def run_main(self, gid, label, n, workers=4):
        """gid on the first n items of the main set; records are shared across stages (one phase), so each stage only
        runs the new items. The cumulative result is stored under "main" and under the stage's label."""
        b = primary()
        r = self.evaluate(self.st.data["genomes"][gid], "main", self.sets["main"][:n], [b], workers={b: workers})[b]
        with self._lock:
            self.st.data["evals"][gid][label] = {b: r}
            self.st.save()
        return r

    def quick_pass(self, g):
        return quick_ok(g, self.run_main(g["id"], "quick", stage_n("quick")))

    def screening(self, passed):
        """Section 2 of the module doc. Returns (survivors, history, 35)."""
        d = self.st.data
        cf = self.family(d["champion"])
        hist = []

        def stage(ids, label, n):
            par = d4().get("parallel_children", 3)
            with ThreadPoolExecutor(max_workers=par) as pool:
                list(pool.map(lambda gid: self.run_main(gid, label, n, workers=max(1, 12 // par)), ids))
            cis = {gid: self.vs_champion(gid) for gid in ids}
            keep = [gid for gid in ids if not (cis[gid]["hi"] is not None and cis[gid]["hi"] < 0)]
            hist.append({"stage": label, "items": n, "scores": {gid: self.mean(gid) for gid in ids},
                         "vs_champion": {gid: {k: cis[gid][k] for k in ("est", "lo", "hi")} for gid in ids},
                         "kept": keep, "cut": [gid for gid in ids if gid not in keep]})
            with self._lock:                # the dashboard shows each cut as soon as it is made
                d.setdefault("pending", {})["halving4"] = hist
                self.st.save()
            return keep, cis

        ids = [c["id"] for c in passed]
        alive, _ = stage(ids, "screen35", stage_n("screen35"))
        alive, cis = stage(alive, "screen65", stage_n("screen65"))
        k = EV["full_top_k"]
        ranked = sorted(alive, key=lambda g: (-(cis[g]["est"] if cis[g]["est"] is not None else -9), g))
        top = ranked[:k]
        other = [g for g in ranked if self.family(g) != cf]
        if other and not any(self.family(g) != cf for g in top):        # one slot for another family (improvement 5)
            top = ranked[:k - 1] + [other[0]] if k > 1 else top
        hist.append({"stage": "to_full", "kept": top, "family_slot": other[0] if other and other[0] in top
                     and other[0] not in ranked[:k] else None})
        for gid in top:
            self.run_main(gid, "full", len(self.sets["main"]), workers=12 // max(1, min(len(top), 3)))
        return top, hist, stage_n("screen35")

    # -- report cards (improvement 2) and strengths cards (improvement 4)
    def practice_report(self, gid, max_wrong=8):
        """Parent gid on the practice items (run once per design, then kept): accuracy by size, wrong answers with the
        question, the builder's answer and the correct one, and its own representation of one k = 8 practice system.
        Practice items are never used to score designs, so the designer may see them in full."""
        d = self.st.data
        have = d.setdefault("practice_reports", {}).get(gid)
        if have:
            return have
        from v3 import items as I
        b, g = primary(), d["genomes"][gid]
        with ThreadPoolExecutor(max_workers=8) as pool:
            recs = list(pool.map(lambda i: self.runner(g, i, b, "practice"), self.sets["practice"]))
        out, wrong, rep_shown = {}, [], None
        for r in recs:
            if r.get("error"):
                continue
            item = I.load(r["item"])
            key, got = I.answer_key(item), I.parse_answers(r.get("final_raw"))
            k = item.get("k")
            out.setdefault(f"k={k}", []).append(sum(got.get(q) == a for q, a in key.items()) / max(1, len(key)))
            blocks = dict(re.findall(r"\[(s\d+\.p\d+)\]\s*(.*?)(?=\n\[s\d+\.p\d+\]|\Z)", item["question"], re.S))
            for q, a in sorted(key.items()):
                if got.get(q) != a and len(wrong) < max_wrong:
                    lines = blocks.get(q, "").strip().splitlines()
                    opts = {int(m.group(1)): m.group(2) for ln in lines[1:] if (m := re.match(r"\s*(\d+)\.\s*(.*)", ln))}
                    wrong.append({"system_size": f"k={k}", "question": lines[0] if lines else q,
                                  "builder_answered": opts.get(got.get(q), "(no answer)"),
                                  "correct": opts.get(a, str(a))})
            if rep_shown is None and k == 8 and r.get("representation"):
                rep_shown = str(r["representation"])[:3000]
        rep = {"practice_accuracy_by_size": {k: round(statistics.mean(v), 3) for k, v in sorted(out.items())},
               "wrong_answers": wrong, "its_representation_of_one_k8_practice_system": rep_shown}
        if recs and not all(r.get("error") for r in recs):
            with self._lock:
                d["practice_reports"][gid] = rep
                self.st.save()
        return rep

    def report_card(self, gid):
        f = self.failure_profile(gid)
        card = {"overall": f.get("overall"), "build_rate": f.get("build_rate"), "score_by_size": f.get("score_by_k"),
                "builds_are_fine": (f.get("build_rate") or 0) >= d4().get("repair_below_build", 0.95)}
        if not card["builds_are_fine"]:
            card["common_build_problems"] = f.get("common_build_problems")
        card.update(self.practice_report(gid))
        return {"report_card": card}

    def gene_stats(self, a, b):
        """Per gene: A's and B's variant, and the mean full e2e of fully evaluated designs carrying it minus the mean of
        all fully evaluated designs (how many designs carry it)."""
        G = self.st.data["genomes"]
        done = [g for g in G if self.e2e(g) is not None]
        mean = statistics.mean(self.e2e(g) for g in done) if done else 0
        out = {}
        for k in S.GENES:
            row = {}
            for side, gid in (("A", a), ("B", b)):
                h = (G[gid].get("gene_hashes") or {}).get(k)
                carry = [self.e2e(g) for g in done if (G[g].get("gene_hashes") or {}).get(k) == h]
                row[side] = {"same_as_other": h == (G[b if side == "A" else a].get("gene_hashes") or {}).get(k),
                             "vs_average": round(statistics.mean(carry) - mean, 3) if carry else None,
                             "designs": len(carry)}
            out[k] = row
        return out

    def gene_stats3(self, ids):
        """gene_stats for three parents (V3-038): per gene, each parent's variant scored against the average."""
        G = self.st.data["genomes"]
        done = [g for g in G if self.e2e(g) is not None]
        mean = statistics.mean(self.e2e(g) for g in done) if done else 0
        out = {}
        for k in S.GENES:
            hs = {x: (G[gid].get("gene_hashes") or {}).get(k) for x, gid in zip("ABC", ids)}
            row = {}
            for x, h in hs.items():
                carry = [self.e2e(g) for g in done if (G[g].get("gene_hashes") or {}).get(k) == h]
                row[x] = {"same_as": [y for y in hs if y != x and hs[y] == h],
                          "vs_average": round(statistics.mean(carry) - mean, 3) if carry else None,
                          "designs": len(carry)}
            out[k] = row
        return out

    def trio(self, rng):
        """Three parents from three different families (V3-038): the best fully evaluated design of each family among
        the leaders, the shelf and the champion; three of the (up to) four strongest families. None if fewer than 3."""
        d = self.st.data
        pool = [g for g in dict.fromkeys([d["champion"]] + d.get("shelf", []) + self.leaders())
                if g in d["genomes"] and self.e2e(g) is not None]
        best = {}
        for gid in pool:
            f = self.family(gid)
            if f not in best or (self.e2e(gid), gid) > best[f]:
                best[f] = (self.e2e(gid), gid)
        fams = sorted(best, key=lambda f: (-best[f][0], f))[:4]
        G = d["genomes"]

        def sensible(ids):          # every parent has a gene that matters that the other two do not share
            return all(any(all(S.canonical(G[x]["genes"][k]) != S.canonical(G[y]["genes"][k]) for y in ids if y != x)
                           for k in S.live_genes(G[x]["genes"])) for x in ids)
        trios = [t for t in __import__("itertools").combinations(fams, 3) if sensible([best[f][1] for f in t])]
        if not trios:
            return None             # no three-parent mix makes sense: the slot becomes an ordinary combine
        pick = trios[rng.randrange(len(trios))]
        return [best[f][1] for f in sorted(pick, key=lambda f: (-best[f][0], f))]

    def strengths_card(self, gid):
        rc = self.report_card(gid)["report_card"]
        return {k: rc.get(k) for k in ("overall", "build_rate", "score_by_size", "practice_accuracy_by_size")}

    # -- breeding
    def plan(self, n):
        mix = d4().get("mix", {"fix": 0.3, "combine": 0.35, "big": 0.35})
        n_comb, n_big = round(n * mix["combine"]), round(n * mix["big"])
        kinds = ["fix"] * (n - n_comb - n_big) + ["combine"] * n_comb + ["big"] * n_big
        res = set(range(0, n, max(1, round(1 / d4().get("reserve", 0.2)))))     # spread over every kind
        return [(k, i in res) for i, k in enumerate(kinds)]

    def shelf_pair(self, rng):
        d = self.st.data
        pool = [g for g in dict.fromkeys(d.get("shelf", []) + [d["champion"]]) if g in d["genomes"]]
        if len(pool) < 2:
            return None
        a = pool[rng.randrange(len(pool))]
        rest = [g for g in pool if g != a]
        return a, rest[rng.randrange(len(rest))]

    def breed_one(self, kind, reserve, op, rng, gen, shelf=False, parent=None):
        d = self.st.data
        a = parent or (self.reserve_parent(rng) if reserve else None) or self.pick_leader(rng)
        if op == "repair" and (self.failure_profile(a).get("build_rate") or 0) >= d4().get("repair_below_build", 0.95):
            op = "point"                    # builds already pass: nothing to repair (improvement 3)
        if op == "smart_combine3":
            ids = self.trio(rng)
            if not ids:
                return None
            bad = [m for m in d.get("mixes", []) if m.get("bad")][-5:]
            child = AI.smart_combine3([d["genomes"][g] for g in ids], [self.strengths_card(g) for g in ids],
                                      self.gene_stats3(ids), [{"picks": m["picks"], "result": m["result"]} for m in bad],
                                      call=self.designer, generation=gen)
            if child is None or child["id"] in d["genomes"]:
                return None
            child.update(generation=gen, island=0, kind=kind, reserve=False)
            return child
        if op == "smart_combine":
            pair = self.shelf_pair(rng) if shelf else None
            if pair:
                a, b = pair
            else:
                others = [g for g in self.leaders() if g != a]
                if not others:
                    return None
                b = self.pick_leader(rng, others)
            bad = [m for m in d.get("mixes", []) if m.get("bad")][-5:]
            child = AI.smart_combine(d["genomes"][a], d["genomes"][b], self.strengths_card(a), self.strengths_card(b),
                                     self.gene_stats(a, b), [{"picks": m["picks"], "result": m["result"]} for m in bad],
                                     call=self.designer, generation=gen)
            if child is None or child["id"] in d["genomes"]:
                return None
            child.update(generation=gen, island=0, kind=kind, reserve=reserve, shelf=bool(pair))
            return child
        if op in AI.ALLOWED:
            child = AI.apply(op, [d["genomes"][a]], self.report_card(a), [], call=self.designer, generation=gen,
                             island=0)
            if child is None or child["id"] in d["genomes"]:
                return None
            child.update(generation=gen, island=0, kind=kind, reserve=reserve)
            return child
        return super().breed_one(kind, reserve, op, rng, gen)

    def breed(self, gen, rng, credits):
        n = EV["children_per_generation"]
        children, seen, comb_i = [], set(), 0
        plan = self.plan(n)
        # V3-038: the last `three_parent` combine slots that are not shelf slots are three-parent combines
        comb = [i for i, (k, _) in enumerate(plan) if k == "combine"]
        free = [i for j, i in enumerate(comb) if not (j % 2 == 0 and self.st.data.get("shelf"))]
        three = set(free[len(free) - d4().get("three_parent", 0):]) if d4().get("three_parent", 0) else set()
        # V3-040: from round `champion_children_from_round` on, the first non-reserve fix slot and the first non-reserve
        # big slot breed from the champion (the fix operator drawn as usual; the big slot is answer_method), so every
        # round tries to improve the champion itself from its own report card
        champ_slots = {}
        start = d4().get("champion_children_from_round")
        if start and gen >= start:
            for want, op_forced in (("fix", None), ("big", "answer_method")):
                i = next((i for i, (k, r) in enumerate(plan) if k == want and not r), None)
                if i is not None:
                    champ_slots[i] = op_forced
        for slot, (kind, reserve) in enumerate(plan):
            shelf = False
            if kind == "combine":
                shelf = comb_i % 2 == 0 and len(self.st.data.get("shelf", [])) > 0   # half the combines use the shelf
                comb_i += 1
            for t in range(cfg().get("tries_per_child", 4)):
                # a champion slot (V3-040) keeps the champion as parent for two tries, then breeds as usual
                champ = slot in champ_slots and t < 2
                # a three-parent slot gets two tries; if Claude's picks keep clashing it becomes an ordinary combine
                op = (champ_slots[slot] or self.pick_op(credits[kind], rng)) if champ else "smart_combine" if shelf \
                    else "smart_combine3" if slot in three and t < 2 else self.pick_op(credits[kind], rng)
                c = self.breed_one(kind, reserve and not shelf and op != "smart_combine3", op, rng, gen, shelf=shelf,
                                   parent=self.st.data["champion"] if champ else None)
                if c is None or c["id"] in seen:
                    if op in credits[kind].base:
                        credits[kind].record(op, False)
                    continue
                children.append(c)
                seen.add(c["id"])
                break
        return children

    # -- after screening: bad mixes, shelf, family shares
    def record_mixes(self, children):
        d = self.st.data
        done = {m["child"] for m in d.setdefault("mixes", [])}
        for c in children:
            if c.get("operator") not in ("smart_combine", "smart_combine3") or c["id"] in done \
                    or len(self.scores(c["id"])) < stage_n("screen35"):
                continue
            s = self.scores(c["id"])
            ps = [self.scores(p, "full") for p in c["parents"][:3]]
            common = [i for i in s if all(i in p for p in ps)]
            if not common:
                continue
            m = [round(statistics.mean(x[i] for i in common), 3) for x in [s] + ps]
            res = {"child": m[0], **{f"parent_{x}": v for x, v in zip("ABC", m[1:])}, "items": len(common)}
            d["mixes"].append({"child": c["id"], "parents": c["parents"][:3], "bad": m[0] < min(m[1:]),
                               "picks": (c.get("op_notes") or {}).get("pick"), "result": res})

    def update_shelf(self, gen):
        d = self.st.data
        shelf = d.setdefault("shelf", [])
        for gid, recs in d.get("proofs", {}).items():
            for r in recs:
                if r.get("round") != gen or r.get("kind") or not r.get("steps"):
                    continue
                est = ((r["steps"][-1].get("vs") or {}).get(r["champion"]) or {}).get("est")
                if est is not None and est >= -d4().get("shelf_margin", 0.02) and gid not in shelf \
                        and gid != d["champion"]:
                    shelf.append(gid)
        d["shelf"] = [g for g in shelf if g != d["champion"]]

    def family_shares(self, children):
        n = len(children) or 1
        c = {}
        for ch in children:
            f = self.family(ch["id"])
            c[f] = c.get(f, 0) + 1
        return {f: round(v / n, 2) for f, v in sorted(c.items(), key=lambda x: -x[1])}

    def generation(self, gen, excerpts=()):
        summary = super().generation(gen, excerpts)
        d = self.st.data
        children = [d["genomes"][i] for i in [x["child"] for x in d["lineage"] if x["gen"] == gen]]
        self.record_mixes(children)
        self.update_shelf(gen)
        shares = self.family_shares(children)
        summary.update(family_shares=shares, shelf=list(d.get("shelf", [])),
                       bad_mixes=sum(1 for m in d.get("mixes", []) if m.get("bad") and m["child"] in
                                     {c["id"] for c in children}))
        top = max(shares.values()) if shares else 0
        prev = (d["generations"][-2].get("family_shares") or {}) if len(d["generations"]) > 1 else {}
        if top > d4().get("family_alarm", 0.7) and prev and max(prev.values()) > d4().get("family_alarm", 0.7):
            self.st.event(f"family alarm: one family over {d4().get('family_alarm', 0.7):.0%} of children for two "
                          f"rounds ({shares})")
        self.st.save()
        return summary
