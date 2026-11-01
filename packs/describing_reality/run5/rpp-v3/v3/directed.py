"""Run 3 (V3-034): directed breeding toward one goal, a design with which GPT-6 Luna describes complex systems most
accurately (Luna's end-to-end score). Not a simulation of natural evolution: the budget goes where it pays.

Each round:
  1. children (config evolution.children_per_generation): a fixed mix of three kinds,
       fix      AI designer changes aimed at the parent's own failures (point, repair, guidance, big_jump),
       combine  two leaders combined: one leader plus the change that made another leader better than its parent
                (transplant), or a gene-level swap,
       explore  code-only changes (simplify, param, numeric; graft once the gene library has entries);
     within each kind the operators' shares adapt to their results (a failed attempt counts as a try);
     about 10% of children are "reserve" children of the best design of another family (core kind) than the
     champion's, so the pool does not collapse to one line. Parents are the leaders: the top fully evaluated designs
     by Luna e2e, picked with rank weights. One pool, no islands, no micro level.
  2. quick build test -> successive halving (10 then 30 items) -> full evaluation (120 items) of the top few.
  3. growing proof (V3-036): the round's best fully evaluated child, whatever its full score, and any other child whose
     full e2e beats the champion's, are compared with the champion and the control on fresh items only: extension
     sets of 120 items that no design was selected on, pooled while undecided (120 -> 240 -> 360). The full set, where
     the children and the champion were all selected, is not used to judge. Proven = paired cluster-bootstrap 95% CI
     above 0 against both. No lead over the champion, or no proof at 360, ends it.
  4. fresh confirmation: a proven candidate is tested once on a new set of 120 composite items never used before (30
     base systems x k = 1, 2, 4, 8); it becomes champion if its mean is above both the champion's and the control's.
     At most one confirmation per round; other proven candidates wait in a queue and are re-proven against the
     champion of the day.
  5. stop after 4 rounds without a new champion, or 20 rounds (or the human's PAUSE, or an allowance guard).
"""
import re

from v3 import stats
from v3.config import CFG
from v3.evolve import EV, Evolution, primary, systems
from v3.genome import schema as S
from v3.operators import ai as AI
from v3.operators import deterministic as D
from v3.selection import algorithms as A

KS = (1, 2, 4, 8)
# item ranges (base seeds, COMPOSITE_STRIDE apart), disjoint from runs 1-2 (validation 1,500,000-1,616,500), the
# cross probe (1,800,000+), calibration (1,900,000+) and the sealed final sets (test split)
MAIN_START = 1_200_000          # quick, screen, full
EXT_START = 1_650_000           # extension sets ext1, ext2, ext3 (V3-036)
CONFIRM_START = 20_000          # fresh confirmation sets (train split; one block per confirmation, never reused)
KRE = re.compile(r"^[KG]\d*x?(\d+)$")


def cfg():
    return EV.get("directed") or {}


def directed():
    return EV.get("mode") in ("directed", "directed4")      # run 4 (V3-037) is directed breeding too


def base_of(iid):
    """The base system of a composite item (K<k>x<base>): items sharing a base share subsystems, so they are one
    cluster in the bootstrap. Other items are their own cluster."""
    m = re.match(r"^K\d+x(\d+)$", iid)
    return f"B{m.group(1)}" if m else iid


def shared_base_set(bases):
    return [f"K{k}x{b}" for b in bases for k in KS]


def run3_sets():
    """The fixed, pre-registered item sets of run 3: quick / screen / full as in run 2's layout (fresh bases), plus three
    extension sets of 30 bases x 4 k (120 items each) for the growing proof (ext3 added by V3-036; ext1 and ext2 are
    unchanged because the base sequence is a fixed prefix)."""
    from v3 import items as I
    from v3.evolve import dev_subset
    ladder = {s: v for s, v in (EV.get("ladder") or {}).items() if s in ("quick", "screen", "full")}
    bases = iter(I.ladder_seeds(sum(n for per_k in ladder.values() for n in per_k.values()), start=MAIN_START))
    out = {s: [f"K{k}x{next(bases)}" for k in sorted(ladder[s], key=int) for _ in range(ladder[s][k])]
           for s in ("quick", "screen", "full")}
    out["full"] = dev_subset(EV.get("dev_items", True)) + out["full"]
    n_ext = cfg().get("ext_bases", 30)
    ext = I.ladder_seeds(3 * n_ext, start=EXT_START)
    out["ext1"], out["ext2"], out["ext3"] = (shared_base_set(ext[:n_ext]), shared_base_set(ext[n_ext:2 * n_ext]),
                                             shared_base_set(ext[2 * n_ext:]))
    return out


def confirm_set(previous):
    """The next fresh confirmation set: `confirm_bases` bases from the train split, starting after every base used by
    earlier confirmation sets (`previous` = their id lists), so no item is ever reused. Deterministic."""
    from v3 import items as I
    from v3.items import COMPOSITE_STRIDE
    used = [int(m.group(1)) for ids in previous for i in ids if (m := re.match(r"^K\d+x(\d+)$", i))]
    start = max(used) + COMPOSITE_STRIDE if used else cfg().get("confirm_start", CONFIRM_START)
    return shared_base_set(I.ladder_seeds(cfg().get("confirm_bases", 30), start=start))


class Directed(Evolution):
    # -- setup
    def baseline(self, seeds, controls=("A0P",)):
        super().baseline(seeds, controls)
        d = self.st.data
        if not d.get("champion"):
            best = max((g["id"] for g in seeds.values() if self.objectives(g["id"])),
                       key=lambda gid: (self.objectives(gid)["e2e"] or 0, gid))
            d["champion"] = best
            d["champions"] = [{"gid": best, "round": 0, "how": "best starting design on the full set"}]
            self.st.event(f"run 3: starting champion {best} (best starting design by Luna full e2e)")
        d.setdefault("proofs", {}); d.setdefault("confirm_sets", []); d.setdefault("confirm_queue", [])
        self.st.save()

    # -- evaluation that reuses a finished stage (the champion and control meet each extension set once)
    def ensure(self, gid, stage, items):
        g = self.st.data["genomes"][gid]
        b = primary()
        have = self.st.data["evals"].get(gid, {}).get(stage, {}).get(b)
        if have and have.get("n") == len(items) and not (have["n"] - len(have.get("build_items") or {})):
            return have
        return self.evaluate(g, stage, items, [b])[b]

    def e2e(self, gid):
        o = self.objectives(gid)
        return (o or {}).get("e2e")

    # -- parents
    def leaders(self):
        d = self.st.data
        ids = [gid for gid in d["genomes"] if self.e2e(gid) is not None]
        return sorted(ids, key=lambda gid: (-(self.e2e(gid) or 0), gid))[:cfg().get("parent_pool", 6)]

    def pick_leader(self, rng, pool=None):
        pool = pool or self.leaders()
        w = [len(pool) - i for i in range(len(pool))]          # rank weights: the best is most likely
        r, acc = rng.random() * sum(w), 0.0
        for gid, x in zip(pool, w):
            acc += x
            if r <= acc:
                return gid
        return pool[-1]

    def reserve_parent(self, rng):
        """The best fully evaluated design of a family (core kind) other than the champion's; None if there is none."""
        d = self.st.data
        champ_kind = d["genomes"][d["champion"]]["genes"]["core"]["kind"]
        best = {}
        for gid in d["genomes"]:
            e = self.e2e(gid)
            kind = d["genomes"][gid]["genes"]["core"]["kind"]
            if e is not None and kind != champ_kind and (kind not in best or (e, gid) > best[kind]):
                best[kind] = (e, gid)
        if not best:
            return None
        kinds = sorted(best)
        return best[kinds[rng.randrange(len(kinds))]][1]

    def root(self, gid):
        g = self.st.data["genomes"]
        while g[gid]["parents"] and g[gid]["parents"][0] in g:
            gid = g[gid]["parents"][0]
        return gid

    def failure_profile(self, gid):
        """Parent 1's own failures on its full evaluation (V3-034): construction / reasoning / ok counts, the score and
        build rate by k, and the most frequent checker problems (quoted item terms blanked). No answers, no items."""
        f = self.st.data["evals"].get(gid, {}).get("full", {}).get(primary()) or {}
        t = {"construction": 0, "reasoning": 0, "answered_ok": 0}
        by_k = {}
        for item, b in (f.get("build_items") or {}).items():
            s = (f.get("scores") or {}).get(item)
            t["construction" if not b else "reasoning" if s is not None and s < 0.5 else "answered_ok"] += 1
            m = re.match(r"^K(\d+)x", item)
            if m and s is not None:
                by_k.setdefault(f"k={m.group(1)}", []).append(s)
        t["score_by_k"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(by_k.items(), key=lambda kv: int(kv[0][2:]))}
        t["overall"] = f.get("e2e") and round(f["e2e"], 3)
        t["build_rate"] = f.get("build") and round(f["build"], 3)
        t["common_build_problems"] = f.get("build_problems") or []
        return t

    # -- children
    def plan(self, n):
        """The kinds of this round's n children, in order: fix / combine / explore by the fixed mix, and which slots are
        reserve children (taken from fix and explore slots alternately)."""
        mix = cfg().get("mix", {"fix": 0.5, "combine": 0.3, "explore": 0.2})
        n_comb, n_expl = round(n * mix["combine"]), round(n * mix["explore"])
        kinds = ["fix"] * (n - n_comb - n_expl) + ["combine"] * n_comb + ["explore"] * n_expl
        n_res = round(n * cfg().get("reserve", 0.1))
        fix_i = [i for i, k in enumerate(kinds) if k == "fix"]
        exp_i = [i for i, k in enumerate(kinds) if k == "explore"]
        alt = [x for pair in zip(fix_i[::-1], exp_i[::-1]) for x in pair] + fix_i + exp_i
        reserve = set(list(dict.fromkeys(alt))[:n_res])
        return [(k, i in reserve) for i, k in enumerate(kinds)]

    def credit(self):
        """One OperatorCredit per kind, sharing the stored win/trial counts (keyed by operator)."""
        d, out = self.st.data, {}
        for kind, shares in cfg()["ops"].items():
            c = A.OperatorCredit(shares, EV["operator_floor"])
            stored = d.get("credit") or {}
            for op in shares:
                if op in (stored.get("wins") or {}):
                    c.wins[op], c.trials[op] = stored["wins"][op], stored["trials"][op]
            out[kind] = c
        return out

    def save_credit(self, credits):
        mix = cfg().get("mix")
        wins = {op: c.wins[op] for c in credits.values() for op in c.base}
        trials = {op: c.trials[op] for c in credits.values() for op in c.base}
        shares = {op: mix[kind] * s for kind, c in credits.items() for op, s in c.shares().items()}
        self.st.data["credit"] = {"wins": wins, "trials": trials, "shares": shares,
                                  "by_kind": {kind: c.shares() for kind, c in credits.items()}, "mix": mix}

    def pick_op(self, credit, rng):
        sh = credit.shares()
        if not self.st.data.get("library"):
            sh.pop("graft", None)                 # nothing to graft from yet (V3-034: no free credit for graft)
        tot = sum(sh.values())
        r, acc = rng.random() * tot, 0.0
        for k in sorted(sh):
            acc += sh[k]
            if r <= acc:
                return k
        return sorted(sh)[-1]

    def breed_one(self, kind, reserve, op, rng, gen):
        d = self.st.data
        a = (self.reserve_parent(rng) if reserve else None) or self.pick_leader(rng)
        if op in ("transplant", "swap", "one_point"):
            # the second leader comes from another line (a different starting design) when there is one
            others = [g for g in self.leaders() if g != a and self.root(g) != self.root(a)] or \
                [g for g in self.leaders() if g != a]
            if not others:
                return None
            b = self.pick_leader(rng, others)
            parents = [d["genomes"][a], d["genomes"][b]]
            key_parent = None
            bp = d["genomes"][b]["parents"][:1]
            if bp and bp[0] in d["genomes"] and self.e2e(bp[0]) is not None and \
                    (self.e2e(b) or 0) > (self.e2e(bp[0]) or 0):
                key_parent = d["genomes"][bp[0]]           # b's own proven change: b beat its parent
            child = D.apply(op, parents, rng, library=dict(d["library"]), cma_state=d["cma"], key_parent=key_parent)
        elif op in AI.ALLOWED:
            child = AI.apply(op, [d["genomes"][a]], self.failure_profile(a), [], call=self.designer, generation=gen,
                             island=0)
        else:
            child = D.apply(op, [d["genomes"][a]], rng, library=dict(d["library"]), cma_state=d["cma"])
            if child is not None and op == "numeric":
                d["cma"] = child["op_notes"]["cma"]
        if child is None or child["id"] in d["genomes"]:
            return None
        child.update(generation=gen, island=0, kind=kind, reserve=reserve)
        return child

    def breed(self, gen, rng, credits):
        n = EV["children_per_generation"]
        children, seen = [], set()
        for kind, reserve in self.plan(n):
            for _ in range(cfg().get("tries_per_child", 4)):
                op = self.pick_op(credits[kind], rng)
                c = self.breed_one(kind, reserve, op, rng, gen)
                if c is None or c["id"] in seen:
                    credits[kind].record(op, False)     # a failed attempt is a try (V3-034)
                    continue
                children.append(c)
                seen.add(c["id"])
                break
        return children

    # -- growing proof and confirmation
    def pooled(self, gid, stages):
        f = {}
        sysm = {}
        for st in stages:
            r = self.st.data["evals"].get(gid, {}).get(st, {}).get(primary()) or {}
            f.update(r.get("scores") or {})
            sysm.update(r.get("systems") or {})
        return f, sysm

    def compare(self, a, b, stages):
        sa, ma = self.pooled(a, stages)
        sb, mb = self.pooled(b, stages)
        clusters = {i: base_of(i) if re.match(r"^K\d+x", i) else {**mb, **ma}.get(i, i) for i in set(sa) | set(sb)}
        return stats.paired_ci(sa, sb, clusters, n_boot=2000)

    def prove(self, gid):
        """The growing proof against the current champion and the control (see module doc). Returns the record."""
        d = self.st.data
        champ, ctrl = d["champion"], d["controls"][0]
        refs = list(dict.fromkeys([champ, ctrl]))
        levels = cfg().get("proof_levels", ["ext1", "ext2", "ext3"])
        steps, verdict = [], "not_proven"
        for L, stage in enumerate(levels):
            if stage != "full":
                for x in [gid] + refs:
                    self.ensure(x, stage, self.sets[stage])
            cis = {r: self.compare(gid, r, levels[:L + 1]) for r in refs}
            steps.append({"stage": stage, "n_items": cis[champ]["n_items"],
                          "vs": {r: {k: cis[r][k] for k in ("est", "lo", "hi")} for r in refs}})
            if all(cis[r]["improved"] for r in refs):
                verdict = "proven"
                break
            if (cis[champ]["est"] or 0) <= 0:
                verdict = "no_lead"
                break
        rec = {"champion": champ, "control": ctrl, "steps": steps, "verdict": verdict, "round": d["generation"] + 1}
        recs = d["proofs"].setdefault(gid, [])
        if recs and recs[-1].get("round") == rec["round"] and recs[-1].get("champion") == champ and \
                "kind" not in recs[-1]:
            recs[-1] = rec                          # a resumed round recomputes the same proof
        else:
            recs.append(rec)
        return rec

    def confirm(self, gid):
        d = self.st.data
        champ, ctrl = d["champion"], d["controls"][0]
        pend = d.setdefault("pending", {"gen": d["generation"] + 1})
        if (pend.get("confirm") or {}).get("gid") == gid:       # resumed: the same fresh set, never a new one
            c, items = pend["confirm"]["index"], pend["confirm"]["items"]
        else:
            c = len(d["confirm_sets"])
            items = confirm_set(d["confirm_sets"])
            d["confirm_sets"].append(items)
            pend["confirm"] = {"gid": gid, "index": c, "items": items}
            self.st.save()
        stage = f"confirm{c}"
        refs = list(dict.fromkeys([champ, ctrl]))
        for x in [gid] + refs:
            self.ensure(x, stage, items)
        cis = {r: self.compare(gid, r, [stage]) for r in refs}
        passed = all((cis[r]["est"] or 0) > 0 for r in refs)
        rec = {"set": stage, "n_items": len(items), "champion": champ, "control": ctrl, "passed": passed,
               "vs": {r: {k: cis[r][k] for k in ("est", "lo", "hi")} for r in refs}}
        d["proofs"].setdefault(gid, []).append(dict(rec, kind="confirmation", round=d["generation"] + 1))
        pend.pop("confirm", None)
        return rec

    def judge(self, surv, gen):
        """Growing proof for this round's fully evaluated children (best first) and queued candidates; at most
        `confirm_per_round` confirmations. Returns the new champion's id or None."""
        d = self.st.data
        champ = d["champion"]
        queue = [g for g in d["confirm_queue"] if g != champ]
        fresh = sorted((g for g in surv if (self.e2e(g) or 0) > (self.e2e(champ) or 0) and g not in queue),
                       key=lambda g: (-(self.e2e(g) or 0), g))
        if cfg().get("proof_best_always"):      # V3-036: the round's best child meets the champion on fresh items anyway
            best = max((g for g in surv if g != champ and self.e2e(g) is not None), key=lambda g: (self.e2e(g), g),
                       default=None)
            if best and best not in fresh and best not in queue:
                fresh.append(best)
        slots, new_champ, still = cfg().get("confirm_per_round", 1), None, []
        for gid in queue + fresh:
            rec = self.prove(gid)
            if rec["verdict"] != "proven":
                continue
            if slots <= 0:
                still.append(gid)
                continue
            slots -= 1
            c = self.confirm(gid)
            if c["passed"]:
                new_champ = gid
                break
        d["confirm_queue"] = still
        if new_champ:
            d["champion"] = new_champ
            d["champions"].append({"gid": new_champ, "round": gen, "how": "proven and confirmed",
                                   "previous": champ})
            g = d["genomes"][new_champ]
            for k in S.GENES:                       # the champion's genes are the graft library
                d["library"][k] = {"value": g["genes"][k], "source": new_champ, "gen": gen}
        return new_champ

    # -- one round
    def generation(self, gen, excerpts=()):
        d = self.st.data
        rng = A.rng_for("gen", gen)
        pend = d.get("pending") if (d.get("pending") or {}).get("gen") == gen else None
        credits = self.credit()
        if pend and "children" in pend:
            children = [d["genomes"][i] for i in pend["children"]]
        else:
            children = self.breed(gen, rng, credits)
            for c in children:
                d["genomes"][c["id"]] = c
                d["lineage"].append({"gen": gen, "child": c["id"], "parents": c["parents"], "op": c["operator"],
                                     "level": "macro", "kind": c["kind"], "reserve": c["reserve"]})
            self.save_credit(credits)
            d["pending"] = {"gen": gen, "children": [c["id"] for c in children]}
            self.st.save()
            self.log(f"round {gen}: {len(children)} children")
        passed = [c for c in children if self.quick_pass(c)]
        surv, hist, n1 = self.screening(passed)
        for gid in surv:
            self.ensure(gid, "full", self.sets["full"])
        # credit: a child wins if its first screening score beats its first parent's on the same items, or it reaches
        # full evaluation (as V3-027); failed builds of the quick test are losses
        credits = self.credit()
        if not (pend and pend.get("credited")):
            b = primary()
            for c in children:
                won = c["id"] in surv
                if c in passed:
                    s = d["evals"][c["id"]].get(f"screen{n1}", {}).get(b, {}).get("scores", {})
                    ps = (d["evals"].get(c["parents"][0], {}).get("full") or {}).get(b, {}).get("scores", {})
                    common = set(s) & set(ps)
                    won = won or (bool(common) and sum(s[i] for i in common) > sum(ps[i] for i in common))
                kind = c.get("kind") or next(k for k, sh in cfg()["ops"].items() if c["operator"] in sh)
                if c["operator"] in credits[kind].base:
                    credits[kind].record(c["operator"], won)
            self.save_credit(credits)
            d.setdefault("pending", {"gen": gen})["credited"] = True
            self.st.save()
        self.refresh_front()
        for gid in surv:
            self.offer(d["genomes"][gid])
            if gid not in d["islands"][0]:
                d["islands"][0].append(gid)
        before = d["champion"]
        new = self.judge(surv, gen)
        d["streak"] = 0 if new else d["streak"] + 1
        d["generation"] = gen
        summary = {"gen": gen, "children": len(children), "quick_passed": len(passed), "full": surv,
                   "kinds": {k: sum(1 for c in children if c.get("kind") == k) for k in cfg()["ops"]},
                   "reserve": sum(1 for c in children if c.get("reserve")),
                   "improved": [new] if new else [], "champion_before": before, "champion": d["champion"],
                   "front": d["front"], "streak": d["streak"], "halving": hist,
                   "operator_shares": d["credit"]["shares"]}
        d.setdefault("generations", []).append(summary)
        d.pop("pending", None)
        self.st.event(f"round {gen} complete: {len(children)} children, {len(passed)} passed quick build, "
                      f"{len(surv)} fully evaluated, " + (f"NEW CHAMPION {new} (was {before})" if new else
                                                         f"champion unchanged ({before}), {d['streak']} of "
                                                         f"{EV['stop_no_improve_generations']} rounds without one"))
        self.st.save()
        return summary

    def should_stop(self):
        d = self.st.data
        return d["streak"] >= EV["stop_no_improve_generations"] or d["generation"] >= EV["max_generations"]
