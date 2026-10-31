"""Steps 3 to 7: the same stages for every problem, run by fixed rules in code.

  compare         every founder sits one exam (the compare set, plus any compare_also sets); the winner and the
                  runner-up are the highest means on the compare set (ties: the first compare_also set, then name)
  takeapart       the winner is split into its parts; the designer writes one new version of each part; every
                  combination is run (2^m, a full factorial); the gated keep rule names the current best
  round n         (build out, up to len(sets.rounds) times) the designer writes per_slot versions of every part from
                  the brief; each is screened alone against the current best on fresh items (a fixed bar); the
                  slots that pass are confirmed in every combination on more fresh items under the same keep rule.
                  The rounds stop when no slot passes the screen, no mix beats the current best, or the gain is
                  under the convergence bar
  final           the sealed set is opened once; the final design meets the control and its closest rivals in a
                  fixed order (gatekeeping); each lead counts only if its whole 95% range is above zero
  report          the trust report

Every step's result goes to <work>/analysis.json and every decision to the events in <work>/state.json. New versions
are frozen (hashed) before anything runs on them and, with approval "human", wait for a person: <work>/stages/<step>/
APPROVED (or REJECT, at most twice, which archives them and writes new ones).
"""
import hashlib
import json
import pathlib
import time

from engine import brief as B
from engine.pack import Paused
from engine.plan import make_pack, verify
from engine.runner import Halted, Store, run_jobs
from engine.stats import Stats, design_labels, dumps


class Engine:
    def __init__(self, plan, pack=None, check_lock=True):
        self.plan = plan
        if check_lock:
            ok, msg = verify(plan)
            if not ok:
                raise SystemExit(msg)
        self.pack = pack or make_pack(plan)
        self.work = pathlib.Path(plan["_work"])
        self.work.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.work)
        s = plan["stats"]
        self.stats = Stats(self.pack.cluster_of, self.pack.level_of, boot=s["boot"], seed=s["seed"],
                           alpha=s["alpha"], retired=self.pack.retired_items())
        self.sets = self.pack.item_sets()
        self.control = plan.get("control") or self.pack.control
        self.pause = self.work / "PAUSE"
        self.st = self.load()

    # ------------------------------------------------------------------ state, events, files
    @property
    def state_file(self):
        return self.work / "state.json"

    def load(self):
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"step": "compare", "events": [], "history": [], "progress": {}}

    def save(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.st, indent=1, default=str), encoding="utf-8")
        for attempt in range(20):
            try:
                tmp.replace(self.state_file)
                return
            except PermissionError:                       # Windows: a viewer has the file open for a moment
                time.sleep(0.5)
        tmp.replace(self.state_file)

    def event(self, msg):
        self.st.setdefault("events", []).append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "msg": msg})
        self.save()
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    def analysis(self):
        try:
            return json.loads((self.work / "analysis.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write_analysis(self, key, res):
        an = self.analysis()
        an[key] = res
        (self.work / "analysis.json").write_text(dumps(an), encoding="utf-8")

    def keep(self, cand):
        """Every candidate the run builds is saved, so the result can be checked and repeated."""
        d = self.work / "candidates"
        d.mkdir(exist_ok=True)
        p = d / f"{cand['id']}.json"
        if not p.exists():
            p.write_text(json.dumps(cand, indent=1, default=str), encoding="utf-8")
        return cand

    def candidate(self, cid):
        return json.loads((self.work / "candidates" / f"{cid}.json").read_text(encoding="utf-8"))

    def items(self, name):
        if name not in self.sets:
            raise KeyError(f"the pack has no item set {name!r}")
        return list(self.sets[name])

    def founders(self):
        allf = self.pack.founders()
        names = self.plan.get("founders") or list(allf)
        return {n: self.keep(allf[n]) for n in names}

    def run(self, designs, items, stage, folder):
        r = self.plan["run"]
        sealed = set(self.sets.get(self.plan["sets"].get("sealed") or "", []))
        if stage != "final" and sealed & set(items):
            raise Halted(f"{stage} would run sealed items: they are opened only once, for the final check")
        self.st["progress"] = {}
        try:
            run_jobs(self.pack, self.store, folder, designs, items, stage, self.st["progress"], r["workers"],
                     r["retries"], tuple(r["backoff"]), self.pause, self.save)
        finally:
            self.save()

    def stage_dir(self, name):
        d = self.work / "stages" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------ the loop
    def loop(self):
        """Run steps until done or waiting (for a person, a pause or a halt). Returns the step it stopped at."""
        steps = {"compare": self.compare, "takeapart_write": self.takeapart_write,
                 "takeapart_run": self.takeapart_run, "round_write": self.round_write,
                 "round_screen": self.round_screen, "round_prepare": self.round_prepare,
                 "round_confirm": self.round_confirm, "final": self.final, "report": self.report}
        while self.st["step"] != "done":
            if self.pause.exists():
                self.event(f"paused before {self.st['step']} ({self.pause} present)")
                return self.st["step"]
            step = self.st["step"]
            try:
                done = steps[step]()
            except Paused as e:
                self.st["paused"] = str(e)
                self.event(f"paused in {step}: {e}")
                return step
            except Halted as e:
                self.st["halted"] = str(e)
                self.event(f"halted in {step}: {e}")
                return step
            if done is False:
                return step
        return "done"

    # ------------------------------------------------------------------ step 3: compare
    def compare(self):
        F = self.founders()
        main = self.items(self.plan["sets"]["compare"])
        also = {n: self.items(n) for n in self.plan["sets"].get("compare_also") or []}
        allitems = list(dict.fromkeys(main + [i for v in also.values() for i in v]))
        self.event(f"compare: {len(F)} founders x {len(allitems)} items")
        self.run(F, allitems, "compare", "main")
        sc = self.store.scores("main", F, allitems)
        base = [b for b in self.plan.get("baselines") or [] if b in F]
        choose = [f for f in F if f not in base]
        res = {"main": self.stats.compare(sc, choose, main, self.control)}
        res["also"] = {n: self.stats.compare(sc, choose, its, self.control) for n, its in also.items()}
        first = next(iter(res["also"].values()), {}).get("designs") or {}
        order = sorted(res["main"]["designs"], key=lambda f: (-res["main"]["designs"][f]["mean_exact"],
                                                              -(first.get(f) or {}).get("mean_exact", 0.0), f))
        res["main"]["ranking"] = order
        res["winner"], res["runner_up"] = order[0], (order[1] if len(order) > 1 else None)
        res["tie_at_top"] = bool(len(order) > 1 and res["main"]["designs"][order[1]]["could_be_best"])
        res["top"] = {n: sorted(r["designs"], key=lambda f: (-r["designs"][f]["mean_exact"], f))[0]
                      for n, r in res["also"].items() if r.get("designs")}
        res["by_level"] = self.stats.by_level(sc, choose, main)
        if base and self.control:
            res["baselines"] = {b: self.stats.compare(sc, [b, self.control], main, self.control) for b in base}
        if also:
            res["rank_agreement"] = {n: self.stats.rank_agreement(sc, choose, main, its) for n, its in also.items()}
        res["pack"] = self.pack.report_extras("compare", sc, main)
        self.write_analysis("compare", res)
        w = F[res["winner"]]
        self.st.update(winner=res["winner"], winner_id=w["id"], runner_up=res["runner_up"], top=res["top"],
                       best={"label": res["winner"], "id": w["id"], "folder": "main",
                             "items": self.plan["sets"]["compare"], "from": "compare"})
        self.st["step"] = "takeapart_write" if self.plan["sets"].get("takeapart") else \
            ("round_write" if self.plan["sets"].get("rounds") else "final")
        self.st["round"] = 1
        self.event(f"compare done: winner {res['winner']}, runner-up {res['runner_up']}"
                   + (" (tied at the top)" if res["tie_at_top"] else "")
                   + "".join(f"; top on {n}: {t}" for n, t in res["top"].items()))

    # ------------------------------------------------------------------ approvals and frozen versions
    def frozen(self, name, make):
        """The stage's frozen versions: written once by make() -> (alts, extras), hashed, then gated."""
        d = self.stage_dir(name)
        f = d / "candidates.json"
        rej = d / "REJECT"
        if f.exists() and rej.exists() and not (d / "APPROVED").exists():
            n = len(list(d.glob("candidates_rejected_*.json")))
            if n >= 2:
                self.st["waiting_for"] = f"{name}: REJECT not accepted (two rejections already)"
                self.save()
                return None
            f.replace(d / f"candidates_rejected_{n + 1}.json")
            rej.replace(d / f"reject_reason_{n + 1}.txt")
            self.event(f"{name}: versions sent back by a person; new ones are written")
        if not f.exists():
            alts, extras = make()
            doc = {"step": name, "alts": alts, "extras": extras}
            doc["hash"] = _hash({"alts": alts, "extras": extras})
            f.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
            self.event(f"{name}: {sum(len(v) for v in alts.values())} new versions written and frozen "
                       f"(hash {doc['hash']})")
        doc = json.loads(f.read_text(encoding="utf-8"))
        if _hash({"alts": doc["alts"], "extras": doc["extras"]}) != doc["hash"]:
            raise Halted(f"{name}: the versions changed after they were frozen")
        if not self.approved(d, f"read {f} and approve the new versions (APPROVED) or send them back (REJECT)"):
            return None
        return doc

    def approved(self, d, what, file="APPROVED"):
        a = d / file
        if self.plan.get("approval") == "auto" and not a.exists():
            a.write_text("approved automatically (plan approval: auto)\n", encoding="utf-8")
        if not a.exists():
            self.st["waiting_for"] = f"a person: {what}"
            self.save()
            return False
        self.st.pop("waiting_for", None)
        return True

    def brief(self, cand, hist):
        b = self.st["best"]
        return B.build(self.stats, self.store, self.pack, cand, b["folder"], self.items(b["items"]), hist)

    # ------------------------------------------------------------------ step 4: take apart
    def ta_slots(self):
        return self.plan["takeapart"].get("slots") or self.pack.slots("takeapart")

    def takeapart_write(self):
        w = self.candidate(self.st["winner_id"])
        parts = self.pack.parts(w, "takeapart")
        slots = self.ta_slots()

        def make():
            br = self.brief(w, [])
            (self.stage_dir("takeapart") / "brief.json").write_text(json.dumps(br, indent=1), encoding="utf-8")
            alts, extras = self.pack.write("takeapart", w, parts, br, slots, 1)
            return {s: alts[s][:1] for s in slots}, extras
        doc = self.frozen("takeapart", make)
        if doc is None:
            return False
        self.st["step"] = "takeapart_run"
        self.event("take apart: versions approved; the factorial starts")

    def ta_designs(self):
        w = self.candidate(self.st["winner_id"])
        doc = json.loads((self.stage_dir("takeapart") / "candidates.json").read_text(encoding="utf-8"))
        parts, slots = self.pack.parts(w, "takeapart"), self.ta_slots()
        out = {}
        for n, lab in enumerate(design_labels(len(slots))):
            ch = {s: doc["alts"][s][0] for j, s in enumerate(slots) if (n >> j) & 1}
            out[lab] = w if n == 0 else self.keep(self.pack.compose(w, parts, ch, "takeapart", doc["extras"]))
        if self.pack.compose(w, parts, {}, "takeapart", doc["extras"])["id"] != w["id"]:
            raise Halted("compose() with no changes does not give back the winner's id")
        return out, slots

    def takeapart_run(self):
        designs, slots = self.ta_designs()
        items = self.items(self.plan["sets"]["takeapart"])
        self.event(f"take apart: {len(designs)} designs x {len(items)} items")
        self.run(designs, items, "takeapart", "takeapart")
        sc = self.store.scores("takeapart", designs, items)
        res = self.stats.retention(sc, list(designs), items, slots, family=self.plan["takeapart"].get("family"))
        res["pack"] = self.pack.report_extras("takeapart", sc, items)
        self.write_analysis("takeapart", res)
        doc = json.loads((self.stage_dir("takeapart") / "candidates.json").read_text(encoding="utf-8"))
        for s in slots:
            e = res["effects"][s]
            self.st["history"].append({"where": "take apart", "slot": s, "effect": e["effect"], "range": e["range"],
                                       "value": doc["alts"][s][0].get("value"),
                                       "verdict": "kept" if s in res["kept_mains"] else "not kept"})
        fd = designs[res["final_design"]]
        self.st["best"] = {"label": f"take apart {res['final_design']}", "id": fd["id"], "folder": "takeapart",
                           "items": self.plan["sets"]["takeapart"], "from": "takeapart"}
        self.st["step"] = "round_write" if self.plan["sets"].get("rounds") else "final"
        self.event(f"take apart done: {res['final_design']} "
                   f"({'replaces' if res['replaces'] else 'keeps'} the winner; kept parts {res['kept_mains'] or 'none'})")

    # ------------------------------------------------------------------ step 5: build out
    def bo_slots(self):
        return self.plan["buildout"].get("slots") or self.pack.slots("buildout")

    def rname(self):
        return f"round{self.st['round']}"

    def round_set(self, part):
        return self.plan["sets"]["rounds"][self.st["round"] - 1][part]

    def round_write(self):
        r, best = self.st["round"], self.candidate(self.st["best"]["id"])
        parts, slots = self.pack.parts(best, "buildout"), self.bo_slots()
        per = self.plan["buildout"]["per_slot"]

        def make():
            br = dict(self.brief(best, self.st["history"]), round=r)
            (self.stage_dir(self.rname()) / "brief.json").write_text(json.dumps(br, indent=1), encoding="utf-8")
            alts, extras = self.pack.write("buildout", best, parts, br, slots, per)
            out = {}
            for s in slots:
                out[s] = [dict(a, label=f"{s}.{k + 1}", archetype=a.get("archetype", k + 1))
                          for k, a in enumerate(alts.get(s) or [])]
            return out, dict(extras or {}, best_id=best["id"])
        doc = self.frozen(self.rname(), make)
        if doc is None:
            return False
        if doc["extras"].get("best_id") != best["id"]:
            raise Halted(f"{self.rname()}: the versions were written for another current best")
        self.st["step"] = "round_screen"
        self.event(f"round {r}: versions approved; the screen starts")

    def round_doc(self):
        return json.loads((self.stage_dir(self.rname()) / "candidates.json").read_text(encoding="utf-8"))

    def screen_designs(self):
        best = self.candidate(self.st["best"]["id"])
        parts, doc = self.pack.parts(best, "buildout"), self.round_doc()
        out, meta = {"CUR": best}, {}
        for s in self.bo_slots():
            for a in doc["alts"][s]:
                c = self.keep(self.pack.compose(best, parts, {s: a}, "buildout", doc["extras"]))
                if c["id"] in {x["id"] for x in out.values()}:
                    raise Halted(f"{a['label']} builds the same candidate as another design")
                out[a["label"]], meta[a["label"]] = c, {"slot": s, "archetype": a["archetype"]}
        return out, meta

    def round_screen(self):
        designs, meta = self.screen_designs()
        items = self.items(self.round_set("screen"))
        bo = self.plan["buildout"]
        self.event(f"round {self.st['round']} screen: {len(designs)} designs x {len(items)} items")
        folder = f"{self.rname()}_screen"
        self.run(designs, items, f"{self.rname()}_screen", folder)
        sc = self.store.scores(folder, designs, items)
        res = self.stats.screen(sc, "CUR", meta, items, bo["screen_gate"], bo["max_advance"])
        self.write_analysis(f"{self.rname()}_screen", res)
        doc = self.round_doc()
        for s in self.bo_slots():
            mine = [a["label"] for a in doc["alts"][s]]
            worst = min(mine, key=lambda lab: res["candidates"][lab].get("estimate") or 0)
            for a in doc["alts"][s]:
                c = res["candidates"][a["label"]]
                adv = res["advancing"].get(str(s)) == a["label"]
                self.st["history"].append({"where": f"round {self.st['round']} screen", "slot": s,
                                           "effect": c.get("estimate"), "range": c.get("range"),
                                           "value": a.get("value") if a["label"] in (worst, res["slots"][str(s)]["best"]) else None,
                                           "verdict": "advanced to the confirmation" if adv else "screened out"})
        self.st[f"{self.rname()}_advancing"] = res["advancing"]
        if not res["advancing"]:
            self.st["step"] = "final"
            self.event(f"round {self.st['round']} screen: no part's best version leads by +{bo['screen_gate']}; "
                       f"the rounds stop and the final check follows")
            return
        self.st["step"] = "round_prepare"
        self.event(f"round {self.st['round']} screen done: advancing "
                   + ", ".join(f"{s} {lab}" for s, lab in res["advancing"].items()))

    def advancing(self):
        doc = self.round_doc()
        adv = self.st[f"{self.rname()}_advancing"]
        by = {a["label"]: a for s in doc["alts"] for a in doc["alts"][s]}
        return {s: by[adv[s]] for s in self.bo_slots() if s in adv}

    def round_prepare(self):
        d = self.stage_dir(self.rname())
        f = d / "prepared.json"
        if not f.exists():
            best = self.candidate(self.st["best"]["id"])
            doc = self.round_doc()
            extras, needs = self.pack.prepare("buildout", best, self.pack.parts(best, "buildout"), self.advancing(),
                                              doc["extras"])
            f.write_text(json.dumps({"extras": extras, "needs_approval": bool(needs),
                                     "hash": _hash(extras)}, indent=1, default=str), encoding="utf-8")
            if needs:
                self.event(f"round {self.st['round']}: additions for the confirmation written (prepared.json)")
        p = json.loads(f.read_text(encoding="utf-8"))
        if _hash(p["extras"]) != p["hash"]:
            raise Halted(f"{self.rname()}: prepared.json changed after it was written")
        if p["needs_approval"] and not self.approved(d, f"read {f} and approve it (PREPARED_APPROVED)",
                                                     "PREPARED_APPROVED"):
            return False
        self.st["step"] = "round_confirm"
        self.event(f"round {self.st['round']}: the confirmation starts")

    def confirm_designs(self):
        best = self.candidate(self.st["best"]["id"])
        parts = self.pack.parts(best, "buildout")
        adv = self.advancing()
        slots = list(adv)
        extras = json.loads((self.stage_dir(self.rname()) / "prepared.json").read_text(encoding="utf-8"))["extras"]
        out, levels = {}, {}
        for n, lab in enumerate(design_labels(len(slots))):
            ch = {s: adv[s] for j, s in enumerate(slots) if (n >> j) & 1}
            levels[lab] = {s: (adv[s]["label"] if s in ch else "current") for s in slots}
            out[lab] = best if n == 0 else self.keep(self.pack.compose(best, parts, ch, "buildout", extras))
        return out, levels, slots

    def round_confirm(self):
        designs, levels, slots = self.confirm_designs()
        items = self.items(self.round_set("confirm"))
        folder = f"{self.rname()}_confirm"
        self.event(f"round {self.st['round']} confirmation: {len(designs)} designs x {len(items)} items")
        self.run(designs, items, folder, folder)
        sc = self.store.scores(folder, designs, items)
        res = self.stats.retention(sc, list(designs), items, slots)
        res["levels"] = levels
        self.write_analysis(f"{self.rname()}_confirm", res)
        self.st["history"].append({"where": f"round {self.st['round']} confirmation", "slot": "+".join(slots),
                                   "effect": (res.get("direct") or {}).get("estimate"),
                                   "range": (res.get("direct") or {}).get("range"),
                                   "verdict": (f"design {res['final_design']} became the current best"
                                               if res["replaces"] else "no mix beat the current best; it stays")})
        if res["replaces"]:
            fd = designs[res["final_design"]]
            self.st["best"] = {"label": f"round {self.st['round']} {res['final_design']}", "id": fd["id"],
                               "folder": folder, "items": self.round_set("confirm"), "from": self.rname()}
        else:
            self.st["best"].update(folder=folder, items=self.round_set("confirm"))   # its fresh D01 records
        last = self.st["round"] >= len(self.plan["sets"]["rounds"])
        conv = self.plan["buildout"]["converge"]
        if not res["replaces"] or res["gain"] < conv or last:
            why = ("no mix beat the current best" if not res["replaces"] else
                   f"the gain {res['gain']} is under +{conv}" if res["gain"] < conv else "the last round")
            self.st["step"] = "final"
            self.event(f"round {self.st['round']} confirmation done: {res['final_design']}"
                       f"{' replaces the current best' if res['replaces'] else ''}; the rounds stop ({why})")
            return
        self.st["round"] += 1
        self.st["step"] = "round_write"
        self.event(f"confirmation done: {res['final_design']} is the new current best (gain {res['gain']}); "
                   f"round {self.st['round']} next")

    # ------------------------------------------------------------------ step 6: the sealed final check
    def final(self):
        sealed_name = self.plan["sets"].get("sealed")
        if not sealed_name:
            self.st["step"] = "report"
            self.event("no sealed set in the plan: no final check")
            return
        F = self.founders()
        fd = self.candidate(self.st["best"]["id"])
        st = self.st
        named = {"control": self.control, "runner_up": st.get("runner_up"), "winner": st.get("winner")}
        for n, t in (st.get("top") or {}).items():
            named[f"top:{n}"] = t
        order = list(self.plan["final"]["rivals"])
        if fd["id"] != st["winner_id"] and "winner" not in order:
            order.append("winner")                        # did the stages improve on the winner?
        rivals, designs = {}, {"FINAL": fd}
        for k, kind in enumerate(order, 1):
            lab = named.get(kind)
            if not lab or F[lab]["id"] == fd["id"] or lab in rivals.values():
                continue
            rivals[f"{len(rivals) + 1}_vs_{kind.replace(':', '_')}"] = lab
            designs[lab] = F[lab]
        if self.control in F and self.control not in designs and F[self.control]["id"] != fd["id"]:
            designs[self.control] = F[self.control]
        sealed = self.items(sealed_name)
        if "sealed" not in st.setdefault("opened", {}):
            st["opened"]["sealed"] = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "set": sealed_name}
            self.pack.open_sealed(sealed_name, "final check")
            self.event(f"sealed set {sealed_name!r} opened ({len(sealed)} items); it is never closed again")
        self.event(f"final check: {len(designs)} designs x {len(sealed)} sealed items")
        self.run(designs, sealed, "final", "main")
        also = {n: self.items(n) for n in self.plan["final"].get("also_on") or []}
        for n, its in also.items():
            self.run({"FINAL": fd}, its, "final", "main")
        allitems = list(dict.fromkeys(sealed + [i for v in also.values() for i in v]))
        sc = self.store.scores("main", designs | {"winner": F[st["winner"]]}, allitems)
        res = {"final_design": st["best"]["label"], "final_id": fd["id"], "same_as_winner": fd["id"] == st["winner_id"],
               "rivals": rivals, "sequence": self.stats.final_sequence(sc, "FINAL", rivals, sealed)}
        labels = list(designs)
        res["sealed_means"] = self.stats.compare(sc, labels, sealed, self.control).get("designs")
        res["by_level"] = self.stats.by_level(sc, labels, sealed)
        res["also"] = {}
        for n, its in also.items():
            res["also"][n] = {"final_mean": self.stats.compare(sc, ["FINAL"], its).get("designs", {}).get("FINAL")}
            if fd["id"] != st["winner_id"]:
                res["also"][n]["vs_winner"] = self.stats.paired_cluster(sc, "FINAL", "winner", its)
        res["pack"] = self.pack.report_extras("final", sc, sealed)
        self.write_analysis("final", res)
        self.st["step"] = "report"
        self.event("final check done: " + "; ".join(
            f"{k} ({v['rival']}) lead {v.get('estimate')} {v.get('range')} {'counted' if v['counted'] else 'not counted'}"
            for k, v in res["sequence"].items()))

    # ------------------------------------------------------------------ step 7: the trust report
    def report(self):
        from engine import report
        report.write(self)
        self.st["step"] = "done"
        self.event(f"trust report written ({self.work / 'report.html'}); the run is complete")


def _hash(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]
