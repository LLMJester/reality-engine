"""A toy problem pack: a complete pack that runs in about a second, with no model and no network.

A candidate is a set of three text parts (layout, rules, check). Its hidden quality is a fixed function of those texts,
and an item is a test case at one of two difficulty levels: a run answers ten probes, each right with a probability
set by the quality and the level (seeded, so every run is repeatable). The designer is a script, not an AI: it writes
versions named after an archetype, and one of them really helps. The engine does not know any of this; it finds out.
Use it to learn the engine, to test changes to it, and as a template for a real pack.
"""
import hashlib
import json
import random

from engine.pack import Pack

SLOTS = ("layout", "rules", "check")
# the hidden truth the engine has to find: what each kind of text adds to the quality
BONUS = {"plain": 0.0, "facts": 0.12, "table": 0.07, "strict": -0.03, "short": 0.05, "trace": 0.02}


def cid(parts):
    return "t" + hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]


def make(parts, name=""):
    return {"id": cid(parts), "name": name, "parts": dict(parts)}


def quality(parts):
    q = 0.45
    for s in SLOTS:
        for word, b in BONUS.items():
            if word in parts[s].split():
                q += b / (2 if s != "rules" else 1)
    return q


class ToyPack(Pack):
    name = "toy"
    description = "A simulated problem with a known answer, for learning and testing the engine."
    control = "PLAIN"

    def __init__(self, clusters=40, seed=7):
        self.n, self.seed = clusters, seed

    def founders(self):
        return {"PLAIN": make({"layout": "plain", "rules": "plain", "check": "none"}, "PLAIN"),
                "FACTS": make({"layout": "plain", "rules": "facts", "check": "none"}, "FACTS"),
                "TABLE": make({"layout": "table", "rules": "plain", "check": "none"}, "TABLE")}

    def item_sets(self):
        def block(start, n):
            return [f"T{c}-{lv}" for c in range(start, start + n) for lv in (1, 2)]
        n = self.n
        return {"compare": block(0, n), "takeapart": block(1000, n), "r1_screen": block(2000, n // 2),
                "r1_confirm": block(3000, n), "r2_screen": block(4000, n // 2), "r2_confirm": block(5000, n),
                "sealed": block(9000, n)}

    def cluster_of(self, item):
        return item.split("-")[0]

    def level_of(self, item):
        return int(item.split("-")[1])

    def run(self, candidate, item, ctx):
        q = quality(candidate["parts"]) - 0.08 * (self.level_of(item) - 1)
        rng = random.Random(f"{self.seed}|{candidate['id']}|{item}")
        cl = random.Random(f"{self.seed}|{self.cluster_of(item)}").uniform(-0.1, 0.1)   # some systems are harder
        right = sum(rng.random() < min(1, max(0, q + cl)) for _ in range(10))
        return {"score": right / 10, "missed": 10 - right}

    def failure_type(self, record):
        return "missed probes" if record and record.get("missed") else None

    def slots(self, kind):
        return list(SLOTS)

    def parts(self, candidate, kind):
        return dict(candidate["parts"])

    def write(self, kind, candidate, parts, brief, slots, per_slot):
        words = {"layout": ["table", "short", "plain grid", "trace"], "rules": ["facts", "strict", "short", "trace"],
                 "check": ["trace", "strict", "short", "none twice"]}
        out = {}
        for s in slots:
            opts = [w for w in words[s] if w != parts[s]]
            out[s] = [{"value": opts[k % len(opts)], "rationale": f"toy archetype {k + 1}"} for k in range(per_slot)]
        return out, {}

    def compose(self, base, parts, changes, kind, extras):
        p = dict(parts)
        for s, alt in changes.items():
            p[s] = alt["value"]
        return make(p, base.get("name", "") + "".join(f" {s}={a['value']}" for s, a in changes.items()))
