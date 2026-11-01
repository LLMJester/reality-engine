"""Pre-registered selection algorithms (spec §5). The AI never decides survival; these functions do.

Objectives are dicts of name -> value where larger is better (cost is passed negated). Missing objectives (None) are
treated as worst for dominance, so a design is never rewarded for an unmeasured score.
"""
import math
import random

NEG = -math.inf


def _vals(obj, keys):
    return [NEG if obj.get(k) is None else obj[k] for k in keys]


def dominates(a, b, keys):
    va, vb = _vals(a, keys), _vals(b, keys)
    return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))


def nondominated_sort(objs, keys):
    """Fast non-dominated sort (Deb et al. 2002). objs: {id: objectives}. Returns a list of fronts (lists of ids)."""
    ids = list(objs)
    S = {i: [] for i in ids}
    n = {i: 0 for i in ids}
    fronts = [[]]
    for p in ids:
        for q in ids:
            if p == q:
                continue
            if dominates(objs[p], objs[q], keys):
                S[p].append(q)
            elif dominates(objs[q], objs[p], keys):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(sorted(nxt))
    return [sorted(f) for f in fronts if f]


def crowding(front, objs, keys):
    """Crowding distance within one front (boundary points get infinity)."""
    d = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    for k in keys:
        srt = sorted(front, key=lambda i: (_vals(objs[i], [k])[0], i))
        lo, hi = _vals(objs[srt[0]], [k])[0], _vals(objs[srt[-1]], [k])[0]
        d[srt[0]] = d[srt[-1]] = math.inf
        span = hi - lo
        if not math.isfinite(span) or span == 0:
            continue
        for j in range(1, len(srt) - 1):
            a, b = _vals(objs[srt[j - 1]], [k])[0], _vals(objs[srt[j + 1]], [k])[0]
            if math.isfinite(a) and math.isfinite(b):
                d[srt[j]] += (b - a) / span
    return d


def nsga2_rank(objs, keys):
    """Total order: by front, then by crowding distance (descending), then id. Returns [(id, front, crowding)]."""
    out = []
    for r, front in enumerate(nondominated_sort(objs, keys)):
        cd = crowding(front, objs, keys)
        for i in sorted(front, key=lambda i: (-cd[i], i)):
            out.append((i, r, cd[i]))
    return out


def nsga2_select(objs, keys, k):
    return [i for i, _, _ in nsga2_rank(objs, keys)[:k]]


class MapElites:
    """Archive of one elite per niche. A challenger replaces the occupant only if it dominates it (spec §5.3)."""

    def __init__(self, keys, cells=None):
        self.keys = keys
        self.cells = dict(cells or {})          # niche -> {"id", "obj"}

    def offer(self, niche, gid, obj):
        cur = self.cells.get(niche)
        if cur is None or dominates(obj, cur["obj"], self.keys):
            self.cells[niche] = {"id": gid, "obj": dict(obj)}
            return True
        return False

    def ids(self):
        return [c["id"] for c in self.cells.values()]

    def to_json(self):
        return {"|".join(n) if isinstance(n, tuple) else n: c for n, c in self.cells.items()}


def niche_of(genome):
    g = genome["genes"]
    return (g["core"]["kind"], g["build_protocol"]["kind"])


def tournament(pool, objs, keys, rng, size=2, explore=None, score=None):
    """Binary tournament on NSGA-II rank; `explore` = {id: bonus} mildly favours underexplored niches. Ties on rank
    and crowding go to the higher `score(id)` (V3-026), then to the id."""
    pool = list(dict.fromkeys(pool))    # each design is one candidate however often it is listed (V3-026)
    rank = {i: (r, -cd) for i, r, cd in nsga2_rank({i: objs[i] for i in pool}, keys)}
    cands = rng.sample(pool, min(size, len(pool)))
    return min(cands, key=lambda i: (rank[i][0] - (explore or {}).get(i, 0.0), rank[i][1],
                                     -(score(i) if score else 0.0), i))


def successive_halving(ids, score, stages, rng=None):
    """stages = [(n_items, keep_fraction_or_count), ...]; score(id, n_items) -> float (higher better).
    Returns (survivors, history). Deterministic given score; ties broken by id."""
    alive = list(ids)
    hist = []
    for n_items, keep in stages:
        s = {i: score(i, n_items) for i in alive}
        k = keep if isinstance(keep, int) else max(1, math.ceil(len(alive) * keep))
        alive = sorted(alive, key=lambda i: (-(s[i] if s[i] is not None else NEG), i))[:k]
        hist.append({"items": n_items, "scores": s, "kept": list(alive)})
    return alive, hist


class OperatorCredit:
    """Adaptive operator shares (spec §4.3): share ∝ smoothed success rate, with a floor."""

    def __init__(self, shares, floor=0.03, prior=(1, 2)):
        self.base = dict(shares)
        self.floor = floor
        self.wins = {k: prior[0] for k in shares}
        self.trials = {k: prior[1] for k in shares}

    def record(self, op, won):
        self.trials[op] += 1
        self.wins[op] += 1 if won else 0

    def shares(self):
        rate = {k: self.wins[k] / self.trials[k] for k in self.base}
        raw = {k: self.base[k] * rate[k] for k in self.base}
        fixed, free = {}, dict(raw)
        while True:   # operators under the floor get exactly the floor; the rest share what is left, proportionally
            room = 1.0 - self.floor * len(fixed)
            tot = sum(free.values()) or 1.0
            low = [k for k, v in free.items() if v / tot * room < self.floor]
            if not low:
                return dict({k: self.floor for k in fixed}, **{k: v / tot * room for k, v in free.items()})
            for k in low:
                fixed[k] = free.pop(k)

    def pick(self, rng):
        sh = self.shares()
        r, acc = rng.random(), 0.0
        for k in sorted(sh):
            acc += sh[k]
            if r <= acc:
                return k
        return sorted(sh)[-1]


def migrate(islands, objs, keys, n, rng):
    """Ring migration: each island sends its n best (NSGA-II) to the next island. Returns new island lists."""
    best = [nsga2_select({i: objs[i] for i in isl}, keys, n) if isl else [] for isl in islands]
    out = []
    for j, isl in enumerate(islands):
        incoming = [i for i in best[j - 1] if i not in isl]
        out.append(list(isl) + incoming)
    return out


def rng_for(*parts):
    """A reproducible RNG for a named decision (e.g. rng_for('gen', 3, 'parents'))."""
    return random.Random("|".join(map(str, parts)))
