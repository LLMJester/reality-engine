"""Check that run 4's item sets (v3.directed4.run4_sets, confirmation and sealed sets) share no base system with any
item used before: every composite item K<k>x<base> draws its subsystems from [base, base + 500), so two bases closer
than 500 can share subsystems. Scans every run record folder (runs/, archive/*/runs) and reports overlaps.
Read-only. Usage: python scripts/check_item_ranges.py [--run5]
"""
import bisect
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STRIDE = 500


def used_bases(singles=None):
    """Bases of composite items (windows of STRIDE seeds) used anywhere in the run folders; with a set `singles`,
    single-seed items W<seed> go there instead (V3-058: a W item uses only its own seed), else they count as bases."""
    out = set()
    for top in (ROOT / "runs", *sorted((ROOT / "archive").glob("*/runs"))):
        for dp, dn, _ in os.walk(top):
            for d in dn:
                m = re.match(r"^K\d+x(\d+)$", d)
                if m:
                    out.add(int(m.group(1)))
                m = re.match(r"^W(\d+)$", d)
                if m:
                    (singles if singles is not None else out).add(int(m.group(1)))
    return sorted(out)


def main():
    if "--run5" in sys.argv:          # run 5 sets (V3-042): main, sealed, pilot, mini, mini_sealed
        from v3 import run5_sets as R5
        new = {name: sorted({int(i.split("x")[1]) for i in ids}) for name, ids in R5.sets().items()}
    else:
        from v3 import directed4 as D4
        from v3.directed import confirm_set
        sets = D4.run4_sets()
        new = {name: sorted({int(m.group(1)) for i in ids if (m := re.match(r"^K\d+x(\d+)$", i))})
               for name, ids in sets.items() if name not in ("quick", "screen", "full")}
        new["confirm0"] = sorted({int(i.split("x")[1]) for i in confirm_set([])})
    singles = set() if "--run5" in sys.argv else None
    used = used_bases(singles)
    singles = sorted(singles or ())
    bad = 0
    for name, bases in new.items():
        if "--run5" in sys.argv and name in ("pilot", "mini", "mini_sealed"):
            print(f"{name}: {len(bases)} bases (pilot items: their own pilot records are expected; checked only "
                  f"against the other run 5 sets below)")
            continue
        own = set(bases) if "--run5" in sys.argv else set()      # V3-058: a set's own runs are not an overlap
        others = [u for u in used if u not in own]
        hits = []
        for b in bases:
            j = bisect.bisect_left(others, b - STRIDE + 1)
            if j < len(others) and others[j] < b + STRIDE:
                hits.append((b, others[j]))
            j = bisect.bisect_left(singles, b)                   # a single seed inside the window [b, b + STRIDE)
            if j < len(singles) and singles[j] < b + STRIDE:
                hits.append((b, f"W{singles[j]}"))
        bad += len(hits)
        print(f"{name}: {len(bases)} bases {bases[0]}-{bases[-1]}, overlaps with used items: {len(hits)} {hits[:5]}")
    names = list(new)
    for x in range(len(names)):          # the run 4 sets must not overlap each other either
        for y in range(x + 1, len(names)):
            ov = [b for b in new[names[x]] for c in new[names[y]] if abs(b - c) < STRIDE]
            if ov:
                bad += len(ov)
                print(f"overlap {names[x]} / {names[y]}: {ov[:5]}")
    print("OK: no overlaps" if not bad else f"FAIL: {bad} overlaps")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
