"""Choose fresh item bases for a run of the describing-reality pack, once, before anything runs.

    python -m packs.describing_reality.allocate <out.json> --sets compare=8,r1_screen=4,r1_confirm=4 --sealed sealed=4
        [--also ../reality-engine/work]

A base is a free 500-seed window: no folder K<k>x<base> or W<seed> anywhere in rpp-v3 (records of runs 1 to 5,
archives, side experiments) or in the --also folders (earlier engine runs), no base of any run 5 set, no excluded
range (run 5's rules, v3.run5_sets.free_windows). Validation-split windows go to the open sets, train-split windows to
the sealed ones, in round-robin order over problem families. Only each base's family is read; no item is built.
"""
import argparse
import hashlib
import json
import pathlib
import time

from packs.describing_reality import pack  # noqa: F401  (puts rpp-v3 on sys.path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sets", required=True, help="name=bases,... (validation split)")
    ap.add_argument("--sealed", required=True, help="name=bases,... (train split)")
    ap.add_argument("--also", nargs="*", default=[], help="more folders whose K<k>x<base> folders count as used")
    a = ap.parse_args(argv)
    from v3 import items  # noqa: F401  (puts rpp-core's generator on sys.path)
    from v3 import run5_sets as R5
    from discovery.world import make_instance
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: item sets are chosen once and never rewritten")
    k_used, w_used = R5.used_bases()
    for d in a.also:
        k2, w2 = R5.used_bases(pathlib.Path(d))
        k_used |= k2
        w_used |= w2
    with R5._choosing():
        others = set()
        for name in list(R5.N_BASES) + R5.stage_set_names():
            others |= set(R5._bases(name))
    fam = lambda s: make_instance(s).family                                           # noqa: E731
    want = lambda spec: [(n, int(b)) for n, b in (x.split("=") for x in spec.split(","))]  # noqa: E731
    sets = {}
    for spec, lo, hi in ((a.sets, 1_000_000, 2_000_000), (a.sealed, 0, 1_000_000)):
        free = R5.round_robin(R5.free_windows(lo, hi, k_used, w_used, others), fam)
        pos = 0
        for name, nb in want(spec):
            if pos + nb > len(free):
                raise SystemExit(f"only {len(free)} free windows")
            sets[name] = sorted(free[pos:pos + nb])
            pos += nb
    body = json.dumps(sets, sort_keys=True)
    doc = {"made": time.strftime("%Y-%m-%dT%H:%M:%S"), "ks": list(R5.KS), "sealed": [n for n, _ in want(a.sealed)],
           "method": __doc__.split("\n\n")[2].replace("\n", " "), "sets": sets,
           "sha256": hashlib.sha256(body.encode()).hexdigest()}
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"wrote {out}: " + ", ".join(f"{n} {len(v)} bases" for n, v in sets.items()))


if __name__ == "__main__":
    main()
