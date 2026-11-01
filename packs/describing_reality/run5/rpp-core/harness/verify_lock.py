"""Abort if any held-out file changed since data/heldout/LOCK.sha256 was written.

`verify()` checks the held-out lock only (unchanged behaviour). `verify_generalisation()`
checks data/generalisation/LOCK.sha256 the same way; run with `--generalisation` to check it
too (`--generalisation-only` to check only it).
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _verify(lock):
    bad = []
    for line in lock.read_text().splitlines():
        h, path = line.split("  ", 1)
        p = ROOT / path
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(path)
    return bad


def verify():
    return _verify(ROOT / "data" / "heldout" / "LOCK.sha256")


def verify_generalisation():
    return _verify(ROOT / "data" / "generalisation" / "LOCK.sha256")


if __name__ == "__main__":
    code = 0
    if "--generalisation-only" not in sys.argv:
        bad = verify()
        print("held-out lock OK" if not bad else f"LOCK VIOLATED: {bad}")
        code |= 1 if bad else 0
    if "--generalisation" in sys.argv or "--generalisation-only" in sys.argv:
        bad = verify_generalisation()
        print("generalisation lock OK" if not bad else f"GENERALISATION LOCK VIOLATED: {bad}")
        code |= 1 if bad else 0
    sys.exit(code)
