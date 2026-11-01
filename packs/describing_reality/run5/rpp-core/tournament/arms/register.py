"""Register an arm without editing tournament/arms/__init__.py (review fix 5; safe for parallel builders).

    python -m tournament.arms.register G1C2 g1c2 G1C2Arm   ->  tournament/arms/registry.d/G1C2.json
"""
import sys

from tournament.arms import get_arm, register_arm, registry


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    p = register_arm(*a)
    registry()
    try:
        get_arm(a[0])
    except Exception as e:  # registration written; the module may not exist yet
        print(f"registered {a[0]} -> {p} (not importable yet: {e})")
        return 0
    print(f"registered {a[0]} -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
