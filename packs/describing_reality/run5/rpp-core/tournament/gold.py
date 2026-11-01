"""Gold representations: tournament/gold/<arm>/<item>.json.

load_gold() reads one, strips answer-bearing fields (claims / answers / conclusions, via the arm's
strip_answers) and loads it into a read-only arm state. validate() reports the arm's validity().

python -m tournament.gold [--arms A1,A5] [--items S01-1,...]     # validity report, no API calls
"""
import argparse
import json
import pathlib

from tournament.arms import ArmUnavailable, get_arm

HERE = pathlib.Path(__file__).resolve().parent
GOLD_DIR = HERE / "gold"


def gold_path(arm_name, item, gold_dir=GOLD_DIR):
    return pathlib.Path(gold_dir) / arm_name.lower() / f"{item}.json"  # gold folders are lowercase (a1, a3, ...)


def has_gold(arm_name, item, gold_dir=GOLD_DIR):
    return gold_path(arm_name, item, gold_dir).exists()


def read_gold(arm_name, item, gold_dir=GOLD_DIR):
    return json.loads(gold_path(arm_name, item, gold_dir).read_text(encoding="utf-8"))


def stripped(arm_name, rep):
    return get_arm(arm_name).strip_answers(rep)


def load_gold(arm_name, item, gold_dir=GOLD_DIR, given_gold=True):
    """Returns (arm, state, stripped_rep)."""
    arm = get_arm(arm_name)
    rep = arm.strip_answers(read_gold(arm_name, item, gold_dir))
    return arm, arm.load(rep, given_gold=given_gold), rep


def validate(arm_name, item, gold_dir=GOLD_DIR):
    try:
        arm, st, rep = load_gold(arm_name, item, gold_dir)
    except ArmUnavailable as e:
        return {"arm": arm_name, "item": item, "error": str(e)}
    except (OSError, ValueError) as e:
        return {"arm": arm_name, "item": item, "error": f"cannot read gold: {e}"}
    v = arm.validity(st)
    return {"arm": arm_name, "item": item, **v}


def list_gold(gold_dir=GOLD_DIR):
    out = []
    for d in sorted(pathlib.Path(gold_dir).glob("*")):
        if d.is_dir():
            out += [(d.name, f.stem) for f in sorted(d.glob("*.json"))
                    if ".audit" not in f.name and not f.stem.endswith(".notes")]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--items", default="")
    ap.add_argument("--gold-dir", default=str(GOLD_DIR))
    a = ap.parse_args(argv)
    arms = set(filter(None, a.arms.split(",")))
    items = set(filter(None, a.items.split(",")))
    rows = []
    for arm, item in list_gold(a.gold_dir):
        if (arms and arm not in arms) or (items and item not in items):
            continue
        r = validate(arm, item, a.gold_dir)
        rows.append(r)
        print(f"{arm:4s} {item:8s} usable={r.get('usable')} executable={r.get('executable')} "
              f"holes={len(r.get('holes', []))} errors={len(r.get('errors', []))} {r.get('error', '')}")
    return rows


if __name__ == "__main__":
    main()
