"""Paths and configuration for the v3 Discovery Programme (spec: rpp-core/reports/NEXT_ROUND_SPEC.md)."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config" / "v3.json"
RUNS = ROOT / "runs"
STATE_DIR = ROOT / "state"


def load(path=CONFIG_FILE):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


CFG = load()
V2 = (ROOT / CFG["v2_root"]).resolve()   # frozen v2 record, read-only source of items, simulators and arms
