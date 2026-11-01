"""OpenRouter access for v3 (Qwen builder, AI grader, frontier audit): the v2 client with v3's own ledger, cap and
usage baseline, so v3 never writes into the frozen v2 archive and its spend is counted from v3's start.

The API key is read by the v2 client from the environment or ~/.rpp-core/secrets.env; it is never logged.
"""
import json
import sys
import time

from v3.config import CFG, RUNS, STATE_DIR, V2

if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import harness.llm as _L  # noqa: E402

BASELINE_FILE = STATE_DIR / "openrouter_baseline.json"
_L.LEDGER = RUNS / "ledger.jsonl"
# V3-008: the v2 provider order put Parasail first for Qwen3 235B; it returned empty first turns on long prompts
_qcfg = dict(_L.MODEL_CFG.get(CFG["models"]["builder_openrouter"], {}))
if CFG["models"].get("builder_openrouter_providers"):
    _qcfg["providers"] = list(CFG["models"]["builder_openrouter_providers"])
    _L.MODEL_CFG[CFG["models"]["builder_openrouter"]] = _qcfg
# V3-035: each run 3 transfer model's fixed reasoning setting (unbounded thinking used the whole output budget;
# null = no reasoning field sent, as Gemma and Mistral need)
for _m, _r in ((CFG.get("final") or {}).get("transfer_settings") or {}).items():
    _c = {k: v for k, v in _L.MODEL_CFG.get(_m, {}).items() if k != "reasoning"}
    if _r is not None:
        _c["reasoning"] = dict(_r)
    _L.MODEL_CFG[_m] = _c
_L.BUDGET_USD = float(CFG["budget"]["openrouter_cash_usd"]) + float(CFG["budget"]["reserve_usd"]) + \
    float(CFG["budget"]["final_reserve_usd"])


def _baseline():
    """OpenRouter key usage when v3 first used the key (recorded once; v3 spend = usage - baseline)."""
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))["usage"]
    import requests
    d = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {_L._key()}"},
                     timeout=30).json()["data"]
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(json.dumps({"usage": d["usage"], "t": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                             encoding="utf-8")
    return d["usage"]


_configured = False


def configure():
    global _configured
    if not _configured:
        _L.USAGE_BASELINE = _baseline()
        _configured = True


def chat(*a, **kw):
    configure()
    return _L.chat(*a, **kw)


def spent():
    configure()
    return _L.true_spent()


BudgetExceeded = _L.BudgetExceeded
