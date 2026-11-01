"""OpenRouter client with a hard budget cap and a full call ledger.

Every call is appended to runs/ledger.jsonl (model, tokens, cost, purpose).
The run aborts cleanly (BudgetExceeded) once the cap is reached.
"""
import json
import os
import pathlib
import threading
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "runs" / "ledger.jsonl"
SECRETS = pathlib.Path.home() / ".rpp-core" / "secrets.env"
BUDGET_USD = float(os.environ.get("RPP_BUDGET_USD", "68.92"))  # $18.92 spent before the discovery programme + $50 authorised for it (DECISIONS D-031)
URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "small": "qwen/qwen3-30b-a3b-instruct-2507",
    "medium": "qwen/qwen3-next-80b-a3b-instruct",
    "large": "qwen/qwen3-235b-a22b-2507",
}
GRADER_MODEL = "deepseek/deepseek-v4.1-flash"

# Per-model call settings (DECISIONS D-014, D-017): pinned provider order (fallbacks allowed,
# provider logged per call), sampling and reasoning settings, identical across conditions.
# Qwen3 instruct models are non-thinking; sampling follows Qwen's recommended settings
# (temperature 0.7, top_p 0.8) with a fixed seed, since greedy decoding loops on open models.
MODEL_CFG = {
    "qwen/qwen3-30b-a3b-instruct-2507": {"providers": ["Nebius", "DekaLLM"], "temperature": 0.7, "top_p": 0.8},
    "qwen/qwen3-next-80b-a3b-instruct": {"providers": ["Parasail", "DeepInfra"], "temperature": 0.7, "top_p": 0.8},
    "qwen/qwen3-235b-a22b-2507": {"providers": ["Parasail", "Nebius", "DeepInfra"], "temperature": 0.7, "top_p": 0.8},
    "deepseek/deepseek-v4.1-flash": {"providers": ["DeepInfra", "Decart"], "reasoning": {"effort": "low"}},
    # discovery programme reasoning model (DECISIONS D-032); these providers do not support seeds
    "deepseek/deepseek-v4-pro": {"providers": ["Baidu", "StreamLake"], "reasoning": {"effort": "medium"}},
}

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def _key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in SECRETS.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no OpenRouter key")


USAGE_BASELINE = 65.419801256  # key usage (USD) recorded before the study's first call (2026-10-17)
_true = {"t": 0.0, "v": 0.0}


def true_spent(max_age=30):
    """Study spend from OpenRouter's own usage counter (includes calls made outside the harness)."""
    now = time.time()
    if now - _true["t"] > max_age:
        try:
            d = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {_key()}"},
                             timeout=30).json()["data"]
            _true["v"] = d["usage"] - USAGE_BASELINE
            _true["t"] = now
        except Exception:
            pass
    return max(_true["v"], spent())


def spent():
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:  # concurrent appends from several processes can leave a partial line (DECISIONS D-029)
                total += json.loads(line).get("cost", 0.0)
            except ValueError:
                continue
    return total


def chat(model, messages, *, purpose, seed=0, temperature=0.0, max_tokens=8000,
         response_format=None, tools=None, reasoning_max_tokens=None, retries=4):
    """Returns (message_dict, usage_dict). Raises BudgetExceeded."""
    with _lock:
        if true_spent() >= BUDGET_USD:
            raise BudgetExceeded(f"budget {BUDGET_USD} reached")
    body = {
        "model": model,
        "messages": messages,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    cfg = MODEL_CFG.get(model, {})
    body["provider"] = {"order": cfg.get("providers", []), "allow_fallbacks": True}
    if "temperature" in cfg:
        body["temperature"] = cfg["temperature"]
    if "top_p" in cfg:
        body["top_p"] = cfg["top_p"]
    if cfg.get("reasoning") is not None:
        body["reasoning"] = cfg["reasoning"]
    if response_format:
        body["response_format"] = response_format
        body["provider"]["require_parameters"] = True
    if tools:
        body["tools"] = tools
    if reasoning_max_tokens is not None:
        body["reasoning"] = {"max_tokens": reasoning_max_tokens}
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {_key()}"},
                              json=body, timeout=600)
            data = r.json()
        except Exception as e:  # network / json
            last = repr(e)
            time.sleep(2 ** attempt * 3)
            continue
        if "error" in data or not data.get("choices"):
            last = json.dumps(data)[:500]
            time.sleep(2 ** attempt * 3)
            continue
        usage = data.get("usage", {})
        rec = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "provider": data.get("provider"),
            "purpose": purpose,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            "cost": usage.get("cost", 0.0) or 0.0,
            "wall_s": round(time.time() - t0, 2),
            "seed": seed,
            "finish_reason": data["choices"][0].get("finish_reason"),
        }
        with _lock:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return data["choices"][0]["message"], rec
    raise RuntimeError(f"OpenRouter call failed after {retries} attempts: {last}")
