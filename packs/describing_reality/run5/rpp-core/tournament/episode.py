"""Budgeted model episodes shared by every arm (DESIGN.md "Episode protocol").

`Episode` wraps an injected chat client (same signature as harness.llm.chat) and enforces the
shared effort budget: at most `budget_calls` model calls and `budget_tokens` output tokens per
item. When the budget is nearly used up the tool loop ends with a forced "FINAL ANSWER now" turn,
which itself counts within the budget (a token reserve is kept for it).
"""
import json

from tournament.arms.base import BUDGET_CALLS, BUDGET_OUTPUT_TOKENS, ERROR_PREFIX

PER_CALL_MAX_TOKENS = 12000   # cap for one non-final turn
FINAL_RESERVE = 3000          # output tokens kept back for the forced final turn
MIN_CALL_TOKENS = 800         # a turn with fewer tokens available is not started
MAX_TOOL_CALLS = 24           # tool calls per item (all arms)
MAX_RESULT_CHARS = 8000       # tool result shown to the model
FORCE_FINAL_MSG = "The work budget is used up. Give your FINAL ANSWER now."


class PhaseCapReached(RuntimeError):
    pass


class EmptyResponse(RuntimeError):
    """The provider returned a turn with neither content nor tool calls (QA-0300). This is a generation error, not
    an answer: tournament.run records it as `error`, so the D-060 record rule regenerates the item instead of grading
    an empty answer as 0."""


class Episode:
    def __init__(self, client, model, *, purpose, seed=11, budget_calls=BUDGET_CALLS,
                 budget_tokens=BUDGET_OUTPUT_TOKENS, cap_check=None, final_reserve=FINAL_RESERVE,
                 per_call_max=PER_CALL_MAX_TOKENS):
        self.client, self.model, self.purpose, self.seed = client, model, purpose, seed
        self.budget_calls, self.budget_tokens = budget_calls, budget_tokens
        self.cap_check = cap_check
        self.final_reserve = min(final_reserve, budget_tokens // 2)
        self.per_call_max = per_call_max
        self.calls = []
        self.output_tokens = 0
        self.input_tokens = 0
        self.cost = 0.0
        self.wall_s = 0.0
        self.forced_final = False

    @property
    def remaining_calls(self):
        return self.budget_calls - len(self.calls)

    @property
    def remaining_tokens(self):
        return max(0, self.budget_tokens - self.output_tokens)

    def can_continue(self):
        """True if another non-final turn fits while keeping one call and the reserve for the final turn."""
        return self.remaining_calls >= 2 and self.remaining_tokens - self.final_reserve >= MIN_CALL_TOKENS

    def can_answer(self):
        """True if one more plain answer turn fits (no reserve needed: used by A0+)."""
        return self.remaining_calls >= 1 and self.remaining_tokens >= MIN_CALL_TOKENS

    def call(self, messages, tools=None, label="turn", final=False, response_format=None):
        if self.remaining_calls < 1:
            raise RuntimeError("episode call budget exhausted")
        if self.cap_check:
            self.cap_check()
        if final:
            max_tokens = max(1, self.remaining_tokens)
        else:
            max_tokens = max(1, min(self.per_call_max, self.remaining_tokens - (self.final_reserve if tools else 0)))
        kw = dict(purpose=self.purpose, seed=self.seed, max_tokens=max_tokens)
        if tools:
            kw["tools"] = tools
        if response_format:
            kw["response_format"] = response_format
        m, u = self.client(self.model, messages, **kw)
        out = int(u.get("completion_tokens", 0) or 0)
        self.output_tokens += out
        self.input_tokens += int(u.get("prompt_tokens", 0) or 0)
        self.cost += float(u.get("cost", 0.0) or 0.0)
        self.wall_s += float(u.get("wall_s", 0.0) or 0.0)
        self.calls.append({"label": label, "max_tokens": max_tokens, "final": final,
                           **{k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens",
                                                     "cost", "wall_s", "provider", "finish_reason")}})
        if not str(m.get("content") or "").strip() and not m.get("tool_calls"):
            raise EmptyResponse(f"empty model response at {label} (no content, no tool call; provider "
                                f"{u.get('provider')}, finish_reason {u.get('finish_reason')}, "
                                f"{out} completion tokens)")
        return m

    def summary(self):
        return {"n_calls": len(self.calls), "output_tokens": self.output_tokens, "input_tokens": self.input_tokens,
                "cost": self.cost, "wall_s": self.wall_s, "forced_final": self.forced_final,
                "budget": {"calls": self.budget_calls, "output_tokens": self.budget_tokens}}


def _assistant(m):
    am = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("tool_calls"):
        am["tool_calls"] = m["tool_calls"]
    return am


def execute_tool_call(arm, state, tc, offered, n_done, max_tool_calls=MAX_TOOL_CALLS):
    """Run one tool call; never raises. Returns (name, args, result_text)."""
    fn = tc.get("function", {}) or {}
    name = fn.get("name", "")
    raw = fn.get("arguments", "{}")
    try:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            args = {}
        else:
            args = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except Exception as e:
        return name, raw, f"{ERROR_PREFIX}: the arguments of {name} are not a valid JSON object ({str(e)[:200]})."
    if n_done >= max_tool_calls:
        return name, args, f"{ERROR_PREFIX}: tool-call limit ({max_tool_calls}) reached; give your FINAL ANSWER."
    if getattr(state, "given_gold", False) and arm.is_mutating(name):
        return name, args, (f"{ERROR_PREFIX}: in this task the representation is given; it cannot be rebuilt or "
                            f"modified. Use the query tools.")
    if name not in offered:
        return name, args, f"{ERROR_PREFIX}: unknown tool {name!r}. Available: {', '.join(sorted(offered)) or 'none'}."
    try:
        res = arm.handle(state, name, args)
    except Exception as e:  # arms must not raise, but never let one crash the episode
        res = f"{ERROR_PREFIX}: internal error in {name}: {type(e).__name__}: {str(e)[:300]}"
    res = res if isinstance(res, str) else json.dumps(res)
    if len(res) > MAX_RESULT_CHARS:
        res = res[:MAX_RESULT_CHARS] + "\n[... output truncated]"
    return name, args, res


def tool_loop(ep, arm, state, messages, tools_fn, final_msg=FORCE_FINAL_MSG, max_tool_calls=MAX_TOOL_CALLS):
    """Generic build/use loop. tools_fn(state) -> tool schemas offered at this turn.
    Returns (final_text, messages, trace)."""
    trace = []
    n_tools = 0
    while True:
        tools = tools_fn(state) or None
        if tools and not ep.can_continue():
            break
        m = ep.call(messages, tools=tools, label=f"turn{len(ep.calls)}", final=not tools)
        tcs = m.get("tool_calls") or []
        if not tools:
            tcs = []
        am = _assistant(m)
        if not tcs:
            am.pop("tool_calls", None)
            messages.append(am)
            return am["content"], messages, trace
        messages.append(am)
        offered = {t["function"]["name"] for t in tools}
        for k, tc in enumerate(tcs):
            name, args, res = execute_tool_call(arm, state, tc, offered, n_tools, max_tool_calls)
            n_tools += 1
            trace.append({"turn": len(ep.calls), "tool": name, "args": args, "result": res[:3000],
                          "error": res.startswith(ERROR_PREFIX)})
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call{len(trace)}", "content": res})
        # refresh the tool list in the next turn (A7 changes it after `choose`)
    # forced final turn (counts within the budget)
    ep.forced_final = True
    messages.append({"role": "user", "content": final_msg})
    m = ep.call(messages, tools=None, label="forced_final", final=True)
    messages.append({"role": "assistant", "content": m.get("content") or ""})
    return m.get("content") or "", messages, trace
