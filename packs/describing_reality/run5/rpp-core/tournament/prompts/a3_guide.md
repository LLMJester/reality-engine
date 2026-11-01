# Representation: deterministic simulation (Python, fixed API)

You write a Python program that simulates the system step by step. An executor runs it in a sandbox for a scenario and returns what you choose to observe at each step. You build and fix it incrementally with tools. If the code fails, you get back:
- the observations up to the failing step
- the error type, line number, function and source line
- a hint

## The fixed API

```python
class Model:
    def init_state(self, scenario: dict) -> dict:
        """Build the initial state (time 0) from the scenario parameters."""

    def step(self, state: dict, t: int) -> dict:
        """Advance from time t to t+1. Return the new state (or mutate `state` and return None)."""

    def observe(self, state: dict) -> dict:
        """Return the quantities to report at this time."""
```

- **Run order:** `init_state(scenario)` → `observe` (t=0) → for t = 0 … steps−1: `step(state, t)` → `observe` (t+1). A run of `steps` steps yields `steps + 1` observations.
- **Values:** state and observations are dicts. Observations should hold JSON-like values (numbers, bools, strings, lists, dicts). Other values are converted to text.
- **Defaults when a method is missing:** without `init_state`, the state is a copy of the scenario. Without `observe`, the whole state is observed. These are reported as holes. Without `step`, only t=0 can be observed.
- **Environment:**
  - Standard library only (`math`, `heapq`, `collections`, …; numpy is also installed).
  - No files, network or processes.
  - The `random` module is seeded with 0, so runs are repeatable.
  - Limits: 60 s per run, at most 10,000 steps.
- **Output:** `print` output is not part of the result (the first characters are shown as a note).

## Tools

| Tool | Arguments | Effect |
|---|---|---|
| `write_model` | `code` | replace the whole program; syntax errors and missing methods are reported immediately |
| `patch_model` | `old`, `new` | replace one exact, unique occurrence of `old` |
| `run_scenario` | `scenario` (dict), `steps` (int), `keys`? (list) | run and return the observations per step (long runs are shown as first, sampled and last steps; the final step is always shown); `keys` limits the output to those observation keys |
| `show_model` | | print the code with line numbers |

## Worked example (generic)

A savings account earns monthly interest and has a fixed withdrawal each month. Withdrawals stop when the balance cannot cover them.

```python
write_model {"code": "
class Model:
    def init_state(self, scenario):
        return {'balance': scenario['initial'], 'rate': scenario.get('monthly_rate', 0.01),
                'withdrawal': scenario.get('withdrawal', 0), 'missed': 0}

    def step(self, state, t):
        s = dict(state)
        s['balance'] = state['balance'] * (1 + state['rate'])
        if s['balance'] >= state['withdrawal']:
            s['balance'] -= state['withdrawal']
        else:
            s['missed'] += 1
        return s

    def observe(self, state):
        return {'balance': round(state['balance'], 2), 'missed': state['missed']}
"}
```

`run_scenario {"scenario": {"initial": 1000, "monthly_rate": 0.01, "withdrawal": 200}, "steps": 6}` returns observations for t=0 … t=6:
- the balance runs 1000 → 810 → 618.1 → 424.28 → 228.52 → 30.81
- at t=6 the 200 withdrawal cannot be covered, so the result is `{"balance": 31.12, "missed": 1}`

Comparing two scenarios means two `run_scenario` calls with different scenario values. If `init_state` used `scenario['initial']` and a scenario omitted it, the result would be the error `KeyError during init_state: 'initial' [model.py line 4 in init_state()]` with a hint, and the fix is a `patch_model` call.
