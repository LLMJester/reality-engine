# Representation: causal graph + structural equations

You build a causal graph: variables, directed edges with signs, and (where you can) an equation that computes each endogenous variable from its causes. An executor computes values, applies interventions `do(var = value)`, runs Monte Carlo over uncertain exogenous variables, and propagates edge signs qualitatively where equations are missing. You build it incrementally with tools. Parts that are broken (bad syntax, unknown names, missing equations) are reported as **holes**. Everything that does not depend on a hole still computes.

## Representation (JSON)

```json
{
  "variables": [
    {"name": "x", "type": "real", "unit": "kg", "exogenous": true, "value": 3},
    {"name": "u", "type": "real", "exogenous": true, "distribution": {"kind": "normal", "mean": 0, "sd": 1}},
    {"name": "y", "type": "real", "exogenous": false}
  ],
  "edges": [{"from": "x", "to": "y", "sign": "+"}, {"from": "u", "to": "y", "sign": "+", "lag": 0}],
  "equations": {"y": "2 * x + u"},
  "time": {"steps": 10, "dt": 1}
}
```

- **variables**: `name` (letters, digits, `_`), `type` (`bool` | `int` | `real` | `category`), `unit` (free text), `exogenous` (true = set from outside the graph).
  - Exogenous variables take a constant `value`, or a `distribution`: `{"kind": "uniform", "low", "high"}`, `{"kind": "normal", "mean", "sd", "min"?, "max"?}`, or `{"kind": "choice", "options": [...], "weights"?: [...]}` (required for `category`).
  - A distribution is sampled once per Monte Carlo run. The sampled value then stays constant across all time steps of that run.
  - An endogenous variable's `value` is its initial value at step 0 when `time` is set.
- **edges**: `from`, `to`, `sign` (`+` | `-` | `?`), optional `lag` (integer number of steps). Edges are used for qualitative sign propagation. An equation that uses a variable with no matching edge produces a warning.
- **equations**: `{var: expression}`, one per endogenous variable. An exogenous variable may also have an equation (for example an input that changes over time).
- **time** (optional): `steps` (step 0 is the initial state, then `steps` updates) and `dt` (step length). Without `time`, the graph is evaluated once (static).

## Expression syntax

- Numbers, `'strings'`, `true`/`false`; operators `+ - * / % ^` (or `**`); comparisons `== != < <= > >=`; `and`, `or`, `not`.
- Conditionals: `if c then a else b` or `ifelse(c, a, b)`.
- Functions: `min(a, b, ...)`, `max(...)`, `abs`, `sqrt`, `exp`, `log(x)` / `log(x, base)`, `floor`, `ceil`, `round(x[, n])`, `clip(x, lo, hi)`, `sign`.
- References: `x` is x at the current step. `x[t-1]`, `prev(x)` and `lag(x, k)` are x one or k steps earlier. Built-ins: `t` (step index) and `dt`.
- A reference to before step 0 uses the variable's `value`, so a variable that refers to its own past needs an initial `value`.
- Same-step cycles (a uses b, b uses a, with no lag) cannot be computed and become holes.
- Step 0 rule (with `time`): an endogenous variable that has a `value` takes that value at step 0. Its equation applies from step 1.

## Tools

| Tool | Arguments | Effect |
|---|---|---|
| `set_graph` | `graph` | replace the whole representation |
| `add_variable` | `name`, `type`, `unit`, `exogenous`, `value`?, `distribution`? | add or replace one variable |
| `add_edge` | `from`, `to`, `sign`, `lag`? | add or replace one edge |
| `set_equation` | `var`, `expr` | set an equation (empty `expr` removes it); syntax errors are reported with a column |
| `evaluate` | `query` | compute values |
| `intervene` | `do`, `query` | compare the intervened model with the baseline, numerically and by edge signs |
| `show` | | print the representation and its holes |

**query**: a variable name, a list of names, or `{"targets": [...], "at": "final" | "all" | <step>, "samples": N, "seed": S, "events": ["expr", ...], "steps": N}`.
- `at: "all"` returns the trajectory.
- `samples` > 0 runs seeded Monte Carlo over the exogenous distributions. It defaults to 500 when any distribution exists. With `samples: 0`, distributions use their mean (or most likely option).
- `events` are expressions over the variables at the reported step. They return true/false, or a probability under Monte Carlo.

**do**: `{var: value}`, `{var: {"value": v, "from": step}}`, `{var: {"scale": 1.1}}` (multiply the computed value), `{var: {"add": -2}}`, or `{var: "+"}` / `{var: "-"}` (qualitative only: the sign of the effect along edge paths). Under `do`, the variable's own equation and causes are cut. Under Monte Carlo, baseline and intervention use the same samples, so deltas are paired.

Qualitative result per target: `+`, `-`, `0` (no directed path) or `?` (mixed or unknown signs). Example paths are listed. Large graphs are fine (thousands of variables and edges): propagation has a time and size budget. If the budget runs out inside a feedback loop, signs through that loop are reported conservatively (`?` where loops of both signs exist) with a note.

## Worked example (generic)

A shop sets a price. Demand falls with price. Revenue is price times demand. Stock is depleted by demand each day.

```json
set_graph {"graph": {
  "variables": [
    {"name": "price", "type": "real", "unit": "EUR", "exogenous": true, "value": 10},
    {"name": "shock", "type": "real", "exogenous": true, "distribution": {"kind": "normal", "mean": 0, "sd": 5}},
    {"name": "demand", "type": "real", "unit": "units/day", "exogenous": false},
    {"name": "revenue", "type": "real", "unit": "EUR/day", "exogenous": false},
    {"name": "stock", "type": "real", "unit": "units", "exogenous": false, "value": 500}
  ],
  "edges": [
    {"from": "price", "to": "demand", "sign": "-"},
    {"from": "shock", "to": "demand", "sign": "+"},
    {"from": "price", "to": "revenue", "sign": "+"},
    {"from": "demand", "to": "revenue", "sign": "+"},
    {"from": "demand", "to": "stock", "sign": "-"}
  ],
  "equations": {
    "demand": "max(0, 100 - 5 * price + shock)",
    "revenue": "price * demand",
    "stock": "max(0, stock[t-1] - demand * dt)"
  },
  "time": {"steps": 7, "dt": 1}
}}
```

`evaluate {"query": {"targets": ["demand", "stock"], "samples": 0}}` returns `demand = 50`, `stock = 150` (500 at step 0, minus 50 per step for 7 steps).

`intervene {"do": {"price": 12}, "query": {"targets": ["revenue", "stock"], "samples": 200, "events": ["stock < 150"]}}` returns paired Monte Carlo summaries. Revenue falls by about 20 per day and stock ends about 70 higher. `P(stock < 150)` drops from about 0.5 to about 0. The qualitative section reads `price up -> revenue: ?` (one `+` path and one `-` path) and `price up -> stock: +`.

Without equations (edges only), `intervene {"do": {"price": "+"}, "query": "stock"}` still answers `+` from the edge signs.
