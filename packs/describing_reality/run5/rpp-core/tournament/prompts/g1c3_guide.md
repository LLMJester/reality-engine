# Representation: causal graph + structural equations (one document)

You describe the system as variables linked by equations, written as one text document. An executor compiles it to a causal graph and computes the variables over time, with Monte Carlo over uncertain inputs. It applies interventions `do(variable = value)` and propagates signs qualitatively where equations are missing.

## How to work (6 responses; the last one is your answer)

1. **Response 1: write the whole graph at once and run it.**
   - Start the document with `time …` and `targets: …`. List **every quantity the question names or asks for** as a target.
   - Then make **every quantity a what-if could change** its own variable: rates, capacities, thresholds, shares, delays, and policy switches as 0/1 variables. Do this **before** refining equations. Never bury a stated number inside an equation; name it.
   - Add the question's what-ifs as `whatif` lines.
   - Put it all in ONE fenced block labelled `graph` and call `submit_graph` with no arguments. A response that contains a `graph` block and no tool call is also run as `submit_graph`. The result already contains the baseline and every `whatif`.
2. **Responses 2–4:** use `patch_graph` to fix every *undefined name* and *hole* the coverage report lists. Add what-ifs and run several at once with `intervene` + `whatifs`.
3. **Last response: FINAL ANSWER, with no tool calls.** Use the computed numbers. Read each target's series and SHAPE for the whole path, not only the final value.

## Document syntax

````
```graph
time steps=30 dt=1                      # unit: day
targets: backlog, served_cum            # every quantity the question asks about
# name : type [unit] = initial | ~ distribution | := equation      # provenance
orders_base : real orders/day = 100          # stated "about 100 orders a day"
promo_start : int day = 10                   # stated "a promotion from day 10"
promo_extra : real orders/day = 40           # stated "40 extra orders a day"
promo_on : real := if t >= promo_start and t < promo_start + 5 then 1 else 0   # stated "for five days"
capacity : real orders/day = 110             # stated
noise : real ~ normal(0, 8)                  # assumed: day-to-day scatter
demand : real orders/day := orders_base + promo_on * promo_extra + noise
served : real orders/day := min(capacity, backlog[t-1] + demand)
backlog : real orders = 0 := max(0, backlog[t-1] + demand - served)
served_cum : real orders = 0 := served_cum[t-1] + served
edge capacity -> backlog -                   # optional: explicit sign overrides the inferred one
whatif extra_shift: do capacity = 130 ; targets backlog ; events backlog > 50
```
````

One element per line. Text after `#` is the element's **provenance**: `stated "short quote"`, `derived from …`, or `assumed: … (plausible range)`.

- **`time steps=N dt=D`**: step 0 is the initial state, then N updates of length D. Without a time line the graph is evaluated once (static).
- **`targets: a, b, c`**: the quantities the question asks about. They are reported at every submit.
- **Variables**: `name : type [unit] = value`, `name : type [unit] ~ distribution`, or `name : type [unit] [= initial] := expression`.
  - `type` is `bool`, `int`, `real` or `category`. The default is `real`. The unit is free text.
  - A variable **without** `:=` is an **exogenous input**: a constant `= value` or a `~ distribution`. Distributions are sampled once per Monte Carlo run and then stay constant across all steps of that run. They are `normal(mean, sd)`, `normal(mean, sd, min, max)`, `uniform(lo, hi)` and `choice(['a', 'b'], [0.7, 0.3])`.
  - A variable **with** `:=` is **endogenous**. Its `= value` is its step-0 value: with `time`, its equation applies from step 1.
  - `exo name : … := expression` declares an exogenous input that follows an equation, for example a schedule over `t`.
  - A bare `name : type` with no value or equation, but with `edge` lines into it, is an endogenous variable known only by edge signs (qualitative).
- **Edges** are implied by the variables each equation uses, with lags taken from `[t-k]`, `prev` and `lag`.
  - Their sign is **inferred** by a small finite-difference test at the step-0 and final-step baseline values. It is `?` if the sign is not the same at both points, or is zero.
  - `edge a -> b +|-|? [lag k]` sets a sign explicitly, overriding the inferred one. It is also the way to state causes that have no equation.
  - Threshold rules such as `if x > 5 then 1 else 0` have zero local slope, so their inferred sign is `?`. Add an `edge` line with the sign you know.
- **`whatif name: do …`** stores a named intervention, which runs at every submit and can be run by name later. Optional parts follow, separated by `;`: `targets a, b`, `events expr, expr`, `at final|all|k`, `samples N`. The do forms, comma-separated, are:
  - `x = v`;
  - `x = v from k` (from step k);
  - `x scale 1.1` (multiply the computed value);
  - `x add -2`;
  - `x +` / `x -` (qualitative only).

## Expression syntax (equations and events)

- Numbers, `'strings'`, `true`/`false`. Operators `+ - * / % ^` (or `**`), comparisons `== != < <= > >=`, and `and`, `or`, `not`.
- Conditionals: `if c then a else b` or `ifelse(c, a, b)`.
- Functions: `min(a, b, ...)`, `max(...)`, `abs`, `sqrt`, `exp`, `log(x)` / `log(x, base)`, `floor`, `ceil`, `round(x[, n])`, `clip(x, lo, hi)`, `sign`.
- References: `x` is x at the current step. `x[t-1]`, `prev(x)` and `lag(x, k)` are x one or k steps earlier. Built-ins: `t` (step index) and `dt`.
- A reference to before step 0 uses the variable's `= value`, so a variable that refers to its own past needs an initial value.
- **Same-step cycles** (a uses b and b uses a, with no lag) cannot be computed and become holes. Break the cycle with a lag, e.g. `stock[t-1]`.
- Within a step, variables are computed in dependency order from the same step's causes. Accumulate with `x := x[t-1] + inflow * dt - outflow * dt`.

## Tools

Every result ends with a budget line. Failures start with `ERROR`.

| Tool | Arguments | Effect |
|---|---|---|
| `submit_graph` | none (uses the last ```` ```graph ```` block of your response), or `source` (document text), or `graph` (JSON fallback) | compile the whole document and replace the representation; returns the coverage report, the baseline of the targets and every stored `whatif` |
| `patch_graph` | `old`, `new` | exact, unique text replacement in the document, then the same as `submit_graph` |
| `evaluate` | `query` | compute values; default targets are your `targets:` |
| `intervene` | `do` + `query`, **or** `whatifs`: up to 6 of `{name?, do, query?}` or stored whatif names | paired comparison with the baseline: numbers, then qualitative signs, then series |
| `show` | | the numbered document, the compiled graph and its holes |

- **query**: a variable name, a list of names, or `{"targets": [...], "at": "final" | "all" | <step>, "samples": N, "seed": S, "events": ["expr", ...], "steps": N}`.
  - `at: "all"` returns compact series.
  - `samples` > 0 runs seeded Monte Carlo. It defaults to 500 when a distribution exists. With `samples: 0`, distributions use their mean (or most likely option).
  - `events` are conditions on the variables at the reported step. They return true/false, or a probability.
- **do** (JSON): `{"x": 5}`, `{"x": {"value": 5, "from": 10}}`, `{"x": {"scale": 1.1}}`, `{"x": {"add": -2}}`, `{"x": "+"}`. Under `do`, the variable's own equation and causes are cut. Baseline and intervention use the same random samples, so deltas are paired.

**Coverage report** (at every submit and patch):
1. **undefined names**, with "did you mean";
2. **target status**: computed, hole or undefined;
3. **holes**, with line, column and hint;
4. **intervention handles** (the exogenous inputs) and **magic numbers**: numbers other than 0, 1, 2, −1 and 100 written inside equations. Name them as variables so a what-if can change them;
5. **edges** with their signs; `?` signs are listed.

**Series** are one row per target: `step:value`, where `a-b:v` means steps a to b all equal v. At most 60 points are shown, always including the maximum, the minimum and the final step. The row is followed by a SHAPE line: start, end, max and min with their times, and monotone / single peak / single trough / oscillating.

## Limits to know

Quantities are continuous, so flows can be fractional. There are no individual items, and no queues with order. Within a step, everything is computed from the same step's causes in dependency order. Approximate discrete rules with `if … then … else`, `floor`/`round` and lags, and say so in your answer.

## Worked example (generic)

A shop sets a price. Demand falls with price. Revenue is price times demand. Stock is depleted by demand each day.

````
```graph
time steps=7 dt=1                       # days
targets: revenue, stock
price : real EUR = 10                   # stated "sells at 10 EUR"
base_demand : real units/day = 100      # stated "about 100 units a day at zero price"
price_slope : real units/day/EUR = 5    # stated "5 fewer units per euro"
shock : real units/day ~ normal(0, 5)   # assumed: daily scatter, sd 3-8
demand : real units/day := max(0, base_demand - price_slope * price + shock)
revenue : real EUR/day := price * demand
stock : real units = 500 := max(0, stock[t-1] - demand * dt)   # stated "500 units in stock"
whatif dearer: do price = 12 ; events stock < 150
```
````

The single `submit_graph` result shows:
- the coverage report: 0 undefined names, the targets computed, no magic numbers, and edges `price->demand -`, `price->revenue +`, `demand->revenue +`, `demand->stock -`, …;
- the baseline: revenue about 500 per day, and a stock series falling 500 → about 150;
- the paired effect of `dearer`: revenue about −20 per day, stock about +70, and `P(stock < 150)` from about 0.5 to about 0. Its qualitative section reads `price up -> revenue: ?` (one `+` path, one `-` path) and `price up -> stock: +`.

Had the document said `demand := max(0, 100 - 5 * price + shock)`, the report would flag "magic numbers: 100 and 5 in demand". A question about a change in the base demand could then not be run as a what-if.
