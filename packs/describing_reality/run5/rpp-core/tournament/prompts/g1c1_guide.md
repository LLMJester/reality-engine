# Representation: probabilistic program with declared parameters (Monte Carlo)

You write one Python program that simulates the system step by step with a seeded random generator. An executor runs it for many seeds and returns distribution summaries. Every number and rule from the description is declared by name at the top, so any what-if is a re-run with overrides, not a rewrite.

## How to work (the budget is 6 responses; the last one is your answer)

1. **Response 1: plan, then build and run.** Write a short plan (at most 12 lines): the quantities the question asks for, the mechanisms that drive them, the time unit and horizon, and the what-if runs you will need. Then write the whole program in ONE fenced block labelled `model`, that is, a line ```` ```model ````, the program, then a line ```` ``` ````. In the same response, call `submit_model` with `runs` for the question's what-if(s). The program in the block is used, so you do not need to repeat it inside the tool arguments. (If you cannot call tools, a response that contains only the block is also run.)
2. **Responses 2–4: fix and run.** If there is an error, fix it with `patch_model` (it re-runs automatically). Use `run_scenario` with `runs` to try several scenarios in one call. By the end of response 4 you must have a successful run of the baseline and of every what-if the question asks about.
3. **Last response: FINAL ANSWER,** with no tool calls and no `model` block. Base every number on the run results. Read the SHAPE and SERIES lines for the whole path (peaks, rebounds, when things start and stop), not only the final values. Where the question asks for likelihoods, report uncertainty from the quantiles and probabilities.

Every executor result ends with `[budget: k of N responses used; ...]`.

## Declarations

```
PARAMS = {                  # every quantity from the description
    "arrival_prob": 0.3,    # stated: "about 3 in 10 minutes see a customer arrive"
    "service_prob": 0.25,   # derived: mean service 4 min -> 1/4 per minute
    "patience": 6,          # assumed: not given; 4-10 plausible
}
RULES = {                   # every discrete rule, policy, limit or switch a what-if might change
    "max_queue": 8,         # stated
    "priority": "fifo",     # stated: "first come, first served"
}
HORIZON = 60                # steps; one step = 1 minute
```

- `PARAMS`: every quantity from the description, with a comment `# stated`, `# derived` or `# assumed: why`.
- `RULES`: every discrete rule, policy, count limit or switch that a what-if might change (on/off, an order, a count, a mode name). Make the code branch on `RULES[...]` (or on the scenario value).
- `HORIZON = n`, with a comment giving the time unit.
- Both dicts must be literals (numbers, strings, booleans, lists, dicts; simple arithmetic such as `1/6` is fine).
- Read values via `scenario["name"]` in `init_state` (the executor puts PARAMS and RULES there, with overrides applied). `PARAMS[...]` and `RULES[...]` also hold the overridden values inside the methods. Do not compute module-level constants from them: those would not see overrides.
- An override of a name that is not declared is an error that lists the declared names. So declare everything a what-if could touch.

## Time and ordering

- Choose a step small enough for the fastest process that matters. If several things happen within one step, write them in the order the description implies and say so in a comment.
- Count events (arrivals, rejections, drops, failures) in cumulative counters, so their totals can be observed.
- Observe everything the question asks about, plus the main state variables.

## The fixed API

```python
class Model:
    def init_state(self, scenario: dict, rng) -> dict:     # rng is optional: init_state(self, scenario) also works
        """Build the initial state (time 0) from the scenario (PARAMS + RULES + overrides)."""

    def step(self, state: dict, t: int, rng) -> dict:
        """Advance from time t to t+1 using rng (a random.Random). Return the new state
        (or mutate `state` and return None)."""

    def observe(self, state: dict) -> dict:
        """Return the quantities to report at this time."""
```

- **Run order (each seed):** `init_state` → `observe` (t=0) → for t = 0 … steps−1: `step(state, t, rng)` → `observe` (t+1).
- **Randomness:** each seed gets its own `rng = random.Random(seed)`. Use its methods: `rng.random()`, `rng.uniform(a, b)`, `rng.gauss(mu, sigma)`, `rng.expovariate(rate)`, `rng.randint(a, b)`, `rng.choice(seq)`, `rng.choices(seq, weights)`. The global `random` module is seeded the same way, so results are repeatable. A deterministic system is fine too: then every seed gives the same path.
- **Common random numbers:** the baseline and each what-if use the same seeds, so a reported difference is the effect of the change, not noise. Draw random numbers in the same order regardless of the parameters where you can (e.g. draw `rng.random()` for every customer, then compare it with the probability).
- **Observation keys:** observations are flattened to scalar keys. Nested dicts become `a.b`, lists of up to 20 items become `a.0`, `a.1`, …, and longer lists become `a.len`. Numbers get mean, sd, quantiles, min and max. Bools get `P(true)`. Strings get frequencies.
- **Defaults when a method is missing:** without `init_state`, the state is a copy of the scenario. Without `observe`, the whole state is observed. These are reported as holes.
- **Environment:**
  - Standard library only (numpy is also installed).
  - No files, network or processes.
  - Limits: 60 s per tool call for all runs together (seeds are reduced, and reported, if time runs out), at most 1,000 steps, 200 seeds.

## Tools

| Tool | Arguments | Effect |
|---|---|---|
| `submit_model` | `source`? (string), `runs`? (list, ≤ 4) | takes the program from `source`, or else from the last ```` ```model ```` block in the same response; checks it; runs the **baseline** (HORIZON steps, 100 seeds) and each entry of `runs`; returns all summaries in one result |
| `patch_model` | `old`, `new`, `runs`? | replaces one exact, unique occurrence of `old`, then re-runs the baseline and `runs` |
| `run_scenario` | one run's fields, or `runs` (list, ≤ 4) | runs the stored program; each run with overrides or a patch is compared with the baseline (same seeds) unless `compare: false` |
| `show_model` | | the program with line numbers, plus the declared PARAMS/RULES table |

**A run's fields:** `name`?, `overrides`? (dict over PARAMS/RULES names), `steps`? (default HORIZON, max 1000), `n_seeds`? (default 100, max 200), `seed`? (default 0), `events`? (list), `keys`? (observation keys to show first), `compare`? (default true), `what_if_patch`? (`{"old": ..., "new": ...}`: a one-run code change for a what-if the switches don't cover; the stored model is unchanged, and the result says "ephemeral patch applied"). In `run_scenario`, top-level `steps`, `n_seeds`, `seed`, `events`, `keys` apply to every entry of `runs` that does not set them.

**events:** each event is an expression over observation keys, such as `"waiting > 10"` or `"done and cost <= 50"`, or `{"expr": "...", "when": "final" | "any" | "all" | <step>}` (default `final`). `any` means true at some step (the result also gives the **first-passage time**: the first step at which it is true), and `all` means true at every step. Syntax: arithmetic and comparisons; `and`, `or`, `not`; `min`, `max`, `abs`; `if c then a else b`. Each probability is reported with its standard error.

**What a result contains** (bounded to 8,000 characters; if needed SERIES, then SHAPE, then EVENTS lines are dropped, and the output says so):
1. a header per run: name, overrides, steps, seeds completed, and errors with line, function, source line and a hint;
2. **FINAL**: the final-step distribution of each key (mean, sd, p05 … p95);
3. **EFFECT vs baseline** (paired over the same seeds): mean difference, p05/p50/p95 of the difference and P(diff > 0) for numbers; the change in P(true) for bools; keys unchanged on every seed are listed together;
4. **EVENTS**: probabilities, and first-passage times for `any` events;
5. **SHAPE** of each key's mean path: start, end, maximum and minimum with their times, monotone / single-peak / oscillating, and the time to reach 50% and 90% of the total change;
6. **SERIES**: the mean path of up to 8 keys, one row per key (`0×12` means twelve zeros in a row); every step when steps ≤ 60, otherwise 60 points that include the extremes and the final step.

## Worked example (generic)

A single-server shop. Each minute a customer arrives with probability `arrival_prob`; the server finishes the current customer with probability `service_prob`; an arriving customer who finds `max_queue` people waiting walks away (balks).

````
Plan: quantities = queue length, served, balked over 2 hours (1 step = 1 minute, HORIZON 120).
Mechanisms: Bernoulli arrivals, Bernoulli service, balking at the cap. What-if: cap of 5 instead of 8.

```model
PARAMS = {
    "arrival_prob": 0.3,     # stated
    "service_prob": 0.25,    # stated
}
RULES = {
    "max_queue": 8,          # stated
    "priority": "fifo",      # stated
}
HORIZON = 120                # one step = 1 minute

class Model:
    def init_state(self, scenario, rng):
        return {"queue": 0, "served": 0, "balked": 0,
                "pa": scenario["arrival_prob"], "ps": scenario["service_prob"], "cap": scenario["max_queue"]}

    def step(self, state, t, rng):
        s = dict(state)
        u_arrive, u_serve = rng.random(), rng.random()   # drawn every step: same random numbers in every run
        if s["queue"] > 0 and u_serve < s["ps"]:          # service first, then the arrival (order assumed)
            s["queue"] -= 1
            s["served"] += 1
        if u_arrive < s["pa"]:
            if s["queue"] < s["cap"]:
                s["queue"] += 1
            else:
                s["balked"] += 1
        return s

    def observe(self, state):
        return {"queue": state["queue"], "served": state["served"], "balked": state["balked"]}
```
````

with the call `submit_model {"runs": [{"name": "cap5", "overrides": {"max_queue": 5}, "events": [{"expr": "queue >= 5", "when": "any"}]}]}`. The result (abbreviated) reads:

```
RUN baseline: steps 120, seeds 100/100 completed, 100 reached t=120
RUN cap5: steps 120, seeds 100/100 completed, ...; overrides {"max_queue":5}; compared with baseline
FINAL [baseline] (t=120, n=100):
  queue: mean 6.1 sd 1.9 | p05 2 ... p95 8 ...
EFFECT [cap5] vs baseline (paired over 100 common seeds):
  balked: mean diff +14.2 | p05 +5 p50 +14 p95 +24 | P(diff>0)=0.97 P(diff<0)=0.00 (n=100)
EVENTS:
  [cap5] P(queue >= 5) [any] = 0.990 +/- 0.010 (n=100); first passage t: mean 21 p05 9 p50 19 p95 41 (n=99)
SHAPE (mean trajectory):
  [baseline] queue: start 0 (t=0) -> end 6.1 (t=120); max 7.3 at t=31, ...; single-peak (rises, then falls); ...
SERIES (mean per step):
  [baseline] queue: 0 0.3 0.52 ...
```

## Answering a what-if about a given model

When the program is given to you and is read-only, do not rebuild it. Call `run_scenario` with `overrides` (any PARAMS or RULES name, as listed by `show_model`) or, for a change the declarations do not cover, `what_if_patch`; then read the EFFECT block (and EVENTS / SHAPE for timing questions). Several scenarios fit in one call via `runs`.
