# R++ Core world model — guide (v0.1)

You will describe the system as a **world model** in R++ Core: a JSON document that states what exists, what state it is in, how that state changes over discrete time steps, and how you know each thing. A deterministic checker compiles the model to a logic program, runs its scenarios and tests its claims. You build the model with tools (below), and can check, run and fix it.

## Structure

```
{ "rpp_core": "0.1", "system": "<title>", "horizon": <default number of steps>,
  "sorts": [...], "statics": [...], "fluents": [...], "initial": [...],
  "actions": [...], "laws": [...], "derived": [...], "constraints": [...],
  "scenarios": [...], "claims": [...], "answer": {...} }
```

**Naming rules (checked strictly):** names of sorts, instances, fluents, statics, actions and all ids are lowercase `snake_case` starting with a letter (`v4_18`, not `4.18`; `http_429`, not `429`). Variables — including every key of a `params` object — are Capitalised (`Q`, `Version`, `T`), never lowercase, and never contain `__`. All ids share one namespace and must be unique. `t`, `now`, `if`, `then`, `else`, `mod` are reserved words, not names.

**Numbers** are integers, or decimals for a quantity declared `"type": "decimal", "decimals": d` (d = 0..4, default 2). For example, `{"name": "rate", "type": "decimal", "decimals": 2}` holds `0.25`. Write decimal values (facts, initial values, ranges, `plausible_range`, literals in expressions such as `level * 0.5`) in real units with at most d places. Results are kept to the target's d places: extra digits are truncated, and an `int` target keeps the whole part.

## Vocabulary

- **sorts** — types with their members: `{"name": "queue", "instances": ["intake", "review"]}`, or integer ranges `{"name": "day", "range": [0, 30]}`.
- **statics** — fixed facts and parameters: `{"name": "capacity", "args": ["queue"], "type": "int", "facts": [{"args": ["intake"], "value": 3, "status": "asserted", "source": {"kind": "document", "ref": "intake handles 3 per hour"}}]}`. `type` is `int`, `decimal`, `bool`, or a sort name. A parameter you had to guess gets `"status": "assumed"`, an `"id"`, and a `"plausible_range": [lo, hi]`.
- **fluents** — properties that change over time: `{"name": "backlog", "args": ["queue"], "type": "int", "range": [0, 500]}`. If a value cannot go below/above physical limits, give a `range`: values are clamped to it. A fluent with `"derived": true` (plus a `"default"`) is recomputed from the current state at every step by **derived** rules instead of persisting.
- **initial** — the state at step 0: `{"fluent": "backlog", "args": ["intake"], "value": 5, "status": "observed"}`. To initialise every instance: `{"fluent": "backlog", "args": ["Q"], "params": {"Q": "queue"}, "value": 0}`.

## Dynamics

- A non-derived fluent **keeps its value** unless an effect changes it. Every non-derived fluent needs an initial value for every instance; every derived fluent needs a `default`.
- **laws** — what the system does by itself: `{"id": "process", "params": {"Q": "queue"}, "when": "backlog(Q) > 0", "unless": "<optional condition>", "delay": 1, "effects": [{"fluent": "backlog", "args": ["Q"], "add": "-min(backlog(Q), capacity(Q))"}]}`. If `when` holds (and `unless` doesn't) at step t, the effects happen at step t+delay (delay ≥ 1). Effects either `"set"` a new value or `"add"` to the current value; several adds to the same fluent in the same step are summed. Do not `set` and `add` the same fluent in the same step.
- **actions** — external events that a scenario can schedule: `{"name": "arrive", "params": {"Q": "queue"}, "pre": "<optional precondition>", "effects": [...]}`. An action at step t takes effect at t+1. Parameters are positional.
- **derived** — `{"id": "over", "fluent": "overloaded", "params": {"Q": "queue"}, "args": ["Q"], "when": "backlog(Q) > 2 * capacity(Q)", "value": "true"}`.
- **constraints** — things that must never/always be true; the checker reports every violation: `{"id": "no_overflow", "keyword": "NEVER", "cond": "backlog(review) > 40"}`. Keywords: `NEVER`, `ALWAYS`, `ONLY` (`cond` only if `if`), `AT_MOST`/`AT_LEAST`/`EXACTLY` (`over: "X in sort"`, `cond`, `n`), `BEFORE`/`AFTER` (`a`, `b`). Optional `params` and `unless`.

## Expressions (strings)

`and`, `or`, `not`, `= != < <= > >=`, `+ - * /` (`/` truncates), `%` or `mod` (remainder), decimal literals `0.5`, `min(a,b,...)`, `max(...)`, `abs(x)`, `count(X in sort: cond)`, `sum(X in sort: value if cond)`, and fluent/static references like `backlog(Q)` or `capacity(intake)`. A boolean fluent alone means "is true". No nested aggregates.
- `if C then A else B` is a value chosen by a condition: `"set": "if level > 80 then level - 10 else level + 5"`. `A if C else B` means the same. Use it inside a single effect or derived value instead of writing two rules.
- `t` (or `now`) is the current step: the step a law fires at, the step a derived value or constraint is evaluated at, or the claim's step. It is useful for schedules and calendars: `t mod 7 >= 5` (weekend with daily steps), `t >= 30`, `"set": "t"`. Names such as `step`, `day` or `time()` are not built in.

## Scenarios: how you ask questions of the model

`{"id": "s1", "horizon": 12, "description": "...", "init": [<overrides of initial state>], "events": [{"action": "arrive", "args": ["intake"], "time": 3}], "interventions": [...]}`.
Interventions change the model for that scenario only: `{"op": "set_static", "static": "capacity", "args": ["review"], "value": 4}`, `{"op": "disable", "id": "<law/derived/constraint/action id>"}`, `{"op": "add_law", "law": {...}}`, `{"op": "add_derived", "derived": {...}}`, `{"op": "add_constraint", "constraint": {...}}`. The scenario `base` (initial state, no events) always exists.

- **Prediction**: a scenario with the events; inferred claims about values at later steps.
- **Intervention**: a scenario with the intervention; inferred claims about what follows and what breaks.
- **Diagnosis**: one scenario per candidate cause, marked `"candidate": true` (an intervention or event representing the fault); the symptoms as `observed` claims in each candidate scenario. The checker reports which candidates reproduce every symptom; the cause is the one that does.
- **Consistency**: put each stated fact the description gives as an `asserted` claim; the checker tells you which ones the rest of the model contradicts.
- **Assumptions / gaps**: mark guessed elements `assumed` with a `plausible_range`; the checker reports which conclusions change across that range.

## Claims and the answer

`{"id": "c1", "about": {"scenario": "s1", "time": 5, "expr": "backlog(review) = 4"}, "status": "inferred", "derivation": {"rule": "s1", "premises": []}, "text": "review backlog will be 4 at step 5"}`.
- `time`: a step, `"any"` (at some step) or `"all"` (at every step).
- `status`: `observed` (seen/measured in the situation), `asserted` (stated by someone, not verified), `inferred` (follows from your model — needs a `derivation` naming a law/derived/action/constraint id or the scenario id, and premise claim ids), `assumed` (taken as true without support).
- `source`: `{"kind": "document", "ref": "<short quote or section>"}` or `{"kind": "model"}`.

`"answer": {"conclusions": [claim ids that answer the question], "assumptions": [ids of every assumed element], "conflicts": [[ids of claims that cannot all be true], ...], "unexplained": [observed claim ids your model cannot reproduce]}`.

The checker verifies: every inferred claim actually follows from your model; no constraint is violated; no fluent gets two values at once; events respect their preconditions; every assumed element is listed; every observation your model fails to reproduce is listed as unexplained; every asserted claim your model contradicts is listed as a conflict.

## Small example

```json
{"rpp_core": "0.1", "system": "two-stage queue", "horizon": 4,
 "sorts": [{"name": "queue", "instances": ["intake", "review"]}],
 "statics": [{"name": "capacity", "args": ["queue"], "type": "int",
   "facts": [{"args": ["intake"], "value": 3, "status": "asserted", "source": {"kind": "document", "ref": "intake clears 3 per step"}},
             {"id": "cap_review", "args": ["review"], "value": 2, "status": "assumed", "plausible_range": [1, 4]}]}],
 "fluents": [{"name": "backlog", "args": ["queue"], "type": "int", "range": [0, 100]},
             {"name": "overloaded", "args": ["queue"], "type": "bool", "derived": true, "default": false}],
 "initial": [{"fluent": "backlog", "args": ["intake"], "value": 5, "status": "observed"},
             {"fluent": "backlog", "args": ["review"], "value": 1, "status": "observed"}],
 "actions": [{"name": "arrive", "params": {"Q": "queue"}, "effects": [{"fluent": "backlog", "args": ["Q"], "add": "4"}]}],
 "laws": [{"id": "process", "params": {"Q": "queue"}, "when": "backlog(Q) > 0",
           "effects": [{"fluent": "backlog", "args": ["Q"], "add": "-min(backlog(Q), capacity(Q))"}]},
          {"id": "forward", "when": "backlog(intake) > 0",
           "effects": [{"fluent": "backlog", "args": ["review"], "add": "min(backlog(intake), capacity(intake))"}]}],
 "derived": [{"id": "over", "fluent": "overloaded", "params": {"Q": "queue"}, "args": ["Q"], "when": "backlog(Q) > 2 * capacity(Q)"}],
 "constraints": [{"id": "review_cap", "keyword": "NEVER", "cond": "backlog(review) > 10"}],
 "scenarios": [{"id": "s1", "horizon": 4, "events": [{"action": "arrive", "args": ["intake"], "time": 1}]}],
 "claims": [{"id": "c1", "about": {"scenario": "s1", "time": 3, "expr": "backlog(review) = 4"}, "status": "inferred",
             "derivation": {"rule": "s1", "premises": []}, "text": "review backlog reaches 4 at step 3"}],
 "answer": {"conclusions": ["c1"], "assumptions": ["cap_review"]}}
```

## Tools

- `submit_model(model)` — submit the whole world model (a JSON object). It replaces the current model. Structural errors are reported at once, with their location.
- `patch_model(patch)` — change part of the current model, either with a JSON Patch list (`[{"op": "add", "path": "/laws/-", "value": {...}}]`, ops `add`, `replace`, `remove`) or with a fragment such as `{"laws": [{"id": "process", ...}]}`: an element whose `id`/`name` matches an existing one replaces it, others are appended, and `{"id": "x", "delete": true}` removes `x`; any other top-level field is replaced.
- `check()` — compile and run the model: every scenario, claim and constraint. Failures come back with a location and a hint. Parts that cannot be compiled are set aside as **holes** and the rest still runs; the result says which parts are usable.
- `run_scenario(scenario, claims?, show?)` — run one scenario (a scenario object, or `{"id": "<existing id>"}`) and get the value of every fluent at every step. Optional `claims` (`[{"id": "q1", "time": 5, "expr": "backlog(review) > 3"}]`) are evaluated on that run; optional `show` limits the output to the listed fluents.

Several tool calls may be made in one response; they run in order.

## Provenance

Every fact, initial value, law, action, derived rule and claim can carry a `status`: `observed` or `asserted` (the description states it — add a `source` with a short quote), `inferred` (follows from other elements — give a `derivation`), or `assumed` (added to fill a gap — give an `id` and, for numbers, a `plausible_range`).
