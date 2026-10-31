# Writing a problem pack

A problem pack tells the engine everything specific to one problem. The engine itself never changes, so every
problem gets the same stages, the same statistics and the same honesty rules. Start from `packs/toy/pack.py`: it is
a complete pack in about a hundred lines.

A pack is a Python class that subclasses `engine.pack.Pack`. A plan file names it (`"pack": "packs.mine.pack:MyPack"`)
and can pass it options (`"pack_options": {...}`).

## 1. Candidates

A **candidate** is whatever you are choosing between: a prompt, a note format, an agent's instructions, a
configuration. To the engine it is a JSON-serialisable dict with one required key, `"id"`. **Equal contents must give
an equal id.** Hash the parts that change behaviour, not the name. The engine relies on this to reuse a run it has
already done, and to know that "no changes" means "the same candidate".

```python
def founders(self):
    return {"CURRENT": make({...}), "IDEA_1": make({...}), "IDEA_2": make({...})}
```

Include your current setup among the founders and name it as the control (`control = "CURRENT"`, or `"control"` in the
plan). Every lead in the trust report is measured against it.

## 2. Items and scoring

An **item** is one test case, named by a string id. `item_sets()` returns named lists of ids. The plan says which set
each step uses: `compare`, `takeapart`, one `screen` and one `confirm` set per build-out round, and `sealed`. **Use
fresh items for every step.** The sealed set must be cases nobody has tuned anything on.

`run(candidate, item, ctx)` runs one candidate on one item and returns a score from 0 to 1:

```python
return {"score": 0.8, ...anything you want kept in the record...}
```

When a run fails, say whose fault it was:

| Return | Meaning | What the engine does |
|---|---|---|
| `{"error": ..., "failure": "design"}` | The candidate broke a rule or ran out of time | Retries, then scores 0. Never dropped |
| `{"error": ..., "failure": "infra"}` | Connection, service or grader trouble | Retries with a backoff, then stops for a person. Never scored |
| `{"error": ..., "failure": "config"}` | The candidate cannot run at all | Stops at once |

Raise an exception named `Paused` (`engine.pack.Paused`) to stop cleanly, for example when an allowance runs low.
Running jobs finish and nothing is lost. `ctx["dir"]` is a folder your pack may write a detailed record into.

Optional hooks:
- `cluster_of(item)`: items that share a source form a cluster, for example one system asked at four sizes. Ranges
  and tests resample clusters, so related items don't count as independent evidence.
- `level_of(item)`: a difficulty level for the "by level" tables.
- `failure_type(record)`: a short label for why a run lost points. It feeds the "Why it fails" tile.

## 3. Parts: taking the winner apart and building out

Steps 4 and 5 change a candidate one part at a time.

- `slots(kind)`: the part names, in a fixed order (`kind` is `"takeapart"` or `"buildout"`). Use five or fewer for
  step 4, since 2⁵ = 32 combinations.
- `parts(candidate, kind)`: the candidate's current version of each slot.
- `compose(base, parts, changes, kind, extras)`: build a candidate from `base` with the slots in `changes`
  (`{slot: alt}`) replaced. **No changes must give back `base`'s id.** Every combination of changes must build.

## 4. The designer, and the firewall

`write(kind, candidate, parts, brief, slots, per_slot)` returns new versions ("alts") of the given slots:

```python
return {"rules": [{"value": "...", "rationale": "..."}, ...], ...}, extras
```

This is usually where an AI writes the new versions. It gets `brief["text"]`: the engine's summary of what has been
measured. That means the current best's scores overall and by level, how often each failure type occurred, your
`evidence()` aggregates, and every version tried before with its measured effect.

**The firewall is the most important rule in a pack.** The designer must never see an item, a question, an answer, an
answer key, a hidden test, a per-item score or anything from the sealed set. The engine's brief contains none of
these. `evidence(candidate, records)` may add aggregates only: counts, rates, means, lengths. If the designer could
read the test, the experiment would measure memory, not quality.

Check each version mechanically before you return it: it must build, differ from the current part, stay within any
length limits and change only its own slot. Let the designer try again with the reasons. A person approves the
frozen versions before anything runs on them (`"approval": "human"` in the plan). `"auto"` is for tests and pilots.

`prepare(kind, candidate, parts, advancing, extras)` runs before a confirmation, with the versions that passed the
screen. Use it when a combination needs something written first: run 5 rewrote a worked example into a new layout,
for instance.

## 5. The plan, the lock and running it

Copy `plans/toy.json`. Then:

```
python -m engine lock plans/mine.json --reason "first run"
python -m engine run plans/mine.json
python -m engine status plans/mine.json
```

The lock fingerprints the plan, the engine and your pack's code. `run` refuses to start if any of them has changed
since. To change a rule after a run has started, write down why, lock again with that reason (each lock is logged in
`LOCKS.log`) and say so when you report the results.

## Checklist

- [ ] Equal candidates give equal ids. `compose()` with no changes returns the base's id
- [ ] The control is your current setup
- [ ] Every step has its own fresh items, and the sealed set is untouched
- [ ] Failures are labelled design, infra or config
- [ ] The designer sees the brief only: no items, answers, keys or sealed results
- [ ] Versions are checked mechanically before they are returned
- [ ] A toy-sized run works end to end before the real one (`"approval": "auto"`, small sets)
