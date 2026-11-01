"""A4: probabilistic program (see tournament/DESIGN.md).

Same fixed API as A3, but `step(self, state, t, rng)` receives a seeded `random.Random` (and
`init_state(self, scenario, rng)` may optionally take it too). `run_scenario` runs N seeds
(default 100, max 200) in one sandboxed process (60 s limit) and returns distribution summaries:
mean / sd / quantiles per observed quantity at the final step, a coarse trajectory of means, and
probabilities of user-specified events (expressions over the observation, see tournament.arms._expr).

Observations are flattened to scalar keys (nested dicts -> "a.b", short lists -> "a.0", long lists ->
"a.len"). Seeds whose run raises at step k contribute to steps < k; the error location is reported.
"""
import math

from tournament.arms import _expr
from tournament.arms.a3_sim import MAX_OUT_CHARS, SimArm, _short, fmt_err

DEFAULT_SEEDS = 100
MAX_SEEDS = 200


def _quant(xs, p):
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _g(v):
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.4g}"
    return str(v)


def summarize(vals):
    """Text summary of a list of scalar observations (numbers, bools, strings; None = missing)."""
    ok = [v for v in vals if v is not None]
    if not ok:
        return "no values"
    if all(isinstance(v, bool) for v in ok):
        return f"P(true)={sum(ok) / len(ok):.3f} (n={len(ok)})"
    nums = [float(v) for v in ok if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) == len(ok):
        xs = sorted(nums)
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
        return (f"mean {_g(m)} sd {_g(sd)} | p05 {_g(_quant(xs, .05))} p25 {_g(_quant(xs, .25))} "
                f"p50 {_g(_quant(xs, .5))} p75 {_g(_quant(xs, .75))} p95 {_g(_quant(xs, .95))} | "
                f"min {_g(xs[0])} max {_g(xs[-1])} (n={len(xs)})")
    cnt = {}
    for v in ok:
        cnt[str(v)] = cnt.get(str(v), 0) + 1
    top = sorted(cnt.items(), key=lambda x: -x[1])[:8]
    return ", ".join(f"{k!r}: {c / len(ok):.3f}" for k, c in top) + (f" ... ({len(cnt)} distinct)" if len(cnt) > 8
                                                                         else "") + f" (n={len(ok)})"


def _parse_events(events):
    """-> list of (label, ast, when) or (label, None, error-text)."""
    out = []
    if events is None:
        return out
    if isinstance(events, (str, dict)):
        events = [events]
    if not isinstance(events, list):
        return [("events", None, "events must be a list of expressions or {expr, when}")]
    for ev in events[:20]:
        when = "final"
        if isinstance(ev, dict):
            src = ev.get("expr")
            when = ev.get("when", "final")
        else:
            src = ev
        if not (when in ("final", "any", "all") or (isinstance(when, int) and not isinstance(when, bool))):
            out.append((str(src), None, "when must be 'final', 'any', 'all' or a step number"))
            continue
        try:
            out.append((str(src), _expr.parse(src), when))
        except _expr.ExprError as e:
            out.append((str(src), None, str(e)))
    return out


def _eval_event(ast, obs_list, when):
    """True/False for one run, or None if undefined (e.g. the run ended early, or a name is missing)."""
    def ev(o):
        try:
            return bool(_expr.eval_on(ast, {k: v for k, v in o.items() if v is not None}))
        except (_expr.EvalError, _expr.ExprError):
            return None
    if not obs_list:
        return None
    if when == "final":
        return ev(obs_list[-1])
    if isinstance(when, int):
        return ev(obs_list[when]) if 0 <= when < len(obs_list) else None
    rs = [ev(o) for o in obs_list]
    if when == "any":
        if any(r is True for r in rs):
            return True
        return False if all(r is False for r in rs) else None
    if any(r is False for r in rs):
        return False
    return True if all(r is True for r in rs) else None


class ProbArm(SimArm):
    name = "A4"
    mode = "prob"
    guide_file = "a4_guide.md"
    MAX_STEPS = 1000

    def _run_props(self):
        p = super()._run_props()
        p["n_seeds"] = {"type": "integer", "description": f"number of seeded runs (default {DEFAULT_SEEDS}, "
                                                          f"max {MAX_SEEDS})"}
        p["seed"] = {"type": "integer", "description": "base seed (default 0)"}
        p["events"] = {"type": "array", "description": "event expressions over observation keys, e.g. "
                       "\"queue > 10\", or {expr, when: final|any|all|<step>}; returns their probabilities"}
        return p

    def tools(self):
        ts = super().tools()
        for t in ts:
            if t["function"]["name"] == "run_scenario":
                t["function"]["description"] = ("Run the model for N seeds in the sandbox (60 s limit); returns "
                                                "distribution summaries and event probabilities.")
        return ts

    def probe_tools(self):
        return [t for t in self.tools() if t["function"]["name"] in ("run_scenario", "show_model")]

    def _t_run_scenario(self, state, args):
        if not self._code(state).strip():
            return "ERROR: no model code yet (hint: call write_model first)"
        scen, steps, note = self._check_run_args(args)
        if scen is None:
            return note
        try:
            n = int(args.get("n_seeds", DEFAULT_SEEDS) or DEFAULT_SEEDS)
            seed = int(args.get("seed", 0) or 0)
        except (TypeError, ValueError):
            return "ERROR: n_seeds and seed must be integers"
        lines = [note.rstrip()] if note else []
        if n > MAX_SEEDS:
            lines.append(f"note: n_seeds capped at {MAX_SEEDS}")
            n = MAX_SEEDS
        n = max(1, n)
        events = _parse_events(args.get("events"))
        keys = args.get("keys") if isinstance(args.get("keys"), list) else None

        recs, notes = self._execute(state, scen, {"mode": "prob", "steps": steps, "seed": seed, "n_seeds": n})
        load_errs = [b for t, b in recs if t == "ERR"]
        runs = [b for t, b in recs if t == "RUN"]
        lines.append(f"scenario {_short(scen, 200)}, steps {steps}, seeds requested {n}, completed runs {len(runs)}")
        for e in load_errs:
            lines.append("ERROR: " + fmt_err(e))
        failed = [r for r in runs if r.get("err")]
        if failed:
            first = failed[0]
            lines.append(f"{len(failed)} of {len(runs)} run(s) raised; first (seed index {first['i']}): "
                         + fmt_err(first["err"]))
            lines.append("    (those runs contribute observations up to the failing step)")
        full = [r for r in runs if len(r["obs"]) == steps + 1]
        if runs:
            all_keys = []
            for r in runs:
                for o in r["obs"][:1] + r["obs"][-1:]:
                    for k in o:
                        if k not in all_keys:
                            all_keys.append(k)
            show = [k for k in (keys or all_keys)][:40]
            lines.append(f"FINAL (t={steps}), over {len(full)} run(s) that reached it:")
            for k in show:
                lines.append(f"  {k}: " + summarize([r["obs"][-1].get(k) for r in full]))
            if keys:
                missing = [k for k in keys if k not in all_keys]
                if missing:
                    lines.append(f"  note: keys never observed: {missing} (observed: {all_keys[:20]})")
            if steps > 0:
                cps = sorted(set([0] + [round(steps * i / 8) for i in range(1, 9)]))
                num_keys = [k for k in show if any(isinstance(r["obs"][0].get(k), (int, float))
                                                   and not isinstance(r["obs"][0].get(k), bool)
                                                   for r in runs if r["obs"])][:8]
                if num_keys:
                    lines.append("MEAN TRAJECTORY (t: " + ", ".join(num_keys) + ")")
                    for c in cps:
                        row = []
                        for k in num_keys:
                            vs = [r["obs"][c].get(k) for r in runs if len(r["obs"]) > c]
                            vs = [float(v) for v in vs if isinstance(v, (int, float)) and not isinstance(v, bool)]
                            row.append(_g(sum(vs) / len(vs)) if vs else "-")
                        lines.append(f"  t={c}: " + ", ".join(row))
        if events:
            lines.append("EVENTS:")
            for label, ast, when in events:
                if ast is None:
                    lines.append(f"  {label}: error: {when}")
                    continue
                rs = [_eval_event(ast, r["obs"] if when != "final" else r["obs"][-1:], when)
                      for r in (full if when == "final" else runs)]
                defined = [x for x in rs if x is not None]
                if defined:
                    p = sum(defined) / len(defined)
                    se = math.sqrt(p * (1 - p) / len(defined))
                    und = len(rs) - len(defined)
                    lines.append(f"  P({label}) [{when}] = {p:.3f} +/- {se:.3f} (n={len(defined)}"
                                 + (f", {und} undefined" if und else "") + ")")
                else:
                    lines.append(f"  P({label}) [{when}] = undefined (no run defines it; check key names: "
                                 f"{(list(runs[0]['obs'][-1].keys())[:15] if runs and runs[0]['obs'] else [])})")
        lines += notes
        self._record_run(state, ok=bool(runs) and not load_errs and not failed and len(full) == len(runs) == n,
                         scen=scen)
        if load_errs or not any(r["obs"] for r in runs):
            lines.insert(0, "ERROR: no run produced observations")
        out = "\n".join(lines)
        return out if len(out) <= MAX_OUT_CHARS else out[:MAX_OUT_CHARS - 200] + "\n...[truncated]...\n" + out[-180:]


A4 = ProbArm   # registry name (tournament/arms/__init__.py: ARMS["A4"] = "a4_prob:A4")
