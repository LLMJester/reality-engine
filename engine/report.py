"""Step 7, the trust report: six numbers at a glance, then the detail behind them.

  How often it's right   the final design's mean on the sealed items, with its 95% range, and by level
  Reliability            the share of its sealed runs that finished without a failure, and its weakest level
  Why it fails           the pack's failure types on the sealed runs, most common first
  Verdict                the final design against the control and its closest rivals, in the locked order
  How hard we looked     founders, new versions written, designs built, runs, stages and changes kept
  Headroom               the closest rival and the best change not yet proven: is another run worth paying for?

Writes <work>/report.json and <work>/report.html (one self-contained page, light and dark).
"""
import html
import json


def summary(eng):
    an, st = eng.analysis(), eng.st
    fin = an.get("final") or {}
    sealed_name = eng.plan["sets"].get("sealed")
    sealed = eng.items(sealed_name) if sealed_name and fin else []
    fd = st["best"]["id"]
    means = fin.get("sealed_means") or {}
    right = means.get("FINAL") or {}
    by_level = {lv: v.get("FINAL", {}).get("mean") for lv, v in (fin.get("by_level") or {}).items()}
    recs = eng.store.records("main", fd, sealed) if sealed else {}
    done = [r for r in recs.values() if r]
    clean = [r for r in done if not r.get("error")]
    fails = {}
    for r in done:
        t = eng.pack.failure_type(r)
        if t:
            fails[t] = fails.get(t, 0) + 1
    seq = fin.get("sequence") or {}
    verdict = [{"test": k, "rival": v.get("rival"), "lead": v.get("estimate"), "range": v.get("range"),
                "counted": v.get("counted")} for k, v in seq.items()]
    # how hard we looked
    cands = list((eng.work / "candidates").glob("*.json"))
    runs = sum(1 for p in eng.store.root.rglob("record.json")) if eng.store.root.exists() else 0
    versions = 0
    for f in (eng.work / "stages").glob("*/candidates.json"):
        versions += sum(len(v) for v in json.loads(f.read_text(encoding="utf-8"))["alts"].values())
    hist = st.get("history") or []
    kept = [h for h in hist if h["verdict"] in ("kept",) or h["verdict"].startswith("design ")]
    rounds = [k for k in an if k.startswith("round") and k.endswith("_screen")]
    effort = {"founders": len(eng.founders()), "new_versions": versions, "designs_built": len(cands),
              "runs": runs, "stages": 1 + ("takeapart" in an) + len(rounds) + bool(fin), "changes_kept": len(kept)}
    # headroom
    closest = min(verdict, key=lambda v: v["lead"] if v["lead"] is not None else 9, default=None)
    unproven = [h for h in hist if h.get("effect") is not None and h["verdict"] not in ("kept",)
                and not h["verdict"].startswith("design ")]
    best_unproven = max(unproven, key=lambda h: h["effect"], default=None)
    return {"plan": eng.plan["name"], "pack": eng.pack.name, "final_design": fin.get("final_design"),
            "final_id": fd, "sealed_set": sealed_name, "n_sealed": len(sealed),
            "clusters": len({eng.stats.cluster_of(i) for i in sealed}),
            "right": {"mean": right.get("mean"), "range": right.get("range"), "by_level": by_level},
            "reliability": {"finished_cleanly": round(len(clean) / len(done), 4) if done else None,
                            "n": len(done), "weakest_level": min(by_level, key=lambda k: by_level[k] or 0)
                            if by_level else None, "pack": (fin.get("pack") or {}).get("reliability")},
            "why_it_fails": dict(sorted(fails.items(), key=lambda x: -x[1])),
            "verdict": verdict, "effort": effort,
            "headroom": {"closest_rival": closest,
                         "best_unproven": {k: best_unproven.get(k) for k in ("where", "slot", "effect", "range",
                                                                             "verdict")} if best_unproven else None}}


def _pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def _rng(r):
    return "" if not r or r[0] is None else f"{100 * r[0]:.0f}–{100 * r[1]:.0f}%"


def _pts(x):
    return "–" if x is None else f"{x * 100:+.1f} pts"


def page(s):
    e = html.escape
    v = s["verdict"]
    counted = [x for x in v if x["counted"]]
    tiles = [
        ("How often it's right", _pct(s["right"]["mean"]), f"range {_rng(s['right']['range'])} on {s['n_sealed']} "
         f"sealed cases" + ("; by level " + ", ".join(f"{k}: {_pct(m)}" for k, m in s["right"]["by_level"].items())
                            if s["right"]["by_level"] else "")),
        ("Reliability", _pct(s["reliability"]["finished_cleanly"]), "of its sealed runs finished without a failure"
         + (f"; weakest level {s['reliability']['weakest_level']}" if s["reliability"]["weakest_level"] else "")),
        ("Why it fails", next(iter(s["why_it_fails"]), "–"),
         ", ".join(f"{k} {n}" for k, n in s["why_it_fails"].items()) or "no failure types reported"),
        ("Verdict", f"{len(counted)} of {len(v)}", "leads counted: " + "; ".join(
            f"vs {x['rival']} {_pts(x['lead'])} ({'counted' if x['counted'] else 'not shown'})" for x in v)),
        ("How hard we looked", f"{s['effort']['runs']:,} runs", f"{s['effort']['founders']} founders, "
         f"{s['effort']['new_versions']} new versions, {s['effort']['designs_built']} designs, "
         f"{s['effort']['stages']} stages, {s['effort']['changes_kept']} changes kept"),
        ("Headroom", _pts((s["headroom"]["closest_rival"] or {}).get("lead")),
         (f"closest rival {s['headroom']['closest_rival']['rival']}" if s["headroom"]["closest_rival"] else "")
         + (f"; best unproven change {s['headroom']['best_unproven']['slot']} "
            f"{_pts(s['headroom']['best_unproven']['effect'])} ({s['headroom']['best_unproven']['where']})"
            if s["headroom"]["best_unproven"] else "")),
    ]
    cards = "\n".join(f'<div class="tile"><h2>{e(t)}</h2><b>{e(str(big))}</b><p>{e(sub)}</p></div>'
                      for t, big, sub in tiles)
    rows = "\n".join(f"<tr><td>{e(x['test'])}</td><td>{e(str(x['rival']))}</td><td>{_pts(x['lead'])}</td>"
                     f"<td>{_rng(x['range']).replace('%', ' pts') if x['range'] else ''}</td>"
                     f"<td>{'counted' if x['counted'] else 'not counted'}</td></tr>" for x in v)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Trust report</title>
<style>
:root{{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#5d5d58;--line:#dcdcd5;--card:#fff;--good:#2f6b3f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--muted:#a5a59f;--line:#34342f;--card:#1e1e1c;--good:#7cc08c}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1040px;margin:0 auto}} h1{{font-size:24px;margin:0 0 4px}} .muted{{color:var(--muted)}}
.glance{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}}
@media (max-width:760px){{.glance{{grid-template-columns:1fr}}}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.tile h2{{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}}
.tile b{{font-size:28px;color:var(--good);font-family:Georgia,serif}} .tile p{{margin:6px 0 0;color:var(--muted)}}
table{{border-collapse:collapse;width:100%;background:var(--card)}} td,th{{border-bottom:1px solid var(--line);
padding:6px 8px;text-align:left}} .wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>Trust report: {e(s['plan'])}</h1>
<p class="muted">Pack {e(s['pack'])} · final design {e(str(s['final_design']))} ({e(s['final_id'])}) · sealed set
{e(str(s['sealed_set']))}: {s['n_sealed']} cases in {s['clusters']} clusters, opened once</p>
<div class="glance">{cards}</div>
<h2>The final check, in the locked order</h2>
<p class="muted">Each lead counts only if its whole 95% range is above zero and every earlier one counted.</p>
<div class="wrap"><table><tr><th>Test</th><th>Rival</th><th>Lead</th><th>95% range</th><th></th></tr>
{rows}</table></div>
</main></body></html>"""


def write(eng):
    s = summary(eng)
    (eng.work / "report.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    (eng.work / "report.html").write_text(page(s), encoding="utf-8")
    return s
