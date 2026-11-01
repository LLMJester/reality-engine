"""The human's spot check of the run 5 hand-written-style items (RUN5_BUILD_PLAN step 3): 20 accepted items, chosen
at random (fixed seed) spread over the six question types, written to items/hand5/spotcheck.html (a local file; the
items never leave this machine). Each item shows the document, the question, the answer key and what the checks
found. Usage: python scripts/hand5_spotcheck.py [--n 20]
"""
import argparse
import html
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "items" / "hand5" / "spotcheck.html"


def pick(items, n, seed=20261028):
    by = {}
    for it in items:
        by.setdefault(it["task_type"], []).append(it)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out, types = [], sorted(by)
    while len(out) < min(n, len(items)):
        for t in types:
            if by[t] and len(out) < n:
                out.append(by[t].pop())
    return out


def page(chosen):
    from v3.hand5.factory import CHECKS, LEAK_MAX
    e = html.escape
    cards = []
    for k, it in enumerate(chosen, 1):
        rec = json.loads((CHECKS / f"{it['id']}.json").read_text(encoding="utf-8"))
        key = it["answer_key"]
        kps = "".join(f"<li>{e(p['text'])}</li>" for p in key.get("key_points", []))
        wrong = "".join(f"<li>{e(w)}</li>" for w in key.get("wrong_answers", []))
        adj = rec.get("adjudication") or {}
        cards.append(f"""
<section class="card"><h2>{k}. {e(it['id'])} · {e(it['task_type'])} · {e(it['kind'])} · {e(it['messiness'])} · {e(it['domain'])}</h2>
<p class="t">{e(it.get('system_title', ''))}</p>
<div class="cols"><div><h3>Document Luna reads</h3><div class="doc md">{e(it['context'])}</div></div>
<div><h3>Question</h3><div class="md">{e(it['question'])}</div>
<h3>Answer key</h3><div class="md">{e(key.get('summary', ''))}</div><h4>Key points</h4><ul>{kps}</ul><h4>Tempting wrong answers</h4><ul>{wrong}</ul>
<h3>Checks</h3><p>Independent Sonnet answer graded {rec['validator_grade'].get('score', 0):.2f}; no-document leak check
{(rec['leak']['grade'] or {}).get('score', 0):.2f} (must be at most {LEAK_MAX}, with no top-weight key point hit); adjudicator: <b>{e(adj.get('verdict', ''))}</b>
{'(key corrected)' if it.get('key_fixed_by_adjudicator') else ''} - {e(adj.get('reason', ''))}</p>
<p class="ask">Your verdict: is the document realistic, the question fair, and the key right?</p></div></div></section>""")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Item Spot Check</title><style>
:root {{ --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --ring:rgba(11,11,11,.1); }}
@media (prefers-color-scheme: dark) {{ :root {{ --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --ring:rgba(255,255,255,.1); }} }}
body {{ margin:0; background:var(--page); color:var(--ink); font:14px/1.55 system-ui,sans-serif; }}
main {{ max-width:1300px; margin:0 auto; padding:16px; }} h1 {{ font-size:20px; }} h2 {{ font-size:15px; margin:0; }}
h3 {{ font-size:13px; margin:12px 0 4px; color:var(--ink2); text-transform:uppercase; letter-spacing:.04em; }} h4 {{ font-size:13px; margin:8px 0 2px; }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:14px; margin-bottom:16px; }}
.t {{ color:var(--ink2); margin:2px 0 8px; }} .cols {{ display:grid; grid-template-columns:1.2fr 1fr; gap:16px; }}
.doc {{ white-space:pre-wrap; max-height:640px; overflow:auto; border:1px solid var(--ring); border-radius:8px; padding:10px; font-size:13px; }}
.ask {{ font-weight:600; }} @media (max-width:800px) {{ .cols {{ grid-template-columns:1fr; }} }}
.md {{ white-space:pre-wrap; }} .md.rendered {{ white-space:normal; }} .md.rendered > :first-child {{ margin-top:0; }}
.md h1 {{ font-size:16px; }} .md h2 {{ font-size:15px; margin:14px 0 6px; }} .md h3 {{ font-size:14px; text-transform:none; letter-spacing:0; color:var(--ink); }}
.md table {{ border-collapse:collapse; margin:8px 0; font-size:12.5px; display:block; overflow-x:auto; }}
.md th, .md td {{ border:1px solid var(--ring); padding:3px 8px; text-align:left; vertical-align:top; }} .md th {{ background:var(--ring); }}
.md code {{ font-size:12px; background:var(--ring); padding:0 3px; border-radius:3px; }} .md pre {{ overflow-x:auto; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<script>
// render the Markdown in place; without the libraries (offline) the raw text stays readable
addEventListener("DOMContentLoaded", () => {{
  if (!window.marked || !window.DOMPurify) return;
  document.querySelectorAll(".md").forEach(el => {{
    el.innerHTML = DOMPurify.sanitize(marked.parse(el.textContent));
    el.classList.add("rendered");
  }});
}});
</script></head><body><main><h1>Run 5 hand-written items: spot check ({len(chosen)} of the 500)</h1>
<p>Chosen at random, spread over the six question types. Each card shows exactly what Luna reads, the question, the
answer key computed from the hidden simulator, and what the automatic checks found.</p>{''.join(cards)}</main></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args(argv)
    items = json.loads((ROOT / "items" / "hand5" / "items.json").read_text(encoding="utf-8"))
    chosen = pick(items, a.n)
    OUT.write_text(page(chosen), encoding="utf-8")
    print(OUT, len(chosen), "items:", ", ".join(c["id"] for c in chosen))


if __name__ == "__main__":
    main()
