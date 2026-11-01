"""Human spot check for the grader test (V3-001): the answers where candidate graders disagree most, on one local page.
The human scores each answer against its key points; the page downloads spotcheck_marks.json. Kept local (it shows
dev answer keys); never published."""
import html
import json
import statistics

from v3.config import RUNS
from v3.grading.grader_test import OUT, sample


def pick(n=20):
    marks = json.loads((OUT / "marks.json").read_text(encoding="utf-8"))
    keys = set().union(*[set(v) for v in marks.values()])
    spread = {}
    for k in keys:
        firsts = [v[k][0]["score"] for v in marks.values() if k in v and v[k] and v[k][0]["score"] is not None]
        if len(firsts) >= 2:
            spread[k] = (max(firsts) - min(firsts), statistics.median(firsts))
    return sorted(spread, key=lambda k: (-spread[k][0], k))[:n]


def page(n=20):
    from v3 import items as I
    chosen = pick(n)
    smp = {k: (iid, raw) for k, iid, raw in sample(100)}
    cards = []
    for j, k in enumerate(chosen):
        iid, raw = smp[k]
        it = I.v2_item(iid)
        key = it["answer_key"]
        kps = "".join(f"<li><b>{html.escape(kp['id'])}</b> (weight {kp.get('weight', 1)}): {html.escape(kp.get('text') or kp.get('point') or json.dumps(kp))}</li>"
                      for kp in key.get("key_points", []))
        wrong = "".join(f"<li>{html.escape(str(w))}</li>" for w in key.get("wrong_answers", []))
        opts = "".join(f'<label><input type="radio" name="{k}" value="{v}"> {v}</label> ' for v in ("0", "0.25", "0.5", "0.75", "1"))
        cards.append(f"""<section><h2>{j + 1} / {len(chosen)} · item {html.escape(iid)}</h2>
<details><summary>System description</summary><pre>{html.escape(it['context'])}</pre></details>
<p><b>Question:</b> {html.escape(it['question'])}</p>
<p><b>Answer key summary:</b> {html.escape(key.get('summary', ''))}</p><ol>{kps}</ol>
{f'<p><b>Known wrong answers:</b></p><ul>{wrong}</ul>' if wrong else ''}
<p><b>Answer to mark:</b></p><pre class="ans">{html.escape(raw[-6000:])}</pre>
<p class="opts">Score (share of weighted key points hit; partial = half; subtract 0.1 per material error): {opts}</p></section>""")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Grader spot check</title><style>
body{{font:15px/1.5 system-ui;max-width:900px;margin:24px auto;padding:0 16px;background:#16171d;color:#e6e6e6}}
section{{background:#22242c;border-radius:8px;padding:16px;margin:18px 0}} pre{{white-space:pre-wrap;background:#111;padding:10px;border-radius:6px}}
.ans{{border-left:3px solid #e9b44c}} label{{margin-right:14px;cursor:pointer}} button{{font-size:16px;padding:10px 18px;margin:20px 0}}
h1{{font-size:22px}} h2{{font-size:17px;color:#e9b44c}}</style></head><body>
<h1>Grader spot check: {len(chosen)} answers</h1>
<p>Mark each answer against its answer key, as the rubric does: the share of key points the answer gets right (partial = half credit), minus 0.1 for each material error. These are the answers the AI graders disagreed on most. Your marks decide which grader is trusted. When done, click <b>Save marks</b>: a file <code>spotcheck_marks.json</code> downloads. Tell Claude when it's saved.</p>
{''.join(cards)}
<button onclick="save()">Save marks</button> <span id="n"></span>
<script>
function save(){{const m={{}};document.querySelectorAll('input[type=radio]:checked').forEach(r=>m[r.name]=parseFloat(r.value));
const n=Object.keys(m).length;document.getElementById('n').textContent=n+' of {len(chosen)} marked';
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(m,null,1)],{{type:'application/json'}}));
a.download='spotcheck_marks.json';a.click();}}
</script></body></html>"""
    out = RUNS / "grader_test" / "spotcheck.html"
    out.write_text(doc, encoding="utf-8")
    return out, chosen


if __name__ == "__main__":
    p, c = page()
    print(p, len(c))
