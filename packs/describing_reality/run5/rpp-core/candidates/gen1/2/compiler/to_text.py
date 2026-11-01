"""Canonical text form of an R++ Core AST (read-only pretty printer)."""
import json
import sys


def _ann(o):
    bits = []
    if o.get("status"):
        bits.append(o["status"])
    src = o.get("source")
    if src:
        bits.append(f'{src["kind"]}' + (f':"{src["ref"]}"' if src.get("ref") else ""))
    if o.get("id"):
        bits.insert(0, o["id"])
    return f"  [{', '.join(bits)}]" if bits else ""


def _params(p):
    return ", ".join(f"{v}: {s}" for v, s in (p or {}).items())


def _call(name, args):
    return f"{name}({', '.join(str(a) for a in args)})" if args else name


def _type(d):
    t = d.get("type", "bool")
    return f"{t}({d['decimals']})" if "decimals" in d else t


def _effect(e):
    tgt = _call(e["fluent"], e.get("args", []))
    return f"{tgt} := {e['set']}" if "set" in e else f"{tgt} += {e['add']}"


def to_text(ast):
    L = [f'SYSTEM "{ast.get("system", "")}" {{ horizon: {ast.get("horizon", 10)} }}', ""]
    if ast.get("sorts"):
        L.append("SORTS {")
        for s in ast["sorts"]:
            body = " | ".join(s["instances"]) if "instances" in s else f"{s['range'][0]}..{s['range'][1]}"
            L.append(f"  {s['name']}: {body}")
        L += ["}", ""]
    if ast.get("statics"):
        L.append("STATICS {")
        for s in ast["statics"]:
            d = f" default {s['default']}" if "default" in s else ""
            L.append(f"  {_call(s['name'], s.get('args', []))} : {_type(s)}{d}")
            for f in s.get("facts", []):
                pr = f" plausible {f['plausible_range'][0]}..{f['plausible_range'][1]}" if f.get("plausible_range") else ""
                L.append(f"    {_call(s['name'], f.get('args', []))} = {f['value']}{pr}{_ann(f)}")
        L += ["}", ""]
    if ast.get("fluents"):
        L.append("FLUENTS {")
        for f in ast["fluents"]:
            r = f" {f['range'][0]}..{f['range'][1]}" if f.get("range") else ""
            d = " DERIVED" + (f" default {f['default']}" if "default" in f else "") if f.get("derived") else ""
            L.append(f"  {_call(f['name'], f.get('args', []))} : {_type(f)}{r}{d}")
        L += ["}", ""]
    if ast.get("initial"):
        L.append("INITIAL {")
        for e in ast["initial"]:
            q = f"FOR {_params(e['params'])}: " if e.get("params") else ""
            L.append(f"  {q}{_call(e['fluent'], e.get('args', []))} = {e['value']}{_ann(e)}")
        L += ["}", ""]
    for a in ast.get("actions", []):
        L.append(f"ACTION {a['name']}({_params(a.get('params'))}){_ann(a)} {{")
        if a.get("pre"):
            L.append(f"  PRE {a['pre']}")
        for e in a.get("effects", []):
            L.append(f"  EFFECT {_effect(e)}")
        L += ["}", ""]
    for law in ast.get("laws", []):
        L.append(f"LAW {law['id']}({_params(law.get('params'))}){_ann({**law, 'id': None})} {{")
        if law.get("when"):
            L.append(f"  WHEN {law['when']}")
        if law.get("unless"):
            L.append(f"  UNLESS {law['unless']}")
        for e in law.get("effects", []):
            L.append(f"  THEN {_effect(e)}" + (f" AFTER {law['delay']}" if law.get("delay", 1) != 1 else ""))
        L += ["}", ""]
    for d in ast.get("derived", []):
        L.append(f"DERIVED {d['id']}({_params(d.get('params'))}){_ann({**d, 'id': None})} {{")
        cond = f" WHEN {d['when']}" if d.get("when") else ""
        unl = f" UNLESS {d['unless']}" if d.get("unless") else ""
        L.append(f"  {_call(d['fluent'], d.get('args', []))} = {d.get('value', 'true')}{cond}{unl}")
        L += ["}", ""]
    if ast.get("constraints"):
        L.append("CHECKLIST {")
        for c in ast["constraints"]:
            kw = c["keyword"]
            q = f"FOR {_params(c['params'])}: " if c.get("params") else ""
            if kw in ("BEFORE", "AFTER"):
                body = f"{c['a']} {kw} {c['b']}"
            elif kw in ("AT_MOST", "AT_LEAST", "EXACTLY"):
                body = f"{kw.replace('_', ' ')} {c['n']} {c['over']}: {c['cond']}"
            elif kw == "ONLY":
                body = f"{c['cond']} ONLY IF {c['if']}"
            else:
                body = f"{kw.replace('_', ' ')} {c['cond']}"
            unl = f" UNLESS {c['unless']}" if c.get("unless") else ""
            L.append(f"  [ ] {c['id']}: {q}{body}{unl}{_ann({**c, 'id': None})}")
        L += ["}", ""]
    for s in ast.get("scenarios", []):
        L.append(f"SCENARIO {s['id']} {{ horizon: {s.get('horizon', ast.get('horizon', 10))}")
        if s.get("description"):
            L.append(f"  // {s['description']}")
        for e in s.get("init", []):
            L.append(f"  INIT {_call(e['fluent'], e.get('args', []))} = {e['value']}")
        for ev in s.get("events", []):
            L.append(f"  EVENT {_call(ev['action'], ev.get('args', []))} @ {ev['time']}")
        for iv in s.get("interventions", []):
            L.append(f"  INTERVENE {json.dumps(iv)}")
        L += ["}", ""]
    if ast.get("claims"):
        L.append("CLAIMS {")
        for c in ast["claims"]:
            ab = c["about"]
            d = c.get("derivation")
            via = f" via {d['rule']}" + (f" from {', '.join(d.get('premises', []))}" if d.get("premises") else "") if d else ""
            src = _ann({"status": None, "source": c.get("source")})
            L.append(f"  {c['id']} [{c['status']}{via}] @ {ab.get('scenario', 'base')} t={ab.get('time', 0)}: "
                     f"{ab['expr']}{src}")
        L += ["}", ""]
    ans = ast.get("answer") or {}
    if ans:
        L.append("ANSWER {")
        for k in ("conclusions", "assumptions", "unexplained"):
            if ans.get(k):
                L.append(f"  {k}: {', '.join(ans[k])}")
        for g in ans.get("conflicts", []) or []:
            L.append(f"  conflict: {', '.join(g)}")
        L.append("}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(to_text(json.load(open(sys.argv[1], encoding="utf-8"))))
