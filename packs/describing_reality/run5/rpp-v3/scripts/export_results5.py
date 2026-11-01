"""Export run 5's results for the public release (white paper Appendix B) into the release repository.

    python scripts/export_results5.py [--dest ../describing-reality-for-ai/results]

Writes, from the run's records and state (read only):
  episodes.csv     one row per scored test run: stage, design, genome, item id, system size, item set, score, whether
                   the notes passed their check, output tokens (writing and answering), seconds
  designs.json     every design's genome by stage and label (founders, the 32 stage 2 mixes, stage 3's screen and
                   confirmation designs)
  new_parts/       what the designer wrote and was given: stage 2's five parts with reasons, stage 3's 20 versions,
                   and the firewalled evidence briefs
  analysis.json    the run's analysis: every pre-registered test, effect, range and verdict
  events.json      the run's event log
Never written: item text, questions, answer keys, notes, transcripts, credentials, or anything from the retired sealed
set (V3-058: its records are not opened; only their count is given). Publishing is the human's call.
"""
import argparse
import csv
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from v3 import run5 as R                      # noqa: E402
from v3.analysis5 import k_of                 # noqa: E402
from v3.run5_sets import retired_items        # noqa: E402

RUNS = ROOT / "runs"


def item_set(mode, iid, sets):
    for name, items in sets.items():
        if iid in items:
            return name
    return "other"


def rows_for(mode, stage, phase, designs, sets, retired):
    """One row per scored run of these designs in this phase; retired items are skipped before any file is opened."""
    out = []
    for label, g in designs.items():
        d = RUNS / phase / R.BUILDER / g["id"]
        if not d.exists():
            continue
        for item_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            iid = item_dir.name
            if iid in retired:
                continue
            s = R.score(mode, g["id"], iid, phase)
            if s is None:
                continue
            r = R.record(mode, g["id"], iid, phase) or {}
            build = r.get("build") or {}
            val = r.get("validity") or build.get("validity") or {}
            out.append({"stage": stage, "design": label, "genome": g["id"], "item": iid, "k": k_of(iid) or "",
                        "item_set": item_set(mode, iid, sets), "score": round(float(s), 4),
                        "notes_passed_check": "" if not val else bool(val.get("usable") and val.get("executable")),
                        "zero_scored_failure": (item_dir / "zero.json").exists(),
                        "output_tokens": r.get("output_tokens") or "", "writing_tokens": build.get("output_tokens") or "",
                        "answering_tokens": r.get("answer_output_tokens") or "",
                        "seconds": round(r.get("wall_s_total") or r.get("wall_s") or 0, 1) or ""})
    return out


def main(dest):
    mode = R.Mode("real")
    st = json.loads((mode.dir / "state.json").read_text(encoding="utf-8"))
    retired = retired_items()
    sets = {"generated": set(mode.gen), "hand_written": set(mode.hand), "sealed_final": set(mode.final_sealed)}
    for p in ("screen", "confirm"):
        sets[f"stage3_{p}"] = set(mode.stage_items(3, p))
    founders = R.founder_genomes(mode.founders)
    s2 = R.stage2_designs(mode, st)
    s3a = R.screen_designs(mode, st, 3)
    s3b = R.confirm_designs(mode, st, 3)[0]
    dest.mkdir(parents=True, exist_ok=True)
    rows = (rows_for(mode, "stage1+final", mode.phase, founders, sets, retired)
            + rows_for(mode, "stage2", mode.phase2, s2, sets, retired)
            + rows_for(mode, "stage3_screen", mode.stage_phase(3, "screen"), s3a, sets, retired)
            + rows_for(mode, "stage3_confirm", mode.stage_phase(3, "confirm"), s3b, sets, retired))
    for r in rows:                                   # the final check's runs sit in stage 1's folder, on their own set
        if r["stage"] == "stage1+final":
            r["stage"] = "final_check" if r["item_set"] == "sealed_final" else "stage1"
    with open(dest / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    designs = {"stage1_founders": founders, "stage2": s2, "stage3_screen": s3a, "stage3_confirm": s3b}
    (dest / "designs.json").write_text(json.dumps(designs, indent=1, ensure_ascii=False), encoding="utf-8")
    parts = dest / "new_parts"
    parts.mkdir(exist_ok=True)
    shutil.copy2(mode.dir / "plus.json", parts / "stage2_new_parts.json")
    shutil.copy2(mode.dir / "plus_evidence.json", parts / "stage2_evidence_brief.json")
    shutil.copy2(mode.dir / "stage3" / "candidates.json", parts / "stage3_new_versions.json")
    shutil.copy2(mode.dir / "stage3" / "evidence.json", parts / "stage3_evidence_brief.json")
    shutil.copy2(RUNS / mode.phase / "analysis.json", dest / "analysis.json")
    (dest / "events.json").write_text(json.dumps(st.get("events") or [], indent=1, ensure_ascii=False), encoding="utf-8")
    man = json.loads((mode.dir / "retired_V3_058.json").read_text(encoding="utf-8"))   # V3-058 manifest: counts only
    n_ret = man["records"]
    counts = {}
    for r in rows:
        counts[r["stage"]] = counts.get(r["stage"], 0) + 1
    (dest / "COUNTS.json").write_text(json.dumps({"episodes": counts, "total": len(rows),
                                                  "retired_sealed_set_runs_not_exported": n_ret}, indent=1),
                                      encoding="utf-8")
    print(json.dumps(counts), "total", len(rows), "retired not exported", n_ret)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT.parent / "describing-reality-for-ai" / "results"))
    main(pathlib.Path(ap.parse_args().dest).resolve())
