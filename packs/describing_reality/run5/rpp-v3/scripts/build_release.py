"""Build the clean, public release of the run 5 harness (white paper Appendix B) into the release repository.

    python scripts/build_release.py [--dest ../describing-reality-for-ai/harness] [--with-items] [--replace]

Copies a snapshot (no git history) of what runs run 5: the v3 code, tests, sandbox, config, PREREG, DECISIONS, the run 5
design and build plan, the launcher, the prereg lock, and the read-only rpp-core files run 5 actually opens
(`scripts/release_rppcore_files.txt`, recorded with an audit hook while the test suite ran). Never copied: git
history, runs/ and state/ (except the lock), credentials, usage logs, rpp-core data (held-out sets) and, unless
--with-items, the 500 hand-written items. Fails if a file looks like it holds a secret. Writes MANIFEST.json (sha256
of every file, the source commits) and README.md. Publishing is the human's call; this only builds the folder.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE = (ROOT / ".." / "rpp-core").resolve()
RPPCORE_LIST = ROOT / "scripts" / "release_rppcore_files.txt"

V3_FILES = ["config/v3.json", "PREREG.md", "DECISIONS.md", "docs/RUN5_DESIGN.md", "docs/RUN5_BUILD_PLAN.md",
            "docs/RUN5_STAGES_DESIGN.md", "items/run5_stages.json", "scripts/export_results5.py",
            "docs/CODEX_BUILDER.md", "scripts/run_v3_loop.cmd", "scripts/check_item_ranges.py",
            "scripts/hand5_spotcheck.py", "scripts/build_release.py", "scripts/release_rppcore_files.txt"]
V3_GLOBS = ["v3/**/*.py", "v3/genome/seeds/*.json", "v3/dashboard5/index.html", "sandbox/*", "tests/**/*.py",
            "tests/**/*.json"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", "dashboard"}          # v3/dashboard is the run 1-4 dashboard
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|sk-or-v1-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|"
                       r"-----BEGIN [A-Z ]*PRIVATE KEY-----|(?i:(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]))")


def git_head(path):
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def v3_sources(with_items):
    out = {ROOT / f for f in V3_FILES}
    for pat in V3_GLOBS:
        out |= {p for p in ROOT.glob(pat) if p.is_file()}
    if with_items:
        out |= {ROOT / "items" / "hand5" / "items.json", ROOT / "items" / "hand5" / "pilot_items.json"}
    return sorted(p for p in out if p.exists() and not (SKIP_PARTS & set(p.relative_to(ROOT).parts)))


def core_sources():
    rel = [ln.strip() for ln in RPPCORE_LIST.read_text(encoding="utf-8").splitlines()
           if ln.strip() and not ln.startswith("#")]
    # the A5 founder's rule language (tournament/arms/a5_logic.py RUNTIME_DEPS): its checker and compiler code and
    # its schema only; the language folder's examples and tests stay out
    a5 = sorted(p.relative_to(CORE).as_posix() for pat in ("candidates/gen1/2/checker/*.py",
                                                            "candidates/gen1/2/compiler/*.py")
                for p in CORE.glob(pat)) + ["candidates/gen1/2/lang/schema.json", "tournament/arms/_rpp_exec.py"]
    for r in rel:
        parts = pathlib.PurePosixPath(r).parts
        if parts[0] in {"data", "runs", "jobs", "candidates", "diagnostics"} or "gold" in parts:
            raise SystemExit(f"refusing rpp-core data path {r} (held-out sets and records never leave)")
    return [(CORE / r) for r in sorted(set(rel) | set(a5))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT.parent / "describing-reality-for-ai" / "harness"))
    ap.add_argument("--with-items", action="store_true", help="include the 500 hand-written items (default: held back)")
    ap.add_argument("--replace", action="store_true", help="replace an earlier build (files it listed are removed)")
    a = ap.parse_args(argv)
    dest = pathlib.Path(a.dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        old = dest / "MANIFEST.json"
        if not (a.replace and old.exists()):
            raise SystemExit(f"{dest} is not empty; pass --replace to rebuild over an earlier build")
        for rel in json.loads(old.read_text(encoding="utf-8"))["files"]:
            (dest / rel).unlink(missing_ok=True)
    plan = [(p, "rpp-v3/" + p.relative_to(ROOT).as_posix()) for p in v3_sources(a.with_items)]
    plan += [(p, "rpp-core/" + p.relative_to(CORE).as_posix()) for p in core_sources()]
    plan.append((ROOT / "state" / "PREREG.LOCK", "rpp-v3/state/PREREG.LOCK"))
    missing = [str(s) for s, _ in plan if not s.is_file()]
    if missing:
        raise SystemExit(f"missing sources: {missing[:10]}")
    hits = []
    for src, rel in plan:
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in SECRET_RE.finditer(text):
            hits.append(f"{rel}: {m.group(0)[:12]}...")
    if hits:
        raise SystemExit("possible secrets; nothing copied:\n" + "\n".join(hits[:20]))
    files = {}
    for src, rel in plan:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        files[rel] = hashlib.sha256(out.read_bytes()).hexdigest()
    lock = json.loads((ROOT / "state" / "PREREG.LOCK").read_text(encoding="utf-8"))
    manifest = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "rpp_v3_commit": git_head(ROOT),
                "rpp_core_commit": git_head(CORE), "prereg_lock": {"t": lock["t"], "reason": lock["reason"]},
                "hand_written_items_included": a.with_items, "files": files}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.copy2(ROOT / "docs" / "RELEASE_README.md", dest / "README.md")
    print(f"built {len(files)} files into {dest} ({sum((dest / r).stat().st_size for r in files) / 1e6:.1f} MB); "
          f"secrets scan clean; hand-written items {'included' if a.with_items else 'held back'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
