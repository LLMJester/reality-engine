"""Container entry point: read {"files": {name: text}} as JSON on stdin, write the files into /scratch (a tmpfs),
then run main.py under the v2 audit-hook runner. Nothing from the host is mounted."""
import json
import os
import sys

job = json.loads(sys.stdin.read())
for name, text in job["files"].items():
    if os.path.basename(name) != name:
        sys.exit("sandbox: bad file name")
    with open(os.path.join("/scratch", name), "w", encoding="utf-8") as f:
        f.write(text)
sys.stdout.flush()
os.execv(sys.executable, [sys.executable, "-I", "/opt/sandbox/sandbox_runner.py", "/scratch", "/scratch/main.py"])
