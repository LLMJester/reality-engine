"""Docker sandbox for model-written code (V3-043, run 5).

The v2 own-code cores (A3 simulation: tournament/arms/a3_sim.py; G1C1 probabilistic: tournament/arms/g1c1.py) run the
model's program in a host subprocess under an audit hook. For Luna in run 5 every such program runs instead in a
throwaway container: no network, read-only file system, a 64 MB tmpfs scratch folder, 1 GB memory, two CPUs, 128
processes (sized for the workstation: 20 cores; the WSL VM has 22 GB, so 20 containers fit at once), a non-root user, no capabilities or added privileges, removed after the call. The audit-hook runner still
runs inside the container as a second layer. Nothing from the host is mounted: the files go in on stdin.

`install()` swaps the two v2 functions that start the subprocess (a3_sim.run_in_sandbox, g1c1._run_driver) for
Docker-backed ones with the same signature, in this process only; rpp-core itself is not modified. If Docker is not
available the call raises SandboxUnavailable: model code never falls back to running on the host.
"""
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path

IMAGE = "rpp-sandbox:run5"
CONTEXT = Path(__file__).resolve().parent.parent / "sandbox"
MAX_OUT = 64_000
FLAGS = ["--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/scratch:rw,size=64m,mode=1777",
         "--tmpfs", "/tmp:rw,size=16m,mode=1777", "--memory", "1g", "--memory-swap", "1g", "--cpus", "2",
         "--pids-limit", "128", "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
         "--ulimit", "fsize=33554432", "--ulimit", "nofile=256"]


class SandboxUnavailable(RuntimeError):
    pass


# set when the sandbox could not run a program in this process (code review): the MCP server writes it to its state
# file and the Codex builder turns it into an infrastructure error, whoever caught the exception in between
UNAVAILABLE = []


def _docker(args, **kw):
    return subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def available():
    try:
        return _docker(["image", "inspect", IMAGE, "--format", "{{.Id}}"], timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def build():
    r = _docker(["build", "-t", IMAGE, str(CONTEXT)], timeout=900)
    if r.returncode:
        raise SandboxUnavailable("docker build failed:\n" + r.stderr[-2000:])
    return r


def command(name):
    return ["docker", "run", "--name", name, *FLAGS, IMAGE]


def run(driver, files, timeout):
    """Run `driver` as main.py with `files` beside it in a fresh container.
    Returns (stdout, stderr, timed_out, returncode), like the v2 functions it replaces."""
    if not available():
        msg = f"Docker image {IMAGE} is not available (is Docker Desktop running?)"
        UNAVAILABLE.append(msg)
        raise SandboxUnavailable(msg)
    job = json.dumps({"files": {**files, "main.py": driver}}).encode("utf-8")
    name = "rpp5-" + uuid.uuid4().hex[:12]
    p = subprocess.Popen(command(name), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bufs, flooded = {"out": bytearray(), "err": bytearray()}, threading.Event()

    def reader(stream, key):                             # code review: never hold more than MAX_OUT x 2 per stream
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            b = bufs[key]
            if len(b) < 2 * MAX_OUT:
                b.extend(chunk[: 2 * MAX_OUT - len(b)])
            else:
                flooded.set()
    threads = [threading.Thread(target=reader, args=(p.stdout, "out"), daemon=True),
               threading.Thread(target=reader, args=(p.stderr, "err"), daemon=True)]
    for t in threads:
        t.start()
    timed_out = False
    try:
        p.stdin.write(job)
        p.stdin.close()
    except OSError:
        pass
    deadline = time.time() + timeout
    while p.poll() is None:
        if time.time() > deadline or flooded.is_set():
            timed_out = time.time() > deadline
            try:
                _docker(["kill", name], timeout=30)
            except (OSError, subprocess.SubprocessError):   # Docker hanging must not escape as a tool error
                pass
            p.kill()
            break
        time.sleep(0.05)
    p.wait()
    for t in threads:
        t.join(timeout=5)
    try:
        _docker(["rm", "-f", name], timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass
    out = bufs["out"].decode("utf-8", "replace")[:MAX_OUT]
    err = bufs["err"].decode("utf-8", "replace")[-MAX_OUT:]
    if p.returncode == 125 or (p.returncode not in (0, None) and not available()):   # a timeout too (review 6)
        msg = f"docker run failed (rc={p.returncode}): {err[-300:]}"   # the container never ran (code review)
        UNAVAILABLE.append(msg)
        raise SandboxUnavailable(msg)
    if flooded.is_set():
        err += "\n[sandbox: output limit reached; the program was stopped]"
    return out, err, timed_out, p.returncode


def install():
    """Route the v2 own-code cores through Docker in this process. Idempotent."""
    from tournament.arms import a3_sim, g1c1
    if getattr(a3_sim, "_v3_docker", False):
        return
    a3_sim.run_in_sandbox = lambda files, timeout: run(a3_sim.DRIVER, files, timeout)
    g1c1._run_driver = lambda files, timeout: run(g1c1.DRIVER, files, timeout)
    a3_sim._v3_docker = g1c1._v3_docker = True


def installed():
    from tournament.arms import a3_sim, g1c1
    return bool(getattr(a3_sim, "_v3_docker", False) and getattr(g1c1, "_v3_docker", False))
