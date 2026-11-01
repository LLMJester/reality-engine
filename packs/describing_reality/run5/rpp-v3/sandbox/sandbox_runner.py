"""Executes untrusted model-written code (condition C2) with an audit hook that
blocks network, process spawning, and file access outside the sandbox dir and
the Python installation. Usage: python -I sandbox_runner.py <workdir> <codefile>
"""
import os
import sys

WORK = os.path.realpath(sys.argv[1])
CODE = os.path.realpath(sys.argv[2])
ALLOWED_READ = [os.path.realpath(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix} if p] + [WORK]
for p in list(sys.path):
    if p and os.path.isdir(p):
        ALLOWED_READ.append(os.path.realpath(p))
# ctypes.dlopen is allowed: numpy/clingo/z3 need it to load (blocking it broke C2's tools; DECISIONS D-021).
BLOCKED = ("socket.connect", "socket.bind", "subprocess.Popen", "os.system", "os.exec", "os.spawn",
           "os.posix_spawn", "os.startfile", "os.kill", "os.remove", "os.rmdir", "shutil.rmtree",
           "os.rename", "winreg")


def _inside(path, roots):
    try:
        rp = os.path.realpath(path)
    except Exception:
        return False
    return any(rp == r or rp.startswith(r + os.sep) for r in roots)


def hook(event, args):
    if event.startswith(BLOCKED):
        raise PermissionError(f"sandbox: {event} is not allowed")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
        writing = any(c in mode for c in "wax+")
        if writing and not _inside(path, [WORK]):
            raise PermissionError("sandbox: writes allowed only in the working directory")
        if not writing and not _inside(path, ALLOWED_READ):
            raise PermissionError("sandbox: reading outside the sandbox is not allowed")


def _watchdog(limit_bytes=1_500_000_000):
    """Kill this sandboxed process if it uses more than ~1.5 GB (runaway model code; DECISIONS D-023)."""
    import threading
    import time
    try:
        import psutil
    except ImportError:
        return
    me = psutil.Process()

    def loop():
        while True:
            try:
                if me.memory_info().rss > limit_bytes:
                    sys.stderr.write("\n[sandbox: memory limit 1.5 GB exceeded; process stopped]\n")
                    sys.stderr.flush()
                    os._exit(137)
            except Exception:
                pass
            time.sleep(0.5)
    threading.Thread(target=loop, daemon=True).start()


_watchdog()
os.chdir(WORK)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
src = open(CODE, encoding="utf-8").read()
sys.addaudithook(hook)
sys.argv = ["main.py"]
exec(compile(src, "main.py", "exec"), {"__name__": "__main__"})
