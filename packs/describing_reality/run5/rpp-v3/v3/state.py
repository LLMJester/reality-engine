"""Single JSON state store with atomic writes and a single-writer lock (v2 lesson: two writers race on state.json)."""
import json
import os
import pathlib
import time

from v3.config import STATE_DIR


class LockHeld(RuntimeError):
    pass


class Store:
    def __init__(self, path=None):
        self.path = pathlib.Path(path or STATE_DIR / "state.json")
        self.lock = self.path.with_suffix(".lock")
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "done": {}, "events": [], "spend": {},
            "generation": 0}

    def acquire(self):
        """Take the writer lock; refuse if another live process holds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock.exists():
            try:
                pid = int(self.lock.read_text().strip())
            except ValueError:
                pid = 0
            if pid and pid != os.getpid() and _alive(pid):
                raise LockHeld(f"state is held by live process {pid}")
        self.lock.write_text(str(os.getpid()))
        return self

    def release(self):
        if self.lock.exists() and self.lock.read_text().strip() == str(os.getpid()):
            self.lock.unlink()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        # V3-024: on Windows the replace fails with "Access is denied" while another process (dashboard, OneDrive,
        # antivirus) has state.json open for a moment; retry for about 2 s before giving up
        for attempt in range(6):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.1 * 2 ** attempt / 2)

    def event(self, msg, level="info"):
        self.data["events"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "level": level, "msg": msg})
        self.data["events"] = self.data["events"][-2000:]


def _alive(pid):
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
