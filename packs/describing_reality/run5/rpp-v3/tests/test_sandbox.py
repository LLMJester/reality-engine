"""Attack tests for the run 5 Docker sandbox (V3-043). Every attack must fail.

Two layers are tested: `full` runs code exactly as the cores do (container + audit-hook runner); `wall` runs it in
the same container with the audit hook bypassed, to prove the container alone holds. Skipped if Docker is not up.
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v3 import sandbox  # noqa: E402

UP = sandbox.available()


def full(code, timeout=30):
    return sandbox.run(code, {}, timeout)


def wall(code, timeout=30):
    """Same container and flags, the audit hook bypassed (python -c directly)."""
    name = "rpp5-test-" + str(time.time_ns())
    cmd = ["docker", "run", "--name", name, *sandbox.FLAGS, "--entrypoint", "python", sandbox.IMAGE, "-c", code]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr, False, r.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return "", "", True, None
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@unittest.skipUnless(UP, "Docker sandbox image not available")
class Attacks(unittest.TestCase):
    def test_normal_code_runs(self):
        out, err, to, rc = full("import numpy as np\nprint('ok', int(np.arange(5).sum()))")
        self.assertEqual((out.strip(), to, rc), ("ok 10", False, 0), err)

    def test_network(self):
        code = ("import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3); print('CONNECTED')\n"
                "except Exception as e:\n    print('blocked', type(e).__name__)")
        for f in (full, wall):
            out = f(code)[0]
            self.assertNotIn("CONNECTED", out)
        out = wall("import socket\ntry:\n    socket.getaddrinfo('example.com', 80); print('RESOLVED')\n"
                   "except Exception as e:\n    print('blocked')")[0]
        self.assertNotIn("RESOLVED", out)

    def test_host_files_unreachable(self):
        with tempfile.TemporaryDirectory() as d:
            secret = Path(d, "planted_secret.txt")
            secret.write_text("TOP SECRET", encoding="utf-8")
            code = ("import os\nhits=[]\nfor root, dirs, files in os.walk('/'):\n"
                    "    if root.startswith(('/proc','/sys')): dirs[:]=[]; continue\n"
                    "    hits += [f for f in files if f == 'planted_secret.txt']\nprint('FOUND' if hits else 'none')")
            self.assertIn("none", wall(code, timeout=60)[0])
        out = full("print(open('/etc/passwd').read()[:20])")
        self.assertIn("PermissionError", out[1])          # the audit hook refuses reads outside the folder

    def test_cannot_write_outside_scratch(self):
        code = ("import os\nfor p in ['/opt/sandbox/x', '/x', '/usr/x', '/home/x']:\n    try:\n"
                "        open(p, 'w').write('x'); print('WROTE', p)\n    except Exception:\n        pass\nprint('done')")
        for f in (full, wall):
            out = f(code)[0]
            self.assertNotIn("WROTE", out)
        self.assertIn("PermissionError", full("open('/tmp/x','w').write('x')")[1])

    def test_no_programs(self):
        out, err, _, _ = full("import subprocess\nsubprocess.run(['ls','/'])")
        self.assertIn("PermissionError", err)
        out, err, _, _ = full("import os\nos.system('ls /')")
        self.assertIn("PermissionError", err)

    def test_not_root_no_privileges(self):
        out = wall("import os\nprint(os.getuid())\ntry:\n    os.setuid(0); print('ROOT')\nexcept Exception:\n"
                   "    print('no')")[0]
        self.assertNotIn("ROOT", out)
        self.assertTrue(out.startswith("1000"), out)

    def test_fork_bomb_is_contained(self):
        code = ("import os\nn=0\ntry:\n    while True:\n        if os.fork()==0:\n            import time; "
                "time.sleep(20); os._exit(0)\n        n+=1\nexcept Exception as e:\n    print('stopped at', n)")
        t = time.time()
        out, err, to, rc = wall(code, timeout=60)
        self.assertLess(time.time() - t, 61)
        if not to:
            n = int(out.split()[-1]) if "stopped at" in out else 0
            self.assertLessEqual(n, 128)

    def test_memory_limit(self):
        code = "x = bytearray(10 * 1024**3)\nprint('ALLOCATED')"
        for f in (full, wall):
            out, err, to, rc = f(code, timeout=60)
            self.assertNotIn("ALLOCATED", out)
        code = "xs=[]\nfor i in range(100):\n    xs.append(bytearray(100*1024**2))\nprint('ALLOCATED')"
        out, err, to, rc = wall(code, timeout=60)
        self.assertNotIn("ALLOCATED", out)

    def test_endless_loop_is_killed(self):
        t = time.time()
        out, err, to, rc = full("while True:\n    pass", timeout=5)
        self.assertTrue(to)
        self.assertLess(time.time() - t, 40)

    def test_disk_filling_stops(self):
        code = ("try:\n    with open('/scratch/big','wb') as f:\n        for i in range(200):\n"
                "            f.write(b'x'*1024*1024)\n    print('WROTE 200MB')\nexcept Exception as e:\n"
                "    print('stopped', type(e).__name__)")
        for f in (full, wall):
            out, err, to, rc = f(code, timeout=60)
            self.assertNotIn("WROTE 200MB", out)

    def test_no_secrets_in_environment(self):
        os.environ["RPP_TEST_SECRET_KEY"] = "should-not-pass"
        out = wall("import os\nprint(sorted(os.environ))")[0]
        self.assertNotIn("RPP_TEST_SECRET_KEY", out)
        # only the image's own variables (GPG_KEY is the public id of the key that signs Python releases)
        image_vars = {"GPG_KEY", "HOME", "HOSTNAME", "LANG", "PATH", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING",
                      "PYTHON_SHA256", "PYTHON_VERSION"}
        self.assertLessEqual(set(eval(out)), image_vars)

    def test_output_flood_is_capped(self):
        out, err, to, rc = full("import sys\nwhile True:\n    sys.stdout.write('x' * 65536)", timeout=60)
        self.assertLessEqual(len(out), sandbox.MAX_OUT)
        self.assertIn("output limit", err)

    def test_docker_hanging_during_cleanup_is_not_a_tool_error(self):
        """Sixth review F1: docker kill / rm hanging must not raise out of run()."""
        import subprocess as sp
        real = sandbox._docker

        def flaky(args, **kw):
            if args and args[0] in ("kill", "rm"):
                real(args, **kw)                         # the container really goes; then Docker "hangs"
                raise sp.TimeoutExpired(args, 30)
            return real(args, **kw)
        sandbox._docker = flaky
        try:
            out, err, to, rc = sandbox.run("while True:\n    pass", {}, 3)
            self.assertTrue(to)
        finally:
            sandbox._docker = real

    def test_daemon_gone_after_a_timeout_is_an_outage(self):
        """Sixth review F1: a container that timed out because Docker died is an outage, not a model timeout."""
        calls = []
        real = sandbox.available

        def first_up_then_down():
            calls.append(1)
            return len(calls) == 1
        sandbox.available = first_up_then_down
        try:
            with self.assertRaises(sandbox.SandboxUnavailable):
                sandbox.run("while True:\n    pass", {}, 3)
        finally:
            sandbox.available = real

    def test_containers_are_removed(self):
        full("print(1)")
        # containers that have stopped but were left behind (running ones may belong to other jobs, e.g. item writers)
        r = subprocess.run(["docker", "ps", "-a", "--filter", "name=rpp5-", "--filter", "status=exited",
                            "--format", "{{.Names}}"], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "")

    def test_twenty_at_once(self):
        t = time.time()
        with ThreadPoolExecutor(20) as pool:
            res = list(pool.map(lambda i: full(f"print({i}*2)"), range(20)))
        self.assertEqual([r[0].strip() for r in res], [str(i * 2) for i in range(20)])
        print(f"\n20 concurrent sandbox calls: {time.time() - t:.1f}s", file=sys.stderr)


@unittest.skipUnless(UP, "Docker sandbox image not available")
class Cores(unittest.TestCase):
    def test_install_routes_v2_cores_through_docker(self):
        import v3.genome.render  # noqa: F401  (puts rpp-core on sys.path)
        sandbox.install()
        self.assertTrue(sandbox.installed())
        from tournament.arms import a3_sim
        out, err, to, rc = a3_sim.run_in_sandbox({}, 30)
        self.assertFalse(to)


if __name__ == "__main__":
    unittest.main()
