"""Arm registry with lazy imports.

ARMS maps an arm name to "module:Class" (a bare module name is relative to this package; a dotted
name is imported as given). Arms built later (A2, A3, A4, A6) plug in by existing as
tournament/arms/<module>.py with that class; nothing else needs to change. `get_arm(name)` raises
ArmUnavailable if the module or class does not exist (yet).

Per-arm registration files (review fix 5): an evolved arm registers itself by writing
tournament/arms/registry.d/<ID>.json = {"module": "<module>", "class": "<Class>"} (see `register_arm` /
`python -m tournament.arms.register`). One file per arm, written atomically, so builders running in parallel
never lose each other's registration (editing this file concurrently could). The files are merged into ARMS at
import time, and `registry()` re-reads both this file's ARMS literal and registry.d/ from disk, so a long-running
process (the orchestrator) sees arms registered after it imported this package. Static entries below win over a
registry.d file of the same name.
"""
import ast
import importlib
import json
import os
import pathlib
import re

ARMS = {
    "A0": "a0_nl:A0",
    "A0+": "a0_nl:A0Plus",
    "A1": "a1_prose:A1",
    "A2": "a2_causal:CausalArm",
    "A3": "a3_sim:SimArm",
    "A4": "a4_prob:ProbArm",
    "A5": "a5_logic:A5",
    "A6": "a6_hybrid:A6HybridArm",
    "A7": "a7_select:A7",
    "A8": "a8_spec:SpecArm",
    "C2": "c2_tools:C2",
}
BUILDABLE = ["A1", "A2", "A3", "A4", "A5", "A6", "A8"]   # the menu A7 chooses from

HERE = pathlib.Path(__file__).resolve().parent
REGISTRY_D = HERE / "registry.d"
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+_-]{0,31}$")
_TARGET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


class ArmUnavailable(ImportError):
    pass


_cache = {}


def _registry_d(dirpath=None):
    """{arm id: "module:Class"} from registry.d/*.json; malformed files are skipped (reported by registry_problems)."""
    out = {}
    d = pathlib.Path(dirpath or REGISTRY_D)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            x = json.loads(p.read_text(encoding="utf-8-sig"))
            target = f"{x['module']}:{x['class']}"
        except Exception:
            continue
        if _ID_RE.match(p.stem) and _TARGET_RE.match(target):
            out[p.stem] = target
    return out


def _static_from_file(path=None):
    """The ARMS literal of this file as it is on disk now (agents may have added lines after import)."""
    try:
        tree = ast.parse(pathlib.Path(path or __file__).read_text(encoding="utf-8"))
    except Exception:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ARMS" for t in node.targets):
            try:
                v = ast.literal_eval(node.value)
                return {str(k): str(x) for k, x in v.items()} if isinstance(v, dict) else {}
            except Exception:
                return {}
    return {}


def registry(refresh=True):
    """Fresh registry: {arm id: "module:Class"} = this file's ARMS literal (re-parsed from disk) + registry.d/*.json.
    With refresh=True the module-level ARMS is updated in place (entries are added or changed, never removed), so
    get_arm() sees arms registered after import."""
    fresh = dict(_registry_d())
    fresh.update(_static_from_file() or {})
    if refresh:
        for k, v in fresh.items():
            if ARMS.get(k) != v:
                ARMS[k] = v
                _cache.pop(k, None)
        return dict(ARMS)
    out = dict(ARMS)
    out.update(fresh)
    return out


def register_arm(arm_id, module, cls, dirpath=None):
    """Register an arm via its own registry.d file (atomic write; safe with concurrent builders)."""
    if not _ID_RE.match(arm_id) or not _TARGET_RE.match(f"{module}:{cls}"):
        raise ValueError(f"bad registration {arm_id!r} -> {module}:{cls}")
    d = pathlib.Path(dirpath or REGISTRY_D)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{arm_id}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"module": module, "class": cls}, indent=1), encoding="utf-8")
    os.replace(tmp, d / f"{arm_id}.json")
    return d / f"{arm_id}.json"


def registry_files():
    """Files that define the registry (for hash locks)."""
    return [pathlib.Path(__file__).resolve()] + (sorted(REGISTRY_D.glob("*.json")) if REGISTRY_D.is_dir() else [])


for _k, _v in _registry_d().items():   # merge per-arm registrations at import time (static entries win)
    ARMS.setdefault(_k, _v)


def _canon(name):
    """Case-insensitive arm names (gold folders are lowercase)."""
    for k in ARMS:
        if k.lower() == str(name).lower():
            return k
    return name


def get_arm(name):
    name = _canon(name)
    if name not in ARMS:
        registry()              # registered on disk after import?
        name = _canon(name)
    if name not in ARMS:
        raise ArmUnavailable(f"unknown arm {name!r}")
    if name in _cache:
        return _cache[name]
    mod, cls = ARMS[name].split(":")
    try:
        m = importlib.import_module(mod if "." in mod else f"{__name__}.{mod}")
        klass = getattr(m, cls, None)
        if klass is None:   # fall back to the module's Arm subclass whose `name` is this arm
            from tournament.arms.base import Arm
            klass = next((v for v in vars(m).values() if isinstance(v, type) and issubclass(v, Arm)
                          and v is not Arm and getattr(v, "name", None) == name), None)
        if klass is None:
            raise AttributeError(f"no Arm class named {cls!r} or with name {name!r} in {mod}")
        arm = klass()
    except (ImportError, AttributeError) as e:
        raise ArmUnavailable(f"arm {name} is not available: {e}") from e
    _cache[name] = arm
    return arm


def available(names=None):
    out = []
    for n in names or list(ARMS):
        try:
            get_arm(n)
            out.append(n)
        except ArmUnavailable:
            pass
    return out
