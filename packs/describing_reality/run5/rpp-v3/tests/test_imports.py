import importlib
import pathlib

import pytest

MODS = sorted(".".join(p.with_suffix("").parts) for p in pathlib.Path("v3").rglob("*.py")
              if "__pycache__" not in p.parts and p.name != "__init__.py")


@pytest.mark.parametrize("mod", MODS)
def test_module_imports(mod):
    importlib.import_module(mod)
