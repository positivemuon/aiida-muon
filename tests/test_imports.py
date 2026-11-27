# -*- coding: utf-8 -*-
"""aiida-muon does not depend on `muesr` and `mulfc` any more (the dipolar field uses `pylocfield`)."""
import subprocess
import sys

MODULES = [
    "aiida_muon.utils.clustering",
    "aiida_muon.utils.magnetism",
    "aiida_muon.utils.export_findmuon",
    "aiida_muon.workflows.utils",
    "aiida_muon.workflows.find_muon",
]


def test_modules_import_without_muesr_and_mulfc():
    """The modules are imported in a fresh interpreter in which `import muesr` / `import mulfc` fail."""
    code = (
        "import importlib, sys\n"
        "sys.modules['muesr'] = None\n"
        "sys.modules['mulfc'] = None\n"
        f"for name in {MODULES!r}:\n"
        "    importlib.import_module(name)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

