# -*- coding: utf-8 -*-
"""aiida-muon does not depend on `muesr` and `mulfc` any more: nothing but the dipolar field needs them."""
import subprocess
import sys

import pytest

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


def test_dipolar_field_needs_muesr(monkeypatch):
    """Without muesr, computing the dipolar field fails with a message that says what to do."""
    from aiida_muon.utils.magnetism import compute_dip_field

    monkeypatch.setitem(sys.modules, "muesr", None)
    with pytest.raises(ImportError, match="muesr"):
        compute_dip_field(None, None, None, None, None)
