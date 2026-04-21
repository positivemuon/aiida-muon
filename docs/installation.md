# Installation

## Prerequisites

Before installing `aiida-muon`, make sure you have:

- Python **3.9** or later
- A working **AiiDA** installation (≥ 2.0) with a configured profile
- A configured **Quantum ESPRESSO** `pw.x` code in AiiDA (for DFT calculations)

If AiiDA is not yet set up on your system, follow the
[AiiDA core installation guide](https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/get_started.html) first.

---

## Install from source

```bash
git clone https://github.com/positivemuon/aiida-muon.git
cd aiida-muon
pip install -e .
```

This installs the core package together with its required dependencies:

| Dependency | Purpose |
|---|---|
| `aiida-core >=2.0` | AiiDA engine and ORM |
| `aiida-quantumespresso >=4.2` | QE plugin (PwRelaxWorkChain, etc.) |
| `aiida-impuritysupercellconv` | Automated supercell size convergence |
| `aiida-qe-restart` | Robust QE restart handling |
| `aiida-monitor` | Optional job monitoring |
| `pymatgen` | Structure manipulation and symmetry analysis |
| `muesr` / `muLFC` | Dipolar field computation |

!!! note "Quantum ESPRESSO version"
    `aiida-muon` requires Quantum ESPRESSO **≥ 7.1** because it depends on the
    updated Hubbard input-card format introduced in that version.

---

## Optional: machine-learning features

The experimental MLIP pre-relaxation and active-learning modules require
additional packages **not** installed by default:

```bash
# For MLIP pre-relaxation via aiida-pythonjob
pip install git+https://github.com/mikibonacci/aiida-pythonjob@fix_serializer
pip install numpy==2

# Choose one or more MLIP backends
pip install mace-torch       # MACE
pip install chgnet            # CHGNet
pip install mattersim         # MatterSim
```

!!! warning "Machine-learning features are experimental"
    The MLIP pre-relaxation (`ML_pre_relax`) and the active-learning workflow
    (`ActiveLearningWorkChain`) are **experimental** and subject to change without
    notice.  Expect rough edges and consult the
    [advanced topics](advanced/ml_features.md) for details before using them in
    production runs.

---

## Install pseudopotential families

The workflow requires a pseudopotential family. The default is `SSSP/1.3/PBE/efficiency`:

```bash
aiida-pseudo install sssp -v 1.3 -x PBE -p efficiency
```

---

## Verify the installation

```python
from aiida import load_profile
load_profile()

from aiida.plugins import WorkflowFactory
FindMuonWorkChain = WorkflowFactory('muon.find_muon')
print(FindMuonWorkChain)
# <class 'aiida_muon.workflows.find_muon.FindMuonWorkChain'>
```

You can also list all registered entry points:

```bash
verdi plugin list aiida.workflows muon
```

Expected output:
```
muon.find_muon     aiida_muon.workflows.find_muon:FindMuonWorkChain
muon.fine_tuning   aiida_muon.workflows.finetuning:FineTuningWorkChain
muon.active_learning  aiida_muon.workflows.active_learning:ActiveLearningWorkChain
```
