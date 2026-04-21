# Machine-Learning Features (experimental)

!!! warning "Experimental"
    All features described on this page are **experimental**.  APIs, input
    names, and behaviour may change without notice in any future release.
    Do not rely on them for production calculations without thorough testing.

---

## Overview

`aiida-muon` provides three ML-related features:

1. **MLIP pre-relaxation** (`ML_pre_relax`) inside `FindMuonWorkChain` — uses a
   machine-learning interatomic potential to cheaply pre-screen candidate muon
   sites before the full DFT relaxation.
2. **`FineTuningWorkChain`** (`muon.fine_tuning`) — fine-tunes a pre-trained MLIP
   model on a user-supplied DFT dataset.
3. **`ActiveLearningWorkChain`** (`muon.active_learning`) — iteratively generates
   DFT training data, fine-tunes, validates, and repeats.

All ML features rely on [`aiida-pythonjob`](https://github.com/mikibonacci/aiida-pythonjob)
to submit Python functions (carrying the MLIP code) to remote computers managed
by AiiDA.

---

## Dependency setup

```bash
# aiida-pythonjob (use the patched fork until upstream is stable)
pip install git+https://github.com/mikibonacci/aiida-pythonjob@fix_serializer
pip install numpy==2    # pinned due to serialiser compatibility

# Choose one or more MLIP backends
pip install mace-torch       # MACE-MP foundation model
pip install chgnet            # CHGNet
pip install mattersim         # Microsoft MatterSim
```

Register a `pythonjob` code in AiiDA for each remote computer:

```bash
verdi code create core.code.installed \
    --label pythonjob-mace \
    --computer my-gpu-cluster \
    --default-calc-job-plugin pythonjob \
    --filepath-executable "$(which python)"
```

The remote Python environment must have the chosen MLIP package installed.

---

## MLIP pre-relaxation in `FindMuonWorkChain`

### How it works

When `ML_pre_relax=True`, the workflow dispatches one `PythonJob` per candidate
supercell.  Each job runs `optimize_structure` from
`aiida_muon.pythonjobs.relax`, which calls an ASE optimizer (default: BFGS)
with the chosen MLIP calculator.

After all ML relaxations finish, the structures are clustered using the same
symmetry-based algorithm as the full DFT stage.  Only the representative unique
sites are sent to the subsequent DFT relaxation.

### Usage

```python
from mace.calculators import mace_mp
from aiida.plugins import WorkflowFactory
from aiida import orm

FindMuonWorkChain = WorkflowFactory('muon.find_muon')

pythonjob_code = orm.load_code('pythonjob-mace@my-cluster')

# The calculator must be a callable (it will be pickled and sent remotely)
def mace_calculator():
    from mace.calculators import mace_mp
    return mace_mp(model='medium', device='cpu', default_dtype='float64')

builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=orm.load_code('pw@my-cluster'),
    structure=my_structure,
    ML_pre_relax=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=mace_calculator,
    pre_clustering=True,       # cluster after ML relaxation
    full_dft_relax=True,       # run DFT on the unique sites
)
```

### ASE optimiser options

These are forwarded via `additional_pythonjob_inputs`:

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    ...
    ML_pre_relax=True,
    additional_pythonjob_inputs={
        'fmax': 1e-3,              # force convergence (eV/Å)
        'optimizer': 'FIRE',       # BFGS, LBFGS, FIRE, MDMin
        'fix_symmetry': True,      # apply FixSymmetry ASE constraint
    },
)
```

---

## MLIP-based supercell size determination

When `ML_supercell_size=True` is passed to `get_builder_from_protocol`, the
`IsolatedImpurityWorkChain` used for supercell convergence also uses MLIP
forces instead of DFT:

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    ML_supercell_size=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=mace_calculator,
)
```

---

## `FineTuningWorkChain` — `muon.fine_tuning`

Fine-tunes a pre-trained MLIP model on a training dataset provided either as
an `.xyz` / `.pkl` file path or as a set of DFT-labelled structures.

### Entry point

```python
from aiida.plugins import WorkflowFactory
FineTuningWorkChain = WorkflowFactory('muon.fine_tuning')
```

### Builder

```python
builder = FineTuningWorkChain.get_builder_from_protocol(
    pythonjob_code=pythonjob_code,
    load_model_path='/path/to/pretrained.pth',
    train_data_path='/path/to/train_data.xyz',
    save_path='./finetuned',
    epochs=200,
    batch_size=4,
    lr=2e-4,
    device='cpu',
)
```

### Supported MLIP backends

The workflow dispatches to the appropriate fine-tuning function based on the
model type detected from the checkpoint path:

| Backend | Helper module |
|---|---|
| MatterSim | `aiida_muon.pythonjobs.finetuning_mattersim` |
| MACE | `aiida_muon.pythonjobs.finetuning_mace` |
| metatrain | `aiida_muon.pythonjobs.finetuning_metatrain` |

---

## `ActiveLearningWorkChain` — `muon.active_learning`

Implements an iterative active-learning loop:

```
setup
if should_run_dft:
    run_dft_calculations
    collect_dft_results
if should_run_score_filtering:
    run_score_calculation        ← selects most informative frames
    collect_score_results
while should_iterate:
    run_finetuning               ← FineTuningWorkChain
    collect_finetuning
    run_validation               ← TODO
    collect_validation           ← TODO
set_outputs
```

!!! note "Validation step is not yet implemented"
    The `run_validation` and `collect_validation` steps are placeholders.
    Currently the loop always terminates after `max_iterations` iterations.

### Score-based frame selection

Before fine-tuning, the workflow can optionally use `ScoreCalculator` from
`aiida_muon.utils.score` to rank DFT frames by their MLIP disagreement and
select only the most informative ones as training data.

The frame selection is controlled by:

| Input | Default | Description |
|---|---|---|
| `score_callback_calculator` | — | ASE-compatible MLIP callable for scoring (omit to skip) |
| `score_num_frames` | `5` | Maximum number of frames to select |
| `score_w_E` | `0.2` | Weight for per-atom energy disagreement |
| `score_w_F` | `0.8` | Weight for force RMSE |
| `score_similarity_thr` | `0.01` | Structural RMSE threshold (Å) for similarity filtering |

### Builder example

```python
from aiida.plugins import WorkflowFactory
ActiveLearningWorkChain = WorkflowFactory('muon.active_learning')

builder = ActiveLearningWorkChain.get_builder_from_protocol(
    pythonjob_code=pythonjob_code,
    load_model_path='/path/to/pretrained.pth',
    structures={'s0': structure_0, 's1': structure_1},  # DFT labelling requested
    pw_code=pw_code,
    pseudo_family='SSSP/1.3/PBE/efficiency',
    save_path='./active_learning_output',
    max_iterations=3,
    epochs=100,
    device='cpu',
)
```

### Outputs

| Output | Type | Description |
|---|---|---|
| `finetuned_model_path` | `orm.Str` | Path to the final fine-tuned checkpoint |
| `iterations_performed` | `orm.Int` | Number of loop iterations completed |
| `train_data_path_used` | `orm.Str` | Path to the training file used |
| `score_reliability` | `orm.Dict` | Model reliability summary (if scoring was run) |
| `score_selected_indices` | `orm.List` | Indices of frames selected for training |
