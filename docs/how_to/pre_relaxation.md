# Pre-Relaxation Strategies

A full DFT relaxation of every candidate muon site can be expensive when the
grid is dense or the supercells are large.  `FindMuonWorkChain` provides two
optional pre-relaxation stages that cheaply discard sites converging to the
same local minimum before the costly full-mesh DFT step.

---

## Gamma-point pre-relaxation

Running at a single Γ k-point is orders of magnitude cheaper than a full
k-point mesh.  When enabled, the workflow:

1. Relaxes all initial supercells at Γ only (with an automatically loosened
   force threshold: 50× the full-mesh value).
2. Clusters the Γ-relaxed positions.
3. Runs only the unique sites at the full k-point mesh.

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    gamma_pre_relax=True,     # enable Gamma-point pre-relaxation
    full_dft_relax=True,      # still run full DFT after pre-relaxation
)
```

!!! tip
    If the k-point mesh for your supercell already collapses to Γ (because the
    supercell is large enough), the workflow detects this automatically and skips
    the Gamma pre-relaxation step to avoid redundant calculations.

### Combining with `pre_clustering`

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    gamma_pre_relax=True,
    pre_clustering=True,      # cluster and reduce sites after each pre-relax stage
    full_dft_relax=True,
)
```

With `pre_clustering=True` the workflow re-clusters after *every* pre-relaxation
stage (not only after the full DFT).  The clustering tolerance is tighter in the
pre-clustering phase (0.25 Å vs 0.5 Å for the full run) to avoid discarding
sites that are only loosely converged.

---

## MLIP pre-relaxation (experimental)

!!! warning "Experimental feature"
    Machine-learning interatomic potential (MLIP) support is **experimental**.
    It requires additional optional dependencies (`aiida-pythonjob` and an MLIP
    package such as MACE or MatterSim) and may change without notice.

When `ML_pre_relax=True` the workflow replaces the Gamma-point DFT step with
a fully ML-driven relaxation using an ASE-compatible calculator submitted via
`aiida-pythonjob`.  This allows the pre-relaxation to run remotely on any
computer configured in AiiDA.

### Requirements

```bash
pip install git+https://github.com/mikibonacci/aiida-pythonjob@fix_serializer
pip install mace-torch    # or chgnet / mattersim
```

A `pythonjob` code must be set up in AiiDA pointing to a Python executable that
has the chosen MLIP package installed:

```bash
verdi code create core.code.installed \
    --label pythonjob \
    --computer my-computer \
    --default-calc-job-plugin pythonjob \
    --filepath-executable "$(which python)"
```

### Usage

```python
from mace.calculators import mace_mp

pythonjob_code = orm.load_code('pythonjob@my-computer')

# The calculator must be a *callable* that returns an ASE calculator.
# It is serialised (pickled) and sent to the remote computer.
def my_calculator():
    from mace.calculators import mace_mp
    return mace_mp(model='medium', device='cpu', default_dtype='float64')

builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    ML_pre_relax=True,           # enable MLIP pre-relaxation
    pythonjob_code=pythonjob_code,
    callback_calculator=my_calculator,
    full_dft_relax=True,         # run DFT after ML pre-screening
    pre_clustering=True,         # cluster after ML relaxation
)
```

### Supported MLIP backends

| Backend | Package | Notes |
|---|---|---|
| MACE | `mace-torch` | `mace_mp` foundation model |
| CHGNet | `chgnet` | Pretrained CHGNet |
| MatterSim | `mattersim` | Microsoft MatterSim |

Any ASE-compatible calculator can in principle be used by providing a
`callback_calculator` that returns it.

---

## Choosing a strategy

| Scenario | Recommended strategy |
|---|---|
| Small unit cell, few initial sites | No pre-relaxation (`full_dft_relax=True`) |
| Medium unit cell, grid with >20 sites | `gamma_pre_relax=True, pre_clustering=True` |
| Large supercell, >50 initial sites | `ML_pre_relax=True, pre_clustering=True` (experimental) |
| Want control over site selection | Provide `supercells_list` directly |
