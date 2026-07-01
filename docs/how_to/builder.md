# Build the Workflow Inputs

`FindMuonWorkChain` follows the AiiDA *protocol* pattern: the class method
`get_builder_from_protocol` returns a fully populated `ProcessBuilder` with
sensible defaults that can be overridden as needed.

---

## Minimal call

```python
from aiida import load_profile, orm
from aiida.plugins import WorkflowFactory

load_profile()
FindMuonWorkChain = WorkflowFactory('muon.find_muon')

pw_code = orm.load_code('pw@my-computer')

builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,    # orm.StructureData or HubbardStructureData
)
```

This already sets:

- The `SSSP/1.3/PBE/efficiency` pseudopotential family.
- k-point distance of 0.301 Å⁻¹.
- Charged supercell (positive muon).
- Automatic supercell size convergence via `IsolatedImpurityWorkChain`.
- Full DFT relaxation of all candidate sites.

---

## Key parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pw_code` | `orm.Code` | *required* | Configured `quantumespresso.pw` code |
| `structure` | `StructureData` | *required* | Input crystal structure |
| `pp_code` | `orm.Code` | `None` | pp.x code; required for hyperfine calculation |
| `protocol` | `str` | `'moderate'` | QE calculation protocol (`'fast'`, `'moderate'`, `'precise'`) |
| `sc_matrix` | `list` | `None` | Explicit supercell matrix, e.g. `[[2, 0, 0], [0, 2, 0],[0, 0, 2]]`. If `None`, determined automatically. |
| `mu_spacing` | `float` | `1.0` | Minimum distance (Å) between two starting muon grid points |
| `niche_atom` | `str` | `'H'` | Chemical symbol used as placeholder for the muon in NICHE |
| `kpoints_distance` | `float` | `0.301` | k-point sampling density (Å⁻¹) |
| `pseudo_family` | `str` | `'SSSP/1.3/PBE/efficiency'` | Label of the pseudopotential family |
| `charge_supercell` | `bool` | `True` | Run a charged (+1) supercell to model the positive muon |
| `hubbard` | `bool` | `True` | Detect and apply DFT+U corrections automatically |
| `hubbard_dict` | `dict` | `None` | Override the automatic U values per species |
| `magmom` | `list` | `None` | Per-site 3D magnetic moments (µB) |
| `spin_pol_dft` | `bool` | `True` | Use spin-polarised DFT when `magmom` is set |
| `gamma_pre_relax` | `bool` | `False` | Run a cheap Gamma-point pre-relaxation before the full mesh |
| `ML_pre_relax` | `bool` | `False` | (experimental) MLIP pre-relaxation before DFT |
| `full_dft_relax` | `bool` | `True` | Run the full DFT relaxation step |
| `pre_clustering` | `bool` | `False` | Cluster after each pre-relaxation step to further reduce site count |
| `activate_monitors` | `bool` | `True` | Attach monitors to relaxation steps (requires `aiida-monitor`) |
| `monitor_entry_point_list` | `list` | `[]` | Additional `aiida-monitor` entry-point strings to attach |

---

## Runtime monitors

When [`aiida-monitor`](https://github.com/mikibonacci/aiida-monitor) is installed,
`get_builder_from_protocol` automatically attaches `aiida_monitor.default_monitor`
to every `PwBaseWorkChain` relaxation call.  This monitor can detect stalled or
hung calculations and terminate them gracefully so the workflow can continue.

The behaviour is controlled by two parameters:

```python
# Disable all monitors:
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    activate_monitors=False,
)

# Add an extra monitor on top of the default one:
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    monitor_entry_point_list=['my_package.my_monitor'],
    # activate_monitors=True is the default
)
```

| Parameter | Effect |
|---|---|
| `activate_monitors=True` (default) | `aiida_monitor.default_monitor` is appended automatically if `aiida-monitor` is installed |
| `activate_monitors=False` | No monitors are set, even if `monitor_entry_point_list` is non-empty |
| `monitor_entry_point_list=[...]` | Additional monitors attached alongside (or instead of) the default |

If `aiida-monitor` is **not** installed the builder silently skips monitor
attachment — the workflow runs normally.

---

## Overriding scheduler options

Scheduler options (walltime, number of MPI tasks, etc.) must be set on the
nested sub-builders **after** calling `get_builder_from_protocol`:

```python
options = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 8},
    'max_wallclock_seconds': 3600,
    'queue_name': 'short',            # optional
}

# Apply to all DFT relaxation calculations
builder.relax.base.pw.metadata.options = options

# Apply to the final SCF (hyperfine only)
builder.pwscf.pw.metadata.options = options
```

---

## Using a different protocol

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    protocol='fast',         # lower cutoffs, fewer k-points — good for testing
)
```

Available protocols are defined in the `aiida-quantumespresso` package:
`'fast'`, `'moderate'` (default), `'precise'`.

---

## Overriding QE parameters

Pass a nested `overrides` dict that mirrors the `PwRelaxWorkChain` input namespace:

```python
overrides = {
    'base': {
        'pw': {
            'parameters': {
                'SYSTEM': {'ecutwfc': 60, 'ecutrho': 480},
                'ELECTRONS': {'conv_thr': 1e-8},
            }
        }
    }
}

builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    overrides=overrides,
)
```

---

## Providing an explicit supercell matrix

Skipping the automatic supercell convergence step saves significant compute time
if you already know the right supercell:

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    sc_matrix=[3, 3, 3],   # 3×3×3 supercell
)
```

---

## Providing a custom list of supercells

If you have already generated the supercell structures with the muon inside
(e.g. from a previous run or an external tool), pass them directly:

```python
# Load previously stored StructureData nodes by UUID
supercells = [node1.uuid, node2.uuid, node3.uuid]

builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    sc_matrix=[[2, 0, 0], [0, 2, 0],[0, 0, 2]],       # still required when supercells_list is given
    supercells_list=supercells,
)
```

---

## Submitting and checking inputs before submission

You can inspect and validate the builder before submitting:

```python
# Dry-run check (raises if required inputs are missing)
from aiida.engine import run_get_node
# or simply print the builder to inspect all set inputs
print(builder)
```

Submit with:

```python
from aiida.engine import submit
node = submit(builder)
print(f"PK: {node.pk}")
```
