# Tutorial 1: Finding Muon Sites (Si, Fe, MnO)

In this tutorial you will submit a `FindMuonWorkChain` for three prototype materials:

| Material | Magnetic? | DFT+U? |
|---|---|---|
| Si (diamond cubic) | No | No |
| Fe (BCC) | Yes | No |
| MnO (rock-salt) | Yes | Yes |

The tutorial uses intentionally loose parameters so that calculations finish quickly on a
single CPU core.

---

## 1. Load the AiiDA profile

```python
from aiida import load_profile, orm
load_profile()
```

---

## 2. Load a structure

Any of the three AiiDA structure types accepted by `FindMuonWorkChain` can be used:

- `orm.StructureData` (AiiDA legacy)
- `aiida_quantumespresso.data.hubbard_structure.HubbardStructureData`
- `aiida_atomistic.data.StructureData` (if `aiida-atomistic` is installed)

The simplest way is to build from ASE:

```python
from ase.build import bulk
from aiida.orm import StructureData

# Silicon
si_atoms = bulk('Si', 'diamond', a=5.43)
si_structure = StructureData(ase=si_atoms)

# BCC Iron
fe_atoms = bulk('Fe', 'bcc', a=2.87)
fe_structure = StructureData(ase=fe_atoms)

# MnO (rock-salt, approximate lattice parameter)
from ase.build import rocksalt
from ase import Atoms
mno_atoms = bulk('MnO', crystalstructure='rocksalt', a=4.45)
mno_structure = StructureData(ase=mno_atoms)
```

---

## 3. Build the workflow inputs

`FindMuonWorkChain` ships with `get_builder_from_protocol`, which pre-fills sensible
defaults.  You only need to supply the mandatory arguments and override what you need.

### Silicon (non-magnetic)

```python
from aiida.plugins import WorkflowFactory
from aiida import orm

FindMuonWorkChain = WorkflowFactory('muon.find_muon')

pw_code = orm.load_code('pw-7.3@mpc3129')   # replace with your code label

builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=si_structure,
    mu_spacing=1.0,          # grid spacing in Å
    sc_matrix=[[2, 0, 0], [0, 2, 0],[0, 0, 2]],     # explicit supercell matrix (skips supercell convergence)
    charge_supercell=True,   # simulate positive muon (charged supercell)
    full_dft_relax=True,     # run full k-mesh DFT relaxation
)

# Adjust scheduler options to match your cluster
builder.relax.base.pw.metadata.options = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
    'max_wallclock_seconds': 3600,
}
```

!!! tip "Runtime monitors"
    When [`aiida-monitor`](https://github.com/mikibonacci/aiida-monitor) is installed,
    a default monitor is attached automatically to each relaxation step and can kill
    stalled calculations so the workflow keeps running.  Pass
    `activate_monitors=False` to `get_builder_from_protocol` to disable this.

### BCC Iron (collinear ferromagnet)

For magnetic materials, pass the magnetic moments as a flat list of 3-component vectors
(one per site in the **unit cell**):

```python
builder_fe = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=fe_structure,
    mu_spacing=1.0,
    sc_matrix=[[2, 0, 0], [0, 2, 0],[0, 0, 2]],
    charge_supercell=True,
    full_dft_relax=True,
    magmom=[[0, 0, 2.2]],   # 2.2 µB along z for each Fe site
    spin_pol_dft=True,
)
builder_fe.relax.base.pw.metadata.options = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
    'max_wallclock_seconds': 7200,
}
```

### MnO (antiferromagnet + Hubbard U)

MnO has a 2-site unit cell (Mn↑, Mn↓) and benefits from a DFT+U correction on Mn 3d:

```python
builder_mno = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=mno_structure,
    mu_spacing=1.5,
    sc_matrix=[[2, 0, 0], [0, 2, 0],[0, 0, 2]],
    charge_supercell=True,
    full_dft_relax=True,
    magmom=[[0, 0, 4.5], [0, 0, -4.5]],   # AFM ordering
    spin_pol_dft=True,
    hubbard=True,          # apply DFT+U automatically based on element heuristics
)
builder_mno.relax.base.pw.metadata.options = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
    'max_wallclock_seconds': 7200,
}
```

---

## 4. Submit the workflow

```python
from aiida.engine import submit

node_si  = submit(builder)
node_fe  = submit(builder_fe)
node_mno = submit(builder_mno)

print(f"Si  workflow PK: {node_si.pk}")
print(f"Fe  workflow PK: {node_fe.pk}")
print(f"MnO workflow PK: {node_mno.pk}")
```

Monitor progress with:

```bash
verdi process list -a
verdi process show <PK>
```

---

## 5. Inspect the results

Once the workflow finishes (`State: Finished`, `Exit status: 0`):

```python
node = orm.load_node(<PK>)

# All relaxed sites (before clustering)
all_sites = node.outputs.all_sites.get_dict()
print("All sites:", all_sites)

# Unique candidate sites after symmetry clustering
unique_sites = node.outputs.unique_sites.get_dict()
print("Unique sites:", unique_sites)
```

For magnetic systems:

```python
# Contact hyperfine field at each unique site (requires pp_code and spin_pol_dft=True)
if node.outputs.unique_sites_hyperfine:
    print("Hyperfine fields:", node.outputs.unique_sites_hyperfine.get_dict())

# Classical dipolar field
if node.outputs.unique_sites_dipolar:
    print("Dipolar fields:", node.outputs.unique_sites_dipolar.get_list())
```

### Export to a human-readable table

The `export_findmuon` utility converts the workflow outputs into a pandas DataFrame:

```python
from aiida_muon.utils.export_findmuon import get_clustering_after_run

# Load workchain data from the AiiDA node
df = get_clustering_after_run(node)
print(df)
```

---

## Summary of the steps

The workflow executed the following steps:

1. **Supercell size check** — either uses the provided `sc_matrix` or calls
   `IsolatedImpurityWorkChain` to determine the minimum converged supercell.
2. **Initial muon sites** — the NICHE algorithm places candidate muon positions
   on a grid, avoiding interstitial positions too close to host atoms.
3. **Supercell generation** — each starting site is embedded in a supercell.
4. **DFT relaxation** — one `PwRelaxWorkChain` per supercell.
5. **Clustering** — relaxed muon positions are grouped by symmetry and spatial
   proximity; duplicates are discarded.
6. **Hyperfine and dipolar fields** (magnetic only) — a final SCF places the
   muon at the origin (`PwBaseWorkChain`) and `pp.x` evaluates the spin density.
   The dipolar field is computed with `muesr`.
