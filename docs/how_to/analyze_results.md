# Analyse and Export Results

Once `FindMuonWorkChain` finishes you have several ways to inspect and export the results.

---

## Workflow outputs overview

| Output | Type | Always present? | Description |
|---|---|---|---|
| `all_index_uuid` | `orm.Dict` | Yes | Mapping from site index to AiiDA node UUID |
| `all_sites` | `orm.Dict` | Yes | All relaxed sites (before clustering) |
| `unique_sites` | `orm.Dict` | Yes | Unique candidate sites after symmetry clustering |
| `unique_sites_hyperfine` | `orm.Dict` | Magnetic only | Contact hyperfine field (T) per unique site |
| `unique_sites_dipolar` | `orm.List` | Magnetic only | Classical dipolar field (T) per unique site |

---

## Inspecting outputs in Python

```python
from aiida import load_profile, orm
load_profile()

node = orm.load_node(<PK>)

# All relaxed structures and their energies
all_sites = node.outputs.all_sites.get_dict()
for label, data in all_sites.items():
    print(f"{label}: position={data['position']}, energy_diff={data['energy']} eV")

# Unique sites
unique = node.outputs.unique_sites.get_dict()
for label, data in unique.items():
    print(f"{label}: position={data['position']}")
```

---

## Exporting to a pandas DataFrame

The `aiida_muon.utils.export_findmuon` module provides a high-level helper that
assembles a human-readable summary table:

```python
from aiida_muon.utils.export_findmuon import get_clustering_after_run, get_distortions

# Summary table (unique sites, energies, cluster mapping)
df = get_clustering_after_run(node)
print(df.to_string())
```

The DataFrame includes columns such as:

- `muon_index` — index of the relaxed supercell
- `label` — cluster label
- `position` — fractional coordinates of the relaxed muon
- `energy_diff` — energy relative to the lowest-energy site (eV)
- `multiplicity` — number of symmetry-equivalent sites in the unit cell

### Per-site distortions

```python
from aiida.orm import StructureData

# Load the unrelaxed and relaxed supercell nodes
unrelaxed = orm.load_node(uuid_unrelaxed).get_ase()
relaxed   = orm.load_node(uuid_relaxed).get_ase()

distortions = get_distortions(unrelaxed, relaxed)
print("Distortion norms (Å):", distortions['distortion_norm'])
print("Δ distances from muon (Å):", distortions['delta_distance'])
```

---

## Accessing individual relaxation calculations

The mapping between site index and AiiDA UUID stored in `all_index_uuid` lets
you drill into any individual sub-calculation:

```python
uuid_map = node.outputs.all_index_uuid.get_dict()
# e.g. {'0': 'xxxxxxxx-...', '1': 'yyyyyyyy-...', ...}

# Load one of the PwRelaxWorkChain nodes
pw_node = orm.load_node(uuid_map['0'])
print(pw_node.outputs.output_parameters.get_dict())

# Get the relaxed structure as ASE Atoms
ase_struct = pw_node.outputs.output_structure.get_ase()
```

---

## Visualising structures

```python
# Use ASE's built-in viewer
from ase.visualize import view

for label, data in unique.items():
    uuid = uuid_map.get(str(data['index']))
    if uuid:
        rlx_node = orm.load_node(uuid)
        view(rlx_node.outputs.output_structure.get_ase())
```

---

## Querying the database

For large studies, use the AiiDA `QueryBuilder` to find all finished
`FindMuonWorkChain` nodes:

```python
from aiida.orm import QueryBuilder, WorkChainNode

qb = QueryBuilder()
qb.append(WorkChainNode, filters={
    'attributes.process_label': 'FindMuonWorkChain',
    'attributes.process_state': 'finished',
    'attributes.exit_status': 0,
})

results = qb.all(flat=True)
print(f"Found {len(results)} finished FindMuonWorkChain runs.")
```
