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

`export_findmuon_data` assembles the summary tables, the distortions and the structures with all the muon sites:

```python
from aiida_muon.utils.export_findmuon import export_findmuon_data, get_distortions

exported = export_findmuon_data(node.outputs)   # `node` is the finished FindMuonWorkChain
exported["table"]       # the unique sites (after clustering), one row per site
exported["table_all"]   # all the relaxed sites (before clustering), with the clustering information
exported["distortions"] # the distortion around each site, see below
exported["unit_cell"]   # unit cell of the host with the unique muon sites
exported["unit_cell_all"]  # unit cell of the host with all the muon sites
```

### How the muon sites are identified

A site is identified by several quantities, which are **not** interchangeable:

| Column | Meaning |
|---|---|
| `supercell_index` | The id of the relaxation (supercell) of the site: the key of the site in the `all_index_uuid`, `all_sites` and `unique_sites` outputs. It is also the (named) index of both tables. It is *not* a position: it is unrelated to the order of the rows, to `label` and to the energy. |
| `atom_index_in_unit_cell` | The **0-based** index of the muon *atom* in the unit cell with all the muon sites: `exported["unit_cell"]` for `table`, `exported["unit_cell_all"]` for `table_all`. Use it with ASE or with a structure viewer to pick the muon of that site: `exported["unit_cell"].get_ase()[atom_index]`. |
| `label` | A letter, in order of increasing energy: `A` is the lowest energy site. The sites that are not in `table` get the following letters. |
| `cluster group` | Only in `table_all`: the group of the clustering to which the site belongs. |
| `structure_id_pk` | The pk of the relaxed supercell structure. |

Other columns: `delta_E` and `tot_energy` (meV), `muon_position_cc` (crystal coordinates of the muon in the supercell),
and, if magnetic fields were computed, `B_T`, `Bdip`, `B_T_norm`, `Bdip_norm`, `B_hf_norm` (Tesla).

!!! note "Renamed columns"
    `supercell_index` was called `muon_index` (as a string) and `atom_index_in_unit_cell` was called
    `muon_index_global_unitcell` (1-based, and wrong in `table` when the clustering had removed sites).
    The old columns are still returned, for one release, with `include_deprecated_columns=True` (the default); pass
    `include_deprecated_columns=False` to drop them. `muon_index_global_unitcell` is now `atom_index_in_unit_cell + 1`.

### Per-site distortions

```python
from aiida.orm import StructureData

# Load the unrelaxed and relaxed supercell nodes
unrelaxed = orm.load_node(uuid_unrelaxed).get_ase()
relaxed   = orm.load_node(uuid_relaxed).get_ase()

distortions = get_distortions(unrelaxed, relaxed)
for element, data in distortions.items():          # one entry per element, other than the muon
    print(element, "distortion norms (Å):", data["distortion"])
    print(element, "Δ distances from muon (Å):", data["delta_distance"])
```

`exported["distortions"]` contains the same, for every site, with the `supercell_index` (as a string) as key.

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
