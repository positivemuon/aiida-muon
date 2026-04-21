# DFT+U / Hubbard Corrections

For transition-metal compounds, DFT calculations are often improved by applying
on-site Hubbard *U* corrections (DFT+U / LDA+U).  `FindMuonWorkChain` supports
two ways to apply them.

---

## Automatic DFT+U (heuristics)

When `hubbard=True` (default) and the structure is passed as a plain
`orm.StructureData` together with a `magmom` list, the workflow tries to look up
reasonable *U* values for each 3d transition-metal species using a built-in
element heuristic provided by `aiida_muon.utils.hubbard.check_get_hubbard_u_parms`.

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=mno_structure,    # MnO — Mn 3d gets U automatically
    magmom=[[0, 0, 4.5], [0, 0, -4.5]],
    hubbard=True,               # default
)
```

!!! note
    Automatic DFT+U is only activated when **both** `magmom` is supplied and the
    structure contains species for which the heuristic has data.  For non-magnetic
    structures the `hubbard` flag has no effect.

---

## Custom U values via `hubbard_dict`

Override or supplement the automatic values with a per-species dictionary:

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=mno_structure,
    magmom=[[0, 0, 4.5], [0, 0, -4.5]],
    hubbard=True,
    hubbard_dict={'Mn': 5.0, 'O': 0.0},   # U in eV
)
```

Passing any value in `hubbard_dict` **overrides** the automatic lookup entirely
for the species listed.

---

## Using `HubbardStructureData`

For the greatest control, construct a
`aiida_quantumespresso.data.hubbard_structure.HubbardStructureData` directly
and pass it as the input structure.  The workflow will forward the embedded
Hubbard parameters to every supercell calculation unchanged:

```python
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida_quantumespresso.common.hubbard import Hubbard

hubbard_structure = HubbardStructureData.from_structure(mno_structure)
hubbard_structure.initialize_onsites_hubbard('Mn', '3d', 5.0, 'U', use_kinds=True)
hubbard_structure.hubbard = Hubbard.from_list(
    hubbard_structure.hubbard.to_list(), projectors='atomic'
)

builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=hubbard_structure,   # Hubbard info is embedded in the structure
    magmom=[[0, 0, 4.5], [0, 0, -4.5]],
)
```

When a `HubbardStructureData` is provided the `hubbard` flag controls only
whether the Hubbard information is propagated to the supercells (`True`) or
stripped away (`False`).

---

## Disabling DFT+U

To force a plain DFT (GGA) calculation even for a structure that would trigger
the automatic heuristic, set `hubbard=False`:

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=mno_structure,
    magmom=[[0, 0, 4.5], [0, 0, -4.5]],
    hubbard=False,    # no DFT+U, plain GGA
)
```

---

## How Hubbard parameters are propagated to supercells

The workflow calls `aiida_muon.utils.hubbard.create_hubbard_structure` to
construct a `HubbardStructureData` for each supercell by mapping the unit-cell
Hubbard parameters onto the expanded, muon-containing structure.  The muon
(represented as an H atom in the QE calculation) does not receive a *U* value.
