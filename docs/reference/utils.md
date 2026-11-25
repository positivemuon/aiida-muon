# Utility Modules Reference

---

## `aiida_muon.utils.sites_supercells`

Functions for generating the initial grid of candidate muon sites and building
supercells.

### `niche_add_impurities`

```python
niche_add_impurities(structure, niche_atom, niche_spacing, niche_distance, verbose=False)
```

Generates a grid of candidate muon (impurity) positions using the NICHE algorithm.
Returns a list of fractional coordinate vectors, each offset by +0.001 to break
symmetry-exact positions.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `structure` | `pymatgen.core.Structure` | Host crystal structure (no muon) |
| `niche_atom` | `str` | Chemical symbol used as muon placeholder (typically `'H'`) |
| `niche_spacing` | `float` | Minimum spacing (Å) between two grid points |
| `niche_distance` | `float` | Minimum distance from host atoms (Å); hardcoded to 1 in the workflow |

**Returns** `list[np.ndarray]` — fractional positions of candidate sites.

---

### `gensup`

```python
gensup(p_st, mu_list, sc_mat, only_one_cell=False, validate_proximity=True)
```

Creates one supercell per candidate site by making a copy of the unit cell,
expanding it by `sc_mat`, and appending the muon.

**Returns** `list[pymatgen.core.Structure]` — supercells with embedded muon.

---

### `compute_suggest_supercell_size`

```python
compute_suggest_supercell_size(structure, pbc=[True, True, True], verbose=False)
```

Heuristic that estimates a suitable supercell size from the unit-cell lattice
parameters (targets a minimum cell dimension of roughly 9 Å).

---

## `aiida_muon.utils.clustering`

Symmetry-aware clustering of relaxed muon positions.

### `analyze_structures`

```python
analyze_structures(reference_supercell, relaxed_outputs, input_structure,
                   magmom=None, d_tol=0.5)
```

Groups the relaxed muon positions by spatial proximity (within `d_tol` Å) and,
if `magmom` is given, by magnetic symmetry.  Returns a dict with:

- `unique_pos` — list of unique fractional positions
- `mapping` — integer array mapping each site to its cluster label
- `mag_inequivalent` — list of supercells for magnetically inequivalent sites
  that require an additional DFT calculation

---

### `prune_too_close_pos`

```python
prune_too_close_pos(frac_positions, host_lattice, min_distance, energies=None, e_tol=0.05)
```

Returns an index array marking positions that are within `min_distance` Å of a
lower-index position (and, optionally, within `e_tol` eV of its energy).
Marked positions should be discarded as duplicates.

---

## `aiida_muon.utils.magnetism`

Utilities for handling collinear magnetism and computing local magnetic fields.

### `get_collinear_mag_kindname`

```python
get_collinear_mag_kindname(p_st, magm, half=True)
```

Assigns spin-up/spin-down *kind names* to a pymatgen `Structure` based on the
supplied magnetic moment list.  The kind names are used by `aiida-quantumespresso`
to set the starting magnetisation for `pw.x`.

---

### `make_collinear_getmag_kind`

Convenience wrapper that calls `get_collinear_mag_kindname` and converts the
result to AiiDA `StructureData` format with the appropriate `starting_magnetization`
QE parameters.

---

### `compute_dipolar_field`

Computes the classical dipolar field at each unique muon site using
`muesr` / `muLFC`.  Called internally by `FindMuonWorkChain.get_dipolar_field`.

---

## `aiida_muon.utils.hubbard`

### `check_get_hubbard_u_parms`

```python
check_get_hubbard_u_parms(structure, u_dict=None)
```

Looks up Hubbard *U* values for the species present in `structure`.  Uses a
built-in element heuristic unless `u_dict` is provided.  Returns an empty dict
if the structure has fewer than two distinct species (the muon placeholder is
excluded from the count).

---

### `create_hubbard_structure`

```python
create_hubbard_structure(supercell_structure, reference)
```

Maps Hubbard parameters from `reference` (either a `HubbardStructureData` or a
`{species: U}` dict) onto `supercell_structure` and returns a
`HubbardStructureData`.

---

## `aiida_muon.utils.export_findmuon`

Post-processing helpers for use after a workflow finishes.

### `export_findmuon_data`

```python
export_findmuon_data(findmuon_output_node, include_deprecated_columns=True)
```

Returns a dict with the DataFrames `table` (unique sites) and `table_all` (all the sites, before clustering, with the
`cluster group`), the `distortions`, and the structures `unit_cell`, `unit_cell_all` and `supercell_all`, which contain
the muon sites (`unit_cell` only the unique ones). `findmuon_output_node` is the `outputs` of a finished
`FindMuonWorkChain`.

Each row of a table is a muon site, identified by:

- `supercell_index`: the id of its relaxation (the key in `all_index_uuid`, `all_sites`, `unique_sites`). It is also
  the named index of the tables. It is not a position.
- `atom_index_in_unit_cell`: the 0-based index of its muon in `unit_cell` (for `table`) or `unit_cell_all` (for
  `table_all`).
- `label`: a letter, in order of increasing energy.

The old columns `muon_index` (now `supercell_index`) and `muon_index_global_unitcell` (now
`atom_index_in_unit_cell`, 1-based) are returned too, for one release, unless `include_deprecated_columns=False`.

### `get_clustering_after_run`

```python
get_clustering_after_run(findmuon_node, d_tol=0.5, s_tol=0.05, e_tol=0.05, a_tol=1e-3)
```

Clusters the relaxed muon positions of a finished `FindMuonWorkChain`. Returns a dict with the unique positions
(`unique_pos`), the magnetically inequivalent structures still to be computed (`mag_inequivalent`) and the `mapping`:
one entry per site, in the order of the `all_sites` output, with the (1-based) position of the representative of the
cluster of the site.

---

### `get_distortions`

```python
get_distortions(unrelaxed_supercell, relaxed_supercell,
                muon_atomic_number=1, only_final_muon_position=True, verbose=False)
```

Computes atomic displacements (distortions) induced by the muon relaxation.
Useful for visualising the local lattice distortion around each candidate site.

**Returns** a dict with one entry per element (other than the muon); each entry is a dict with, for the atoms of that
element sorted by their initial distance from the muon:

- `atm_distance_init`, `atm_distance_final` — distance from the muon before and after the relaxation (Å)
- `distortion` — norm of the displacement vector (Å)
- `delta_distance` — change in distance from the muon (Å)

---

## `aiida_muon.utils.score`

!!! warning "Experimental"
    This module is part of the experimental machine-learning features.

### `ScoreCalculator`

```python
ScoreCalculator(calculator, w_E=0.2, w_F=0.8)
```

Computes per-frame disagreement scores between a DFT trajectory and an MLIP
calculator.  The composite score for frame *i* is:

$$s_i = w_E \cdot |\Delta E_i| / N_\text{atoms} + w_F \cdot \text{RMSE}(\mathbf{F}_i)$$

**Key methods**

| Method | Description |
|---|---|
| `add_dft_trajectory(traj, shift=None)` | Load a DFT trajectory (ASE-readable) |
| `evaluate_on_dft_trajectory()` | Run the MLIP on all frames |
| `compute_scores()` | Compute composite scores |
| `find_frames(num_frames, similarity_thr)` | Select the most informative frames |
| `model_reliability()` | Return a reliability summary dict |

---

## `aiida_muon.utils.trajectory`

### `atoms_list_to_trajectory_data`

```python
atoms_list_to_trajectory_data(atoms_list)
```

Converts a list of ASE `Atoms` objects to an AiiDA `TrajectoryData` node.

---

## `aiida_muon.utils.niche`

### `Niche`

```python
Niche(structure, impurity_atom)
```

Core NICHE class.  Places an impurity atom on a grid over the unit cell,
filtering positions that are too close to existing atoms.  Called internally by
`niche_add_impurities`.
