# Workflow Internals

This page explains in detail how `FindMuonWorkChain` works internally.
It is useful for developers, for debugging unexpected behaviour, or for
understanding provenance in the AiiDA graph.

---

## Overview

`FindMuonWorkChain` is a `ProtocolMixin`-enabled `WorkChain`.  It is designed
so that each logical sub-task can be individually skipped or replaced, enabling
a wide range of calculation strategies through a single, unified entry point.

---

## Context variables

The workflow uses the following `ctx` attributes to hold state between steps:

| Variable | Set by | Description |
|---|---|---|
| `ctx.structure` | `setup` | The AiiDA StructureData used for all supercell generation |
| `ctx.sc_matrix` | `check_converge_supercell_size` or `setup` | 3×3 supercell matrix as a Python list |
| `ctx.magmom` | `setup` | Magnetic moments as a Python list (if provided) |
| `ctx.hubbardu_dict` | `setup` | Hubbard U dict, populated from `inputs.hubbard_dict` if given |
| `ctx.mu_lst` | `get_initial_muon_sites` | List of fractional positions for candidate sites |
| `ctx.supc_list` | `get_initial_supercell_structures` | List of `StructureData` supercells with muon |
| `ctx.run_type` | `should_run_*` | `'gamma'`, `'full'`, or `'ASE'` — controls submission mode |
| `ctx.set_gamma_only` | `should_run_gamma_relaxations` | Whether to use `GAMMA_ONLY` QE setting |
| `ctx.non_collinear` | `setup` | `True` if `noncolin=True` in the QE parameters |
| `ctx.n` | `collect_relaxed_structures` | Running counter of successful relaxations |
| `ctx.n_uuid_dict` | `collect_relaxed_structures` | Mapping: site index → workchain UUID |
| `ctx.relaxed_outputs` | `collect_relaxed_structures` | List of dicts with `idx`, `rlxd_struct`, `energy` |
| `ctx.unique_cluster` | `analyze_relaxed_structures` | Unique fractional positions after clustering |
| `ctx.cluster_mapping` | `analyze_relaxed_structures` | Integer mapping of each site to its cluster |
| `ctx.offset` | `new_struct_after_analyze` | Index offset for magnetically inequivalent sites |

---

## Step-by-step walkthrough

### 1. `pre_check_structure_data_compatibility`

Determines whether the input is a plain `orm.StructureData` (or `LegacyStructureData`)
or a `HubbardStructureData`, and stores the result in `ctx.structure_type`.
This decides whether Hubbard parameters need to be propagated to supercells.

### 2. Supercell size determination (optional)

If `sc_matrix` is **not** in the inputs, `IsolatedImpurityWorkChain` is used to
find the minimum supercell such that the impurity–impurity interaction is below a
convergence threshold.  The converged matrix is stored in `ctx.sc_matrix`.

If `sc_matrix` **is** provided, this sub-workflow is skipped entirely.

### 3. `setup`

Initialises all context variables.  Importantly:

- Sets `ctx.set_gamma_only = True` if the k-point mesh for the supercell would
  collapse to Γ anyway, to enable the `GAMMA_ONLY` QE optimisation.
- Checks for `noncolin=True` in the relax inputs to disable `GAMMA_ONLY`.
- Loads any pre-supplied supercells from `inputs.supercells_list`.

### 4. Site generation (conditional)

If `ctx.supc_list` is empty (no pre-supplied supercells):

1. **`get_initial_muon_sites`** — calls `niche_add_impurities` to place a grid.
   The grid spacing is `inputs.mu_spacing`; the minimum distance from host atoms
   is hardcoded to 1 Å.
2. **`get_initial_supercell_structures`** — calls `gensup` to embed each muon
   in a supercell, then calls `generate_supercells_list` to convert pymatgen
   objects to AiiDA `StructureData` (with Hubbard parameters if needed).

### 5. Pre-relaxation stages (conditional)

Each stage follows the same pattern:

```
should_run_*  →  compute_supercell_structures  →  collect_relaxed_structures  →  run_cluster_analysis
```

**`compute_supercell_structures`** regenerates the supercell list (to re-apply
Hubbard parameters after the list was filtered) and then dispatches to:

- `submit_dft_relaxations(enforce_gamma=True)` for the Gamma stage
- `submit_dft_relaxations(enforce_gamma=False)` for the full-mesh stage
- `submit_ase_relaxations()` for the MLIP stage

**`collect_relaxed_structures`** iterates over all submitted workchains and
extracts the relaxed structure and final energy.  Up to 40% failures are
tolerated before the workflow aborts.

**`run_cluster_analysis`** calls `analyze_relaxed_structures`, which uses
`prune_too_close_pos` to discard duplicates within `d_tol` Å.  After the full
DFT stage it also checks for magnetically inequivalent sites that were not in
the original grid.

### 6. `collect_all_results`

Merges results from all stages into `ctx.relaxed_outputs_all` and
`ctx.unique_cluster`.

### 7. Magnetic post-processing (conditional)

Only executed when `inputs.magmom` is present:

- **`run_final_scf_mu_origin`** — for each unique site, translates the muon to
  the unit-cell origin and runs a `PwBaseWorkChain` SCF.
- **`compute_spin_density`** — runs a `pp.x` calculation to extract the spin
  density on a real-space grid.
- **`compute_contact_hyperfine`** — evaluates the spin density at the muon
  position using the Fermi contact formula.
- **`get_dipolar_field`** — computes the dipolar field with `muesr`.
- **`set_field_outputs`** — stores the hyperfine and dipolar results as outputs.

### 8. `set_relaxed_muon_outputs`

Stores `all_index_uuid`, `all_sites`, and `unique_sites` as final outputs.

---

## Hubbard parameter propagation

Every time `generate_supercells_list` is called (which happens at the start of
each `compute_supercell_structures` step), the workflow recreates the
`HubbardStructureData` supercells from scratch.  This ensures that if the
supercell list was filtered by clustering in between, the Hubbard metadata is
re-attached correctly to the remaining structures.

---

## Failure handling

The workflow tolerates up to **40%** failed `PwRelaxWorkChain` sub-calculations
per stage before returning `ERROR_RELAX_CALC_FAILED`.  Failed calculations are
simply skipped; their indices are not included in the clustering analysis.

---

## Debugging tips

- Check reported messages with `verdi process report <PK>`.
- Use `verdi process show <PK>` to list all called sub-processes and their status.
- Individual relaxation nodes can be loaded by UUID from `outputs.all_index_uuid`.
- If clustering discards too many sites, lower `mu_spacing` or increase `d_tol`
  in the workflow source (there is currently no public input for `d_tol`).
