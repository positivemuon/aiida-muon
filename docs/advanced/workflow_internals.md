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
- **`get_dipolar_field`** — computes the dipolar field with `pylocfield`.
- **`set_field_outputs`** — stores the hyperfine and dipolar results as outputs.

### Ewald cutoffs of the dipolar field: why these values

The dipolar field is computed by `compute_dip_field` (`aiida_muon.utils.magnetism`) with `pylocfield`. The sum over the
infinite lattice is done with the Ewald method, in which the sum is split in two parts that converge quickly: one over
real-space lattice vectors $\mathbf{R}$ and one over reciprocal-space lattice vectors $\mathbf{G}$. Each one is
truncated with a cutoff, and `get_ewald_cutoffs` sets them from the cell.

**Why not the defaults of `pylocfield`?** `compute_dipolar_tensors` uses `r_c = 12` Å and `g_c = 12` (the reciprocal
vectors that are summed have length up to $2\pi g_c \approx 75$ Å$^{-1}$). This is very far beyond what is needed. The
number of lattice vectors, and so the cost, grows as the cube of the cutoffs and with the volume of the cell, and
the sum is repeated for each atom of the supercell. With the defaults, a 54-atoms bcc Fe supercell (3×3×3) takes about
45 s and a 64-atoms MnO supercell about 25 s, while the former implementation took less than a second.

**What is needed.** With the Ewald parameter $\rho$ (`rho`, 1.8 Å$^{-1}$, the default of `pylocfield`, passed
explicitly), the terms of the sums are damped by Gaussians:

| Sum | Terms | Negligible beyond |
|---|---|---|
| reciprocal space | $\propto e^{-|\mathbf{G}|^2/(4\rho^2)}$ | $G_\text{conv} = 2\rho\sqrt{\ln(1/\text{tol})}$ |
| real space | $\propto e^{-(\rho r)^2}$ | $R_\text{conv} = \sqrt{\ln(1/\text{tol})}/\rho$ |

With a relative tolerance of $10^{-10}$ ($\ln(1/\text{tol}) = 23.03$) this is $G_\text{conv} = 17.3$ Å$^{-1}$, i.e.
$g_c = G_\text{conv}/2\pi = 2.75$, and $R_\text{conv} = 2.67$ Å. We multiply both by a **safety factor of 1.5**, so the
neglected terms are of order $e^{-52}$ instead of $e^{-23}$.

- **`g_c`** (reciprocal space) is $1.5 \times 2.75 = 4.12$. The reciprocal vectors do not depend on where the atoms
  are, so this is all that is needed, for any cell.
- **`r_c`** (real space) needs one more term. `r_c` selects *cells* $\mathbf{R}$, and all the atoms of a selected cell
  are summed. The muon sits at the centre of the supercell, so an atom within $R_\text{conv}$ of the muon can be in
  a cell whose $|\mathbf{R}|$ is as large as $R_\text{conv}$ plus the largest distance from the centre of a cell to
  one of its points, i.e. half of its longest diagonal ($\mathbf{a}_1 \pm \mathbf{a}_2 \pm \mathbf{a}_3$). So
  $r_c = 1.5\,R_\text{conv} + \tfrac12 \max|\mathbf{a}_1 \pm \mathbf{a}_2 \pm \mathbf{a}_3|$. With this, no atom that
  matters is left out, whatever the shape or size of the supercell, and the extra cells make the real-space sum only
  slightly longer. For a cubic supercell of 8.6 Å this gives $r_c \approx 11.5$ Å.

**Check.** The result was compared with the one obtained with the defaults of `pylocfield`:

| Supercell | Difference in $B_\text{dip+L}$ | Time with the defaults | Time with `get_ewald_cutoffs` |
|---|---|---|---|
| bcc Fe 3×3×3 (54 atoms) | $< 10^{-14}$ T | 43–85 s | 1.4 s |
| MnO 2×2×2 (64 atoms), antiferromagnetic | $< 10^{-14}$ T | 25–42 s | 1.0 s |
| hexagonal, tilted moments, 2×2×1 (oblique cell) | $3 \times 10^{-16}$ T | 4 s | 0.1 s |
| bcc Fe 1×1×1, 2×2×1, 2×2×2 | $< 10^{-15}$ T | 0.1–3.5 s | 0.01–0.1 s |

(Differences below $10^{-10}$ T are the round-off of the calculation; `tests/test_dipolar_field.py` requires
$10^{-10}$ T.) With smaller cutoffs the error appears where the formulas predict it: it stays below $10^{-7}$ T for $g_c \ge 2.5$ and
becomes $3 \times 10^{-5}$ T at $g_c = 2$ ($e^{-(2\pi g_c)^2/4\rho^2}$ is then about $10^{-5}$).

**Difference from `muesr`.** The result differs from the one of the former implementation (`muesr`) by about
$10^{-4}$ T on fields of 1–3 T (less than 0.01 %). This is not related to the cutoffs, since it does not change with
them: it comes from the different summation methods.

**If you change something.** If you change `rho`, the cutoffs follow (they are computed from it); if you change
the tolerance or the safety factor, edit `EWALD_LOG_INVERSE_TOLERANCE` and `EWALD_SAFETY_FACTOR` in
`aiida_muon/utils/magnetism.py`, where the same derivation is written as a comment, and run
`tests/test_dipolar_field.py`, which compares the result with the former `muesr` values and with the default cutoffs
of `pylocfield`.

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
