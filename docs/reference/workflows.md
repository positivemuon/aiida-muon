# Workflows Reference

This page documents all workflows registered as AiiDA entry points in `aiida-muon`.

---

## `FindMuonWorkChain` — `muon.find_muon`

The main workflow.  Finds candidate muon implantation sites by DFT supercell
relaxation and analyses the results.

```python
from aiida.plugins import WorkflowFactory
FindMuonWorkChain = WorkflowFactory('muon.find_muon')
```

### Inputs

#### Structure

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `structure` | `StructureData` / `HubbardStructureData` | Yes | — | Input crystal structure |
| `sc_matrix` | `orm.List` | No | — | Supercell matrix. If omitted, determined by `IsolatedImpurityWorkChain` |
| `supercells_list` | `orm.List` | No | — | UUIDs of pre-built supercell+muon structures (skips site generation) |

#### Muon grid

| Name | Type | Default | Description |
|---|---|---|---|
| `mu_spacing` | `orm.Float` | `1.0` | Minimum grid spacing between candidate sites (Å) |
| `niche_atom` | `orm.Str` | `'H'` | Chemical symbol used as muon placeholder in NICHE |

#### Magnetism

| Name | Type | Default | Description |
|---|---|---|---|
| `magmom` | `orm.List` | — | Per-site 3D magnetic moments (µB) in unit-cell order |
| `spin_pol_dft` | `orm.Bool` | `True` | Use spin-polarised DFT |

#### DFT options

| Name | Type | Default | Description |
|---|---|---|---|
| `pseudo_family` | `orm.Str` | `'SSSP/1.3/PBE/efficiency'` | Pseudopotential family label |
| `kpoints_distance` | `orm.Float` | `0.301` | k-point sampling density (Å⁻¹) |
| `charge_supercell` | `orm.Bool` | `True` | Charged (+1) supercell for positive muon |
| `hubbard` | `orm.Bool` | `True` | Apply DFT+U corrections |
| `hubbard_dict` | `orm.Dict` | — | Override U values per species (eV) |
| `qe_settings` | `orm.Dict` | — | Extra QE `settings` dict |

#### Calculation control

| Name | Type | Default | Description |
|---|---|---|---|
| `gamma_pre_relax` | `orm.Bool` | `False` | Gamma-point pre-relaxation stage |
| `full_dft_relax` | `orm.Bool` | `True` | Full k-mesh DFT relaxation stage |
| `pre_clustering` | `orm.Bool` | `False` | Cluster after each pre-relaxation step |
| `ML_pre_relax` | `orm.Bool` | `False` | (experimental) MLIP pre-relaxation |

#### Post-processing

| Name | Type | Required | Description |
|---|---|---|---|
| `pp_code` | `orm.Code` | No | pp.x code; required for hyperfine calculation |
| `pp_metadata` | `dict` | No | Non-DB metadata/options for pp.x |

#### Exposed sub-workflow namespaces

| Namespace | Workflow | Description |
|---|---|---|
| `relax` | `PoweredPwRelaxWorkChain` | DFT relaxation inputs (k-points, cutoffs, etc.) |
| `pwscf` | `PwBaseWorkChain` | Final SCF for hyperfine; excludes `structure` and `kpoints` |
| `impuritysupercellconv` | `IsolatedImpurityWorkChain` | Supercell convergence; only used when `sc_matrix` is not given |
| `pythonjob` | `PythonJob` | MLIP relaxation via aiida-pythonjob (experimental) |

### Outputs

| Name | Type | Always? | Description |
|---|---|---|---|
| `all_index_uuid` | `orm.Dict` | Yes | Site-index → UUID mapping for all relaxed supercells |
| `all_sites` | `orm.Dict` | Yes | All relaxed muon sites (before clustering) |
| `unique_sites` | `orm.Dict` | Yes | Unique sites after symmetry clustering |
| `unique_sites_hyperfine` | `orm.Dict` | Magnetic + `pp_code` | Contact hyperfine field (T) per site |
| `unique_sites_dipolar` | `orm.List` | Magnetic | Classical dipolar field (T) per site |

### Exit codes

| Code | Name | Description |
|---|---|---|
| 404 | `ERROR_MUSCONV_CALC_FAILED` | `IsolatedImpurityWorkChain` subprocess failed |
| 405 | `ERROR_RELAX_CALC_FAILED` | More than 40% of `PwRelaxWorkChain` subprocesses failed |
| 406 | `ERROR_BASE_CALC_FAILED` | A `PwBaseWorkChain` subprocess failed |
| 407 | `ERROR_PP_CALC_FAILED` | A `pp.x` subprocess failed |
| 408 | `ERROR_NO_SUPERCELLS` | No supercells generated; try reducing `mu_spacing` |

### Workflow outline

```
pre_check_structure_data_compatibility
if check_converge_supercell_size:
    run_converge_supercell_size
    check_supercell_convergence
setup
if should_generate_supercells:
    get_initial_muon_sites
    get_initial_supercell_structures
if should_run_mlip_relaxation:          # experimental, ML_pre_relax=True
    compute_supercell_structures
    collect_relaxed_structures
    run_cluster_analysis
if should_run_gamma_relaxations:        # gamma_pre_relax=True
    compute_supercell_structures
    collect_relaxed_structures
    run_cluster_analysis
if should_run_full_relaxations:         # full_dft_relax=True
    compute_supercell_structures
    collect_relaxed_structures
    run_cluster_analysis
    if new_struct_after_analyze:        # magnetic inequivalent sites found
        compute_supercell_structures
        collect_relaxed_structures
collect_all_results
if structure_is_magnetic:
    if spin_polarized_dft:
        run_final_scf_mu_origin
        compute_spin_density
        compute_contact_hyperfine
    get_dipolar_field
    set_field_outputs
set_relaxed_muon_outputs
```

---

## `FineTuningWorkChain` — `muon.fine_tuning`

!!! warning "Experimental"
    This workflow is experimental and subject to change.

Wraps a fine-tuning run for an MLIP model (MatterSim, MACE, or metatrain)
on a user-supplied training dataset using `aiida-pythonjob`.

```python
from aiida.plugins import WorkflowFactory
FineTuningWorkChain = WorkflowFactory('muon.fine_tuning')
```

See the [ML Features documentation](../advanced/ml_features.md) for usage details.

---

## `ActiveLearningWorkChain` — `muon.active_learning`

!!! warning "Experimental"
    This workflow is experimental and subject to change.

Iteratively fine-tunes an MLIP model by selecting informative frames from
DFT calculations, fine-tuning, and validating — repeating until convergence.

```python
from aiida.plugins import WorkflowFactory
ActiveLearningWorkChain = WorkflowFactory('muon.active_learning')
```

See the [ML Features documentation](../advanced/ml_features.md) for usage details.
