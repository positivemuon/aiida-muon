# Tutorial 2: Magnetic Materials

This tutorial focuses on magnetic systems and the additional outputs that
`FindMuonWorkChain` produces for them: the **contact hyperfine field** (from
the DFT spin density) and the **classical dipolar field** (computed analytically
with `muesr`).

We use BCC iron as the example throughout.

---

## 1. Understanding magnetism inputs

`FindMuonWorkChain` detects a magnetic structure through two inputs:

| Input | Type | Description |
|---|---|---|
| `magmom` | `orm.List` | Per-site 3D magnetic moments (µB), one vector per unit-cell site |
| `spin_pol_dft` | `orm.Bool` | Whether to use spin-polarised DFT (default: `True`) |

When `magmom` is present the workflow:

- Assigns spin-up/spin-down *kind names* to the QE input structure so that
  `pw.x` uses a proper starting magnetisation.
- After finding the unique muon sites, runs a final SCF with the muon placed at
  the unit-cell origin to compute the spin density via `pp.x`.
- Computes the contact hyperfine field from that density.
- Computes the dipolar field using the `muesr` / `muLFC` library.

!!! note "Collinear vs non-collinear"
    By default the workflow treats the magnetism as collinear and projects the
    supplied 3D `magmom` vectors onto the z-axis.  For non-collinear (spin-orbit)
    calculations, pass `noncollinear=True` to `get_builder_from_protocol` and
    ensure the `relax` inputs include `noncolin = .true.` in the QE SYSTEM namelist.

---

## 2. Setting up BCC iron

```python
from aiida import load_profile, orm
load_profile()

from ase.build import bulk
from aiida.orm import StructureData
from aiida.plugins import WorkflowFactory

FindMuonWorkChain = WorkflowFactory('muon.find_muon')

fe_atoms = bulk('Fe', 'bcc', a=2.87)
fe_structure = StructureData(ase=fe_atoms)

pw_code  = orm.load_code('pw@my-computer')
pp_code  = orm.load_code('pp@my-computer')   # needed for the hyperfine calculation

builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    pp_code=pp_code,           # supply pp_code to activate hyperfine calculation
    structure=fe_structure,
    mu_spacing=1.0,
    sc_matrix=[[2, 0, 0], [0, 2, 0],[0, 0, 2]],
    charge_supercell=True,
    full_dft_relax=True,
    magmom=[[0, 0, 2.2]],      # ferromagnetic moment along z
    spin_pol_dft=True,
)

builder.relax.base.pw.metadata.options = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
    'max_wallclock_seconds': 7200,
}
```

---

## 3. Submitting and monitoring

```python
from aiida.engine import submit

node = submit(builder)
print(f"Submitted workflow PK: {node.pk}")
```

```bash
verdi process list -a -L FindMuonWorkChain
```

---

## 4. Retrieving magnetic outputs

```python
node = orm.load_node(<PK>)

# Unique muon sites after clustering
unique_sites = node.outputs.unique_sites.get_dict()

# Contact hyperfine field (T) at each unique site
hyperfine = node.outputs.unique_sites_hyperfine.get_dict()

# Dipolar field (T) at each unique site
dipolar = node.outputs.unique_sites_dipolar.get_list()

for site_label, hf_value in hyperfine.items():
    print(f"Site {site_label}:")
    print(f"  Fractional position : {unique_sites[site_label]['position']}")
    print(f"  Contact hyperfine   : {hf_value} T")

for i, B_dip in enumerate(dipolar):
    print(f"Site {i}: dipolar field = {B_dip} T")
```

---

## 5. Understanding the hyperfine calculation

The contact hyperfine field originates from the Fermi contact interaction between
the muon spin and the spin density of the host electrons:

$$B_\text{hf} = \frac{2\mu_0}{3} \mu_B \, \rho_s(\mathbf{r}_\mu)$$

where $\rho_s = \rho_\uparrow - \rho_\downarrow$ is the spin density at the
muon site $\mathbf{r}_\mu$.

The workflow computes $\rho_s(\mathbf{r}_\mu)$ with `pp.x` after placing the
muon at the origin of the unit cell.

!!! tip "Why spin_pol_dft?"
    Setting `spin_pol_dft=False` skips the final SCF + `pp.x` step and produces only
    the dipolar field.  Use this if you want to save compute time and the contact
    contribution is expected to be negligible.

---

## 6. Understanding the dipolar field

The dipolar field is evaluated by `muesr` / `muLFC` using the converged magnetic
structure and the candidate muon positions.  It is a purely classical quantity
that depends only on the atomic magnetic moments and the muon position:

$$\mathbf{B}_\text{dip}(\mathbf{r}_\mu) =
\frac{\mu_0}{4\pi} \sum_i \frac{3(\hat{r}_i \cdot \mathbf{\mu}_i)\hat{r}_i
- \mathbf{\mu}_i}{r_i^3}$$

The sum runs over all magnetic atoms in a large sphere centred on the muon.
`muesr` determines the sphere radius automatically from the supercell geometry.
