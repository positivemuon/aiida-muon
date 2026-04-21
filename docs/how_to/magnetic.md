# Magnetic Calculations

`FindMuonWorkChain` has first-class support for magnetically ordered materials.
When magnetic moments are provided the workflow automatically:

1. Assigns spin-up / spin-down *kind names* for `pw.x` starting magnetisation.
2. Runs a final SCF with `pp.x` to compute the spin density at the muon site.
3. Computes the **contact hyperfine field** from the spin density.
4. Computes the classical **dipolar field** with `muesr`.

---

## Providing magnetic moments

Pass magnetic moments as a Python list of 3-component vectors, one per site of
the **input unit cell** (not the supercell):

```python
# Ferromagnetic Fe BCC — one site
magmom = [[0, 0, 2.2]]

# Antiferromagnetic MnO — two sites (Mn↑ and Mn↓)
magmom = [[0, 0, 4.5], [0, 0, -4.5]]
```

The workflow internally uses pymatgen's
`CollinearMagneticStructureAnalyzer` to project the 3D vectors onto a collinear
axis and derive the appropriate kind names for Quantum ESPRESSO.

---

## Enabling the contact hyperfine calculation

Supply a `pp_code` to `get_builder_from_protocol`:

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    pp_code=pp_code,           # enables the pp.x step
    structure=fe_structure,
    magmom=[[0, 0, 2.2]],
    spin_pol_dft=True,
)
```

If `pp_code` is not given, the hyperfine step is skipped and only the dipolar
field is computed.

---

## Non-collinear / spin-orbit calculations

For non-collinear magnetism (e.g. spin spirals or spin-orbit coupling), pass
`noncollinear=True` and include the relevant QE parameters in the overrides:

```python
overrides = {
    'base': {
        'pw': {
            'parameters': {
                'SYSTEM': {
                    'noncolin': True,
                    'lspinorb': True,
                }
            }
        }
    }
}

builder = FindMuonWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=my_structure,
    magmom=[[0, 0, 2.5]],
    spin_pol_dft=True,
    noncollinear=True,
    overrides=overrides,
)
```

!!! warning
    When `noncolin=True` the workflow automatically disables the `GAMMA_ONLY`
    optimisation (which is incompatible with non-collinear calculations) and
    skips the Gamma-point pre-relaxation shortcut.

---

## Outputs for magnetic systems

Once the workflow finishes you can retrieve:

```python
node = orm.load_node(<PK>)

# Unique muon sites (fractional coordinates + energy)
unique_sites = node.outputs.unique_sites.get_dict()

# Contact hyperfine field (Tesla)
# Key: site label (string), Value: Bhf in T
hf = node.outputs.unique_sites_hyperfine.get_dict()

# Classical dipolar field (Tesla)
# List of field magnitudes at each unique site
dip = node.outputs.unique_sites_dipolar.get_list()
```

---

## Interpreting the outputs

The **contact hyperfine field** $B_\text{hf}$ is related to the local electron
spin density at the muon nucleus:

$$B_\text{hf} = \frac{2\mu_0}{3}\mu_B \, \rho_s(\mathbf{r}_\mu)$$

It can be positive or negative depending on the sign of the spin density at
the muon site, which is not always the same as the majority spin of the host.

The **dipolar field** $\mathbf{B}_\text{dip}$ is computed by summing the
classical magnetic dipole contributions from all atoms in a large sphere
around the muon.  The Lorentz and demagnetisation corrections are not
included by default; they must be added analytically if needed.

The **total internal field** experienced by the muon (in Tesla) is approximately:

$$B_\mu \approx |\mathbf{B}_\text{hf} + \mathbf{B}_\text{dip}|$$

This quantity can be compared directly to the precession frequency measured
in a transverse-field µSR experiment:

$$f_\mu = \frac{\gamma_\mu}{2\pi} B_\mu, \quad \gamma_\mu / 2\pi = 135.5 \text{ MHz/T}$$
