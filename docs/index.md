# aiida-muon

An [AiiDA](https://www.aiida.net) workflow plugin for finding candidate muon implantation sites and computing the local magnetic field at those sites.

[![GitHub](https://img.shields.io/badge/GitHub-positivemuon%2Faiida--muon-blue?logo=github)](https://github.com/positivemuon/aiida-muon)
[![AiiDA](https://img.shields.io/badge/AiiDA-%E2%89%A52.0-orange)](https://www.aiida.net)

---

Positive muon spin rotation/relaxation/resonance (µSR) is a powerful experimental probe for studying magnetism,
superconductivity, and other phenomena in condensed matter. A key step in interpreting µSR data is knowing
**where the muon stops** inside the host material.

`aiida-muon` automates this search by running a battery of DFT supercell relaxations — via
[Quantum ESPRESSO](https://www.quantum-espresso.org/) through
[aiida-quantumespresso](https://aiida-quantumespresso.readthedocs.io) — and analysing the results with
symmetry-based clustering. For magnetic materials it also computes the **contact hyperfine field** and the
**classical dipolar field** at each candidate site.

<div class="grid cards" markdown>

-   **Installation**

    ---

    Install `aiida-muon` and its dependencies, then verify the setup.

    [To the installation guide](installation.md)

-   **Tutorials**

    ---

    Step-by-step guides covering non-magnetic (Si), ferromagnetic (Fe), and
    antiferromagnetic (MnO) test cases.

    [To the tutorials](tutorials/index.md)

</div>

<div class="grid cards" markdown>

-   **How-To Guides**

    ---

    Concise recipes for common tasks: building inputs, handling magnetic
    structures, DFT+U, pre-relaxation, and results export.

    [To the how-to guides](how_to/index.md)

-   **Advanced Topics**

    ---

    Workflow internals and the experimental machine-learning features.

    [To the advanced topics](advanced/index.md)

</div>

## Key features

- Automated generation of a grid of candidate muon stopping sites using the
  [NICHE](https://github.com/positivemuon/aiida-muon) algorithm.
- Full-mesh DFT relaxation of muon supercells via `PwRelaxWorkChain` (Quantum ESPRESSO).
- Optional **Gamma-point pre-relaxation** to cheaply reduce the number of starting sites.
- Optional **MLIP pre-relaxation** (experimental) for fast prescreening with machine-learning
  interatomic potentials.
- Optional **automated supercell size determination** via
  [aiida-impuritysupercellconv](https://github.com/positivemuon/aiida-impuritysupercellconv).
- Symmetry-aware clustering of relaxed sites to identify unique candidate positions.
- Contact hyperfine field from DFT spin density (pp.x) for magnetic systems.
- Classical dipolar field computed with [muesr](https://github.com/bonfus/muesr).
- Full AiiDA provenance: every intermediate result is stored in the database.

## How to cite

If you use this package for published research, please cite:

> Ifeanyi J. Onuorah, Miki Bonacci et al.,
> [*Automated computational workflows for muon spin spectroscopy*](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00314d),
> Digital Discovery **4**, 523-538 (2025).

Also cite the underlying AiiDA infrastructure:

> Sebastiaan P. Huber et al.,
> [*AiiDA 1.0, a scalable computational infrastructure for automated reproducible workflows and data provenance*](https://doi.org/10.1038/s41597-020-00638-4),
> Scientific Data **7**, 300 (2020).

## Acknowledgements

We acknowledge support from:

- The [NCCR MARVEL](http://nccr-marvel.ch/) funded by the Swiss National Science Foundation.
- The PNRR MUR project [ECS-00000033-ECOSISTER](https://ecosister.it/).

<img src="source/images/MARVEL_logo.png" width="200"/>
<img src="source/images/ecosister_logo.png" width="200"/>
