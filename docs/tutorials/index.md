# Tutorials

Welcome to the `aiida-muon` tutorials. These step-by-step guides walk you through
the most common use cases, from a simple non-magnetic material to a magnetically ordered system.

<div class="grid cards" markdown>

-   **Finding Muon Sites (Si, Fe, MnO)**

    ---

    **Duration**: ~15 minutes (setup) + HPC time

    **Level**: Beginner

    Learn the full workflow from start to finish: load a structure, build the
    workflow inputs, submit to AiiDA, and inspect the candidate muon sites.

    [Start Tutorial 1](1_basic_findmuon.md)

-   **Magnetic Materials**

    ---

    **Duration**: ~20 minutes (setup) + HPC time

    **Level**: Intermediate

    Extend the basic workflow to magnetic systems: provide magnetic moments,
    enable spin-polarised DFT, and retrieve the contact hyperfine and dipolar
    fields at the candidate sites.

    [Start Tutorial 2](2_magnetic_materials.md)

</div>

!!! tip "Prerequisites"
    Before starting, make sure you have:

    - Completed the [installation](../installation.md).
    - A configured AiiDA profile (`verdi status` should show everything green).
    - A `pw.x` code set up in AiiDA (`verdi code list`).
    - The `SSSP/1.3/PBE/efficiency` pseudopotential family installed
      (`verdi data core.upf listfamilies`).
