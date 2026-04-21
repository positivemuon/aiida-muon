# How-To Guides

Concise recipes for common tasks.  Each guide assumes you have a working AiiDA
profile and a configured `pw.x` code.

<div class="grid cards" markdown>

-   **Build the Workflow Inputs**

    ---

    Use `get_builder_from_protocol` and customise options for your specific
    calculation: scheduler resources, k-point density, pseudopotentials, and more.

    [Builder guide](builder.md)

-   **Magnetic Calculations**

    ---

    Provide magnetic moments, enable spin-polarised DFT, and retrieve
    hyperfine and dipolar fields.

    [Magnetic guide](magnetic.md)

</div>

<div class="grid cards" markdown>

-   **DFT+U / Hubbard**

    ---

    Apply DFT+U corrections with `HubbardStructureData` or via the automatic
    element-lookup heuristics.

    [Hubbard guide](hubbard.md)

-   **Pre-Relaxation Strategies**

    ---

    Speed up large calculations using Gamma-point or MLIP pre-relaxation to
    reduce the number of sites sent to the full DFT relaxation.

    [Pre-relaxation guide](pre_relaxation.md)

</div>

<div class="grid cards" markdown>

-   **Analyse and Export Results**

    ---

    Convert workflow outputs to pandas DataFrames, inspect per-site distortions,
    and generate input files for µSR analysis codes.

    [Analysis guide](analyze_results.md)

</div>
