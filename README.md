# aiida-muon
An [AiiDA](www.aiida.net) workflow plugin for finding candidate implantation site(s) for the muon and contact hyperfine contribution to the local field. Candidate sites are obtained DFT relaxation calculations with  the [Quantum-Espresso code using its aiida plugin](https://aiida-quantumespresso.readthedocs.io/en/latest/) and subsequent symmetry analysis.

**Please note**: the code supports Quantum ESPRESSO versions higher or equal than v7.1, due to the change in the Hubbard card format in the pw.x input file.

## Dependencies
To run the FindMuonWorkChain, aiida-core, plugin installations and aiida-quantum espresso code and computer setups are required.

## Available Workflows and utilities
```
aiida_muon/
└── workflows
    ├── __init__.py
    └── find_muon.py
    └── utils.py
```

## Installation
install this repository as:

```
git clone https://github.com/positivemuon/aiida-muon.git
cd aiida-muon/
pip install -e .
```
## Example

These examples are for Si, Fe and MnO to demonstrate the capabilities of the workchain. For the purpose of quick run "loose" calculation parameters have been chosen for a quick 1 processor run.

Run the workflow following the example as;

```
cd examples/
python run_example_si_fe_mno.py
```
* (caveat: labels of codes to be edited)

## Acknowledgements
We acknowledge support from:
* the [NCCR MARVEL](http://nccr-marvel.ch/) funded by the Swiss National Science Foundation;
* the PNRR MUR project [ECS-00000033-ECOSISTER](https://ecosister.it/);

<img src="https://raw.githubusercontent.com/positivemuon/aiida-muon/main/docs/source/images/MARVEL_logo.png" width="250px" height="131px"/>
<img src="https://raw.githubusercontent.com/positivemuon/aiida-muon/main/docs/source/images/ecosister_logo.png" width="300px" height="84px"/>