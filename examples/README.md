# AiiDA-Muon Examples

This directory contains examples demonstrating various features of `aiida-muon`.

## Available Examples

### 1. MLIP Relaxation (`run_mlip_relaxation.py`)

Python script demonstrating structure relaxation using Machine Learning Interatomic Potentials (MLIPs) through `aiida-pythonjob`.

**Features:**
- MACE-MP calculator example
- CHGNet calculator example  
- Custom calculator configurations
- Direct PythonJob usage

**Usage:**
```bash
# Run MACE example (default)
python run_mlip_relaxation.py mace

# Run CHGNet example
python run_mlip_relaxation.py chgnet

# Run custom MACE configuration
python run_mlip_relaxation.py custom

# Direct PythonJob submission
python run_mlip_relaxation.py direct
```

**Requirements:**
- `aiida-pythonjob`
- MLIP package: `mace-torch`, `chgnet`, or `m3gnet`
- Configured `pythonjob` code in AiiDA

**Setup PythonJob code:**
```bash
verdi code create core.code.installed \
    --label pythonjob \
    --computer localhost \
    --default-calc-job-plugin pythonjob \
    --filepath-executable "$(which python)" \
    --prepend-text "export OMP_NUM_THREADS=1"
```

### 2. MLIP Relaxation Notebook (`mlip_relaxation_example.ipynb`)

Interactive Jupyter notebook with comprehensive examples of MLIP-based relaxations.

**Contents:**
- Loading and visualizing structures
- Setting up different MLIP calculators (MACE, CHGNet)
- Submitting and monitoring calculations
- Analyzing relaxed structures
- Comparing initial vs. relaxed geometries
- Batch relaxations for multiple structures

**To use:**
```bash
jupyter notebook mlip_relaxation_example.ipynb
```

### 3. FindMuon Workflow (`run_example_si_fe_mno.py`)

Example of running the full `FindMuonWorkChain` for muon site determination using DFT.

**Features:**
- Silicon (non-magnetic)
- BCC Iron (magnetic)
- MnO (magnetic, Hubbard U)
- Supercell generation
- Muon site clustering
- Hyperfine field calculations

**Usage:**
```bash
python run_example_si_fe_mno.py
```

Edit the `system` variable in the script to choose between `"Si"`, `"Fe"`, or `"MnO"`.

### 4. Query and Analysis (`LaCoPO_run_builder_query.ipynb`)

Jupyter notebook demonstrating how to query and analyze completed `FindMuonWorkChain` calculations.

## Data Files

The `data/` directory contains example structure files:
- `Si.cif` - Silicon crystal structure
- `Fe_bcc.mcif` - BCC Iron with magnetic information
- `MnO.mcif` - Manganese oxide with magnetic information

## General Requirements

All examples require:
- `aiida-core` (with configured profile and database)
- `aiida-muon`
- `pymatgen`
- `ase`

Specific examples may have additional requirements (see above).

## Configuration

Before running examples, ensure:

1. **AiiDA profile is loaded:**
   ```python
   from aiida import load_profile
   load_profile()
   ```

2. **Codes are configured:**
   - For DFT examples: `pw` and `pp` codes from Quantum ESPRESSO
   - For MLIP examples: `pythonjob` code

3. **Pseudopotentials are available:**
   ```bash
   aiida-pseudo install sssp
   ```

## Troubleshooting

### MLIP Calculator Issues

If you encounter import errors for MLIP calculators, ensure the package is installed:

```bash
# For MACE
pip install mace-torch

# For CHGNet  
pip install chgnet

# For M3GNet
pip install m3gnet
```

### PythonJob Not Found

Install `aiida-pythonjob`:
```bash
pip install aiida-pythonjob
```

### Force Convergence Units

The MLIP relaxation functions use **Ry/bohr** units for `fmax` (matching QE conventions), which are automatically converted to eV/Å internally. The default value of `1e-4 Ry/bohr` corresponds to the moderate protocol in QE for muon calculations.

## Contributing

To add new examples:
1. Create your example script/notebook
2. Add documentation here in the README
3. Include any required data files in `data/`
4. Test the example with a clean environment

## Citation

If you use aiida-muon in your research, please cite:
```
[Citation information to be added]
```
