# -*- coding: utf-8 -*-
"""
Example: Finding muon sites in Si, Fe and MnO using FindMuonWorkChain.

This script mirrors the Tutorial 1 in the documentation:
  docs/tutorials/1_basic_findmuon.md

Edit the code labels and choose a system (Si / Fe / MnO) before running.
"""

from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory
from ase.build import bulk
from aiida.orm import StructureData

load_profile()

FindMuonWorkChain = WorkflowFactory('muon.find_muon')

# ---------------------------------------------------------------------------
# 1. Choose the system and load a structure
# ---------------------------------------------------------------------------

system = "Fe"   # change to "Fe" or "MnO"

if system == "Si":
    atoms = bulk('Si', 'diamond', a=5.43)
    structure = StructureData(ase=atoms)
    magmom = None
    spin_pol_dft = False

elif system == "Fe":
    atoms = bulk('Fe', 'bcc', a=2.87)
    structure = StructureData(ase=atoms)
    magmom = [[0, 0, 2.2]]   # 2.2 µB along z for each Fe site
    spin_pol_dft = True

elif system == "MnO":
    atoms = bulk('MnO', crystalstructure='rocksalt', a=4.45)
    structure = StructureData(ase=atoms)
    magmom = [[0, 0, 4.5], [0, 0, -4.5]]   # AFM ordering
    spin_pol_dft = True

# ---------------------------------------------------------------------------
# 2. Load codes  –  edit labels to match your installation
# ---------------------------------------------------------------------------

pw_code = orm.load_code('pw-7.3@mpc3129')   # required for all systems
pp_code = orm.load_code('pp-7.3@mpc3129')  # required for magnetic systems

# ---------------------------------------------------------------------------
# 3. Build workflow inputs via the protocol helper
# ---------------------------------------------------------------------------

kwargs = dict(
    pw_code=pw_code,
    pp_code=pp_code,
    structure=structure,
    mu_spacing=0.5,
    sc_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], # very minimal supercell for a quick test; increase if you have time and resources
    charge_supercell=True,
    full_dft_relax=True,
    spin_pol_dft=spin_pol_dft,
    pre_clustering=False,  # analyze and recompute after relaxations
    gamma_pre_relax=True,  # pre-relax with Gamma-only k-point mesh
)

if magmom is not None:
    kwargs['magmom'] = magmom

if system == "MnO":
    kwargs['mu_spacing'] = 1.5
    kwargs['hubbard'] = True    # apply DFT+U automatically

builder = FindMuonWorkChain.get_builder_from_protocol(**kwargs)

# Adjust scheduler options to match your cluster
builder.relax.base.pw.metadata.options = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 4},
    'max_wallclock_seconds': 7200,
}


# ---------------------------------------------------------------------------
# 4. Submit
# ---------------------------------------------------------------------------

node = submit(builder)
print(f"{system} workflow submitted with PK: {node.pk}")
print(f"Monitor with:  verdi process status {node.pk}")

# ---------------------------------------------------------------------------
# 5. Retrieve results (run after the workflow finishes)
# ---------------------------------------------------------------------------
#
# node = orm.load_node(<PK>)
#
# all_sites    = node.outputs.all_sites.get_dict()
# unique_sites = node.outputs.unique_sites.get_dict()
# print("Unique muon sites:", unique_sites)
#
# For magnetic systems:
# if hasattr(node.outputs, 'unique_sites_hyperfine'):
#     print("Hyperfine fields:", node.outputs.unique_sites_hyperfine.get_dict())
# if hasattr(node.outputs, 'unique_sites_dipolar'):
#     print("Dipolar fields:",   node.outputs.unique_sites_dipolar.get_list())
#
# Export to a pandas DataFrame:
# from aiida_muon.utils.export_findmuon import get_clustering_after_run
# df = get_clustering_after_run(node)
# print(df)


# NB: FOR A PROPER RUN IT IS SUFFICIENT TO PROVIDE ONLY
# (I)INPUT STRUCTURE/MAGMOM (II) SC MATRIX (III)THE PW AND PP CODES
