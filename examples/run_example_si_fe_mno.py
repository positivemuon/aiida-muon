# -*- coding: utf-8 -*-
import numpy as np
from aiida import load_profile, orm
from aiida.plugins import DataFactory
from aiida.engine import run, submit
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

load_profile()

import sys

sys.path.append("../")

from aiida_muon.workflows.find_muon import FindMuonWorkChain
from aiida_muon.workflows.utils import read_wrkc_output

# NB: FOR A PROPER RUN IT IS SUFFICIENT TO PROVIDE ONLY
# (I)INPUT STRUCTURE/MAGMOM (II) SC MATRIX (III)THE PW AND PP CODES


# SCMATRIX
# scmat=np.array([ [[2,0,0],[0,2,0],[0,0,2]],])
# scmat_node = orm.ArrayData()
# scmat_node.set_array('sc_matrix',np.array(scmat))
# scmat_node=orm.List([ [[2,0,0],[0,2,0],[0,0,2]], ])
scmat_node = orm.List(
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ]
)


# EDIT PW AND PP CODE TO SUIT YOURS
codename = "pw-7.2@localhost" # edit
code = orm.Code.get_from_string(codename)

from aiida.orm import StructureData as LegacyStructureData

#choose the StructureData to be used in the simulation.
structuredata="old"
if structuredata=="new":    
    StructureData = DataFactory("atomistic.structure")
else:
    StructureData = LegacyStructureData

builder = FindMuonWorkChain.get_builder()

builder.sc_matrix = scmat_node
builder.relax.base.pw.code = code
builder.pseudo_family = orm.Str("SSSP/1.3/PBE/efficiency")

system = "Si" #Si, Fe, MnO

if system == "Si":
    kpoints_distance = orm.Float(0.601)
    charge_supercell = orm.Bool(False)
    parser           = CifParser("data/Si.cif")
    smag1            = parser.get_structures(primitive = True)[0]
    aiida_structure  = orm.StructureData(pymatgen = smag1)
    mu_spacing = orm.Float(1.0) #for Si primitive three mu sites
    ppcode = None
elif system == "Fe":
    kpoints_distance = orm.Float(0.601)
    charge_supercell = orm.Bool(False)
    parser           = CifParser("data/Fe_bcc.mcif")
    smag1            = parser.get_structures(primitive = True)[0]
    aiida_structure  = orm.StructureData(pymatgen = smag1)
    mu_spacing = orm.Float(0.75) #for Fe  no primitive 4 mu sites
    magmoms = smag1.site_properties["magmom"]
    magmom = orm.List([list(magmom) for magmom in magmoms])        
    ppcode = orm.load_code("pw-7.2@localhost")
elif system == "MnO":
    kpoints_distance = orm.Float(0.601)
    charge_supercell = orm.Bool(False)
    parser           = CifParser("data/MnO.mcif")
    smag1            = parser.get_structures(primitive = True)[0]
    aiida_structure  = orm.StructureData(pymatgen = smag1)
    mu_spacing = orm.Float(1.6)  # for mno primitive 2 mu sites,  4 mno atoms
    magmoms = smag1.site_properties["magmom"]
    magmom = orm.List([list(magmom) for magmom in magmoms])         
    ppcode = orm.load_code("pw-7.2@localhost")

## However, this works only with old version of QE... so the best example is the one in the jupyter notebook,
## which uses the protocols which set automatically the U.
builder.qe.hubbard_u = orm.Bool(True)

builder.structure = aiida_structure
builder.mu_spacing = mu_spacing
builder.kpoints_distance = kpoints_distance
builder.pp_code = ppcode

if system in ["Fe","MnO"] and structuredata == "new":
    aiida_structure.magnetization.set_from_components(magnetic_moment_per_site = magmom)
else:
    builder.qe.magmom = magmom

pw_metadata = {
    "description": "Muons site calculations for " + smag1.formula,
    #'dry_run' : True,
    "options": {"max_wallclock_seconds": 50000, "resources": {"num_machines": 1}},
    "label": f"{system} relax test",
}

pw_settings = {"ONLY_INITIALIZATION": True}

# TO DO.put a check on  parameters that cannot be set by hand in the overrides eg mag, hubbard
parameters_dict = {
    "CONTROL": {
        "calculation":"relax",
        "max_seconds": 45000, 
        "forc_conv_thr": 0.1, 
        "etot_conv_thr": 0.1},
    "SYSTEM": {
        "ecutwfc": 30.0,
        "ecutrho": 240.0,
        "occupations": "smearing",
        "smearing":"gaussian",
        "degauss": 0.01,
    },
    "ELECTRONS": {
        "electron_maxstep": 100,
        "conv_thr": 1.0e-3,
    },
}


builder.relax.base.pw.parameters = orm.Dict(parameters_dict)
builder.relax.base.pw.metadata = pw_metadata
# builder.qe.settings =orm.Dict(dict=pw_settings)

node = run(builder)
print(node)
outdata = node["unique_sites"]
# print(outdata)
aa = read_wrkc_output(outdata)
# print(aa)
