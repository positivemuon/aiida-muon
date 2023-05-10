# -*- coding: utf-8 -*-
import numpy as np
from aiida import load_profile, orm
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
codename = "pw7_0@localhost_serial1"  # edit to user
code = orm.Code.get_from_string(codename)

pcodename = "pp7_0@localhost_serial1"  # edit to user
pcode = orm.Code.get_from_string(pcodename)


builder = FindMuonWorkChain.get_builder()
builder.sc_matrix = scmat_node
builder.qe.pw_code = code
builder.qe.pp_code = pcode
builder.qe.pseudofamily = orm.Str("SSSP/1.2/PBE/efficiency")
builder.qe.k_dist = orm.Float(0.601)
builder.qe.charge_supercell = orm.Bool(False)


# uncomment below for Si
"""
parser          = CifParser("data/Si.cif")
smag1        = parser.get_structures(primitive = True)[0]
aiida_structure2 = orm.StructureData(pymatgen = smag1)
builder.qe.structure = aiida_structure2
builder.qe.mu_spacing = orm.Float(1.0) #for Si primitive three mu sites
"""


# uncomment below for Fe
# """
smag1 = Structure.from_file("data/Fe_bcc.mcif", primitive=False)
aiida_structure = orm.StructureData(pymatgen=smag1)
smag = aiida_structure.get_pymatgen_structure()
magmoms = smag1.site_properties["magmom"]
magmom = orm.List([list(magmom) for magmom in magmoms])
builder.structure = aiida_structure
builder.qe.magmom = magmom
builder.mu_spacing = orm.Float(0.75)  # for Fe  no primitive 4 mu sites
##builder.mu_spacing = orm.Float(0.6) #for Fe primitive 4 mu sites
# """

# uncomment below for MnO
"""
smag1 = Structure.from_file('data/MnO.mcif',primitive=True)
aiida_structure = orm.StructureData(pymatgen = smag1)
smag = aiida_structure.get_pymatgen_structure()
magmoms = smag1.site_properties['magmom']
magmom = orm.List([list(magmom) for magmom in magmoms])
builder.structure = aiida_structure
builder.qe.magmom = magmom
builder.mu_spacing = orm.Float(1.6) #for mno primitive 2 mu sites,  4 mno atoms
"""


pw_metadata = {
    "description": "Muonss site calculations for " + smag1.formula,
    #'dry_run' : True,
    "options": {"max_wallclock_seconds": 50000, "resources": {"num_machines": 1}},
    "label": "Si Fe MnO  relax test",
}


pw_settings = {"ONLY_INITIALIZATION": True}


# TO DO.put a check on  parameters that cannot be set by hand in the overrides eg mag, hubbard
parameters_dict = {
    "CONTROL": {"max_seconds": 45000, "forc_conv_thr": 0.1, "etot_conv_thr": 0.1},
    "SYSTEM": {
        "ecutwfc": 30.0,
        "ecutrho": 240.0,
    },
    "ELECTRONS": {
        "electron_maxstep": 100,
        "conv_thr": 1.0e-3,
    },
}


builder.qe.parameters = orm.Dict(dict=parameters_dict)
builder.qe.metadata = orm.Dict(dict=pw_metadata)
# builder.qe.settings =orm.Dict(dict=pw_settings)

node = run(builder)
print(node)
outdata = node["unique_sites"]
# print(outdata)
aa = read_wrkc_output(outdata)
# print(aa)
