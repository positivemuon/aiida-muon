# -*- coding: utf-8 -*-
import numpy as np
from muesr.core import Sample
from muesr.core.atoms import Atoms
from muesr.engines.clfc import find_largest_sphere, locfield
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.core import PeriodicSite, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry import analyzer
from pymatgen.util.coord import pbc_shortest_vectors

##########################################################################


# revisit
def get_struct_wt_distortions(prist_stc, rlxd_stc, n_mupos, ipt_st):
    """
    Experimental Function!

    Translates displacement due to the muon from one muon to a
    magnetically inequivalent site.
    Returns: Structure with translated displ and muon position

    This function assumes that H is the particle of interest.
    This is probably a problem when H atoms are already present.
    """
    tol = 0.001

    # get symmetry operations
    spp = analyzer.SpacegroupAnalyzer(ipt_st)
    ops = spp.get_symmetry_operations()
    # opsg = spp.get_space_group_operations()
    opg = spp.get_point_group_operations()

    # get  and remove relaxed muon position
    mupos_rlx = rlxd_stc.frac_coords[rlxd_stc.atomic_numbers.index(1)]
    # rlxd_stc.pop()
    rlxd_stc.remove_sites([rlxd_stc.atomic_numbers.index(1)])

    # remove initial muon position from prist_stc
    # prist_stc.pop()
    prist_stc.remove_sites([prist_stc.atomic_numbers.index(1)])

    assert len(rlxd_stc.frac_coords) == len(prist_stc.frac_coords)

    # get the symmetry operations that can transform mupos_rlx to n_mupos
    symm_op = []
    for i, op in enumerate(ops):
        newp = op.operate(mupos_rlx) % 1
        if np.all(np.abs(newp - n_mupos) < tol):
            symm_op.append(i)

    nw_stc = prist_stc.copy()

    # get and transform displacement vectors
    if len(symm_op) > 0:
        disp = rlxd_stc.frac_coords - prist_stc.frac_coords
        t_disp = opg[symm_op[0]].operate_multi(disp)

        ##instead get  disp with transforming atoms
        # t_disp2 = np.zeros([nw_stc.num_sites, 3])
        for i in range(len(nw_stc)):
            nw_stc.translate_sites(i, t_disp[i], frac_coords=True, to_unit_cell=False)
            #
            # new_rlx_pos=ops[symm_op[0]].operate(rlxd_stc[i].frac_coords)
            # t_disp2[i] = new_rlx_pos - prist_stc[i].frac_coords
            # nw_stc.translate_sites(i, t_disp2[i],frac_coords=True, to_unit_cell=False)
    else:
        print(
            "Check symm op in get_struct_wt_distortions func, this should never happen"
        )

    nw_stc.append(
        species="H",
        coords=n_mupos,
        coords_are_cartesian=False,
        validate_proximity=True,
        properties={"kind_name": "H"},
    )

    return nw_stc

def read_wrkc_output(outdata):
    """read workchain output dictionary and convert pymatgen structure dict to structure object"""
    # outdata is aiida dictionary
    out_dict = {}
    data = outdata.get_dict()
    for dd in data.keys():
        py_st = Structure.from_dict(data[dd][0])
        enrgy = data[dd][1]
        out_dict.update({dd: [py_st, enrgy]})
    return out_dict