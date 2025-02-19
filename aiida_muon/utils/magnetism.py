# -*- coding: utf-8 -*-
import numpy as np
import typing as t 

from muesr.core import Sample
from muesr.core.atoms import Atoms
from muesr.engines.clfc import find_largest_sphere, locfield
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.core import PeriodicSite, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry import analyzer
from pymatgen.util.coord import pbc_shortest_vectors

from aiida.engine import calcfunction
from aiida import orm
from aiida.orm import StructureData as LegacyStructureData



def get_collinear_mag_kindname(p_st, magm, half=True):
    """
    Using pymatgen structure and magmom properties, provides the kind name for
    magnetically distinct species for spin polarized calculations with aiida-QuantumESPRESSO.

    Parameters
    ----------
        p_st: pymatgen.core.Structure
              Structure to be analyzed.

        magm: list
            corresponding magmom properties of the pymatgen structure instance.
        half: bool
            if True, only sets magnetic moments to 0,+-0.5.

    Raises
    ------
        assertion error.

    Returns: the input pymatgen structure instance but with additional kind name property
             and a dict of the new distinct magnetic site specie and +-0.5 moment value.


    """
    assert p_st.num_sites == len(magm)

    # get and force collinear example for 0.411_Tb5Ge4.mcif and 0.271_Tb2MnNiO6.mcif
    coll_m, drx = Magmom.get_consistent_set_and_saxis(magm)
    for i, m in enumerate(coll_m):
        # returns mtm [0, 0, t], t is the total mtm, and saxis rotated.
        mtm = Magmom(m).get_00t_magmom_with_xyz_saxis()
        p_st[i].properties["magmom"] = Magmom([0.0, 0.0, mtm[2]])
        # p_st[i].properties['magmom'] = Magmom(m)
        # p_st[i].properties['colmom'] = m

    p_st2 = CollinearMagneticStructureAnalyzer(p_st, make_primitive=False)

    # remove this check later
    try:
        assert p_st2.is_magnetic and p_st2.is_collinear
    except AssertionError:
        print(" Not collinear, should not happen")

    st_spin = p_st2.get_structure_with_spin()

    # collect magnetic elements by name. For each of them create kinds
    kind_values = []
    magnetic_elements_kinds = {}
    # n_sites = len(st_spin)

    for s_idx, site in enumerate(st_spin):
        spin = site.specie.spin
        element = site.specie.element.symbol

        kind_name = None
        if not np.allclose(np.abs(spin), 0.0):
            # check if element was already found to be magnetic, otherwise return empty dictionary
            kinds_for_element = magnetic_elements_kinds.get(element, {})

            # if the element spin is already found, give same kind name, otherwise add new kind name
            for kind, kind_spin in kinds_for_element.items():
                if np.allclose(spin, kind_spin):
                    kind_name = kind
                    break
            else:
                kind_name = "{}{}".format(element, len(kinds_for_element) + 1)
                kinds_for_element[kind_name] = spin

            # store the updated list of kinds for this element in the full dictionary.
            magnetic_elements_kinds[element] = kinds_for_element

        kind_values.append(kind_name)

    # if element == 'H':
    #    kind_values[-1] = 'No'

    # merge below with above
    # get dictionary of +-0.5 starting magnetization for spin polarized calculation
    import copy

    mgek = copy.deepcopy(magnetic_elements_kinds)
    for k in mgek.keys():
        for idx in mgek[k].keys():
            spin = mgek[k][idx]
            if half:
                mgek[k][idx] = round(spin * (0.5 / abs(spin)), 1)
            else:
                mgek[k][idx] = round(spin, 1)
    start_mag_dict = {}
    for val in mgek.values():
        start_mag_dict.update(val)

    # return p_st.copy(site_properties={'kind_name': kind_values}), magnetic_elements_kinds,  start_mag
    return p_st.copy(site_properties={"kind_name": kind_values}), start_mag_dict

@calcfunction
def make_collinear_getmag_kind(aiid_st, magmm, half=True):
    """
    This calls the 'get_collinear_mag_kindname' utility function.
    It takes the provided magmom, make it collinear and then with
    assign kind_name property for each atom site relevant
    spin polarized calculation.

    Returns: Structure data and dictionary of pw starting magnetization card.
    """
    p_st = aiid_st.get_pymatgen_structure()
    # magmm = magmom_node.get_array('magmom')
    # from array to Magmom object
    magmoms = [Magmom(magmom) for magmom in magmm]

    st_k, st_m_dict = get_collinear_mag_kindname(p_st, magmoms, half)

    aiida_st2 = LegacyStructureData(pymatgen=st_k)
    aiid_dict = orm.Dict(dict=st_m_dict)

    return {"struct_magkind": aiida_st2, "start_mag_dict": aiid_dict}

def compute_dip_field(p_st, magm, sc_mat, r_supst, cnt_field):
    """
    Computes the dipolar contribution to the muon local field
    using the MUESR code (10.7566/JPSCP.21.011052).

    Parameters:
    -----------
        p_st: Input pymatgen structure instance
        magm: corresponding magmom properties of the pymatgen structure instance
        sc_mat: Input supercell matrix
        r_supst: pymatgen  relaxed structure instance  with the muon position
        cnt_field : DFT calculated contact field at the muon site in r_supst


    Returns:
    ---------
        The B_dip+B_L vectors, its norm and B_Total=B_dip+B_L +B_C vectors

    """

    # set-up required parameters
    assert p_st.num_sites == len(magm)

    for i, m in enumerate(magm):
        p_st[i].properties["magmom"] = Magmom(m)

    p_scst = p_st.copy()
    p_scst.make_supercell(sc_mat)

    # center the supercell around the muon for no-PBC cases. muon at (0.5,0.5,0.5)
    musite = r_supst.frac_coords[r_supst.atomic_numbers.index(1)]
    # remove muon atom from cell
    r_supst.remove_sites([r_supst.atomic_numbers.index(1)])
    r_supst.translate_sites(
        range(r_supst.num_sites), 0.5 - musite, frac_coords=True, to_unit_cell=True
    )
    p_scst.translate_sites(
        range(p_scst.num_sites), 0.5 - musite, frac_coords=True, to_unit_cell=True
    )

    # Gen supercell fourier comp. mag moments in complex form
    momt = p_scst.site_properties["magmom"]
    fc_sup = np.zeros([len(momt), 3], dtype=complex)
    for i, m in enumerate(momt):
        # fc_sup[i] = m.get_moment_relative_to_crystal_axes(p_scst.lattice).astype(complex)
        fc_sup[i] = m.get_moment().astype(complex)

    # get the s_axis for transforming the contact field that is isotropic
    s_axis = Magmom.get_suggested_saxis(momt)

    # start dipolar calculations

    smp = Sample()

    # get structure from pymatgen-->ase_atoms-->Muesr_atoms
    ase_atom = AseAtomsAdaptor.get_atoms(p_scst)
    # smp.cell = ase_atom  #raise TypeError('Cell is invalid.') for MnO.mcif
    atoms = Atoms(
        symbols=ase_atom.symbols,
        scaled_positions=ase_atom.get_scaled_positions(),
        cell=ase_atom.cell,
        pbc=True,
    )
    smp.cell = atoms

    smp.new_mm()
    smp.mm.k = np.array([0.0, 0.0, 0.0])
    # smp.mm.fc_set(fc_sup, coord_system=2)
    smp.mm.fc = fc_sup

    # smp.add_muon(musite+0.5-musite)
    smp.add_muon([0.5, 0.5, 0.5])
    # smp.current_mm_idx=0
    radius = find_largest_sphere(smp, [50, 50, 50])

    # compute B in full(50x50x50 supercell) in the pristine structre
    r_f_ps = locfield(smp, "s", [50, 50, 50], radius)

    # compute B only within the supercell  using the pristine structre,
    # To include muon induced relaxation effects
    radius_n = np.min(r_supst.lattice.abc)
    r_s_ps = locfield(smp, "s", [50, 50, 50], radius_n)

    # change the cell to the relaxed
    # smp.cell = AseAtomsAdaptor.get_atoms(r_supst)
    ase_atom_r = AseAtomsAdaptor.get_atoms(r_supst)
    atoms_r = Atoms(
        symbols=ase_atom_r.symbols,
        scaled_positions=ase_atom_r.get_scaled_positions(),
        cell=ase_atom_r.cell,
        pbc=True,
    )
    smp.cell = atoms_r

    # compute B only within the supercell the using the relaxed structre
    r_s_rlx = locfield(smp, "s", [50, 50, 50], radius_n)

    # B (B_dip+B_lor) vector with muon distortion effects in tesla (https://doi.org/10.1016/j.cpc.2022.108488)
    B_D = r_f_ps[0].D + r_s_rlx[0].D - r_s_ps[0].D
    B_DL = B_D + r_f_ps[0].L

    return (
        B_DL,
        B_DL + s_axis * cnt_field,
        np.linalg.norm(B_DL + s_axis * cnt_field, axis=0),
    )


@calcfunction
def compute_dipolar_field(
    p_st: LegacyStructureData,
    magmm: orm.List,
    sc_matr: orm.List,
    r_supst: LegacyStructureData,
    cnt_field: orm.Float,
):
    """
    This calcfunction calls the compute dipolar field
    """

    pmg_st = p_st.get_pymatgen_structure()
    r_sup = r_supst.get_pymatgen_structure()

    b_fld = compute_dip_field(pmg_st, magmm, sc_matr, r_sup, cnt_field.value)

    return orm.List([b_fld])