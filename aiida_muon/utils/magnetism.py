# -*- coding: utf-8 -*-
import numpy as np
import typing as t 

from pylocfield.ewald import compute_dipolar_tensors, compute_field as ewald_sum
from pylocfield.realspace import compute_field as direct_sum

from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.core import PeriodicSite, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry import analyzer
from pymatgen.util.coord import pbc_shortest_vectors

from aiida.engine import calcfunction
from aiida import orm
from aiida.orm import StructureData as LegacyStructureData
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

from aiida_muon.utils.hubbard import create_hubbard_structure


# ---------------------------------------------------------------------------------------------------------------------
# Cutoffs of the Ewald sum of the dipolar field (see `get_ewald_cutoffs`).
#
# `pylocfield.ewald.compute_dipolar_tensors` computes, for every atom of the supercell, a sum over real-space lattice
# vectors R (cutoff `r_c`, in Angstrom) and a sum over reciprocal-space lattice vectors G (cutoff `g_c`, in
# 1/Angstrom *without* the factor 2*pi: the vectors that are summed have length 2*pi*|g|, |g| <= g_c). Its defaults,
# r_c = g_c = 12, are enormous: |G| up to 2*pi*12 = 75 1/Angstrom. The number of lattice vectors, and so the cost,
# grows as the cube of the cutoffs and with the volume of the cell, and it is repeated for every atom: for a 54 atoms
# supercell of bcc Fe (3x3x3) the default takes ~45 s, for a 64 atoms MnO supercell ~25 s. `muesr` took < 1 s.
#
# But the Ewald method splits the sum in two parts that are damped by Gaussians, whose width is set by the parameter
# rho (`rho`, in 1/Angstrom; the default of pylocfield is 1.8, and we pass it explicitly to be sure that the cutoffs
# below always refer to the rho in use):
#
#   reciprocal space:  terms ~ exp(-|G|^2 / (4 rho^2))
#   real space:        terms ~ exp(-(rho |r|)^2)         (|r| = distance between the muon and the atom)
#
# so that, to have the terms below a relative tolerance `tol`, we only need
#
#   |G| >= G_conv = 2 rho sqrt(ln(1/tol))     and      |r| >= R_conv = sqrt(ln(1/tol)) / rho
#
# With rho = 1.8 and tol = 1e-10 (ln(1/tol) = 23.03): G_conv = 17.3 1/Angstrom (i.e. g_c = G_conv/(2 pi) = 2.75)
# and R_conv = 2.67 Angstrom. We then multiply them by a safety factor (1.5), so that the neglected terms are
# ~exp(-52) instead of ~exp(-23).
#
# For the reciprocal sum this is all: the G vectors do not depend on where the atoms are.
#
# For the real-space sum there is one more thing. The sum runs over the atoms of the cells R + (cell with the atoms),
# and `r_c` selects the cells R, not the atoms: all the atoms of the cell R are included if |R| <= r_c. The muon is
# at the centre of the supercell (see `compute_dip_field`), so an atom at a distance <= R_conv from the muon
# belongs to a cell with |R| <= R_conv + (the largest distance from the centre of the cell to one of its points), i.e.
# half the longest diagonal of the cell. With r_c = safety*R_conv + half diagonal, no atom that matters is lost,
# whatever the shape and size of the supercell. Few cells are added by this, and the real-space sum is cheap.
#
# Result: same B as with the defaults, within ~1e-12 T (it was verified up to 1e-15 T on bcc Fe and MnO supercells),
# 30-100 times faster. The remaining difference (~1e-4 T on 1-3 T, i.e. < 0.01 %) with respect to the former
# implementation, which used `muesr`, is not due to the cutoffs: it does not change with them.
# ---------------------------------------------------------------------------------------------------------------------
EWALD_RHO = 1.8  # Ewald parameter (1/Angstrom), the default of pylocfield
EWALD_LOG_INVERSE_TOLERANCE = np.log(1e10)  # the terms that are neglected are < 1e-10 (relative), before the safety factor
EWALD_SAFETY_FACTOR = 1.5


def get_ewald_cutoffs(cell, rho: float = EWALD_RHO) -> t.Tuple[float, float]:
    """The cutoffs ``(r_c, g_c)`` of the Ewald sum of the dipolar field for the given cell: see the comment above.

    :param cell: the cell (an ``ase.cell.Cell`` or a 3x3 array, in Angstrom) of the supercell in which the dipolar field
        is computed, with the muon at its centre.
    :param rho: the Ewald parameter, in 1/Angstrom.
    :return: ``r_c`` in Angstrom and ``g_c`` in 1/Angstrom (without the factor 2*pi), as
        :func:`pylocfield.ewald.compute_dipolar_tensors` wants them.
    """
    lattice = np.asarray(cell, dtype=float)
    sqrt_log = np.sqrt(EWALD_LOG_INVERSE_TOLERANCE)

    # |G| beyond which the reciprocal-space terms are negligible; pylocfield wants |G| / (2 pi).
    g_c = EWALD_SAFETY_FACTOR * 2.0 * rho * sqrt_log / (2.0 * np.pi)

    # |r| beyond which the real-space terms are negligible, plus the cells needed to reach it from the muon, which is at
    # the centre of the cell: half of the longest diagonal (the 4 diagonals are a1 +- a2 +- a3).
    half_diagonal = 0.5 * max(
        np.linalg.norm(lattice[0] + sign_2 * lattice[1] + sign_3 * lattice[2])
        for sign_2 in (1, -1)
        for sign_3 in (1, -1)
    )
    r_c = EWALD_SAFETY_FACTOR * sqrt_log / rho + half_diagonal

    return float(r_c), float(g_c)


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
    using the pylocfield package (https://github.com/bonfus/pylocfield, the successor of MUESR,
    10.7566/JPSCP.21.011052).

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
    fc_sup = np.zeros([len(momt), 1, 3], dtype=complex)
    for i, m in enumerate(momt):
        # fc_sup[i] = m.get_moment_relative_to_crystal_axes(p_scst.lattice).astype(complex)
        fc_sup[i, 0, :] = m.get_moment().astype(complex)

    # get the s_axis for transforming the contact field that is isotropic
    s_axis = Magmom.get_suggested_saxis(momt)

    #### start dipolar calculations

    # get structure from pymatgen-->ase_atoms (with the magnetic moments and the muon position, for pylocfield)
    pristine_scst = AseAtomsAdaptor.get_atoms(p_scst)
    pristine_scst.info["q"] = np.array([ [0.0, 0.0, 0.0] ])
    pristine_scst.set_array("fc", 0.5*fc_sup)

    # smp.add_muon(musite+0.5-musite)
    pristine_scst.info["mu"] = np.array([[0.5, 0.5, 0.5]])
    # smp.current_mm_idx=0

    # Compute dipolar tensors with Ewald method
    # (the cutoffs of the sums are set explicitly: with the defaults of pylocfield this takes minutes: see
    # `get_ewald_cutoffs` and the comment on `EWALD_RHO`)
    r_c, g_c = get_ewald_cutoffs(pristine_scst.cell)
    compute_dipolar_tensors(pristine_scst, r_c=r_c, g_c=g_c, rho=EWALD_RHO)
    
    r_f_ps = ewald_sum(pristine_scst)

    # compute B only within the supercell  using the pristine structre,
    # To include muon induced relaxation effects
    radius_n = np.min(r_supst.lattice.abc)
    #r_s_ps = locfield(smp, "s", [50, 50, 50], radius_n)
    r_s_ps = direct_sum(pristine_scst, r_c=radius_n)

    # change the cell to the relaxed
    relaxed_scst = AseAtomsAdaptor.get_atoms(r_supst)
    relaxed_scst.info["q"] = np.array([ [0.0, 0.0, 0.0] ])
    relaxed_scst.set_array("fc", 0.5*fc_sup)
    relaxed_scst.info["mu"] = np.array([[0.5, 0.5, 0.5]])

    # compute B only within the supercell the using the relaxed structre
    r_s_rlx = direct_sum(relaxed_scst, r_c=radius_n)

    # B (B_dip+B_lor) vector with muon distortion effects in tesla (https://doi.org/10.1016/j.cpc.2022.108488)
    B_DL = r_f_ps[0,0] + r_s_rlx[0,0] - r_s_ps[0,0]

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
