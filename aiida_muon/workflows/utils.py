# -*- coding: utf-8 -*-
import numpy as np
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.core import PeriodicSite, Structure
from pymatgen.electronic_structure.core import Magmom


def get_collinear_mag_kindname(p_st, magm):
    """
    Using pymatgen structure and magmom properties, provides the kind name for
    magnetic ditinct species for spin polarized caculations with aiida-quntum espresso

    Parm:
        p_st: pymatgen structure instance
        magm: corresponding magmom properties of the pymatgen structure instance

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
            mgek[k][idx] = round(spin * (0.5 / abs(spin)), 1)

    start_mag_dict = {}
    for val in mgek.values():
        start_mag_dict.update(val)

    # return p_st.copy(site_properties={'kind_name': kind_values}), magnetic_elements_kinds,  start_mag
    return p_st.copy(site_properties={"kind_name": kind_values}), start_mag_dict


def check_get_hubbard_u_parms(p_st):
    """Set Hubbard U parameters for each kind of specie
    Returns: A dictionary of hubbard U species and their values
    """
    # materials_project
    U_dict1 = {
        "Co": 3.32,
        "Cr": 3.7,
        "Fe": 5.3,
        "Mn": 3.9,
        "Mo": 4.38,
        "Ni": 6.2,
        "V": 3.25,
        "W": 6.2,
        "Cu": 0.0,
    }
    # PRB 73, 195107 2006
    U_dict2 = {
        "Co": 3.3,
        "Cr": 3.5,
        "Fe": 4.0,
        "Mn": 4.0,
        "Mo": 0.0,
        "Ni": 6.4,
        "V": 3.1,
        "W": 0.0,
        "Cu": 4.0,
    }

    if "kind_name" in list(p_st.site_properties.keys()):
        spc = p_st.site_properties["kind_name"]
        d_spc = list(set(spc))
    else:
        spc = [str(atm.specie.symbol) for atm in p_st]
        d_spc = list(set(spc))

    # 1 element compounds are not given any U values, 2 for the muon specie
    if len(d_spc) > 2:
        hub_u = {}
        for spk in d_spc:
            spk2 = "".join(filter(str.isalpha, spk))
            if spk2 in U_dict1:
                hub_u[spk] = U_dict2[spk2]
        if hub_u:
            return hub_u
        else:
            return None
    else:
        return None


##########################################################################

from pymatgen.symmetry import analyzer
from pymatgen.util.coord import pbc_shortest_vectors

# from pymatgen.electronic_structure.core import Magmom


def find_equivalent_positions(
    frac_coords, host_lattice, atol=1e-3, energies=None, e_tol=0.05
):
    """
    Returns eqivalent atoms list of
    Energies and energy tolerance (e_tol) are in eV

    """

    lattice = host_lattice.lattice
    # Bring to unit cell
    frac_coords %= 1

    # prepare list of equivalent atoms. -1 mean "not yet analyzed".
    eq_list = np.zeros(len(frac_coords), dtype=np.int32) - 1

    spg = analyzer.SpacegroupAnalyzer(host_lattice, symprec=atol)

    ops = spg.get_symmetry_operations()

    # This hosts all the equivalent positions obtained for each of the
    # lattice points using all symmetry operations.
    eq_pos = np.zeros([len(ops), len(frac_coords), 3])

    for i, op in enumerate(ops):
        eq_pos[i] = op.operate_multi(frac_coords) % 1

    # Compute equivalence list
    for i in range(len(frac_coords)):
        if eq_list[i] >= 0:
            continue

        for j in range(i, len(frac_coords)):
            diff = pbc_shortest_vectors(
                lattice, eq_pos[:, j, :], frac_coords[i]
            ).squeeze()
            if (energies is not None) and (len(energies) == len(frac_coords)):
                if (np.linalg.norm(diff, axis=1) < atol).any() and (
                    abs(energies[i] - energies[j]) < e_tol
                ):
                    eq_list[j] = i
            else:
                if (np.linalg.norm(diff, axis=1) < atol).any():
                    eq_list[j] = i
    return eq_list


def prune_too_close_pos(
    frac_positions, host_lattice, min_distance, energies=None, e_tol=0.05
):
    """Returns index of too close atoms"""
    # energies and tolerance should be in eV
    lattice = host_lattice.lattice

    s_idx = np.arange(len(frac_positions))

    for i, pi in enumerate(frac_positions):
        for j, pj in enumerate(frac_positions):
            if j > i:
                diff = pbc_shortest_vectors(lattice, pi, pj).squeeze()
                # print(i,j,diff,np.linalg.norm(diff, axis=0))
                if (energies is not None) and (len(energies) == len(frac_positions)):
                    if (np.linalg.norm(diff, axis=0) < min_distance) and (
                        abs(energies[i] - energies[j]) < e_tol
                    ):
                        s_idx[j] = -1
                else:
                    if np.linalg.norm(diff, axis=0) < min_distance:
                        s_idx[j] = -1

    # frac_positions = np.delete(frac_positions,s_idx,0) #use if append
    # frac_positions = frac_positions[s_idx == np.arange(len(frac_positions))]
    return s_idx


def get_poslist1_not_in_list2(pos_lst1, pos_lst2, host_lattice, d_tol=0.5):
    """
    Function that compares two position list
    and returns position of list1 not in list2
    """
    lattice = host_lattice.lattice
    s_idx = np.zeros(len(pos_lst1), dtype=np.int32) - 1
    for i, pi in enumerate(pos_lst1):
        for j, pj in enumerate(pos_lst2):
            diff = pbc_shortest_vectors(lattice, pi, pj).squeeze()
            if np.linalg.norm(diff, axis=0) < d_tol:
                s_idx[i] = i

    pos_not_in_list = pos_lst1[s_idx != np.arange(len(pos_lst1))]
    return pos_not_in_list


def cluster_unique_sites(pk_list, mu_list, enrg_list, p_st, p_smag):
    """
    Function that clusters + get symmetry unique muon positions
    from list of muon sites from relax calculations.

    The clustering is in three steps.
    Step1: Prune equivalent (same position) positions in the list
           to a distance threshold of 0.5 Angstrom and energy difference within 0.05 eV.

    Step2: Find and remove magnetically+symmetrically (using p_smag ) eqvivalent sites
           within symmetry tolerance of 0.05 Angstrom and energy difference within 0.05 eV.

    Step3: Check to see if all the magnetically inquivalent sites of given muon list are
            all captured, else find them and give new listof magnetically inequivalent sites
            to be calculated.

    Params:
        pk_list: list of the pk_lists corresponding to the calc. that gives the muon sites
        mu_list: list of the muon sites in fractional coordinates
        enrg_list: list of their corresponding relative DFT energies in units of eV
        p_st: A pymatgen "unitcell" structure instance
        p_smag: A pymatgen "magnetic unitcell" structure instance

    Returns:
          (i) list of symmterically unique muon positions from the initial
                list (mu_list) provided. The corresponding pk_lists and
                energies in eV are returned as well
          (ii) list of magnetically inequivalent positions to be sent
               back to the daemon for relaxations.

    """

    assert len(pk_list) == len(mu_list) == len(enrg_list)

    # if no magnetic symmetry
    # if p_smag is None:
    #    p_smag = p_st.copy()

    # assert pymatgen structure instance
    # points to consider:
    # 1what of when we have only the mcif? how to we get and decide the corresponding cif?
    # 2We can set two sets of threshold, normal and loose?
    # 3For step 3 of checking  non present magnetic inequival sites,
    # we can decide to check this only for sites with energy less than 0.6 eV?

    # We can set two sets of threshold, normal and loose?
    # Normal thresholds
    d_tol = 0.5  # inter-site distance tolerance for clustering in Ang
    s_tol = 0.05  # symmetry tolerance for clustering in Ang
    e_tol = 0.05  # energy difference tolerance for clustering in eV
    a_tol = 1e-3  # symmetry tolerance for printing equi sites in Ang

    # Step1
    idx = prune_too_close_pos(mu_list, p_smag, d_tol, enrg_list)
    mu_list2 = mu_list[idx == np.arange(len(mu_list))]
    enrg_list2 = enrg_list[idx == np.arange(len(enrg_list))]
    pk_list2 = pk_list[idx == np.arange(len(pk_list))]

    # Step 2
    ieq = find_equivalent_positions(
        mu_list2, p_smag, s_tol, energies=enrg_list, e_tol=e_tol
    )
    mu_list3 = mu_list2[ieq == np.arange(len(mu_list2))]
    enrg_list3 = enrg_list2[ieq == np.arange(len(enrg_list2))]
    pk_list3 = pk_list2[ieq == np.arange(len(pk_list2))]

    # The cluster/unque positions from the given muon list
    clus_pos = list(zip(pk_list3, mu_list3, enrg_list3))
    clus_pos_sorted = sorted(clus_pos, key=lambda x: x[2])

    # TODO: EDIT THIS FOR ONLY WHEN MAGNETIC? CAN THIS CHANGE ANYTHING FOR NON_MAGNETIC SYSTEMS?
    # Step 3: Now check if there are magnetic inequivalent sites not in the given list.
    # we can decide to check this only for sites with energy less than 0.6 eV?
    spg = analyzer.SpacegroupAnalyzer(p_st)
    ops = spg.get_symmetry_operations(cartesian=False)

    new_pos_to_calc = []
    for i, pp in enumerate(mu_list3):
        # get all the equivalent positions with unitcell symmetry
        pos = [x.operate(pp) % 1 for x in ops]
        pos = np.unique(pos, axis=0)

        # find magnetically inequivalent in pos
        ieq_l = find_equivalent_positions(pos, p_smag, atol=a_tol)
        pos2 = pos[ieq_l == np.arange(len(pos))]

        # if magnetically inequivalent pos. exists
        if len(pos2) > 1:
            # check to see if already in the given muon list
            new_pos = get_poslist1_not_in_list2(
                pos2, mu_list, host_lattice=p_st, d_tol=d_tol
            )
            if new_pos.any():
                for j in new_pos:
                    # new_pos_to_calc.append(j.tolist())
                    # identify which site it is magnetic equivalent to with the label and append
                    new_pos_to_calc.append((pk_list3[i], j.tolist()))

    return clus_pos_sorted, new_pos_to_calc


# revisit
def get_struct_wt_distortions(prist_stc, rlxd_stc, n_mupos, ipt_st_mag):
    """
    Translates displacement due to the muon from one muon to a
    magnetically inequivalent site.
    Returns: Structure with translated displ and muon position
    """
    tol = 0.0001

    # get symmetry operations
    spp = analyzer.SpacegroupAnalyzer(ipt_st_mag)
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
        newp = op.operate(mupos_rlx)
        if np.all(np.abs(newp - n_mupos) < tol):
            symm_op.append(i)

    nw_stc = prist_stc.copy()

    # get and transform displacement vectors
    disp = rlxd_stc.frac_coords - prist_stc.frac_coords
    t_disp = opg[symm_op[1]].operate_multi(disp)

    ##instead get  disp with transforming atoms
    # t_disp2 = np.zeros([nw_stc.num_sites, 3])
    for i in range(len(nw_stc)):
        nw_stc.translate_sites(i, t_disp[i], frac_coords=True, to_unit_cell=False)
        #
        # new_rlx_pos=ops[symm_op[0]].operate(rlxd_stc[i].frac_coords)
        # t_disp2[i] = new_rlx_pos - prist_stc[i].frac_coords
        # nw_stc.translate_sites(i, t_disp2[i],frac_coords=True, to_unit_cell=False)

    nw_stc.append(
        species="H",
        coords=n_mupos,
        coords_are_cartesian=False,
        validate_proximity=True,
        properties={"kind_name": "H"},
    )

    return nw_stc


def load_workchain_data(data):
    pk_list = []
    mu_list = []
    enrg_list1 = []
    for j, d in enumerate(data):
        pk_list.append(d["pk"])
        scst = d["rlxd_struct"]
        mu_list.append(scst.frac_coords[scst.atomic_numbers.index(1)])  # for muon
        enrg_list1.append(d["energy"])

    # get energy diff and convert to eV
    ry2ev = 13.605698066
    e_min = min(enrg_list1)
    enrg_list = [(enrg - e_min) * ry2ev for enrg in enrg_list1]

    return np.array(pk_list), np.array(mu_list), np.array(enrg_list)
