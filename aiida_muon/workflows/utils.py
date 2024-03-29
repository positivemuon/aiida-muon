# -*- coding: utf-8 -*-
import numpy as np
from muesr.core import Sample
from muesr.core.atoms import Atoms
from muesr.engines.clfc import find_largest_sphere, locfield
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.core import PeriodicSite, Structure
from pymatgen.electronic_structure.core import Magmom

from pymatgen.symmetry import analyzer
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.io.ase import AseAtomsAdaptor



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


def check_get_hubbard_u_parms(p_st, new_structuredata=False):
    """Set Hubbard U parameters for each kind of specie

    Parameters
    ----------
    p_st : pymatgen.core.Structure
           Structure data to be analyzed.


    Returns
    -------
    dict or None
        A dictionary of hubbard U species and their values

    suggestions:
        1. should we return {} instead of None ?
        2. should the list provided as input?
        
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
    if len(d_spc) > 2: # TODO: is this correct? Is it really needed?
        hub_u = {}
        for spk in d_spc:
            # remove numbers from species name
            spk2 = "".join(filter(str.isalpha, spk))
            # check is in the dictorionary
            if spk2 in U_dict2:
                hub_u[spk] = U_dict2[spk2]
        if hub_u:
            return hub_u
        else:
            return None
    else:
        return None


##########################################################################


def find_equivalent_positions(
    frac_coords, host_lattice, atol=1e-3, energies=None, e_tol=0.05
):
    """
    Returns equivalent positions in a list of fractional coordinates given the
    symmetry of host_lattice.

    If energies are not passed, only a threshold on distance is considered.
    Otherwise both conditions (distance and same energy) must be verified.

    Parameters
    ----------
    frac_positions : numpy.array
        The nAtoms x 3 array containing scaled atomic positions.

    host_lattice : pymatgen.core.Structure
        The lattice structure. Used to identify the symmetry operations of the lattice.

    atol: float
         Absolute tolerance (in Angstrom) for the interatimic distance used to
         assume that two positions are the same.

    energies: list or numpy.array
         Energy (or any other scalar property) associated with positions
         reported in frac_positions.

    e_tol: float
        Absolute difference between the scalar property associated with atomic sites.

    Returns
    -------
    np.array
        A list of integers.
        If the value of the item equals its index, the atoms is equivalent to itself.
        If the value equals the another index, the index of the equivalent atom is reported.

    Suggestions:
                 2. change `energies` into `scalar_value` to make it more general.
    
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
    """
    Returns index of atom too close to another one in the cell.

    If energies are not passed, only inter-atomic distance is considered.
    Otherwise both conditions (distance and same energy) must be verified.

    Parameters
    ----------
    frac_positions : numpy.array
        The nAtoms x 3 array containing scaled atomic positions.

    host_lattice : pymatgen.core.Structure
        The lattice structure. Only its lattice property is used.

    min_distance: float
         Minimum distance in Angstrom between atoms. Atoms closer than this
         will be considered the same unless they have different energy associated.

    energies: list or numpy.array
         Energy (or any other scalar property) associated with positions
         reported in frac_positions.

    e_tol: float
        Absolute difference between the scalar property associated with atomic sites.

    Returns
    -------
    np.array
        A list of integers.
        If the value of the item equals its index, the atoms is not within
        `min_distance` from others (or the energy threshold is not satisfied).
        If the value is -1, the atom (and possibly the energy) is close to another
        one in the cell.

    Suggestions:
                 1. modify -1 into the index of the first atom that matched the conditions
                    on energy and distance.
                 2. change `energies` into `scalar_value` to make it more general.
    
    """
    
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
    and returns position of pos_lst1 not in pos_lst2

    Parameters
    ----------
    pos_lst1 : numpy.array (2D)
               First set of scaled coordinates in the form [nat, 3]

    pos_lst1 : numpy.array (2D)
               Second set of scaled coordinates in the form [nat, 3]

    host_lattice: pymatgen.core.Structure
                  Used to access lattice class

    d_tol: float
           Absolute tolerance in Angstrom.
    
    
    Suggestion: this can be obtained with prune_too_close_pos.
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


def cluster_unique_sites(idx_list, mu_list, enrg_list, p_st, p_smag):
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
        idx_list: list of the index corresponding to the calc. that gives the muon sites
        mu_list: list of the muon sites in fractional coordinates
        enrg_list: list of their corresponding relative DFT energies in units of eV
        p_st: A pymatgen "unitcell" structure instance
        p_smag: A pymatgen "magnetic unitcell" structure instance

    Returns:
          (i) list of symmterically unique muon positions from the initial
                list (mu_list) provided. The corresponding idx_lists and
                energies in eV are returned as well
          (ii) list of magnetically inequivalent positions to be sent
               back to the daemon for relaxations.

    """

    assert len(idx_list) == len(mu_list) == len(enrg_list)

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
    idx_list2 = idx_list[idx == np.arange(len(idx_list))]

    # Step 2
    ieq = find_equivalent_positions(
        mu_list2, p_smag, s_tol, energies=enrg_list, e_tol=e_tol
    )
    mu_list3 = mu_list2[ieq == np.arange(len(mu_list2))]
    enrg_list3 = enrg_list2[ieq == np.arange(len(enrg_list2))]
    idx_list3 = idx_list2[ieq == np.arange(len(idx_list2))]

    # The cluster/unque positions from the given muon list
    clus_pos = list(zip(idx_list3, mu_list3, enrg_list3))
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
                    new_pos_to_calc.append((idx_list3[i], j.tolist()))

    return clus_pos_sorted, new_pos_to_calc


# revisit
def get_struct_wt_distortions(prist_stc, rlxd_stc, n_mupos, ipt_st):
    """
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
        newp = op.operate(mupos_rlx)
        if np.all(np.abs(newp - n_mupos) < tol):
            symm_op.append(i)

    nw_stc = prist_stc.copy()

    # get and transform displacement vectors
    disp = rlxd_stc.frac_coords - prist_stc.frac_coords
    #the list must contain a symm operation, if not there is an error somewhere not here
    t_disp = opg[symm_op[0]].operate_multi(disp)

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
    """load and extract relaxed structures for analysis"""
    idx_list = []
    mu_list = []
    enrg_list1 = []
    for j, d in enumerate(data):
        idx_list.append(d["idx"])
        scst = Structure.from_dict(d["rlxd_struct"])
        mu_list.append(scst.frac_coords[scst.atomic_numbers.index(1)])  # for muon
        enrg_list1.append(d["energy"])

    # get energy diff and convert to eV
    # ry2ev = 13.605698066
    e_min = min(enrg_list1)
    enrg_list = [(enrg - e_min) for enrg in enrg_list1]

    return np.array(idx_list), np.array(mu_list), np.array(enrg_list)


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

    # compute B in the single-supercell (1x1x1) using the pristine structre
    r_s_ps = locfield(smp, "s", [1, 1, 1], radius)

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

    # compute B in the single-supercell (1x1x1) using the relaxed structre
    r_s_rlx = locfield(smp, "s", [1, 1, 1], radius)

    # B (B_dip+B_lor) vector with muon distortion effects in tesla (https://doi.org/10.1016/j.cpc.2022.108488)
    B_D = r_f_ps[0].D + r_s_rlx[0].D - r_s_ps[0].D
    B_DL = B_D + r_f_ps[0].L

    return (
        B_DL,
        B_DL + s_axis * cnt_field,
        np.linalg.norm(B_DL + s_axis * cnt_field, axis=0),
    )
