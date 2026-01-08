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

from aiida_muon.workflows.utils import get_struct_wt_distortions

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
                    on energy and distance. -> this is the mapping
                 2. change `energies` into `scalar_value` to make it more general.

    """

    # energies and tolerance should be in eV
    lattice = host_lattice.lattice

    s_idx = np.arange(len(frac_positions))
    mapping = np.arange(len(frac_positions)) + 1  
    mapping[0] = 1

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
                        
                        mapping[j] = mapping[i]
                        #print(i,j,mapping)
                else:
                    if np.linalg.norm(diff, axis=0) < min_distance:
                        s_idx[j] = -1
                        mapping[j] = mapping[i]

    # frac_positions = np.delete(frac_positions,s_idx,0) #use if append
    # frac_positions = frac_positions[s_idx == np.arange(len(frac_positions))]
    return s_idx, mapping

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
    idx, mapping = prune_too_close_pos(mu_list, p_smag, d_tol, enrg_list)
    #print(f"idx: {idx}")
    #print(f"mapping: {mapping}")
    mu_list2 = mu_list[idx == np.arange(len(mu_list))]
    enrg_list2 = enrg_list[idx == np.arange(len(enrg_list))]
    idx_list2 = idx_list[idx == np.arange(len(idx_list))]

    # Step 2
    mu_list20 = mu_list2.copy()
    ieq = find_equivalent_positions(
        mu_list20, p_smag, s_tol, energies=enrg_list, e_tol=e_tol
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
        pos = [x.operate(pp) for x in ops]
        pos = np.unique(pos, axis=0)

        # find magnetically inequivalent in pos
        pos0 = pos.copy()
        ieq_l = find_equivalent_positions(pos0, p_smag, atol=a_tol)
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

    return clus_pos_sorted, new_pos_to_calc, mapping

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

def analyze_structures(init_supc, rlxd_results, input_st, magmom=None):
    """
    This calls "cluster_unique_sites" function that analyzes and clusters
    the relaxed muon positions.

    Returns:
    (i) List of relaxed unique candidate sites supercell structures
    (ii) List of to be calculated magnetic inequivalent supercell structures
    """
    idx_lst, mu_lst, enrg_lst = load_workchain_data(rlxd_results)

    if magmom:
        assert input_st.num_sites == len(magmom)
        st_smag = input_st.copy()
        for i, m in enumerate(magmom):
            st_smag[i].properties["magmom"] = Magmom(m)
    else:
        st_smag = input_st.copy()

    clus_pos, new_pos, mapping = cluster_unique_sites(
        idx_lst, mu_lst, enrg_lst, p_st=input_st, p_smag=st_smag
    )

    # REVISIT
    # TODO-clean: lines below can go in the function 'cluster_unique_sites' with much less lines.

    # get input supercell structure with distortions of new mag inequivalent position
    nw_stc_calc = []
    if len(new_pos) > 0:
        for i, nwp in enumerate(new_pos):
            for j, d in enumerate(rlxd_results):
                if nwp[0] == d["idx"]:
                    init_supc2 = init_supc.get_pymatgen_structure().copy()
                    nw_st = get_struct_wt_distortions(
                        init_supc2,
                        Structure.from_dict(d["rlxd_struct"]),
                        nwp[1],
                        input_st,
                    )
                    if nw_st: nw_stc_calc.append(nw_st)

    uniq_clus_pos = []
    for i, clus in enumerate(clus_pos):
        for j, d in enumerate(rlxd_results):
            if clus[0] == d["idx"]:
                uniq_clus_pos.append(d)

    assert len(clus_pos) == len(uniq_clus_pos)

    return {"unique_pos": uniq_clus_pos, "mag_inequivalent": nw_stc_calc, "mapping": mapping}


def get_clustering_after_run(findmuon_node):
    """
    This function is called after the findmuon workchain has been run.
    It loads the results of the workchain and the input structure and
    calls the function "analyze_structures" to cluster the relaxed muon
    positions.

    Returns:
    (i) List of relaxed unique candidate sites supercell structures
    (ii) List of to be calculated magnetic inequivalent supercell structures
    """
    # load the results of the workchain
    rlxd_results = []
    for idx, result in findmuon_node.outputs.all_sites.get_dict().items():
        rlxd_result = {
            'idx': idx,
            'rlxd_struct': result[0],
            'energy': result[1]
        }
        rlxd_results.append(rlxd_result)
        
    input_st = findmuon_node.inputs.structure.get_pymatgen()
    
    # get the initial supercell structure
    init_supc = input_st.copy()
    sc_matrix = [
        findmuon_node.outputs.all_index_uuid.creator.caller.inputs.sc_matrix.get_list()
    ]
    init_supc.make_supercell(sc_matrix[0])

    # get the magnetic moments
    if hasattr(findmuon_node.inputs, 'magmom'):
        magmom = findmuon_node.inputs.magmom
    else:
        magmom = None

    return analyze_structures(init_supc, rlxd_results, input_st, magmom)
