import numpy as np

from aiida_muon.utils.niche import Niche
from pymatgen.core import Structure
from ase import Atoms

def compute_suggest_supercell_size(
    structure: Atoms,
    pbc = [True, True, True],
    verbose: bool = False,
    ):
    """
    This calcfunction computes the supercell size based on the input structure.
    
    """
    suggested_3D = 9 // np.array(structure.cell.cellpar()[:3]) + 1
    return [int(suggested_3D[i]) if pbc[i] else 1 for i in range(3)]

def niche_add_impurities(
    structure: Structure,
    niche_atom: str,
    niche_spacing: float,
    niche_distance: float,
    verbose: bool = False,
):
    """
    This calcfunction calls Niche. 
    
    Supplies structure, atom index and impurity spacing 
    required to get the grid initial sites.

    Return: Adapted here to return only lists of generated muon sites.
    """
    niche_instance = Niche(structure, niche_atom)

    n_st = niche_instance.apply(niche_spacing, niche_distance)

    # +0.001 to break symmetry if at symmetry pos
    mu_lst = [
        i + 0.001
        for j, i in enumerate(n_st.frac_coords)
        if n_st.species[j].value == niche_atom
    ]

    if verbose: 
        print(f"Muon sites: {mu_lst}")
        
    return mu_lst

def gensup(
    p_st, 
    mu_list, 
    sc_mat,
    only_one_cell = False,
    validate_proximity=True
    ):
    """
    This makes the supercell with the given SC matrix.
    It also appends the muon.
    
    if only_one_cell is True, then the supercell is made only once and 
    we append all the sites in the same supercell. Useful for pre-processing visualizations.

    the validate_proximity is set to False to avoid the error of muon being too close to each
    other, when we use only_one_cell=True.
    
    Returns: list of supercell (pymatgen) structures with muon.
              Number of supercells depends on number of imput mulist
    """
    supc_list = []
    if only_one_cell:
        # we do it only once for all
        p_scst = p_st.copy() 
        p_scst.make_supercell(sc_mat)
    for ij in mu_list:
        if not only_one_cell:
            p_scst = p_st.copy()
            p_scst.make_supercell(sc_mat)
            
        ij_sc = (np.dot(ij, np.linalg.inv(sc_mat))) % 1
        # ij_sc = [x + 0.001 for x in ij_sc]
        p_scst.append(
            species="H",
            coords=ij_sc,
            coords_are_cartesian=False,
            validate_proximity=validate_proximity,
            properties={"kind_name": "H"},
        )
        supc_list.append(p_scst)
    if only_one_cell:
        supc_list = supc_list[-1:]
    return supc_list

def generate_supercell_with_impurities(
    structure: Structure, # pymatgen
    sc_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    mu_spacing = 1,
    mu_list = None,
):
    """ Generates one supercell - undistorted - with all the predicted impurities
    """


    # 2 get the muon list
    if not mu_list:
        mu_list = niche_add_impurities(
            structure,
            niche_atom = "H",
            niche_spacing = mu_spacing,
            niche_distance = 1, # distance from hosting atoms,
        )

    # 3 produce the supercells
    supc_list = gensup(structure, mu_list, sc_matrix, only_one_cell=True, validate_proximity=False)
    
    return supc_list[0]
        