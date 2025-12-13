import numpy as np
import copy

from aiida.orm import StructureData as LegacyStructureData
from aiida.engine import calcfunction

from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida_quantumespresso.common.hubbard import Hubbard

def check_get_hubbard_u_parms(p_st, u_dict=None, new_structuredata=False):
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
    
    U_dict_used = U_dict2 if not u_dict else u_dict

    if "kind_name" in list(p_st.site_properties.keys()):
        spc = p_st.site_properties["kind_name"]
        d_spc = list(set(spc))
    else:
        spc = [str(atm.specie.symbol) for atm in p_st]
        d_spc = list(set(spc))

    if 1 in p_st.atomic_numbers:
        is_muon_there = 1
    else:
        is_muon_there = 0
    # 1 element compounds are not given any U values, 2 for the muon specie
    hub_u = {}
    if len(d_spc) > 1+is_muon_there:  # TODO: is this correct? Is it really needed?
        for spk in d_spc:
            # remove numbers from species name
            spk2 = "".join(filter(str.isalpha, spk))
            # check is in the dictorionary
            if spk2 in U_dict_used:
                hub_u[spk] = U_dict_used[spk2]
    
    return hub_u
    
    
@calcfunction
def create_hubbard_structure(structure: [LegacyStructureData], hubbard_dict: [dict, HubbardStructureData]):
    """
    Create a Hubbard structure from a given structure and Hubbard parameters.
    
    NOTE: It works in two ways:
    1 - If second input is HubbardStructureData, it creates a new HubbardStructureData with the structure of the first input and the Hubbard parameters of the second input.
    2 - If the second input is a dictionary, it creates a new HubbardStructureData with the structure of the first 
        input and the Hubbard parameters of the second input.

    Parameters:
    structure (HubbardStructureData or LegacyStructureData): The input structure data.
    hubbard_dict (dict or HubbardStructureData): Dictionary containing Hubbard U parameters or a HubbardStructureData object.

    Returns:
    HubbardStructureData: The resulting Hubbard structure with initialized on-site Hubbard parameters.

    Raises:
    TypeError: If the input types are not as expected.
    
    The logic is naive, to be optimized.
    """
    
    if isinstance(hubbard_dict, HubbardStructureData): 
        hubbard_structure = HubbardStructureData.from_structure(structure)
        for p in hubbard_dict.hubbard.parameters:
            kind = hubbard_dict.sites[p.atom_index].kind_name
            manifold = p.atom_manifold
            value = p.value
            hubbard_structure.initialize_onsites_hubbard(kind, manifold, value, 'U', use_kinds=True)
    elif isinstance(hubbard_dict, dict): 
        hubbard_structure = HubbardStructureData.from_structure(structure)
        for kind, U in hubbard_dict.items():
            hubbard_structure.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)

    #hubbard_structure.hubbard = Hubbard.from_list(hubbard_structure.hubbard.to_list(), projectors="atomic")
    return hubbard_structure

# def assign_hubbard_parameters(structure: atomistic.StructureData, hubbard_dict):
#     for kind, U in hubbard_dict.items():
#         structure.hubbard.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)
