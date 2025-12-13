import numpy as np
import copy

from aiida import orm
from aiida.engine import calcfunction

@calcfunction
def reassign_kinds(structure: orm.StructureData, kind_list: orm.List) -> orm.StructureData:
    """
    This function takes a StructureData object and a list of kind names, and returns a new StructureData object with
    the kinds in the order specified by the kind_list.
    
    The function clones the input structure and redefine the kinds base on the kind list.
    The way how we modify the kinds is not the recommended one, but we do in this way to avoid the 
    redefinition of HubbardStructureData.
    
    Parameters:
        structure (StructureData): The input StructureData object.
        kind_list (List): A list of kind names in the desired order, as obtained in ASE atoms.

    Returns:
        StructureData: A new StructureData object with the kinds in the specified order.
    """
    
    new_kinds = copy.deepcopy(structure.base.attributes.get("kinds"))
    new_sites = copy.deepcopy(structure.base.attributes.get("sites"))
    new_structure = structure.clone()
    for i, kind in enumerate(kind_list.get_list()):
        
        # we first change the site value, 
        # then we can change the kind name.
        for j, site in enumerate(new_sites):
            if site["kind_name"] == new_kinds[i]["name"]:
                new_sites[j]["kind_name"] = kind
        
        new_kinds[i]["name"] = kind
        
    new_structure.base.attributes.set("kinds", new_kinds)
    new_structure.base.attributes.set("sites", new_sites)
    
    return new_structure