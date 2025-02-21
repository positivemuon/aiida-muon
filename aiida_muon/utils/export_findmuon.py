from aiida import orm

import base64
import numpy as np
import pandas as pd

from pymatgen.core import Structure
from ase import Atoms

# NOTE: these functions are used in the aiidalab-qe-muon plugin, so a change should be tested against it.

def get_distortions(
    unrelaxed_supercell: Atoms,
    relaxed_supercell: Atoms,
    muon_atomic_number: int = 1,
    verbose: bool = False,
    ) -> dict:
    """
    Get the distortions of the muon site in the relaxed supercell with respect to the unrelaxed supercell.  
    """
    atm_indxes = [atom.index for atom in unrelaxed_supercell]
    mu_index = [atom.index for atom in unrelaxed_supercell if atom.number == muon_atomic_number]
    
    unrelaxed_atm_dist = unrelaxed_supercell.get_distances(
        mu_index, atm_indxes, mic=True, vector=False
    )

    relaxed_atm_dist = relaxed_supercell.get_distances(
            mu_index, atm_indxes, mic=True, vector=False
    )
    
    distortion = (relaxed_atm_dist-unrelaxed_atm_dist)[:-1]
    sort_distances = np.argsort(unrelaxed_atm_dist[:-1])
    #distortion = distortion[sort_distances]
    
    sorted_atoms = [relaxed_supercell.get_chemical_symbols()[sorted_index] for sorted_index in sort_distances]
    
    distortions_dict = {}
    for element in set(relaxed_supercell.get_chemical_symbols()).difference({"H"}):
        distortions_dict[element] = {}
        indices = [i for x,i in zip(sorted_atoms, sort_distances) if x == element]
        distortions_dict[element]["atm_distance_init"] = np.array(unrelaxed_atm_dist[:-1])[indices]
        distortions_dict[element]["atm_distance_final"] = np.array(relaxed_atm_dist[:-1])[indices]
        distortions_dict[element]["distortion"] = distortion[indices]
        
    return distortions_dict

def produce_muonic_dataframe(findmuon_output_node: orm.Node) -> pd.DataFrame:
    """Produce two dataframes: one for the unique sites and one for all sites.
    
    In this way, we can do some post processing.
    """
    bars = {
        "magnetic_units": "tesla",
        "magnetic_keys": [
            "B_T",
            "Bdip",
            "B_T_norm",
            "B_hf_norm",
            #"hyperfine",
            "Bdip_norm",
        ],
        "muons": {},
    }
    
    all_structures = {}
    distortions={}
    for i, (idx, uuid) in enumerate(findmuon_output_node.all_index_uuid.get_dict().items(), start=1):
        relaxwc = orm.load_node(uuid)
        if not relaxwc.is_finished_ok:
            continue
        all_structures[idx] = {}
        tot_energy = np.round(relaxwc.outputs.output_parameters.get_dict()["energy"]*10**3,1)
        structure_pk = relaxwc.outputs.output_structure.pk
        
        # info for all the structures.
        all_structures[idx] = {}
        all_structures[idx]["structure_id_pk"] = structure_pk
        all_structures[idx]["label"] = idx
        all_structures[idx]["tot_energy"] = tot_energy
        all_structures[idx]["muon_position_cc"] = list(
                np.round(
                    np.array(
                        findmuon_output_node.unique_sites.get_dict()[idx][0]["sites"][
                            -1
                        ]["abc"]
                    ),
                    3,
                ),
            )
        all_structures[idx]["muon_index"] = idx
        all_structures[idx]["muon_index_global_unitcell"] = \
            len(findmuon_output_node.all_index_uuid.creator.caller.inputs.structure.sites) + i
        
        # distortions.
        distortions[idx] = get_distortions(
            relaxwc.inputs.structure.get_ase(),
            relaxwc.outputs.output_structure.get_ase(),
        )
        
        # info for the unique sites.
        if idx in findmuon_output_node.unique_sites.get_dict().keys():
            bars["muons"][idx] = {}
            bars["muons"][idx]["structure_id_pk"] = structure_pk
            bars["muons"][idx]["label"] = idx # a label to recognise the supercell/muon site
            bars["muons"][idx]["muon_index"] = idx # sure that we need it?
            bars["muons"][idx]["tot_energy"] = tot_energy
            bars["muons"][idx]["muon_position_cc"] = list(
                np.round(
                    np.array(
                        findmuon_output_node.unique_sites.get_dict()[idx][0]["sites"][
                            -1
                        ]["abc"]
                    ),
                    3,
                ),
            )
            bars["muons"][idx]["muon_index_global_unitcell"] = \
                len(findmuon_output_node.all_index_uuid.creator.caller.inputs.structure.sites) + i # remember: i starts from 1        
        
    fields_list = []
    if "unique_sites_dipolar" in findmuon_output_node:
        fields_list = ["B_T", "Bdip", "B_T_norm", "Bdip_norm", "B_hf_norm"] # "hyperfine", 
        for configuration in findmuon_output_node.unique_sites_dipolar.get_list():
            for B in ["B_T", "Bdip"]:
                bars["muons"][str(configuration["idx"])][B] = list(
                    np.round(np.array(configuration[B]), 3)
                )
                if B in ["B_T"]:
                    bars["muons"][str(configuration["idx"])]["B_T_norm"] = round(
                        np.linalg.norm(np.array(configuration[B])), 3
                    )
                if B in ["Bdip"]:
                    bars["muons"][str(configuration["idx"])]["Bdip_norm"] = round(
                        np.linalg.norm(np.array(configuration[B])), 3
                    )
            if "unique_sites_hyperfine" in findmuon_output_node:
                v = findmuon_output_node.unique_sites_hyperfine.get_dict()[
                    str(configuration["idx"])
                ]
                # bars["muons"][str(configuration["idx"])]["hyperfine"] = v
                bars["muons"][str(configuration["idx"])]["B_hf_norm"] = round(
                    abs(v[-1]), 3
                )  # <-- we select the last, is in T (the first is in Atomic units).


    # exporting the dataframe for the unique sites:
    df = pd.DataFrame.from_dict(bars["muons"])
    df.columns = df.columns.astype(int)
    # sort
    df = df.sort_values("tot_energy", axis=1)
    
    # deltaE
    # round deltaE to integer meV
    df.loc["delta_E"] = (df.loc["tot_energy"] - df.loc["tot_energy"].min()).astype(int)
    
    # Insert the delta_E row as third row
    df = df.reindex(
        [
            "structure_id_pk", 
            "label", 
            "delta_E", 
            "tot_energy", 
            "muon_position_cc",
        ]+ fields_list
        +["muon_index_global_unitcell"]
        +["muon_index"],
    )
    
    # redefine the "label" to be letters from A to Z
    df.loc["label"] = [chr(65 + i) for i in range(len(df.columns))]
    
    
    # exporting the dataframe for all sites; we do the same as above...
    df_all = pd.DataFrame.from_dict(all_structures)
    df_all.columns = df_all.columns.astype(int)
    df_all = df_all.sort_values("tot_energy", axis=1)
    df_all.loc["delta_E"] = (df_all.loc["tot_energy"] - df_all.loc["tot_energy"].min()).astype(int)
    df_all = df_all.reindex(
        [
            "structure_id_pk", 
            "label", 
            "delta_E", 
            "tot_energy", 
            "muon_position_cc",
            "muon_index_global_unitcell",
            "muon_index"]
    )
    
    # I need to order the labels in such a way that are the same as in df, and add new letters for the clustering-wise-excluded sites.
    labels = []
    for i in range(len(df_all.columns)):
        if df_all.loc["structure_id_pk"].values[i] in df.loc["structure_id_pk"].values:
            labels.append(df.loc["label"].values[df.loc["structure_id_pk"].values == df_all.loc["structure_id_pk"].values[i]][0])
        else:
            labels.append(chr(65 + i + len(df_all.columns)))
    df_all.loc["label"] = labels
    
    # then swap row and columns (for sure can be done already above for df, but useful to keep the same order before this point)
    df = df.transpose()
    df_all = df_all.transpose()
    
    return df, df_all, distortions


# (2) unit cell with all muonic sites.
def produce_collective_unit_cell(findmuon_output_node: orm.Node, before_clustering=False) -> orm.StructureData:

    # e_min=np.min([qeapp_node.outputs.unique_sites.get_dict()[key][1] for key in qeapp_node.outputs.unique_sites.get_dict()])
    
    sc_matrix = [
        findmuon_output_node.all_index_uuid.creator.caller.inputs.sc_matrix.get_list()
    ]  # WE NEED TO HANDLE also THE CASE IN WHICH IS GENERATED BY MUSCONV.
    input_str = findmuon_output_node.all_index_uuid.creator.caller.inputs.structure.get_pymatgen().copy()

    # append tags to recognize the muon site.
    input_str.tags = [None] * input_str.num_sites

    for key in findmuon_output_node.all_sites.get_dict():
        if key in findmuon_output_node.unique_sites.get_dict().keys() or before_clustering:
            # print("H"+key, qeapp_node.outputs.unique_sites.get_dict()[key][1], (qeapp_node.outputs.unique_sites.get_dict() [key][1]-e_min))
            # fo.write("%s %16f %16f \n "%  ("H"+key, uniquesites_dict[key][1], (uniquesites_dict[key][1]-e_min)))
            py_struc = Structure.from_dict(
                findmuon_output_node.unique_sites.get_dict()[key][0]
            )
            musite = py_struc.frac_coords[py_struc.atomic_numbers.index(1)]
            mupos = np.dot(musite, sc_matrix) % 1
            # bad workaround for strange bug.
            if len(mupos) == 1:
                mupos = mupos[0]
                if len(mupos) == 1:
                    mupos = mupos[0]
            input_str.append(
                species="H" + key,
                coords=mupos,
                coords_are_cartesian=False,
                validate_proximity=True,
            )
            input_str.tags.append(key)

    kind_properties = []
    for i in input_str.sites:
        i.properties["kind_name"] = i.label
        kind_properties.append(i.properties)
    # raise ValueError(l)

    # We convert from pymatgen Structure to orm.StructureData, so we can use directly StructureDataViewer.
    structure = orm.StructureData(pymatgen=input_str)
    
    # Tags are used for the StructureDataViewer to highlight the muon sites.
    structure.tags = input_str.tags
    
    return structure

    

def export_findmuon_data(findmuon_output_node: orm.Node) -> dict:
    
    df, df_all, distortions = produce_muonic_dataframe(findmuon_output_node)
    return {
        "table": df,
        "table_all": df_all,
        "distortions": distortions,
        "unit_cell": produce_collective_unit_cell(findmuon_output_node),
        "unit_cell_all": produce_collective_unit_cell(findmuon_output_node, before_clustering=True),
    }