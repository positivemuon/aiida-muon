# -*- coding: utf-8 -*-
import numpy as np
import copy
from aiida import orm
from aiida.engine import WorkChain, calcfunction, if_
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida_quantumespresso.common.types import RelaxType
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_quantumespresso.common.types import ElectronicType, RelaxType, SpinType
from pymatgen.core import Structure
from pymatgen.electronic_structure.core import Magmom
from aiida.common import AttributeDict
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from typing import Union

from aiida_impuritysupercellconv.workflows.impuritysupercellconv import input_validator as impuritysupercellconv_input_validator

from aiida.orm import StructureData as LegacyStructureData
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida_quantumespresso.common.hubbard import Hubbard

from aiida_muon.utils.sites_supercells import niche_add_impurities, gensup
from aiida_muon.utils.clustering import analyze_structures
from aiida_muon.utils.magnetism import make_collinear_getmag_kind, compute_dipolar_field
from aiida_muon.utils.hubbard import check_get_hubbard_u_parms, create_hubbard_structure

StructureData = DataFactory("atomistic.structure")
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
IsolatedImpurityWorkChain = WorkflowFactory('impuritysupercellconv')

def IsolatedImpurityWorkChain_override_validator(inputs,ctx=None):
    """validate inputs for impuritysupercellconv.relax; actually, it is
    just a way to avoid defining it if we do not want it. 
    otherwise the default check is done and it will excepts. 
    """
    if "impuritysupercellconv" in inputs.keys():
        if "parameters" in inputs["impuritysupercellconv"]["pwscf"]["pw"].keys():
            if len(inputs["impuritysupercellconv"]["pwscf"]["pw"]["parameters"].get_dict()):
                original_IsolatedImpurityWorkChain.spec().inputs.validator(inputs["impuritysupercellconv"],ctx)
            else:
                return None
        else:
            return None
    
IsolatedImpurityWorkChain.spec().inputs.validator = IsolatedImpurityWorkChain_override_validator


class FindMuonWorkChain(ProtocolMixin, WorkChain):
    """
    FindMuonWorkChain finds the candidate implantation site for a positive muon.
    It first performs DFT relaxation calculations for a set of initial muon sites.
    It then analyzes the results of these calculations and finds candidate muon sites.
    If there are magnetic inequivalent sites not initially, they are recalculated
    It further calculates the muon contact hyperfine field at these candidate sites.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.input(
            "structure",
            valid_type=(StructureData, LegacyStructureData),
            required=False,
            help="Input initial structure",
        )

        spec.input(
            "sc_matrix",
            valid_type=orm.List,
            required=False,   #MB put False by MB
            help=" List of length 1 for supercell size ",
        )

        spec.input(
            "mu_spacing",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0),
            required=False,
            help="Minimum distance in Angstrom between two starting muon positions  generated on a grid.",
        )

        # read as list or array?
        spec.input(
            "magmom",
            valid_type=orm.List,
            required=False,
            help="List of 3D magnetic moments in Bohr magneton of the corresponding input structure if magnetic",
        )
        
        spec.input(
            "spin_pol_dft",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
            help="Spin-polarised DFT simulation or not",
        )

        spec.input(
            "pp_code",
            valid_type=orm.Code,
            required=False,
            help="The pp.x code-computer for post processing only if magmom is supplied",
        )

        spec.input(
            "pseudo_family",
            valid_type=orm.Str,
            default=lambda: orm.Str("SSSP/1.3/PBE/efficiency"),
            required=False,
            help="The label of the pseudo family",
        )

        spec.input(
            "kpoints_distance",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.301),
            help="The minimum desired distance in 1/Å between k-points in reciprocal space.",
        )

        spec.input(
            "hubbard",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
            help="To check and get Hubbard U value or not",
        )
        spec.input(
            "hubbard_dict",
            valid_type=orm.Dict,
            required=False,
            help="Dictionary of Hubbard U values",
        )
        spec.input(
            "charge_supercell",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
            help="To run charged supercell for positive muon or not (neutral supercell)",
        )

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace="relax",
            exclude=("structure","base_final_scf"),
            namespace_options={
                'required': True, 
                'populate_defaults': False,
                'help': 'Inputs for SCF calculations.',
            },
        )  # use the  pw relax workflow
        
        #to run final scf
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="pwscf",
            namespace_options={
                'required': False, 
                'populate_defaults': False,
                'help': 'Inputs for final SCF calculation with the muon at the origin.',
            },
            exclude=("pw.structure", "kpoints"),
        )  # 
        
        #very optional inputs:
        spec.input(
            "qe_settings",
            valid_type=orm.Dict,
            required=False,
            help=" Preferred settings for the calc, otherwise default is used",
        )

        spec.input(
            "pp_metadata",
            valid_type= dict, 
            non_db=True,
            required=False,
            help=" Preferred metadata and scheduler options for pp.x",
        )

        spec.input(
            "impuritysupercellconv_metadata",
            valid_type=dict,
            non_db=True,
            required=False,
            help=" Preferred metadata and scheduler options for impuritysupercellconv",
        )

        # activate IsolatedImpurityWorkChain only if sc_matrix input not present.
        spec.expose_inputs(
            IsolatedImpurityWorkChain,
            namespace="impuritysupercellconv",
            exclude=("structure", "pseudos",),
            namespace_options={
                'required': False, 'populate_defaults': False,
                'help': 'the preprocess IsolatedImpurityWorkChain step, if needed.',
            },
        )  # use the  pw calcjob
        
        # NOTE: we skip the validation.
        # spec.inputs.validator = recursive_consistency_check
        
        spec.outline(
            cls.pre_check_structure_data_compatibility,
            if_(cls.check_converge_supercell_size)(     
                cls.run_converge_supercell_size,         
                cls.check_supercell_convergence,          
            ),
            cls.setup,
            cls.get_initial_muon_sites,
            cls.get_initial_supercell_structures,
            cls.compute_supercell_structures,
            cls.collect_relaxed_structures,
            cls.analyze_relaxed_structures,
            if_(cls.new_struct_after_analyze)(
                cls.compute_supercell_structures,
                cls.collect_relaxed_structures,
                cls.collect_all_results,
            ),
            if_(cls.structure_is_magnetic)(
                if_(cls.spin_polarized_dft)(
                    cls.run_final_scf_mu_origin,  # to be removed if better alternative
                    cls.compute_spin_density,
                    cls.compute_contact_hyperfine,
                ),
                cls.get_dipolar_field,
                cls.set_field_outputs,
            ),
            cls.set_relaxed_muon_outputs,
        )

        spec.exit_code(
            404,
            "ERROR_MUSCONV_CALC_FAILED",
            message="The IsolatedImpurityWorkChain subprocesses failed",
        )
        spec.exit_code(
            405,
            "ERROR_RELAX_CALC_FAILED",
            message="One of the PwRelaxWorkChain subprocesses failed",
        )
        spec.exit_code(
            406,
            "ERROR_BASE_CALC_FAILED",
            message="One of the PwBaseWorkChain subprocesses failed",
        )
        spec.exit_code(
            407,
            "ERROR_PP_CALC_FAILED",
            message="One of the PPWorkChain subprocesses failed",
        )
        spec.exit_code(
            407,
            "ERROR_NO_SUPERCELLS",
            message="No supercells available: try to decrease mu_spacing.",
        )

        # TODO: more exit codes catch errors and throw exit codes

        spec.output(
            "all_index_uuid", valid_type=orm.Dict, required=True
            )
        spec.output(
            "all_sites", valid_type=orm.Dict, required=True
            )
        spec.output(
            "unique_sites", valid_type=orm.Dict, required=True, help="Unique sites after clustering"
            )
        spec.output(
            "unique_sites_hyperfine", valid_type=orm.Dict, required=False
            )  # return only when magnetic
    
        spec.output(
            "unique_sites_dipolar", valid_type=orm.List, required=False
            )  # return only when magnetic
        
        
    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        structure: Union[StructureData, LegacyStructureData, HubbardStructureData],
        pp_code: orm.Code = None,
        protocol: str =None,
        overrides: dict = {},
        enforce_defaults = True,
        relax_unitcell: bool =False,
        conv_thr: float = 0.0257,
        magmom: list = None,
        options=None,
        sc_matrix: list =None,
        mu_spacing: float = 1.0,
        kpoints_distance: float = 0.301,
        charge_supercell: bool =True,
        hubbard: bool = True,
        hubbard_dict: dict = None,
        spin_pol_dft: bool = True,
        pseudo_family: str ="SSSP/1.3/PBE/efficiency",
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param pp_code: the ``Code`` instance configured for the ``quantumespresso.pp`` plugin.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param enforce_defaults: if True, will enforce that all inputs of the protocol are set, if False, only the required ones.
        :param relax_unitcell: To relax the unit cell or not.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param sc_matrix: List of length 1 for supercell size.
        :param mu_spacing: Minimum distance in Angstrom between two starting muon positions  generated on a grid..
        :param kpoints_distance: the minimum desired distance in 1/Å between k-points in reciprocal space.
        :param charge_supercell: To run charged supercell for positive muon or not (neutral supercell).
        :param hubbard: To check and get Hubbard U value or not.
        :param hubbard_dict: Dictionary of Hubbard U values.
        :param pseudo_family: the label of the pseudo family.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
                
        _overrides, start_mg_dict, structure = get_default_dict(structure, pseudo_family, kpoints_distance, charge_supercell, magmom, spin_pol_dft)
        
        if enforce_defaults:
            overrides = recursive_merge(overrides,_overrides)
        
        #check on the structure: if hubbard is needed, do it with append onsite... if the structure is already stored, clone it. 
        # NOTE: `check_get_hubbard_u_parms` will only return the dictionary if we have more than 2 species in the structure (muon excluded).
        hubbard_params = check_get_hubbard_u_parms(structure.get_pymatgen(), u_dict=hubbard_dict)
        if isinstance(structure,HubbardStructureData): # we do not do anything, we let the user define the Hubbard U 
            pass #print("This is HubbardStructureData, to have backward compatibility with old StructureData and forward compatibility with qe>=7.1 .")
        else: # orm.StructureData
            # we want DFT+U only if magmoms are there. 
            # NOTE: I don't think is always the case. 
            if hubbard_params is not None and magmom is not None and hubbard:
                structure = HubbardStructureData.from_structure(structure)
                for kind, U in hubbard_params.items():
                    structure.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)
                structure.hubbard = Hubbard.from_list(structure.hubbard.to_list(), projectors="atomic")
        
                 
        #### IsolatedImpurityWorkChain
        builder_impuritysupercellconv = IsolatedImpurityWorkChain.get_builder_from_protocol(
                pw_code = pw_code,
                structure = structure,
                pseudo_family = pseudo_family,
                relax_unitcell = relax_unitcell,
                charge_supercell = charge_supercell, # <== by default it is false.
                kpoints_distance = kpoints_distance,
                conv_thr = conv_thr,
                overrides = overrides.pop("impuritysupercellconv",None),
                )
        
        #builder_impuritysupercellconv.pop('structure', None)
        
        #### PwBaseWorkChain for final scf mu-origin
        builder_pwscf = PwBaseWorkChain.get_builder_from_protocol(
                pw_code,
                structure,
                pseudo_family = pseudo_family,
                protocol=protocol,
                overrides=overrides.get("base",None),
                **kwargs,
                )
        
        
        #### PwRelaxWorkChain
        builder_relax = PwRelaxWorkChain.get_builder_from_protocol(
                pw_code,
                structure,
                pseudo_family = pseudo_family,
                protocol=protocol,
                overrides=overrides,
                relax_type=RelaxType.POSITIONS,
                **kwargs,
                )
        
        builder_relax.pop('structure', None)
        builder_relax.pop('base_final_scf', None)
        
        builder_pwscf['pw'].pop('structure', None)
        builder_pwscf.pop('kpoints_distance', None)       
        
        #### Builder
        builder = cls.get_builder()
        
        builder.structure = structure
        builder.pseudo_family = orm.Str(pseudo_family)
        builder_impuritysupercellconv.pseudo_family = orm.Str(pseudo_family)
        
        #setting subworkflows inputs
        #probably, it is better to populate defaults and then pop if not needed, as done later.
        for k,v in builder_impuritysupercellconv.items():
            if k == "relax":
                for k1,v1 in builder_impuritysupercellconv.relax.items():
                    if k1 == "base_final_scf": continue
                    setattr(builder.impuritysupercellconv.relax,k1,v1)
            else:
                setattr(builder.impuritysupercellconv,k,v)
        #builder.impuritysupercellconv = builder_impuritysupercellconv  If you use this instead of the above, it will give ValueError.
        builder.pwscf = builder_pwscf
        builder.relax = builder_relax
        
        #if not relax_unitcell: builder.impuritysupercellconv.pop('relax')
        builder.impuritysupercellconv.pop('structure')
        
        # If magmoms are defined, we need to set the spin_pol_dft to True
        if start_mg_dict: 
            if isinstance(magmom, list):
                magmom = orm.List(magmom)
            builder.magmom = magmom
        
        # If sc_matrix, we do not need to run the IsolatedImpurityWorkChain
        if sc_matrix: 
            builder.sc_matrix=orm.List(sc_matrix)
            builder.pop('impuritysupercellconv')

        builder.mu_spacing=orm.Float(mu_spacing)
        builder.charge_supercell=orm.Bool(charge_supercell)
        builder.kpoints_distance = orm.Float(kpoints_distance)
        builder.hubbard = orm.Bool(hubbard)
        builder.spin_pol_dft = orm.Bool(spin_pol_dft)
        
        # PpCalculation inputs: Only this, the rest is really default and generated on the fly - hardcoded
        if pp_code: builder.pp_code = pp_code
        
        # Checking for additional metadata
        for i in ["pp_metadata","impuritysupercellconv_metadata","qe_settings"]:
            # I don't like this.
            if i in overrides.keys():
                builder[i] = overrides[i] 
        
        return builder
    
    
    def pre_check_structure_data_compatibility(self):
        """
        Pre-process to understand if 
        we use StructureData or LegacyStructureData
        """
        if isinstance(self.inputs.structure, HubbardStructureData):
            self.ctx.structure_type = HubbardStructureData
        else:
            self.ctx.structure_type = LegacyStructureData
        
    def check_converge_supercell_size(self):
        """Understand if impuritysupercellconv is needed.
        
        Search for the sc_matrix in the inputs.
        """
        if hasattr(self.inputs,"sc_matrix"):
            self.ctx.sc_matrix = self.inputs.sc_matrix.get_list()
                    
        return not hasattr(self.inputs,"sc_matrix")
    
    def run_converge_supercell_size(self):
        """Call IsolatedImpurityWorkChain for supercell convergence.
        
        """
        inputs = AttributeDict(self.exposed_inputs(IsolatedImpurityWorkChain, namespace='impuritysupercellconv'))
        inputs.structure = self.inputs.structure
        
        # TODO: remove this line, if we don't want hubbard we should just provide a LegacyStructureData
        if not self.inputs.hubbard: 
            inputs.structure = orm.StructureData(ase=self.inputs.structure.get_ase()) # so we lose the info on hubbard. this is the case where we use protocol but then we set builder.hubbard = False later.
        
        # We ensure we use a kpoints_distance, if not present in the inputs, we use the FindMuonWorkChain one.
        if not "kpoints_distance" in inputs:
            inputs.kpoints_distance = self.inputs.kpoints_distance
        
        # specific metadata for the IsolatedImpurityWorkChain, directly exposed in this workflow for user friendliness.
        if hasattr(self.inputs,"impuritysupercellconv_metadata"):
            inputs.pwscf.pw.metadata = self.inputs.impuritysupercellconv_metadata

        # Specific name and submittions
        inputs.metadata.call_link_label = f'IsolatedImpurityWorkChain'
        future = self.submit(IsolatedImpurityWorkChain, **inputs)
        self.report(
            f"Launching IsolatedImpurityWorkChain (PK={future.pk}) for supercell matrix determination"
        )
        
        # We tell the Workflow to wait until we finish this run. Then, it can proceed in the outline.
        self.to_context(**{"IsolatedImpurityWorkChain": future})

    def check_supercell_convergence(self):
        """ Check that the IsolatedImpurityWorkChain is finished ok
        
        """
        if not self.ctx["IsolatedImpurityWorkChain"].is_finished_ok:
            self.report("The IsolatedImpurityWorkChain (supercell size estimation) failed. Exiting the workflow.")
            return self.exit_codes.ERROR_MUSCONV_CALC_FAILED
        else:
            sc_mat_array = self.ctx["IsolatedImpurityWorkChain"].outputs.Converged_SCmatrix.get_array('sc_mat')
            self.ctx.sc_matrix = sc_mat_array.tolist()
            self.report(f"Supercell size computed to be: {sc_mat_array}")
                
    def setup(self):
        """Setup for the workflow.
        
        In particular, we set the structure and the magnetization information, if any.
        We no more setup the hubbard dictionary here: the Hubbard parameters should be defined in the `get_builder_from_protocols`.
        """
        if not hasattr(self.ctx,"structure"): 
            # TODO: set, if any the final relaxed unit cell as obtained from the IsolatedImpurityWorkChain pre-relaxation.
            self.ctx.structure = self.inputs.structure
            
        if hasattr(self.inputs,"matrix"):
            self.ctx.sc_matrix = self.inputs.sc_matrix.get_list()
        elif not hasattr(self.ctx,"sc_matrix"):
            raise ValueError("No supercell matrix defined. Exiting the workflow.")
        
        # checking if we are using the atomistic.StructureData or not. 
        # NOTE: the atomistic.StructureData is not yet implemented, this is just a placehoder.
        # to use magnetization info, we need always to pass `magmom` as input.
        if "magmom" in self.inputs:
            self.ctx.magmom = self.inputs.magmom.get_list()
            
        if hasattr(self.inputs,"hubbard_dict"):
            self.ctx.hubbardu_dict = self.inputs.hubbard_dict.get_dict()
        else:
            self.ctx.hubbardu_dict = None
        
        # init relaxation calc count
        self.ctx.n = 0
        self.ctx.n_uuid_dict = {}
        
        return
            
    def get_initial_muon_sites(self):
        """Get list of starting muon sites.
        
        """
        self.ctx.mu_lst = niche_add_impurities(
            structure = self.ctx.structure.get_pymatgen_structure(), 
            niche_atom = "H", 
            niche_spacing = self.inputs.mu_spacing.value, 
            niche_distance = 1, # distance from hosting atoms.
        )
        
        if len(self.ctx.mu_lst) == 0:
            self.report("No muon sites found. Exiting the workflow.")
            return self.exit_codes.ERROR_NO_SUPERCELLS
        else:
            self.report(f"Number of muon sites found: {len(self.ctx.mu_lst)}")
            
        return
    
    def get_initial_supercell_structures(self):
        """Get initial supercell+muon list.
        
        """
        self.report("Getting supercell list")
        input_struct = self.ctx.structure.get_pymatgen_structure()
        muon_list = self.ctx.mu_lst

        self.ctx.supc_list = gensup(input_struct, muon_list, self.ctx.sc_matrix)  # ordinary function
        if len(self.ctx.supc_list) == 0:
            self.report("No Supercells, please decrease the mu_spacing parameter. Exiting the workflow...")
            return self.exit_codes.ERROR_NO_SUPERCELLS

    def setup_pw_overrides(self):
        """Get the required overrides i.e pw parameter setup. STILL INCLUDED IN THE OUTLINE"""
        '''
        Miki Bonacci: I think that this overrides are no more needed once we have the MagneticStructureData.
        Also, if we do this in a protocol, we can also tune it before the run, just in case.
        Hubbard can be set by protocol, as we have the defaults. 
        base_final_scf not needed because it is not currently used: but we can use its inputs to run the final scf with muon at the orgin? 
        '''
        self.report("Setting up the relaxation calculation")
        overrides = {
            #'final_scf' : orm.Bool(False),
            "base": {
                "kpoints_distance": orm.Float(self.inputs.kpoints_distance.value),
                #"pseudo_family":self.inputs.pseudo_family,
                "pw": {
                    "parameters": {},
                    "metadata": {},
                },
            },
            #"base_final_scf": {
            #    "pseudo_family": self.inputs.pseudo_family.value,
            #},
            "clean_workdir": orm.Bool(True),
        }

        ##TO DO:put a check on  parameters that cannot be set by hand in the overrides eg mag, hubbard

        # set some cards
        overrides["base"]["pw"]["parameters"] = recursive_merge(
            overrides["base"]["pw"]["parameters"], {"CONTROL": {"nstep": 200}}
        )
        #overrides["base"]["pw"]["parameters"] = recursive_merge(overrides["base"]["pw"]["parameters"], {"CONTROL": {"etot_conv_thr": 1.0e-4}})
        #overrides["base"]["pw"]["parameters"] = recursive_merge(overrides["base"]["pw"]["parameters"], {"CONTROL": {"forc_conv_thr": 1.0e-3}})   #less costlier instead of 1e-4 default
        # overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'smearing': 'gaussian'}})
        overrides["base"]["pw"]["parameters"] = recursive_merge(
            overrides["base"]["pw"]["parameters"],
            {"ELECTRONS": {"electron_maxstep": 500}},
        )
        overrides["base"]["pw"]["parameters"] = recursive_merge(
            overrides["base"]["pw"]["parameters"], {"ELECTRONS": {"mixing_mode": "local-TF", "mixing_beta":0.3}}
        )
        # overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'ELECTRONS':{'conv_thr': 1.0e-6}})
        overrides["base"]["pw"]["metadata"] = recursive_merge(
            overrides["base"]["pw"]["metadata"],
            {
                "description": "Muon site calculations for "
                + self.inputs.structure.get_pymatgen_structure().formula
            },
        )
        if hasattr(self.inputs,"charge_supercell"):
        #
            overrides["base"]["pw"]["parameters"] = recursive_merge(
                overrides["base"]["pw"]["parameters"], {"SYSTEM": {"tot_charge": int(self.inputs.charge_supercell)}}
            )
        # if self.inputs.magmom is not None:
        #MB this should be automatically done in the new implementation with the MagneticStructureData.
        if "magmom" in self.inputs and self.ctx.start_mg_dict and self.inputs.spin_pol_dft:
            overrides["base"]["pw"]["parameters"] = recursive_merge(
                overrides["base"]["pw"]["parameters"], {"SYSTEM": {"nspin": 2}}
            )
            overrides["base"]["pw"]["parameters"] = recursive_merge(
                overrides["base"]["pw"]["parameters"],
                {
                    "SYSTEM": {
                        "starting_magnetization": self.ctx.start_mg_dict.get_dict()
                    }
                },
            )
        self.ctx.overrides = overrides

    def compute_supercell_structures(self):
        """Run relax workflows for each muon supercell.
        
        We first check that kpoints_distance is set, then we loop
        over the supercell list and run the relax workflow, ensuring that
        the structure is HubbardStructureData if needed.
        """

        self.report("Computing muon supercells")
        supercell_list = self.ctx.supc_list
        
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        
        # Make sure we have a kpoints distance
        if not "kpoints_distance" in inputs.base:
            self.report(f"Setting kpoints distance to be: {self.inputs.kpoints_distance.value}")
            inputs.base.kpoints_distance = self.inputs.kpoints_distance
        
        for i_index in range(len(supercell_list)):
            
            if self.ctx.structure_type == HubbardStructureData:
                # we need to create a new HubbardStructureData from the pymatgen supercell
                # using the LegacyStructureData constructor first
                inputs.structure = LegacyStructureData(pymatgen=supercell_list[i_index])
                inputs.structure = HubbardStructureData.from_structure(inputs.structure)
            else:
                inputs.structure = self.ctx.structure_type(pymatgen=supercell_list[i_index])
            
            # we then assign the Hubbard parameters if needed
            if isinstance(self.inputs.structure, HubbardStructureData):
                self.report(f"Generating supercell #{i_index} with Hubbard parameters.")
                inputs.structure = create_hubbard_structure(inputs.structure,self.inputs.structure)
            elif self.ctx.hubbardu_dict and "magmom" in self.inputs:
                self.report(f"Enforcing DFT+U for supercell #{i_index}, as magmoms are defined and U parameters are available.")
                inputs.structure = create_hubbard_structure(inputs.structure,self.ctx.hubbardu_dict)
                
            
            # we define the pseudos again (now we have the structure+H)
            inputs.base.pw.pseudos = get_pseudos(
                inputs.structure, self.inputs.pseudo_family.value
            )
            
            

            # Set the `CALL` link label and submission
            inputs.metadata.call_link_label = f'supercell_{i_index:02d}'
            future = self.submit(PwRelaxWorkChain, **inputs)
            # key = f'workchains.sub{i_index}' #nested sub
            key = f"workchain_{i_index}"
            self.report(
                f"Launching PwRelaxWorkChain (PK={future.pk}) for supercell structure {supercell_list[i_index].formula} with index {i_index}"
            )
            self.to_context(**{key: future})

    def collect_relaxed_structures(self):
        """Retrieve final positions and energy from the relaxed structures.
        
        """

        self.report("Gathering computed positions and energy")
        supercell_list = self.ctx.supc_list

        computed_results = []

        # for nested
        # for key, workchain in self.ctx.workchains.items():
        #    if not workchain.is_finished_ok

        n_notf = 0
        for i_index in range(len(supercell_list)):
            key = f"workchain_{i_index}"
            workchain = self.ctx[key]

            # checking failed calculations and exiting if more than 40% failed
            if not workchain.is_finished_ok:
                self.report(
                    f"PwRelaxWorkChain failed with exit status {workchain.exit_status}"
                )
                n_notf += 1
                # if failed calculation is more than 40%, then exit
                if float(n_notf) / len(supercell_list) > 0.4:
                    return self.exit_codes.ERROR_RELAX_CALC_FAILED
            else:
                self.ctx.n += 1
                uuid = workchain.uuid
                energy = workchain.outputs.output_parameters.get_dict()["energy"]
                rlx_structure = (
                    workchain.outputs.output_structure.get_pymatgen_structure()
                )
                # rlx_structure = workchain.outputs.output_structure

                # computed_results.append((pk,rlx_structure,energy))
                computed_results.append(
                    (
                        {
                            "idx": self.ctx.n,
                            "rlxd_struct": rlx_structure.as_dict(),
                            "energy": energy,
                        }
                    )
                )
                self.ctx.n_uuid_dict.update({self.ctx.n: uuid})

                # print(computed_results)

        self.ctx.relaxed_outputs = computed_results

    def analyze_relaxed_structures(self):
        """Analyze relaxed structures.
        
        Get unique candidate sites and check if there are 
        new magnetic inequivalent (via symmetry operations) 
        sites to calculate.
        
        Basically, this represents the clustering step.
        Different algorithm could in principle be implemented.
        """
        self.report("Analyzing the relaxed structures")
        inpt_st = self.ctx.structure.get_pymatgen_structure()

        if "magmom" in self.ctx:
            r_anly = analyze_structures(
                self.ctx.supc_list[0],
                self.ctx.relaxed_outputs,
                inpt_st,
                self.ctx.magmom,
            )
        else:
            r_anly = analyze_structures(
                self.ctx.supc_list[0], self.ctx.relaxed_outputs, inpt_st
            )

        self.ctx.unique_cluster = r_anly["unique_pos"]
        # print('uniq_positions',self.ctx.unique_cluster)

        # revisit, this so the initial inputs and collected results are not ovewritten with repeated calls in outline
        self.ctx.supc_list_all = self.ctx.supc_list
        self.ctx.relaxed_outputs_all = self.ctx.relaxed_outputs

        self.ctx.supc_list = r_anly["mag_inequivalent"]

    def new_struct_after_analyze(self):
        """Check if there is new magnetic inequivalent sites"""
        self.report("Checking new structures to calculate... {len(self.ctx.supc_list) > 0}")

        return len(self.ctx.supc_list) > 0

    def collect_all_results(self):
        """Collecting results of new structures and then append"""
        self.report("Appending results of new structures ")

        self.ctx.relaxed_outputs_all.extend(self.ctx.relaxed_outputs)
        self.ctx.unique_cluster.extend(self.ctx.relaxed_outputs)

    def structure_is_magnetic(self):
        """Checking if structure is magnetic"""

        # return self.inputs.magmom is not None
        # return 'magmom' in self.inputs
        magnetic = False
        if "magmom" in self.inputs:
            magnetic = self.inputs.magmom is not None
            
        self.report(f"Checking if structure is magnetic... {magnetic}")
            
    def spin_polarized_dft(self):
        """Checking if we need spin polarization in DFT"""
        self.report(f"Checking if we had spin polarization in the simulations: {self.inputs.spin_pol_dft.value}")
        return self.inputs.spin_pol_dft.value

    def run_final_scf_mu_origin(self):
        """Move muon to origin and perform scf"""
        unique_cluster_list = self.ctx.unique_cluster
        
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='pwscf'))
        inputs_pw = inputs["pw"]["parameters"].get_dict()
       
        # we need for the PpCalculation.
        # inputs["clean_workdir"] = orm.Bool(False) 

        #inputs.kpoints_distance = orm.Float(inputs.kpoints_distance.value - 0.1) #denser reciprocal space grid 
        
        if not "kpoints_distance" in inputs:
            inputs.kpoints_distance = self.inputs.kpoints_distance
        
        for j_index, clus in enumerate(unique_cluster_list):
            #
            # rlx_st = clus['rlxd_struct']
            # rlx_struct = StructureData(pymatgen = rlx_st)
            # or
            c_uuid = self.ctx.n_uuid_dict[clus["idx"]]
            rlx_node = orm.load_node(c_uuid)
            rlx_st = rlx_node.outputs.output_structure.get_pymatgen_structure()

            # move muon to origin
            musite = rlx_st.frac_coords[rlx_st.atomic_numbers.index(1)]
            rlx_st.translate_sites(
                range(rlx_st.num_sites), -musite, frac_coords=True, to_unit_cell=False
            )
            if self.ctx.structure_type == HubbardStructureData:
                inputs.pw.structure = LegacyStructureData(pymatgen=rlx_st)
                inputs.pw.structure = HubbardStructureData.from_structure(inputs.pw.structure)
            else:
                inputs.pw.structure = self.ctx.structure_type(pymatgen=rlx_st)
                
            #inputs.pw.structure = self.ctx.structure_type(pymatgen=rlx_st)
            if isinstance(inputs.pw.structure,HubbardStructureData):
                inputs.pw.structure = create_hubbard_structure(inputs.pw.structure,self.inputs.structure)
            elif self.ctx.hubbardu_dict and not isinstance(inputs.pw.structure,HubbardStructureData) and "magmom" in self.inputs:
                inputs.pw.structure = create_hubbard_structure(inputs.pw.structure,orm.Dict(dict=self.ctx.hubbardu_dict))
            
            inputs.pw.pseudos = get_pseudos(
                inputs.pw.structure, self.inputs.pseudo_family.value
            )
            
            inputs.pop("pseudo_family", None)
            
            # Set the `CALL` link label and submit
            inputs.metadata.call_link_label = f'mu_origin_supercell_{j_index:02d}'
            pwb_future = self.submit(PwBaseWorkChain, **inputs)
            pwb_key = f"pwb_workchain_{j_index}"
            self.report(
                f"Launching PwBaseWorkChain (PK={pwb_future.pk}) for PWRelaxed (uuid={c_uuid}) structure"
            )
            self.to_context(**{pwb_key: pwb_future})

    def compute_spin_density(self):
        """Compute spin density at unique candidate sites, via pp.x
        
        """
        self.report("Computing Spin density")

        PpCalculation = CalculationFactory("quantumespresso.pp")
        pp_builder = PpCalculation.get_builder()
        pp_builder.code = self.inputs.pp_code


        if hasattr(self.inputs,"pp_metadata"):
            pp_builder.metadata = self.inputs.pp_metadata #.get_dict()
        #MB: the following should not be done, but for aiidalab qe app we need intranode only:
        if pp_builder.metadata['options']['resources']['num_machines'] > 1:
            pp_builder.metadata['options']['resources']['num_machines'] = 1

        parameters = orm.Dict(
            dict={
                "INPUTPP": {
                    "plot_num": 6,
                },
                "PLOT": {"iflag": 3},
            }
        )
        pp_builder.parameters = parameters

        unique_cluster_list = self.ctx.unique_cluster

        # for direct pp.x without scf
        """
        for j_index, clus in enumerate(unique_cluster_list):
            c_uuid = self.ctx.n_uuid_dict[clus['idx']]
            rlx_node = orm.load_node(c_uuid)
            pp_builder.parent_folder = rlx_node.outputs.remote_folder

            pp_future = self.submit(pp_builder)
            pkey = f'pworkchain_{j_index}'
            self.report(f'Launching PpCalcJOb  with (PK={pp_future.pk}) for PWRelaxed (UUID={c_uuid}) structure')
            self.to_context(**{pkey: pp_future})
        """

        # inspect the scf pw.x run and then run pp.x
        for j_index, clus in enumerate(unique_cluster_list):
            pwb_key = f"pwb_workchain_{j_index}"
            pwb_workchain = self.ctx[pwb_key]

            if not pwb_workchain.is_finished_ok:
                self.report(
                    f"PwbaseWorkChain failed with exit status {pwb_workchain.exit_status}"
                )
                return self.exit_codes.ERROR_BASE_CALC_FAILED
            else:
                pp_builder.parent_folder = pwb_workchain.outputs.remote_folder
                # print('pbasepk',pwb_workchain.pk)

                pp_future = self.submit(pp_builder)
                pkey = f"pworkchain_{j_index}"
                c_uuid = self.ctx.n_uuid_dict[clus["idx"]]
                self.report(
                    f"Launching PpCalcJOb  with (PK={pp_future.pk}) for PWRelaxed \
                (UUID={c_uuid}) structure and PWBase-mu-origin (PK={pwb_workchain.pk}) "
                )
                self.to_context(**{pkey: pp_future})

    def compute_contact_hyperfine(self):
        """compute spin density at unique candidate sites"""
        self.report("Getting Contact field")
        unique_cluster_list = self.ctx.unique_cluster
        # contact_hf = []
        chf_dict = {}

        for j_index, clus in enumerate(unique_cluster_list):
            pwb_key = f"pwb_workchain_{j_index}"  # remove later
            pwb_workchain = self.ctx[pwb_key]

            pkey = f"pworkchain_{j_index}"
            pworkchain = self.ctx[pkey]

            if not pworkchain.is_finished_ok:
                self.report(
                    f"Spin density post-process simulation failed with exit status {pworkchain.exit_status}"
                )
                return self.exit_codes.ERROR_PP_CALC_FAILED
            else:
                p_pk = pworkchain.pk
                sp_density = pworkchain.outputs.output_data.get_array("data")[0, 0, 0]
                # contact_hf.append(({'rlx_idx':clus['idx'],'pwb_pk':pwb_workchain.pk, 'pp_pk':pworkchain.pk, 'spin_density':sp_density, 'hf_T':sp_density*52.430351})) # In Tesla
                chf_dict.update(
                    {str(clus["idx"]): [sp_density, sp_density * 52.430351]}
                )

        # self.ctx.cont_hf = contact_hf
        self.ctx.cont_hf = orm.Dict(dict=chf_dict)
        # print("contact_results ",chf_dict)

    def get_dipolar_field(self):
        unique_cluster_list = self.ctx.unique_cluster
        if self.inputs.spin_pol_dft: cnt_field_dict = self.ctx.cont_hf.get_dict()
        dip_results = []
        for j_index, clus in enumerate(unique_cluster_list):
            #
            # rlx_st = clus['rlxd_struct']
            rlx_st = Structure.from_dict(clus["rlxd_struct"])
            if self.ctx.structure_type == HubbardStructureData:
                rlx_struct = LegacyStructureData(pymatgen=rlx_st)
                rlx_struct = HubbardStructureData.from_structure(rlx_struct)
            else:
                rlx_struct = self.ctx.structure_type(pymatgen=rlx_st)
            if not self.inputs.spin_pol_dft:
                cnt_field = 0
            else:
                cnt_field = cnt_field_dict[str(clus["idx"])][1]
            print(cnt_field)
            b_field = compute_dipolar_field(
                self.inputs.structure,
                self.ctx.magmom,
                self.ctx.sc_matrix,
                rlx_struct,
                orm.Float(cnt_field),
            )
            # dip_results.update({str(clus['idx']):[b_field[0][0], b_field[0][1], b_field[0][2]]})  #as dict
            dip_results.append(
                (
                    {
                        "idx": clus["idx"],
                        "Bdip": b_field[0][0],
                        "B_T": b_field[0][1],
                        "B_T_norm": b_field[0][2],
                    }
                )
            )

        self.ctx.dipolar_dict = orm.List(dip_results)
        print("dipolar_results ", dip_results)

    def set_field_outputs(self):
        """outputs"""
        self.report("Setting field outputs")
        # self.out('unique_sites_hyperfine', get_list(self.ctx.cont_hf))
        if self.inputs.spin_pol_dft:
            self.out("unique_sites_hyperfine", self.ctx.cont_hf)
        self.out("unique_sites_dipolar", self.ctx.dipolar_dict)

    def set_relaxed_muon_outputs(self):
        """outputs"""
        # self.report('Setting Relaxation and analysis Outputs')

        self.out(
            "all_index_uuid",
            get_dict_uuid(orm.List(list(self.ctx.n_uuid_dict.items()))),
        )

        self.out("all_sites", get_dict_output(orm.List(self.ctx.relaxed_outputs_all)))

        self.out("unique_sites", get_dict_output(orm.List(self.ctx.unique_cluster)))
        
        self.report("final output provided, the workflow is completed successfully.")


#################################################################################
# calcfunctions and called functions

def get_pseudos(aiida_struc, pseudofamily):
    """Get pseudos"""
    family = orm.load_group(pseudofamily)
    pseudos = family.get_pseudos(structure=aiida_struc)
    return pseudos

@calcfunction
def get_dict_uuid(outdata):
    """convert list to aiida dictionary for outputting"""
    out_dict = {}

    for i, dd in enumerate(outdata):
        out_dict.update({str(dd[0]): dd[1]})

    return orm.Dict(dict=out_dict)


@calcfunction
def get_dict_output(outdata):
    """convert list to aiida dictionary for outputting"""
    out_dict = {}

    for i, dd in enumerate(outdata):
        out_dict.update({str(dd["idx"]): [dd["rlxd_struct"], dd["energy"]]})

    return orm.Dict(dict=out_dict)

#Creates the default used in the protocols and in the forcing inputs step.
def get_default_dict(structure, pseudo_family, kpoints_distance, charge_supercell,magmom, spin_pol_dft):
    _overrides = {
           "base": {
                #"pseudo_family": pseudo_family,
                "kpoints_distance": kpoints_distance,
                "pw": {
                    "parameters": {
                "CONTROL": {
                    #"etot_conv_thr" =   1.0e-4, 
                    #"forc_conv_thr" =   1.0e-3,   this default is preferred for relax, it is 1e-4 for PWbaseworkchain
                    "nstep": 200
                    },
                "SYSTEM":{
                    "occupations": "smearing",
                    "smearing": "gaussian",
                    "degauss": 0.01,},
                "ELECTRONS": {
                    "electron_maxstep": 500,
                    "mixing_mode": "local-TF",
                    'conv_thr': 1.0e-6,
                    'mixing_beta':0.3,
                },
                },
                    "metadata": {
                    "description": "Muon site calculations for "
                    + structure.get_pymatgen_structure().formula
                },
                },
            },
            #"base_final_scf": {"pseudo_family": pseudo_family,},
            "clean_workdir": orm.Bool(True),
        }

    if charge_supercell:
        _overrides["base"]["pw"]["parameters"]["SYSTEM"]["tot_charge"] = 1.0
        
    # MAGMOMS       
    if magmom and spin_pol_dft: # drop this... correct ?and not hasattr(structure, "magnetic"):
        rst_mg = make_collinear_getmag_kind(
            structure, magmom,
        )
        structure = rst_mg["struct_magkind"]
        start_mg_dict = rst_mg["start_mag_dict"]
        
        _overrides["base"]["pw"]["parameters"]["SYSTEM"]["nspin"]= 2
        _overrides["base"]["pw"]["parameters"]["SYSTEM"]["starting_magnetization"] = start_mg_dict.get_dict()
        
    else:
        start_mg_dict = None
    
    # Produce the overrides for IsolatedImpurityWorkChain: the same pw setup as for PwRelaxWorkChain, but for PwBaseWorkChain.
    _overrides["impuritysupercellconv"] = {
        "base" : copy.deepcopy(_overrides["base"]),
        "pre_relax" : copy.deepcopy(_overrides), 
    }
    # switch off charge in the pre_relax:
    _overrides["impuritysupercellconv"]["pre_relax"]["base"]["pw"]["parameters"]["SYSTEM"]["tot_charge"] = 0
        
        
    
    
    
    '''# HUBBARD
    # check and assign hubbard u
    inpt_st = structure.get_pymatgen_structure()
    ##TO DO:put a check on  parameters that cannot be set by hand in the overrides eg mag, hubbard.
    rst_u = check_get_hubbard_u_parms(inpt_st)
    hubbardu_dict = rst_u 
    if hubbardu_dict and hubbard:
        _overrides["base"]["pw"]["parameters"]["SYSTEM"]["lda_plus_u"] = True
        _overrides["base"]["pw"]["parameters"]["SYSTEM"]["lda_plus_u_kind"] = 0
        _overrides["base"]["pw"]["parameters"]["SYSTEM"]["Hubbard_U"] = hubbardu_dict'''
        
    return _overrides, start_mg_dict, structure


def iterdict(d,key):
  value = None
  for k,v in d.items():
    if isinstance(v, dict):
        value = iterdict(v,key)
    else:            
        if k == key:
          return v
    if value: return value


# NOTE: for now, this is ignored. The user should be free to set whatever he wants, and should be free to fail.
def recursive_consistency_check(input_dict,_):
    
    
    """Validation of the inputs provided for the FindMuonWorkChain. It checks essentially the same of pw_overrides. If you go from protocols you are safe, except for Hubbard: 
    an exception is raise if it is needed, but you have to set it up in your StructureData. PROBLEM: how to deal with supercell generation...
    """
    
    #check hubbard here ore somewhere else. 
    
    parameters = copy.deepcopy(input_dict)
    _overrides, start_mg_dict, structure = get_override_dict(parameters["structure"],parameters["pseudo_family"], parameters["kpoints_distance"], parameters["charge_supercell"],parameters.pop('magmom',None),parameters.pop("spin_pol_dft",None))
    
    inconsistency_sentence = ''
    
    '''#Hubbard validation in the structure:
    hubbard_params = check_get_hubbard_u_parms(structure.get_pymatgen())
    if hubbard_params is not None:
        if "hubbard" not in structure.get_defined_properties() or structure.hubbard.parameters == []:
            if structure.is_stored:
                inconsistency_sentence+="The structure you provided as input is stored but requires hubbard parameters. Please define a new StructureData instance with also hubbard parameters according to this: \n{hubbard_params}."
            else:
                inconsistency_sentence+="The structure you provided as input requires hubbard parameters. Please define hubbard parameters according to this: \n{hubbard_params}."'''
        
    #QE inputs validation:
    keys = ["tot_charge","nspin","occupations","smearing"]
    
    wrong_inputs_relax = []
    wrong_inputs_pwscf = []
    
    impuritysupercellconv_inconsistency = ''
    if "impuritysupercellconv" in parameters:
        impuritysupercellconv_inconsistency = impuritysupercellconv_input_validator(parameters["impuritysupercellconv"],None,caller="FindMuonWorkchain")
    
    if impuritysupercellconv_inconsistency: inconsistency_sentence += impuritysupercellconv_inconsistency
    
    if parameters["relax"]["base"]["pw"]["parameters"].get_dict()["CONTROL"]["calculation"] != 'relax':
        inconsistency_sentence+=f'Checking inputs.relax.base.pw.parameters.CONTROL.calculation: can be only "relax". No cell relaxation should be performed.'
    
    
    if 'base_final_scf' in parameters['relax']:
        if parameters['relax']['base_final_scf'] ==  {'metadata': {}, 'pw': {'metadata': {'options': {'stash': {}}}, 'monitors': {}, 'pseudos': {}}}:
            pass
        elif parameters['relax']['base_final_scf'] ==  {}:
            pass
        else:
            inconsistency_sentence+=f'Checking inputs.relax.base_final_scf: should not be set, the final scf after relaxation is not supported in the FindMuonWorkChain.'
    
    if "pwscf" in parameters: #mu scf origin.
        if not "pp_code" in parameters: 
            inconsistency_sentence+=f'Checking inputs: "pp_code" input not provided but required!'
        elif not parameters["pp_code"]: 
            inconsistency_sentence+=f'Checking inputs: "pp_code" input not provided but required!'

        if not "pp_metadata" in parameters: 
            inconsistency_sentence+=f'Checking inputs: "pp_metadata" input not provided but required!'
        
    for key in keys:
        value_input_relax = iterdict(parameters["relax"]["base"]["pw"]["parameters"].get_dict(),key)
        value_overrides = iterdict(_overrides,key)
        #print(value_input_relax,value_input_pwscf,value_overrides)
        if value_input_relax != value_overrides:
            if value_input_relax in [0, None] and value_overrides in [0, None]:
                continue # 0 is None and viceversa
            wrong_inputs_relax.append(key)
            inconsistency_sentence += f'Checking inputs.relax.base.pw.parameters input: "{key}" is not correct. You provided the value "{value_input_relax}", but only "{value_overrides}" is consistent with your settings.\n'
        
        if "pwscf" in parameters: #mu scf origin.
            value_input_pwscf = iterdict(parameters["pwscf"]["pw"]["parameters"].get_dict(),key)
            if value_input_pwscf != value_overrides:
                if key == "nspin" and value_input_pwscf == 2: 
                    continue
                if value_input_pwscf in [0, None] and value_overrides in [0, None]:
                    continue # 0 is None and viceversa
                wrong_inputs_pwscf.append(key)
                inconsistency_sentence += f'Checking inputs.pwscf.pw.parameters input: "{key}" is not correct. You provided the value "{value_input_pwscf}", but only "{value_overrides}" is consistent with your settings.\n'
    
    if len(wrong_inputs_relax+wrong_inputs_pwscf)>0:
        raise ValueError('\n'+inconsistency_sentence+'\n Please check the inputs of your FindMuonWorkChain instance or use "get_builder_from_protocol()" method to populate correctly the inputs.')

                      
    return 
