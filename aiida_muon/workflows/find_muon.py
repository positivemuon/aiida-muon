from aiida import orm
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida_quantumespresso.common.types import RelaxType
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida.engine import WorkChain, calcfunction
from aiida.engine import if_
import numpy as np
from pymatgen.electronic_structure.core import Magmom
from niche import Niche
from .utils import (get_collinear_mag_kindname, 
    check_get_hubbard_u_parms, 
    get_struct_wt_distortions, 
    load_workchain_data, 
    cluster_unique_sites)



class FindMuonWorkChain(WorkChain):
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
        
        ## pw qe particular inputs
        spec.input_namespace('qe', required = False, 
                            help="Input parameters, settings and options for QE DFT calculations")
        
        
        spec.input("qe.structure", valid_type = orm.StructureData, required = True, 
                   help = 'Input initial structure')
        
        #read as list or array?
        spec.input("qe.magmom", valid_type = orm.List, required = False, 
                   help = 'List of 3D magnetic moments in Bohr magneton of the corresponding input structure if magnetic')
        
        #spec.input("qe.sc_matrix", valid_type = orm.ArrayData, required = True, 
        #           help = ' Array of supercell size, PS: array label is also "sc_matrix". ')
        
        spec.input("qe.sc_matrix", valid_type = orm.List, required = True, 
                   help = ' List of length 1 for supercell size ')
        
        spec.input('qe.mu_spacing', valid_type = orm.Float, default = lambda: orm.Float(1.0), required = False, 
                   help = 'Minimum distance in Angstrom between two starting muon positions  generated on a grid.') 
        
        spec.input("qe.pw_code", valid_type = orm.Code, required = True,
                  help="The pw.x code-computer for dft the calculations")
        
        spec.input("qe.pp_code", valid_type = orm.Code, required = False,
                  help="The pp.x code-computer for post processing only if qe.magmom is supplied")
        
        spec.input("qe.pseudofamily", valid_type = orm.Str, 
                   default = lambda: orm.Str('SSSP/1.2/PBE/efficiency'), required = False,
                  help="The label of the pseudo family")
        
        spec.input('qe.k_dist', valid_type = orm.Float, default = lambda: orm.Float(0.401)
                   , help = 'The minimum desired distance in 1/Å between k-points in reciprocal space.') 
        
        spec.input('qe.charge_supercell', valid_type=orm.Bool, default = lambda: orm.Bool(True),
                   required=False, help = 'To run charged supercell for positive muon or not (neutral supercell)')
        
        spec.input("qe.parameters", valid_type = orm.Dict, required = False,
                  help=" Preferred pw.x set of parameters, otherwise it is set automatically")
        
        spec.input("qe.settings", valid_type = orm.Dict, required = False,
                  help=" Preferred settings for the calc, otherwise default is used")
        
        spec.input("qe.metadata", valid_type = orm.Dict, required = False,
                  help=" Preferred metadata and scheduler options for relax calc, otherwise  default in the Code is used")
        
        spec.input("qe.pp.metadata", valid_type = orm.Dict, required = False,
                  help=" Preferred metadata and scheduler options for pp.x")
        
        
        spec.outline(cls.get_initial_muon_sites, 
                     cls.setup_magnetic_hubbardu_dict, 
                     cls.get_initial_supercell_structures, 
                     cls.setup_pw_overrides, 
                     cls.compute_supercell_structures, 
                     cls.collect_relaxed_structures,
                     cls.analyze_relaxed_structures,
                     if_(cls.new_struct_after_analyze)(
                         cls.compute_supercell_structures, 
                         cls.collect_relaxed_structures,
                         cls.collect_all_results
                     ),
                     if_(cls.structure_is_magnetic)(
                         cls.run_scf_mu_origin,                    #to be removed for better alternative
                         cls.compute_spin_density,
                         cls.inspect_get_contact_hperfine,
                         #cls.set_hperfine_outputs
                     ),
                     #cls.set_relaxed_muon_outputs
                    )
                     
        spec.exit_code(405, 'ERROR_RELAX_CALC_FAILED',
                       message='One of the PwRelaxWorkChain subprocesses failed')
        
        spec.exit_code(406, 'ERROR_BASE_CALC_FAILED',
                       message='One of the PwBaseWorkChain subprocesses failed')
        
        spec.exit_code(407, 'ERROR_PP_CALC_FAILED',
                       message='One of the PPWorkChain subprocesses failed')
        
        #TODO: more exit codes catch errors and throw exit codes
        
        #spec.output('all_sites_outputs',  valid_type = orm.List, required = True)
        
        #spec.output('unique_sites_structures',  valid_type = orm.List, required = True)
        
        #spec.output('unique_sites_hyperfine',  valid_type = orm.List, required = False) #return only when magnetic
        
    
    def get_initial_muon_sites(self):
        """ get list of starting muon sites"""
        
        #repharse Niche, input and outputs?
        #Not clear only spacing parameter, need for minimum number of initial muon?
        
        rst = niche_add_impurities( self.inputs.qe.structure, 
                               orm.Str('H'),
                               self.inputs.qe.mu_spacing,
                               orm.Float(1.0)
                              )
        #self.ctx.mu_lst = rst["mu_lst"]
        self.ctx.mu_lst = rst
    
    def setup_magnetic_hubbardu_dict(self):
        """
        Gets:
        (i)structure with kindname from magmom
        (ii) Dictionary for startuing magnetization
        (iii) Dictionary for Hubbard-U parameters
        """
        
        #get the magnetic kind relevant for pw spin-polarization setup
        if 'magmom' in self.inputs.qe:
            rst_mg = make_collinear_getmag_kind(self.inputs.qe.structure,self.inputs.qe.magmom)
            self.ctx.struct = rst_mg["struct_magkind"]
            self.ctx.start_mg_dict = rst_mg["start_mag_dict"]
        else: self.ctx.struct = self.inputs.qe.structure
            
        #check and get hubbard u
        inpt_st = self.ctx.struct.get_pymatgen_structure()
        rst_u = check_get_hubbard_u_parms(inpt_st)
        self.ctx.hubbardu_dict = rst_u
    
    
     
    def get_initial_supercell_structures(self):
        """Get initial supercell+muon list"""
        
        self.report('Getting supercell list')
        input_struct = self.ctx.struct.get_pymatgen_structure()
        #muon_list    = self.ctx.mu_lst.get_array('mu_list')
        muon_list    = self.ctx.mu_lst
        sc_mat       = self.inputs.qe.sc_matrix[0]
        
        supercell_list = gensup(input_struct, muon_list, sc_mat) #ordinary function
        self.ctx.supc_list = supercell_list
        
    
    def setup_pw_overrides(self):
        """Get the required overrides i.e pw parameter setup"""
        
        self.report("Setting up the relaxation calculation")
        overrides = {
                'base':{
                     'pseudo_family': self.inputs.qe.pseudofamily.value,
                      'kpoints_distance' : self.inputs.qe.k_dist.value,
                    'pw': {
                    	'parameters': {},
                        # 'metadata': self.inputs.qe.metadata.get_dict(),
                        #'settings': self.inputs.qe.settings.get_dict(),
                        }
                    },
                'base_final_scf':{
                     'pseudo_family': self.inputs.qe.pseudofamily.value,
                     'kpoints_distance' : self.inputs.qe.k_dist.value,
                    'pw': {
                    	'parameters': {},
                        # 'metadata': self.inputs.qe.metadata.get_dict(),
                        #'settings': self.inputs.qe.settings.get_dict(),
                        }
                    },
                'clean_workdir' : orm.Bool(False),
                }
        
        ##TO DO:put a check on  parameters that cannot be set by hand in the overrides eg mag, hubbard
        
        if 'parameters' in self.inputs.qe:
            overrides['base']['pw']['parameters'] = self.inputs.qe.parameters.get_dict()
            overrides['base_final_scf']['pw']['parameters'] = self.inputs.qe.parameters.get_dict()
        
        #set some cards
        overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'CONTROL':{'nstep': 200}})
        #overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'smearing': 'gaussian'}})
        overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'ELECTRONS':{'electron_maxstep': 300}})
        overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'ELECTRONS':{'mixing_beta': 0.30}})
        #overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'ELECTRONS':{'conv_thr': 1.0e-6}})
        #
        overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'CONTROL':{'nstep': 200}})
        #overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'SYSTEM':{'smearing': 'gaussian'}})
        overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'ELECTRONS':{'electron_maxstep': 300}})
        overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'ELECTRONS':{'mixing_beta': 0.30}})
        #overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'ELECTRONS':{'conv_thr': 1.0e-6}})
            
        if self.inputs.qe.charge_supercell:  
            overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'tot_charge': 1.0}})
            overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'SYSTEM':{'tot_charge': 1.0}})
        
        #if self.inputs.magmom is not None:
        if 'magmom' in self.inputs.qe and self.ctx.start_mg_dict:
            overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'nspin': 2}})
            overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'starting_magnetization': self.ctx.start_mg_dict.get_dict()}})
            overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'SYSTEM':{'nspin': 2}})
            overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'SYSTEM':{'starting_magnetization': self.ctx.start_mg_dict.get_dict()}})
            
        
        #check and assign hubbard u
        if self.ctx.hubbardu_dict:
            overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'lda_plus_u': True}})
            overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'lda_plus_u_kind': 0}})
            overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'Hubbard_U': self.ctx.hubbardu_dict}})
            overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'SYSTEM':{'lda_plus_u': True}})
            overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'SYSTEM':{'lda_plus_u_kind': 0}})
            overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'SYSTEM':{'Hubbard_U': self.ctx.hubbardu_dict}})
        
        
        if 'metadata' in self.inputs.qe:
            overrides['base']['pw']['metadata'] = self.inputs.qe.metadata.get_dict()
            overrides['base_final_scf']['pw']['metadata'] = self.inputs.qe.metadata.get_dict()
        
        if 'settings' in self.inputs.qe:
            overrides['base']['pw']['settings'] = self.inputs.qe.settings.get_dict()
            overrides['base_final_scf']['pw']['settings'] = self.inputs.qe.settings.get_dict()
        
        self.ctx.overrides = overrides
    
    
    
    def compute_supercell_structures(self):
        """Run relax workflows for each muon supercell"""
        
        self.report('Computing muon supercells')
        supercell_list = self.ctx.supc_list 
        
        
        for i_index in range(len(supercell_list)):
            pw_builder = PwRelaxWorkChain.get_builder_from_protocol(
                code = self.inputs.qe.pw_code,
                structure = orm.StructureData(pymatgen = supercell_list[i_index]) ,
                overrides = self.ctx.overrides,
                relax_type = RelaxType.POSITIONS
            )
            
            #pw_builder['base']['pw']['metadata'] = self.inputs.qe.metadata.get_dict()
            #pw_builder['base']['pw']['metadata'] ['description'] = ' Relaxation for muon supercell index'+str(i_index)
            #inputs.clean_workdir = self.inputs.clean_workdir
                
            future = self.submit(pw_builder) 
            #key = f'workchains.sub{i_index}' #nested sub
            key = f'workchain_{i_index}'
            self.report(f'Launching PwRelaxWorkChain (PK={future.pk}) for supercell structure index {i_index}')
            self.to_context(**{key: future})
    
    
    #work tomorrow
    def collect_relaxed_structures(self):
        """Retrieve final positions and energy from the relaxed structures"""
       
        self.report('Gathering computed positions and energy')
        supercell_list = self.ctx.supc_list 

        computed_results = []
        
        #for nested
        #for key, workchain in self.ctx.workchains.items():
        #    if not workchain.is_finished_ok
    
        for i_index in range(len(supercell_list)):
            key = f'workchain_{i_index}'
            workchain = self.ctx[key]
            
            #TODO:IMPLEMEMENT CHECKS FOR RESTART OF UNFINISHED CALCULATION 
            #     AND/OR NUMBER OF UNCONVERGED CALC IS ACCEPTABLE
            
            if not workchain.is_finished_ok:
                self.report(f'PwRelaxWorkChain failed with exit status {workchain.exit_status}')
                return self.exit_codes.ERROR_RELAX_CALC_FAILED
            else:
                pk     = workchain.pk
                energy = workchain.outputs.output_parameters.get_dict()['energy']
                rlx_structure = workchain.outputs.output_structure.get_pymatgen_structure()
                #rlx_structure = workchain.outputs.output_structure
                
                #computed_results.append((pk,rlx_structure,energy))
                computed_results.append(({'pk':pk,'rlxd_struct':rlx_structure,'energy':energy}))
                
                #print(computed_results)
        
        self.ctx.relaxed_outputs = computed_results
        
    def analyze_relaxed_structures(self):
        """
        Analyze relaxed structures and get unique candidate sites and
        check if there are new magnetic equivalent sites to calculate
        """
        self.report('Analyzing the relaxed structures')
        inpt_st = self.inputs.qe.structure.get_pymatgen_structure()
        
        
        if 'magmom' in self.inputs.qe:
            r_anly = analyze_structures(
                self.ctx.supc_list[0], 
                self.ctx.relaxed_outputs, 
                inpt_st,
                self.inputs.qe.magmom
            )
        else:
            r_anly = analyze_structures(
                self.ctx.supc_list[0], 
                self.ctx.relaxed_outputs, 
                inpt_st
            )
            
        self.ctx.unique_cluster = r_anly['unique_pos']
        print('uniq_positions',self.ctx.unique_cluster)
        
        #revisit, this so the initial inputs and collected results are not ovewritten with repeated calls in outline
        self.ctx.supc_list_all = self.ctx.supc_list
        self.ctx.relaxed_outputs_all  = self.ctx.relaxed_outputs
        
        self.ctx.supc_list = r_anly['mag_inequivalent']
    
    def new_struct_after_analyze(self):
        """Check if there is new magnetic inequivalent sites"""
        self.report('Checking new structures to calculate')
        
        return len(self.ctx.supc_list) > 0
    
    def collect_all_results(self):
        """Collecting results of new structures and then append"""
        self.report('Appending results of new structures ')
        
        self.ctx.relaxed_outputs_all.extend(self.ctx.relaxed_outputs)
        self.ctx.unique_cluster.extend(self.ctx.relaxed_outputs)
        

    def structure_is_magnetic(self):
        """ Checking if structure is magnetic"""
        
        #return self.inputs.qe.magmom is not None
        #return 'magmom' in self.inputs.qe
        if 'magmom' in self.inputs.qe:
            return True
        else:
            return False
    
    #scf first then pp.x ! TODO: NOT NECESSARY REMOVE ON REVISIT
   
    def run_scf_mu_origin(self):
        """Move muon to origin and  perform scf"""
        unique_cluster_list = self.ctx.unique_cluster 
        
        pb_overrides = self.ctx.overrides['base']
        #pb_overrides['pw'] = recursive_merge(pb_overrides['pw'], {'clean_workdir' : orm.Bool(False)})
        pb_overrides = recursive_merge(pb_overrides, {'clean_workdir' : orm.Bool(False)})
        
        for j_index, clus in enumerate(unique_cluster_list):
            #
            #rlx_st = clus['rlxd_struct']
            #rlx_struct = orm.StructureData(pymatgen = rlx_st)
            #or 
            
            rlx_node = orm.load_node(clus['pk'])
            rlx_st = rlx_node.outputs.output_structure.get_pymatgen_structure()
            
            #move muon to origin
            musite = rlx_st.frac_coords[rlx_st.atomic_numbers.index(1)]
            rlx_st.translate_sites(range(rlx_st.num_sites),-musite, frac_coords=True, to_unit_cell=False)
            rlx_struct = orm.StructureData(pymatgen = rlx_st)
            
            pwb_builder = PwBaseWorkChain.get_builder_from_protocol(
                code = self.inputs.qe.pw_code,
                structure = rlx_struct ,
                overrides = pb_overrides
            )
            #pwb_builder['pw']['metadata'] = self.inputs.qe.metadata.get_dict()
            
            pwb_future = self.submit(pwb_builder) 
            pwb_key = f'pwb_workchain_{j_index}'
            c_pk = clus['pk']
            self.report(f'Launching PwBaseWorkChain (PK={pwb_future.pk}) for PWRelaxed (PK={c_pk}) structure')
            self.to_context(**{pwb_key: pwb_future})
    
    
    def compute_spin_density(self):
        """compute spin density at unique candidate sites"""
        self.report('Computing Spin density')
        
        PpCalculation = CalculationFactory('quantumespresso.pp')
        pp_builder = PpCalculation.get_builder()
        pp_builder.code = self.inputs.qe.pp_code
        
        if 'pp.metadata' in self.inputs.qe:
            pp_builder.metadata = self.inputs.qe.pp.metadata.get_dict()
        
        parameters = orm.Dict(dict={
            'INPUTPP': {
                "plot_num" : 6,
            },
            'PLOT': {
                "iflag" : 3
            }
        })
        pp_builder.parameters = parameters
        
        unique_cluster_list = self.ctx.unique_cluster
        
        #for direct pp.x without scf
        """
        for j_index, clus in enumerate(unique_cluster_list):
            rlx_node = orm.load_node(clus['pk'])
            pp_builder.parent_folder = rlx_node.outputs.remote_folder
            
            pp_future = self.submit(pp_builder) 
            pkey = f'pworkchain_{j_index}'
            self.report(f'Launching PpCalcJOb  with (PK={pp_future.pk}) for PWRelaxed (PK={clus['pk']}) structure')
            self.to_context(**{pkey: pp_future})
        """
        
        #inspect the scf pw.x run and then run pp.x    
        for j_index, clus in enumerate(unique_cluster_list):
            pwb_key = f'pwb_workchain_{j_index}'
            pwb_workchain = self.ctx[pwb_key]
            
            
            if not pwb_workchain.is_finished_ok:
                self.report(f'PwbaseWorkChain failed with exit status {pwb_workchain.exit_status}')
                return self.exit_codes.ERROR_BASE_CALC_FAILED
            else:
                pp_builder.parent_folder = pwb_workchain.outputs.remote_folder
                #print('pbasepk',pwb_workchain.pk)
                
                pp_future = self.submit(pp_builder) 
                pkey = f'pworkchain_{j_index}'
                c_pk = clus['pk']
                self.report(f'Launching PpCalcJOb  with (PK={pp_future.pk}) for PWRelaxed \
                (PK={c_pk}) structure and PWBase-mu-origin (PK={pwb_workchain.pk}) ')
                self.to_context(**{pkey: pp_future})
                
        
    def inspect_get_contact_hperfine(self):
        """compute spin density at unique candidate sites"""
        self.report('Getting Contact field')
        unique_cluster_list = self.ctx.unique_cluster
        contact_hf = []
        
        for j_index, clus in enumerate(unique_cluster_list):
            pwb_key = f'pwb_workchain_{j_index}'  #remove later
            pwb_workchain = self.ctx[pwb_key]
            
            pkey = f'pworkchain_{j_index}'
            pworkchain = self.ctx[pkey]
            
            if not pworkchain.is_finished_ok:
                self.report(f'PpWorkChain failed with exit status {pworkchain.exit_status}')
                return self.exit_codes.ERROR_PP_CALC_FAILED
            else:
                p_pk     = pworkchain.pk
                sp_density = pworkchain.outputs.output_data.get_array('data')[0,0,0]
                contact_hf.append(({'rlx_pk':clus['pk'],'pwb_pk':pwb_workchain.pk, 'pp_pk':pworkchain.pk, 'spin_density':sp_density, 'hf_T':sp_density*52.430351})) # In Tesla
        
        self.ctx.cont_hf = contact_hf
        print("contact_results ",self.ctx.cont_hf)
    
    
    #def set_hperfine_outputs(self):
    #    """outputs"""
    #    self.report('Setting hypferfine Outputs')
    
    
    #YET TO DECIDE HOW TO ORDER THE OUTPUTS. STRUCTURES ARE NOT SERIALIZABLE IN A LIST; SO NOT AN OPTION.
    #DYNAMIC OUTPUT AND PUT STRUCTURES IN A DO LOOP
    
    
    #def set_relaxed_muon_outputs(self):
        #outputs
        #self.report('Setting Relaxation and analysis Outputs')
        
 






















#GENERAL TODO: FOR SPEEDUP CHANGE MOSTLY TO ARRAY INSTEAD OF LIST ESP. IF ELEMENTS CAN BE MADE HOMOGENOUS                   
#################################################################################
# calcfunctions and called functions


    

@calcfunction
def niche_add_impurities(
    structure: orm.StructureData,
    niche_atom: orm.Str,
    niche_spacing: orm.Float,
    niche_distance: orm.Float) :
    """
    This calcfunction calls Niche. Supplies structure, atom index and impurity 
    spacing required to get the grid initial sites
    
    Return: Adapted here to return only lists of generated muon sites.
    """

    #niche_class = get_object_from_string("niche.Niche")

    pmg_st = structure.get_pymatgen_structure()
    #niche_instance = niche_class(pmg_st, niche_atom.value)
    niche_instance = Niche(pmg_st, niche_atom.value)

    n_st = niche_instance.apply(niche_spacing.value, niche_distance.value)
    
    new_structure_data = orm.StructureData()
    new_structure_data.set_pymatgen(n_st)
    #print(n_st)
    
    #+0.001 to break symmetry if at symmetry pos
    mu_lst = [i+ 0.001 for j,i in enumerate(n_st.frac_coords) if n_st.species[j].value == niche_atom.value]
    
    #mu_lst_node = orm.ArrayData()
    #mu_lst_node.set_array('mu_list', np.array(mu_lst))

    #return new_structure_data
    #return {"mu_lst":mu_lst_node}
    return orm.List(mu_lst)


#@calcfunction
#def gensup(aiida_struc, mu_list, sc_matrix):
    #p_st = aiida_struc.get_pymatgen_structure()
    #imp_list = mu_list.get_array('mu_list')
    #sc_mat = sc_matrix.get_array('sc_matrix')
    
# Do we really need to keep this in the provenance?
def gensup(p_st, mu_list, sc_mat): 
    """ 
    This makes the supercell with the given SC matrix.
    It also appends the muon.
    
    Returns: list of supercell structures with muon. 
              Number of supercells depends on number of imput mulist
    """
    supc_list=[]
    for ij in mu_list:
        p_scst = p_st.copy()
        p_scst.make_supercell(sc_mat)
        ij_sc  = (np.dot(ij,np.linalg.inv(sc_mat)))%1
        #ij_sc = [x + 0.001 for x in ij_sc]
        p_scst.append(species = "H", 
                      coords = ij_sc, 
                      coords_are_cartesian = False, 
                      validate_proximity = True, 
                      properties = {'kind_name': 'H'}
                     )
        supc_list.append(p_scst)
    return supc_list



@calcfunction
def make_collinear_getmag_kind(aiid_st,magmm):
    """ 
    This calls the 'get_collinear_mag_kindname' utility function.
    It takes the provided magmom, make it collinear and then with
    assign kind_name property for each atom site relevant 
    spin polarized calculation.
    
    Returns: Structure data and dictionary of pw starting magnetization card.
    """
    p_st = aiid_st.get_pymatgen_structure()
    #magmm = magmom_node.get_array('magmom')
    #from array to Magmom object
    magmoms = [Magmom(magmom) for magmom in magmm]
    
    st_k, st_m_dict = get_collinear_mag_kindname(p_st,magmoms)
    
    
    aiida_st2 = orm.StructureData(pymatgen =st_k)
    aiid_dict  = orm.Dict(dict=st_m_dict)
    
    return  {'struct_magkind':aiida_st2, 'start_mag_dict': aiid_dict}



def analyze_structures(init_supc, rlxd_results, input_st, magmom = None):
    """
    This calls "cluster_unique_sites" function that analyzes and clusters
    the relaxed muon positions.
    
    Returns: 
    (i) List of relaxed unique candidate sites supercell structures
    (ii) List of to be calculated magnetic inequivalent supercell structures
    """
    pk_lst, mu_lst, enrg_lst = load_workchain_data(rlxd_results)
    
    if magmom:
        assert input_st.num_sites == len(magmom)
        st_smag = input_st.copy()
        for i, m in enumerate(magmom):
            st_smag[i].properties['magmom'] = Magmom(m)
    else:
        st_smag = input_st.copy()
    
    clus_pos, new_pos = cluster_unique_sites(pk_lst, mu_lst, enrg_lst, p_st = input_st, p_smag = st_smag)
    
    #REVISIT
    #TODO-clean: lines below can go in the function 'cluster_unique_sites' with much less lines.

    #get input supercell structure with distortions of new mag inequivalent position 
    nw_stc_calc = []
    if len(new_pos) > 0:
        for i,nwp in enumerate(new_pos):
            for j, d in enumerate(rlxd_results):
                if nwp[0] == d['pk']:
                    nw_st = get_struct_wt_distortions(init_supc, d['rlxd_struct'], nwp[1], st_smag)
                    nw_stc_calc.append(nw_st)
    
    uniq_clus_pos = []
    for i, clus in enumerate(clus_pos):
        for j, d in enumerate(rlxd_results):
            if clus[0] == d['pk']:
                uniq_clus_pos.append(d)
    
    assert len(clus_pos) == len(uniq_clus_pos)    
    
    return {'unique_pos':uniq_clus_pos, 'mag_inequivalent':nw_stc_calc}