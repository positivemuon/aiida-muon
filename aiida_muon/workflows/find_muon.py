# -*- coding: utf-8 -*-
import numpy as np
import copy
from aiida import orm
from aiida.engine import WorkChain, calcfunction, if_
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida_quantumespresso.common.types import RelaxType
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge, ProtocolMixin
from pymatgen.core import Structure, Element
from aiida.common import AttributeDict
from typing import Union

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance,
)

from aiida_impuritysupercellconv.workflows.impuritysupercellconv import input_validator as impuritysupercellconv_input_validator

from aiida.orm import StructureData as LegacyStructureData
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida_quantumespresso.common.hubbard import Hubbard

from aiida_muon.utils.sites_supercells import niche_add_impurities, gensup
from aiida_muon.utils.clustering import analyze_structures
from aiida_muon.utils.magnetism import make_collinear_getmag_kind, compute_dipolar_field
from aiida_muon.utils.hubbard import check_get_hubbard_u_parms, create_hubbard_structure

try:
    from aiida_pythonjob import PythonJob
    HAS_PYTHONJOB = True
except ImportError:
    HAS_PYTHONJOB = False
    PythonJob = None

try:
    StructureData = DataFactory("atomistic.structure")
    HAS_ATOMISTIC = True
    valid_types = (StructureData, LegacyStructureData, HubbardStructureData)
except Exception:
    HAS_ATOMISTIC = False
    valid_types = (LegacyStructureData, HubbardStructureData)

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')

from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain as LegacyPwRelaxWorkChain
from aiida_qe_restart.relax import PoweredPwRelaxWorkChain as PwRelaxWorkChain

IsolatedImpurityWorkChain = WorkflowFactory('impuritysupercellconv')

_original_IsolatedImpurityWorkChain_validator = IsolatedImpurityWorkChain.spec().inputs.validator

def IsolatedImpurityWorkChain_override_validator(inputs,ctx=None):
    """validate inputs for impuritysupercellconv.relax; actually, it is
    just a way to avoid defining it if we do not want it. 
    otherwise the default check is done and it will excepts. 
    """
    if "impuritysupercellconv" in inputs.keys():
        if "parameters" in inputs["impuritysupercellconv"]["pwscf"]["pw"].keys():
            if len(inputs["impuritysupercellconv"]["pwscf"]["pw"]["parameters"].get_dict()):
                _original_IsolatedImpurityWorkChain_validator(inputs["impuritysupercellconv"],ctx)
            else:
                return None
        else:
            return None
    
IsolatedImpurityWorkChain.spec().inputs.validator = IsolatedImpurityWorkChain_override_validator


class FindMuonWorkChain(ProtocolMixin, WorkChain):
    """WorkChain for finding candidate muon implantation sites in a crystal.

    The workflow proceeds through the following stages:

    1. **Supercell size determination** – either uses the ``sc_matrix`` provided
       by the user, or calls ``IsolatedImpurityWorkChain`` to determine the
       minimum converged supercell automatically.
    2. **Initial muon sites** – the NICHE algorithm distributes candidate muon
       positions on a grid over the unit cell, respecting the ``mu_spacing``
       distance constraint.
    3. **Supercell generation** – each starting site is embedded into a supercell.
    4. **Pre-relaxation (optional)** – reduce the number of DFT calculations by
       first relaxing with a cheap method:

       - ``gamma_pre_relax=True``: Γ-point-only DFT relaxation.
       - ``ML_pre_relax=True``: MLIP relaxation via ``PythonJob`` (experimental).

       After each pre-relaxation step, a clustering pass removes duplicates
       before the next (more expensive) step runs.
    5. **Full DFT relaxation** – one ``PwRelaxWorkChain`` per surviving supercell
       (skipped when ``full_dft_relax=False``).
    6. **Clustering** – relaxed muon positions are grouped by spatial proximity
       and energy; symmetry-equivalent sites are merged.
    7. **Hyperfine and dipolar fields** (magnetic materials only) – a final SCF
       calculation places the muon at the origin (``PwBaseWorkChain``); the
       contact hyperfine field is evaluated from the spin density via ``pp.x``,
       and the classical dipolar field is computed with ``muesr``.

    Monitor support
    ---------------
    When ``aiida-monitor`` is installed, the ``aiida_monitor.default_monitor``
    is attached automatically to every ``PwBaseWorkChain`` relaxation step
    (controlled by ``activate_monitors`` in ``get_builder_from_protocol``).
    Additional monitors can be passed via ``monitor_entry_point_list``.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        if HAS_PYTHONJOB:
            spec.expose_inputs(
                PythonJob,
                namespace='pythonjob',
                namespace_options={
                    'required': False,
                    'populate_defaults': False,
                    'help': 'Inputs for MLIPs calculations.',
                },
            )
        spec.input(
            "structure",
            valid_type=valid_types,
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
        
        spec.input(
            "niche_atom",
            valid_type=orm.Str,
            default=lambda: orm.Str("H"),
            required=False,
            help="Chemical symbol of the impurity atom to use for muon site generation.",
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
        spec.input(
            "gamma_pre_relax",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="To run gamma pre-relaxation or not",
        )
        spec.input(
            "ML_pre_relax",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="To run ML pre-relaxation or not",
        )
        spec.input(
            "full_dft_relax",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Whether to run full k-mesh DFT relaxation. Set to False to skip DFT relaxation entirely (e.g. when only MLIP or Gamma pre-relaxation results are needed).",
        )
        spec.input(
            "supercells_list",
            valid_type=orm.List,
            required=False,
            help="List of supercells to be used for the relaxation",
        )
        spec.input(
            "pre_clustering",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Whether to analyze and recompute structures after each pre-relaxation step. For full relax, it is always reanalyzed.",
        )

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace="relax",
            exclude=("structure"),
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
            if_(cls.should_generate_supercells)(
                cls.get_initial_muon_sites,
                cls.get_initial_supercell_structures,
            ),
            if_(cls.should_run_mlip_relaxation)(
                cls.compute_supercell_structures,
                cls.collect_relaxed_structures,
                cls.run_cluster_analysis,
            ),
            if_(cls.should_run_gamma_relaxations)(
                cls.compute_supercell_structures,
                cls.collect_relaxed_structures,
                cls.run_cluster_analysis,
            ),
            if_(cls.should_run_full_relaxations)(
                cls.compute_supercell_structures,
                cls.collect_relaxed_structures,
                cls.run_cluster_analysis,
                if_(cls.new_struct_after_analyze)(   # we do this analysis only for the full mesh case.
                    cls.compute_supercell_structures,
                    cls.collect_relaxed_structures,
                ),
            ),
            cls.collect_all_results,
            if_(cls.structure_is_magnetic)(
                if_(cls.spin_polarized_dft)(
                    cls.run_final_scf_mu_origin,
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
            408,
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
        structure: Union[StructureData, LegacyStructureData, HubbardStructureData] if HAS_ATOMISTIC else Union[LegacyStructureData, HubbardStructureData], # type: ignore
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
        niche_atom: str = "H",
        kpoints_distance: float = 0.301,
        charge_supercell: bool =True,
        hubbard: bool = True,
        hubbard_dict: dict = None,
        spin_pol_dft: bool = True,
        pseudo_family: str ="SSSP/1.3/PBE/efficiency",
        gamma_pre_relax: bool = False,
        ML_pre_relax: bool = False,
        ML_supercell_size: bool = False,
        pythonjob_code: orm.Code = None,
        callback_calculator: callable = None,
        full_dft_relax: bool = True,
        supercells_list: list = [],
        pre_clustering: bool = False,
        noncollinear: bool = False,
        monitor_entry_point_list: list = [],
        activate_monitors: bool = True,
        additional_pythonjob_inputs: dict = {},
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param pp_code: the ``Code`` instance configured for the ``quantumespresso.pp`` plugin.
            Required when ``magmom`` is set and contact hyperfine fields are desired.
        :param protocol: QE calculation protocol (``'fast'``, ``'moderate'``, ``'precise'``).
            Defaults to ``'moderate'`` when not specified.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param enforce_defaults: if ``True``, merge protocol defaults into ``overrides`` before
            building the sub-workflow builders. Default ``True``.
        :param relax_unitcell: pre-relax the unit cell before placing the muon. Default ``False``.
        :param conv_thr: force convergence threshold (eV/Å) for the supercell convergence step.
            Default ``0.0257`` eV/Å (≈ 1 × 10⁻³ a.u.).
        :param magmom: per-site 3-component magnetic moments (µB) in the **unit cell**, e.g.
            ``[[0, 0, 2.2]]`` for one Fe site. Enables spin-polarised DFT and hyperfine
            calculation when combined with ``pp_code``.
        :param options: dictionary of scheduler options (``resources``, ``max_wallclock_seconds``,
            etc.) applied recursively to all nested ``CalcJob`` inputs.
        :param sc_matrix: explicit supercell matrix, e.g. ``[[2, 0, 0], [0, 2, 0], [0, 0, 2]]``.
            When provided, the ``IsolatedImpurityWorkChain`` convergence step is skipped.
        :param mu_spacing: minimum distance (Å) between two starting muon grid points.
            Default ``1.0``.
        :param niche_atom: chemical symbol used as placeholder for the muon in the NICHE
            algorithm (must be a valid element symbol). Default ``'H'``.
        :param kpoints_distance: minimum desired k-point spacing (Å⁻¹). Default ``0.301``.
        :param charge_supercell: run a charge +1 supercell to model the positive muon.
            Default ``True``.
        :param hubbard: detect and apply DFT+U corrections automatically based on element
            heuristics. Default ``True``.
        :param hubbard_dict: override the automatic Hubbard U values; keys are element/kind
            labels, values are U in eV.
        :param spin_pol_dft: run spin-polarised DFT. Default ``True``; automatically set
            to ``True`` when ``magmom`` is provided.
        :param pseudo_family: label of the pseudopotential family. Default
            ``'SSSP/1.3/PBE/efficiency'``.
        :param gamma_pre_relax: run a cheap Γ-point-only DFT pre-relaxation to reduce the
            number of candidate sites before the full k-mesh step. Default ``False``.
        :param ML_pre_relax: run an MLIP pre-relaxation via ``PythonJob`` before DFT
            (experimental). Requires ``pythonjob_code`` and ``callback_calculator``.
            Default ``False``.
        :param ML_supercell_size: use MLIP forces inside ``IsolatedImpurityWorkChain`` to
            determine the supercell size (experimental). Default ``False``.
        :param pythonjob_code: ``Code`` node for the ``PythonJob`` process; required when
            ``ML_pre_relax=True`` or ``ML_supercell_size=True``.
        :param callback_calculator: ASE-compatible MLIP callable used for ML force/relaxation
            calculations; required when ``ML_pre_relax=True`` or ``ML_supercell_size=True``.
        :param full_dft_relax: run the full k-mesh DFT relaxation step. Set to ``False`` to
            skip DFT entirely (e.g. when only MLIP or Gamma pre-relaxation results are needed).
            Default ``True``.
        :param supercells_list: list of UUIDs of already-generated ``StructureData`` nodes
            (supercells with the muon included) to use instead of automatic site generation.
        :param pre_clustering: cluster and prune duplicate sites after each pre-relaxation
            step before proceeding to the next (more expensive) step. Default ``False``.
        :param noncollinear: set to ``True`` for non-collinear magnetic calculations;
            disables automatic Gamma-only optimisation. Default ``False``.
        :param monitor_entry_point_list: list of ``aiida-monitor`` entry-point strings to
            attach to every ``PwBaseWorkChain`` relaxation. The
            ``'aiida_monitor.default_monitor'`` entry point is appended automatically when
            ``aiida-monitor`` is installed and ``activate_monitors=True``.
        :param activate_monitors: enable monitor attachment. When ``True`` (default) and
            ``aiida-monitor`` is installed, ``aiida_monitor.default_monitor`` is added
            automatically. Set to ``False`` to disable all monitors.
        :param additional_pythonjob_inputs: extra keyword arguments forwarded to the
            ``PythonJob`` input-preparation helper (e.g. metadata, custom serialisers).
        :return: a process builder instance with all inputs defined ready for launch.
        """
        
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        # the get defaul also changes the structure, if needed (magmoms and hubbardstructuredata as input)
        _overrides, start_mg_dict, structure, magmom = get_default_dict(structure, pseudo_family, kpoints_distance, charge_supercell, magmom, spin_pol_dft, noncollinear)
        
        if enforce_defaults:
            overrides = recursive_merge(overrides,_overrides)
        
        #check on the structure: if hubbard is needed, do it with append onsite... if the structure is already stored, clone it. 
        # NOTE: `check_get_hubbard_u_parms` will only return the dictionary if we have more than 2 species in the structure (muon excluded).
        hubbard_params = check_get_hubbard_u_parms(structure.get_pymatgen(), u_dict=hubbard_dict)

        if isinstance(structure, HubbardStructureData): # we do not do anything, we let the user define the Hubbard U 
            if not hubbard:
                # we can lose the Hubbard info in this way:
                structure = LegacyStructureData(ase = structure.get_ase())
        else: # orm.StructureData
            # we define automatic DFT+U settings only if magmoms are there, in this case. 
            # NOTE: I don't think is always the case. 
            if len(hubbard_params) > 0 and magmom and hubbard:
                structure = HubbardStructureData.from_structure(structure)
                for kind, U in hubbard_params.items():
                    structure.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)
                structure.hubbard = Hubbard.from_list(structure.hubbard.to_list(), projectors="atomic")
        
        #builder_impuritysupercellconv.pop('structure', None)
        
        overrides["base"]["pw"].pop('pseudos', None)

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
        builder_relax.pop('base_init_relax', None)
        
        # backward compatibility: pop base_final_scf if present, since it is not needed for the relax workflow.
        builder_relax.pop('base_final_scf', None)
        
        builder_pwscf['pw'].pop('structure', None)
        builder_pwscf.pop('kpoints_distance', None)       
        
        #### Builder
        builder = cls.get_builder()
        
        builder.structure = structure
        builder.pseudo_family = orm.Str(pseudo_family)
        
        #### IsolatedImpurityWorkChain
        if not sc_matrix:
            if ML_supercell_size:
                builder_impuritysupercellconv = IsolatedImpurityWorkChain.get_builder_from_protocol(
                    structure=structure,
                    pythonjob_code=pythonjob_code,
                    callback_calculator=callback_calculator,
                    charged_supercell=charge_supercell,
                    ML_forces=True,  # Enable MLIP force calculations
                    **additional_pythonjob_inputs,
                )
            else:
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
                builder_impuritysupercellconv.pseudo_family = orm.Str(pseudo_family)
                #setting subworkflows inputs
                #probably, it is better to populate defaults and then pop if not needed, as done later.
            for k,v in builder_impuritysupercellconv.items():
                if k in ["pwscf","relax"] and ML_supercell_size: continue
                if k == "pythonjob" and not ML_supercell_size: continue
                if k == "relax":
                    for k1,v1 in builder_impuritysupercellconv.relax.items():
                        if k1 == "base_final_scf": continue
                        setattr(builder.impuritysupercellconv.relax,k1,v1)
                else:
                    try:
                        setattr(builder.impuritysupercellconv,k,v)
                    except Exception as e:
                        raise ValueError(f"Error {e} while setting {k} with {v}.")
            builder.impuritysupercellconv.pop('structure', None)
        else:
            builder.sc_matrix=orm.List(sc_matrix)
        
        builder.pwscf = builder_pwscf
        builder.relax = builder_relax  
          
        
        # If magmoms are defined, we need to set the spin_pol_dft to True
        if start_mg_dict: 
            if isinstance(magmom, list):
                magmom = orm.List(magmom)
            builder.magmom = magmom

        # Validate that niche_atom is a valid element
        try:
            Element(niche_atom)
        except Exception as e:
            raise ValueError(f"niche_atom should be a valid chemical symbol, got {niche_atom}")

        builder.mu_spacing=orm.Float(mu_spacing)
        builder.niche_atom=orm.Str(niche_atom)
        builder.charge_supercell=orm.Bool(charge_supercell)
        builder.kpoints_distance = orm.Float(kpoints_distance)
        builder.hubbard = orm.Bool(hubbard)
        builder.spin_pol_dft = orm.Bool(spin_pol_dft)
        
        # PpCalculation inputs: Only this, the rest is really default and generated on the fly - hardcoded
        if pp_code: builder.pp_code = pp_code
        
        # Checking for additional metadata
        for i in ["pp_metadata","qe_settings"]:
            # I don't like this.
            if i in overrides.keys():
                builder[i] = overrides[i] 
                
        builder.gamma_pre_relax = orm.Bool(gamma_pre_relax)
        builder.ML_pre_relax = orm.Bool(ML_pre_relax)
        builder.full_dft_relax = orm.Bool(full_dft_relax)
        builder.pre_clustering = orm.Bool(pre_clustering)
        
        if builder.ML_pre_relax:
            from aiida_muon.pythonjobs.relax import prepare_ase_pythonjob_relaxation_inputs
            pythonjob_inputs = prepare_ase_pythonjob_relaxation_inputs(
                structure=structure,
                pythonjob_code=pythonjob_code,
                callback_calculator=callback_calculator,
                charged_supercell=charge_supercell,
                **additional_pythonjob_inputs,

            )
            builder.pythonjob = pythonjob_inputs
        
        if len(supercells_list)>0:
            builder.supercells_list = orm.List(list=supercells_list)
        
        try:
            from aiida_monitor.monitor import monitor
            from importlib.metadata import entry_points
            registered = {ep.name for ep in entry_points().get('aiida.calculations.monitors', [])}
            if 'aiida_monitor.default_monitor' in registered:
                if 'aiida_monitor.default_monitor' not in monitor_entry_point_list and activate_monitors:
                    monitor_entry_point_list.append('aiida_monitor.default_monitor')
        except Exception:
            pass
        
        # Filter to only entry points that are actually registered
        try:
            from importlib.metadata import entry_points
            registered = {ep.name for ep in entry_points().get('aiida.calculations.monitors', [])}
            monitor_entry_point_list = [ep for ep in monitor_entry_point_list if ep in registered]
        except Exception:
            pass

        if monitor_entry_point_list and activate_monitors:
            builder.relax.base_relax.pw.monitors = {f'monitor_{i}': orm.Dict({'entry_point': monitor_entry_point_list[i]}) for i in range(len(monitor_entry_point_list))}

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
            self.report(f"Supercell size provided in the inputs: {self.ctx.sc_matrix}. Skipping supercell convergence step.")
            return False
        self.report("No supercell size provided in the inputs. We will submit the IsolatedImpurityWorkChain to determine the supercell size.")
        return True
       
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

        # Specific name and submittions
        inputs.metadata.call_link_label = f'IsolatedImpurityWorkChain'
        future = self.submit(IsolatedImpurityWorkChain, **inputs)
        self.report(
            f"Launching IsolatedImpurityWorkChain (PK={future.pk}) for supercell matrix determination"
        )
        
        # We tell the Workflow to wait until we finish this run. Then, it can proceed in the outline.
        self.to_context(**{"IsolatedImpurityWorkChain": future})

    def check_supercell_convergence(self):
        """Check that the IsolatedImpurityWorkChain is finished ok."""
        if not self.ctx["IsolatedImpurityWorkChain"].is_finished_ok:
            self.report("The IsolatedImpurityWorkChain (supercell size estimation) failed. Exiting the workflow.")
            return self.exit_codes.ERROR_MUSCONV_CALC_FAILED
        
        sc_mat_array = self.ctx["IsolatedImpurityWorkChain"].outputs.Converged_SCmatrix.get_array('sc_mat')
        self.ctx.sc_matrix = sc_mat_array.tolist()
        self.report(f"Supercell size computed to be: {sc_mat_array}")
        return
                
    def setup(self):
        """Setup for the find-muon workflow.
        
        In particular, we set the structure and the magnetization information, if any.
        We no more setup the hubbard dictionary here: the Hubbard parameters should be defined in the `get_builder_from_protocols`.
        """
        if not hasattr(self.ctx, "structure"): 
            # TODO: set, if any the final relaxed unit cell as obtained from the IsolatedImpurityWorkChain pre-relaxation.
            self.ctx.structure = self.inputs.structure
            
        if hasattr(self.inputs, "sc_matrix"):
            self.ctx.sc_matrix = self.inputs.sc_matrix.get_list()
        elif not hasattr(self.ctx, "sc_matrix"):
            raise ValueError("No supercell matrix defined. Exiting the workflow.")
        

        if "magmom" in self.inputs:
            self.ctx.magmom = self.inputs.magmom.get_list()
        
        # We can also pass the hubbard info separatly, as we check now... but not ideal. we should use atomistic StructureData or HubbardStructureData.
        if hasattr(self.inputs,"hubbard_dict"):
            self.ctx.hubbardu_dict = self.inputs.hubbard_dict.get_dict()
        else:
            self.ctx.hubbardu_dict = {}
        
        # setting some variable for the workflow 
        self.ctx.n = 0 # init relaxation calc count
        self.ctx.n_uuid_dict = {}
        self.ctx.offset = 0 # offset for the supercell index if we find magnetic inequivalent sites.
        self.ctx.set_gamma_only = False

        self.ctx.has_magnetic_inequivalent = False # to understand if we have magnetic inequivalent sites, in this case we need to set an offset for the index and we cannot set gamma only for the relaxations.

        # check if the calculation is non-collinear; in that case, we cannot set Gamma only even if it is 1x1x1.
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        base = inputs['base_relax'] if 'base_relax' in inputs else inputs['base']
        if base.pw.parameters.get_dict().get('SYSTEM',{}).get('noncolin',False):
            self.report("Non-collinear calculation detected, setting Gamma only to False.")
            self.ctx.non_collinear = True
        else:
            self.ctx.non_collinear = False
        
        # We can also provide a list of supercell to run, if we do not want to use the automated generation.
        self.ctx.supc_list = [orm.load_node(uuid) for uuid in self.inputs.supercells_list.get_list()] if hasattr(self.inputs,"supercells_list") else []

        return
       
    def should_generate_supercells(self):
        """Check if we need to generate supercells.
        If we provide supercell_list, they need to already have the muon inside.
        TODO: add check for that.
        """
        
        return len(self.ctx.supc_list) == 0
         
    def get_initial_muon_sites(self):
        """Get list of starting muon sites.
        
        """
        
        niche_atom = self.inputs.niche_atom.value
        
        # Validate that niche_atom is a valid element
        try:
            Element(niche_atom)
        except Exception as e:
            self.report(f"Invalid niche_atom '{niche_atom}': {e}")
            return self.exit_codes.ERROR_NO_SUPERCELLS
        
        self.ctx.mu_lst = niche_add_impurities(
            structure = self.ctx.structure.get_pymatgen_structure(), 
            niche_atom = niche_atom, 
            niche_spacing = self.inputs.mu_spacing.value, 
            niche_distance = 1, # distance from hosting atoms. Hardcoded.
        )
        
        if len(self.ctx.mu_lst) == 0:
            self.report(f"No muon sites found using niche_atom '{niche_atom}'. Exiting the workflow.")
            return self.exit_codes.ERROR_NO_SUPERCELLS
        
        self.report(f"Number of muon sites found using niche_atom '{niche_atom}': {len(self.ctx.mu_lst)}")
        return
    
    def get_initial_supercell_structures(self):
        """Get initial supercell+muon list.
        
        """
        self.report("Getting supercell list")
        input_struct = self.ctx.structure.get_pymatgen_structure()
        muon_list = self.ctx.mu_lst

        self.ctx.supc_list = gensup(input_struct, muon_list, self.ctx.sc_matrix)
        if len(self.ctx.supc_list) == 0:
            self.report("No Supercells, please decrease the mu_spacing parameter. Exiting the workflow...")
            return self.exit_codes.ERROR_NO_SUPERCELLS
        
        self.ctx.supc_list = self.generate_supercells_list()
        return

    def generate_supercells_list(self):
        """Generate the supercell list from pymatgen objects to 
        StructureData objects.
        """
        self.report("Generating supercell list")
        supercell_list = self.ctx.supc_list
        new_supercell_list = []
        
        for i_index in range(len(supercell_list)):
            
            # Here we put a logic to handle the creation of a supercell if Hubbard params are there.
            # using the LegacyStructureData constructor first
            if isinstance(supercell_list[i_index], Structure): # pymatgen object
                structure = LegacyStructureData(pymatgen=supercell_list[i_index])
            else:
                structure = supercell_list[i_index]
            
            # we then assign the Hubbard parameters only if needed
            if self.ctx.structure_type == HubbardStructureData and self.inputs.hubbard and not isinstance(structure, HubbardStructureData):
                self.report(f"Generating supercell #{i_index} with Hubbard parameters.")
                structure = create_hubbard_structure(structure, self.inputs.structure)
            elif len(self.ctx.hubbardu_dict) > 0 and "magmom" in self.inputs and self.inputs.hubbard and not isinstance(structure, HubbardStructureData):
                self.report(f"Enforcing DFT+U for supercell #{i_index}, as magmoms are defined and U parameters are available.")
                structure = create_hubbard_structure(structure,self.ctx.hubbardu_dict)
            
            new_supercell_list.append(structure)
        
        return new_supercell_list

    def should_run_mlip_relaxation(self):
        """Check if we should run MLIP relaxations.
        """
        if self.inputs.ML_pre_relax:
            # Set context for ML relaxations
            self.ctx.run_type = "ASE"
            return True
        
        return False
    
    def should_run_gamma_relaxations(self):
        """Check if we should run gamma relaxations.
        """

        if self.inputs.gamma_pre_relax.value == False:
            self.report("Skipping gamma pre-relaxation as specified in the inputs.")
            return False

        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        base = inputs['base_relax'] if 'base_relax' in inputs else inputs['base']
            
        if not "kpoints_distance" in base:
            self.report(f"Setting kpoints distance to be: {self.inputs.kpoints_distance.value}")
            base.kpoints_distance = self.inputs.kpoints_distance
        
        mesh = create_kpoints_from_distance(
                    self.ctx.supc_list[0],
                    orm.Float(base.kpoints_distance),
                    orm.Bool(False),
                    metadata={"store_provenance": False},
                ).get_kpoints_mesh()
    
        if np.all(np.array(mesh[0]) == 1) and np.all(np.array(mesh[0]) == 1):
            self.report("We don't need a Gamma point pre-relaxation, Gamma is anyway the only sampled point.")
            self.ctx.set_gamma_only = not self.ctx.non_collinear # so we set gamma point only... in the dft runs
            return False

        if self.inputs.gamma_pre_relax:
            # Set context for gamma relaxations
            self.ctx.run_type = "gamma"
            self.ctx.enforce_gamma = True
            return True
        
        return False
    
    def should_run_full_relaxations(self):
        """Check if we should run full relaxations.
        """
        if self.inputs.full_dft_relax.value == False:
            self.report("Skipping DFT relaxations as specified in the inputs.")
            return False
        
        # Set context for full relaxations
        self.ctx.run_type = "full"
        self.ctx.enforce_gamma = False
        return True

    def should_pre_clustering(self):
        """Check if we should analyze and recompute after relaxations."""
        should_run_pre_clustering = self.inputs.pre_clustering.value and self.inputs.gamma_pre_relax.value
        
        if not should_run_pre_clustering and self.inputs.pre_clustering.value:
            self.report("Pre-clustering is activated but gamma pre-relaxation is not activated. Pre-clustering will be skipped as it is only relevant after the gamma pre-relaxation step.")

        if not hasattr(self.ctx, 'pre_clustering_done'): self.ctx.pre_clustering_done = False

        if should_run_pre_clustering and self.ctx.pre_clustering_done:
            self.report("Pre-clustering is activated but it has already been done after the gamma pre-relaxation step. Skipping pre-clustering.")
            return False

        return should_run_pre_clustering

    def run_cluster_analysis(self):
        # Always analyze the relaxed structures if full k-mesh relaxation, and optionally analyze and recompute if specified in the inputs for Gamma only and MLIPs.
        if self.should_pre_clustering() and not self.ctx.pre_clustering_done:
            self.analyze_relaxed_structures(mode='pre-clustering', d_tol=0.25)
            self.ctx.pre_clustering_done = True
        elif self.ctx.run_type == "full":
            self.analyze_relaxed_structures(mode='full')
        else:
            self.report("Skipping pre-clustering of structures after pre-relaxation as specified in the inputs.")
    
    def submit_dft_relaxations(self, enforce_gamma=False):
        """Submit the DFT relaxations for each supercell.
        if enforce_gamma is True, we use the gamma point only, even if 
        we have a different k-point mesh. This can be useful for pre-relaxation
        calculations.
        """
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        base = inputs['base_relax'] if 'base_relax' in inputs else inputs['base']
        
        gamma_only_suffix = ""

        if self.ctx.run_type == "gamma":
            # we use a looser convergence threshold for the gamma point pre-relaxation
            relax_parameters = base.pw.parameters.get_dict()
            relax_parameters["CONTROL"]["forc_conv_thr"] = relax_parameters["CONTROL"]["forc_conv_thr"] * 50 # TODO: check if this is ok
            base.pw.parameters = orm.Dict(dict=relax_parameters)
        
        # Make sure we have a kpoints distance
        if enforce_gamma:
            # if we enforce_gamma but anyway the mesh is 1x1x1, we skip this step and we run the relaxation as is.
            #self.report("Enforcing gamma point only for the supercell relaxations.")
            mesh = orm.KpointsData()
            mesh.set_kpoints_mesh([1, 1, 1])
            base.kpoints = mesh
            gamma_only_suffix = "_gamma"
            self.report("Using gamma point only for the supercell relaxations.")
            if not self.ctx.non_collinear and not isinstance(self.ctx.supc_list[0], HubbardStructureData):
                settings = base.pw.settings.get_dict() if hasattr(base.pw, "settings") else {}
                settings["GAMMA_ONLY"] = True
                base.pw.settings = orm.Dict(dict=settings)
            else:
                self.report("Non-collinear calculation detected or DFT+U calculation, not setting GAMMA_ONLY but a [1,1,1] mesh.")
            if hasattr(base.pw, "parallelization"):
                if "npool" in base.pw.parallelization.get_dict():
                    base.pw.parallelization = orm.Dict(dict={k:v  for k,v in base.pw.parallelization.get_dict().items() if k != "npool"})
            
        elif self.ctx.set_gamma_only:
            # in this case, we have Gamma as the only sampled point by default, so we set GAMMA_ONLY to True if it is not non-collinear.
            self.report("Using gamma point only for the supercell relaxations.")
            if not self.ctx.non_collinear and not isinstance(self.ctx.supc_list[0], HubbardStructureData):
                settings = base.pw.settings.get_dict() if hasattr(base.pw, "settings") else {}
                settings["GAMMA_ONLY"] = True
                base.pw.settings = orm.Dict(dict=settings)
            else:
                self.report("Non-collinear calculation detected or DFT+U calculation, not setting GAMMA_ONLY but a [1,1,1] mesh.")
            if hasattr(base.pw, "parallelization"):
                if "npool" in base.pw.parallelization.get_dict():
                    base.pw.parallelization = orm.Dict(dict={k:v  for k,v in base.pw.parallelization.get_dict().items() if k != "npool"})
            
        for i_index in range(len(self.ctx.supc_list)):

            inputs.structure = self.ctx.supc_list[i_index]
            
            # we define the pseudos again (now we have the structure+H)
            base.pw.pseudos = get_pseudos(
                inputs.structure, self.inputs.pseudo_family.value
            )
            
            # Set the `CALL` link label and submission
            inputs.metadata.call_link_label = f'supercell_{i_index:02d}' + gamma_only_suffix
            future = self.submit(PwRelaxWorkChain, **inputs)
            # key = f'workchains.sub{i_index}' #nested sub
            key = f"workchain_{i_index}"
            self.report(
                f"Launching PwRelaxWorkChain (PK={future.pk}) for supercell structure {self.ctx.supc_list[i_index].get_formula()} with index {i_index}" \
                    + gamma_only_suffix.replace("_gamma",", Gamma only sampling")
            )
            self.to_context(**{key: future})
        
        return
        
    def compute_supercell_structures(self):
        """Run relax workflows for each muon supercell.
        
        We first check that kpoints_distance is set, then we loop
        over the supercell list and run the relax workflow, ensuring that
        the structure is HubbardStructureData if needed.
        """

        cycle_type = self.ctx.run_type
        self.report(f"Starting {cycle_type} relaxation cycle")

        self.report("Computing muon supercells")
        self.ctx.supc_list = self.generate_supercells_list()
        
        if self.ctx.run_type == "gamma":
            self.report("Running gamma point DFT relaxations")
            self.submit_dft_relaxations(enforce_gamma=True)
        elif self.ctx.run_type == "full":
            self.report("Running DFT relaxations")
            self.submit_dft_relaxations(enforce_gamma=False)
        elif self.ctx.run_type == "ASE":
            self.report("Running ML relaxations")
            self.submit_ase_relaxations()    
        
        return

    def submit_ase_relaxations(self):
        if not HAS_PYTHONJOB:
            raise ImportError(
                'aiida-pythonjob is required for ML relaxations. '
                'Install it with: pip install aiida-pythonjob'
            )

        inputs = AttributeDict(self.exposed_inputs(PythonJob, namespace='pythonjob'))
        
        suffix = "_ase"

        for i_index in range(len(self.ctx.supc_list)):

            inputs.function_inputs.atoms = self.ctx.supc_list[i_index]
            
            # Set the `CALL` link label and submission
            inputs.metadata.call_link_label = f'supercell_{i_index:02d}' + suffix
            future = self.submit(PythonJob, **inputs)
            # key = f'workchains.sub{i_index}' #nested sub
            key = f"workchain_{i_index}"
            self.report(
                f"Launching PythonJob (PK={future.pk}) for supercell structure {self.ctx.supc_list[i_index].get_formula()} with index {i_index}" \
                    + suffix
            )
            self.to_context(**{key: future})
        
        return
    
    def collect_relaxed_structures(self):
        """Retrieve final positions and energy from the relaxed structures.        
        """

        self.report("Gathering computed positions and energy")
        supercell_list = self.ctx.supc_list
        new_supercell_list = [] # we create this list so we can run two relaxations one after the other, like Gamma and full mesh.

        computed_results = []

        # for nested
        # for key, workchain in self.ctx.workchains.items():
        #    if not workchain.is_finished_ok

        n_notf = 0
        self.ctx.n, self.ctx.n_uuid_dict = 0, {}
        for i_index in range(len(supercell_list)):
            key = f"workchain_{i_index}"
            workchain = self.ctx[key]

            # checking failed calculations and exiting if more than 40% failed
            if not workchain.is_finished_ok:
                self.report(
                    f"Relaxation calculation {i_index} failed with exit status {workchain.exit_status}"
                )
                n_notf += 1
                # if failed calculation is more than 40%, then exit
                if float(n_notf) / len(supercell_list) > 0.4:
                    return self.exit_codes.ERROR_RELAX_CALC_FAILED
            else:
                self.ctx.n = i_index+self.ctx.offset
                uuid = workchain.uuid
                if workchain.process_class in [LegacyPwRelaxWorkChain, PwRelaxWorkChain]:
                    energy = workchain.outputs.output_parameters.get_dict()["energy"]
                    rlx_structure = (
                        workchain.outputs.output_structure.get_pymatgen_structure()
                    )
                    new_supercell_list.append(workchain.outputs.output_structure)
                elif 'pythonjob' in workchain.process_type:
                    energy = workchain.outputs.energy.value
                    rlx_structure =  (
                        workchain.outputs.structure.get_pymatgen_structure()
                    )
                    new_supercell_list.append(workchain.outputs.structure)
                else:
                    raise ValueError(f"Unknown workchain type: {workchain.process_type} for uuid={uuid}.")

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

        self.ctx.supc_list = new_supercell_list
        if len(supercell_list)!= len(new_supercell_list):
            self.report(
                f"Relaxation of {len(supercell_list) - len(new_supercell_list)} supercells failed.\n So we skip them and continue with the rest."
            )

        return
    
    def analyze_relaxed_structures(self, mode='pre-clustering', d_tol = 0.5):
        """Analyze relaxed structures.
        
        Get unique candidate sites and check if there are 
        new magnetic inequivalent (via symmetry operations) 
        sites to calculate.
        
        Basically, this represents the clustering step.
        Different algorithm could in principle be implemented.

        NB: here we should put the choice of the thresholds for clustering, at least for the different methods (k-mesh, Gamma, MLIPs).
        """
        self.report("Analyzing the relaxed structures")
        inpt_st = self.ctx.structure.get_pymatgen_structure()

        if "magmom" in self.ctx and mode!='pre-clustering':
            r_anly = analyze_structures(
                self.ctx.supc_list[0],
                self.ctx.relaxed_outputs,
                inpt_st,
                self.ctx.magmom,
            )
        else:
            r_anly = analyze_structures(
                self.ctx.supc_list[0], self.ctx.relaxed_outputs, inpt_st, d_tol = d_tol
            )

        self.ctx.unique_cluster = r_anly["unique_pos"]
        self.ctx.cluster_mapping = r_anly["mapping"]
        self.report(f"Mapping of relaxed structures to unique clusters: {self.ctx.cluster_mapping}")
        self.report(f"Unique clusters found: {len(self.ctx.unique_cluster)} out of {len(self.ctx.relaxed_outputs)} relaxed structures.")
        
        # print('uniq_positions',self.ctx.unique_cluster)

        # revisit, this so the initial inputs and collected results are not ovewritten with repeated calls in outline
        self.ctx.supc_list_all = self.ctx.supc_list
        self.ctx.relaxed_outputs_all = self.ctx.relaxed_outputs

        if mode == 'pre-clustering':
            # cluster_mapping holds 1-based cluster IDs, not list indices.
            # Pick the index of the first structure for each unique cluster ID.
            seen = {}
            for idx, cid in enumerate(self.ctx.cluster_mapping):
                if cid not in seen:
                    seen[cid] = idx
            self.ctx.supc_list = [self.ctx.supc_list_all[i] for i in seen.values()]
            self.report(f"Number of unique clusters found: {len(self.ctx.unique_cluster)}, out of {len(self.ctx.relaxed_outputs)} relaxed structures.")
        else:
            self.ctx.supc_list = r_anly["mag_inequivalent"]
            if len(self.ctx.supc_list) > 0:
                self.ctx.has_magnetic_inequivalent = True

    def new_struct_after_analyze(self):
        """Check if there is new magnetic inequivalent sites. This is done only in the full mesh relaxation."""
        self.report(f"Checking new structures to calculate... {len(self.ctx.supc_list) > 0}")

        if len(self.ctx.supc_list) > 0:
            self.ctx.run_type = "full"
            self.ctx.offset = len(self.ctx.relaxed_outputs_all) if self.ctx.has_magnetic_inequivalent else 0  # offset for the supercell index if we find magnetic inequivalent sites.
            return True
        return False

    def collect_all_results(self):
        """Collecting results of new structures and then append"""
        self.report("Appending results of new structures ")
        if not hasattr(self.ctx, "relaxed_outputs_all"):
            self.ctx.relaxed_outputs_all = []
            self.ctx.relaxed_outputs_all.extend(self.ctx.relaxed_outputs)
        if not hasattr(self.ctx, "unique_cluster"):
            self.ctx.unique_cluster = []
            self.ctx.unique_cluster.extend(self.ctx.relaxed_outputs)

    def structure_is_magnetic(self):
        """Checking if structure is magnetic"""

        # return self.inputs.magmom is not None
        # return 'magmom' in self.inputs
        magnetic = False
        if "magmom" in self.inputs:
            magnetic = self.inputs.magmom is not None
            
        self.report(f"Checking if structure is magnetic... {magnetic}")
        
        return magnetic
            
    def spin_polarized_dft(self):
        """Checking if we need spin polarization in DFT"""
        self.report(f"Checking if we had spin polarization in the simulations: {self.inputs.spin_pol_dft.value}")
        return self.inputs.spin_pol_dft.value

    def run_final_scf_mu_origin(self):
        """Move muon to origin and perform scf"""
        unique_cluster_list = self.ctx.unique_cluster
        self.report(f"Running final SCF calculations with muon at the origin for the {len(unique_cluster_list)} unique clusters.")
        
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='pwscf'))
        inputs_pw = inputs["pw"]["parameters"].get_dict()
       
        # we need for the PpCalculation.
        # inputs["clean_workdir"] = orm.Bool(False) 

        #inputs.kpoints_distance = orm.Float(inputs.kpoints_distance.value - 0.1) #denser reciprocal space grid 

        if self.ctx.set_gamma_only:
            settings = inputs.pw.settings.get_dict() if hasattr(inputs.pw, "settings") else {}
            settings["GAMMA_ONLY"] = True
            inputs.pw.settings = orm.Dict(dict=settings)
            if hasattr(inputs.pw, "parallelization"):
                if "npool" in inputs.pw.parallelization.get_dict():
                    inputs.pw.parallelization = orm.Dict(dict={k:v  for k,v in inputs.pw.parallelization.get_dict().items() if k != "npool"})
        
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
            
            inputs.pw.structure = LegacyStructureData(pymatgen=rlx_st)
            
            # we then assign the Hubbard parameters only if needed
            if self.ctx.structure_type == HubbardStructureData and self.inputs.hubbard:
                self.report(f"Generating supercell #{j_index} with Hubbard parameters.")
                inputs.pw.structure = create_hubbard_structure(inputs.pw.structure, self.inputs.structure)
            elif len(self.ctx.hubbardu_dict) > 0 and "magmom" in self.inputs and self.inputs.hubbard:
                self.report(f"Enforcing DFT+U for supercell #{j_index}, as magmoms are defined and U parameters are available.")
                inputs.pw.structure = create_hubbard_structure(inputs.pw.structure, self.ctx.hubbardu_dict)
                
            inputs.pw.pseudos = get_pseudos(
                inputs.pw.structure, self.inputs.pseudo_family.value
            )
            
            inputs.pop("pseudo_family", None)
            
            # check if we need to set gamma only
            mesh = create_kpoints_from_distance(
                    inputs.pw.structure,
                    orm.Float(inputs.kpoints_distance),
                    orm.Bool(False),
                    metadata={"store_provenance": False},
                ).get_kpoints_mesh()
    
            if np.all(np.array(mesh[0]) == 1) and np.all(np.array(mesh[0]) == 1):
                settings = inputs.pw.settings.get_dict() if hasattr(inputs.pw, "settings") else {}
                settings["GAMMA_ONLY"] = True
                inputs.pw.settings = orm.Dict(dict=settings)
            
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
        if pp_builder.metadata.get("options",{}).get("resources",{}).get("num_machines",1) > 1:
            if pp_builder.metadata['options']['resources']['num_machines'] > 1:
                pp_builder.metadata['options']['resources']['num_machines'] = 1
        else:
            pp_builder.metadata['options']['resources'] = {
                "num_machines": 1,
                "num_mpiprocs_per_machine": pp_builder.metadata.get("options",{}).get("resources",{}).get("num_mpiprocs_per_machine", 1),
            }
            pp_builder.metadata['options']['max_wallclock_seconds'] = pp_builder.metadata.get("options",{}).get("max_wallclock_seconds", 3600)
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
            rlx_struct = LegacyStructureData(pymatgen=rlx_st)
            
            # we then assign the Hubbard parameters only if needed
            if self.ctx.structure_type == HubbardStructureData and self.inputs.hubbard:
                self.report(f"Generating supercell #{j_index} with Hubbard parameters.")
                rlx_struct = create_hubbard_structure(rlx_struct, self.inputs.structure)
            elif len(self.ctx.hubbardu_dict) > 0 and "magmom" in self.inputs and self.inputs.hubbard:
                self.report(f"Enforcing DFT+U for supercell #{j_index}, as magmoms are defined and U parameters are available.")
                rlx_struct = create_hubbard_structure(rlx_struct, self.ctx.hubbardu_dict)

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
# helper (calc-) functions

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
def get_default_dict(structure, pseudo_family, kpoints_distance, charge_supercell,magmom, spin_pol_dft, noncollinear=False,):
    """
    Here, the noncollinear is used to not set the nspin parameter in the overrides.
    FOR NOW: we set the non colin params in the overrides provided by the user. It is a bit involved, but it is temporary solution!
    """


    formula = structure.get_formula()
    
    _overrides = {
           "base": {
                "pseudo_family": pseudo_family,
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
                    + formula
                },
                },
            },
            "base_final_scf": {"pseudo_family": pseudo_family,},
            "clean_workdir": orm.Bool(True),
        }

    if charge_supercell:
        _overrides["base"]["pw"]["parameters"]["SYSTEM"]["tot_charge"] = 1.0
        
    if magmom and not noncollinear:
        rst_mg = make_collinear_getmag_kind(
            structure, magmom,
        )
        new_structure = rst_mg["struct_magkind"]        
        start_mg_dict = rst_mg["start_mag_dict"]
        
        # we need to convert the structure into HubbardStructureData, if needed:
        if isinstance(structure, HubbardStructureData):
            from aiida_muon.utils.manage_new_structure import reassign_kinds
            # first I need to remap the kinds of the old structure to the new ones.
            old_structure = reassign_kinds(
                    structure,
                    new_structure.get_kind_names(),
                )
            
            # then I can create the new hubbard structure with the new kinds.
            structure = create_hubbard_structure(new_structure, old_structure)
        else:
            structure = new_structure
        
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

    # Mirror "base" to "base_relax" so both the legacy PwRelaxWorkChain (key: "base")
    # and the new PoweredPwRelaxWorkChain (key: "base_relax") can pick up the right key.
    _overrides["base_relax"] = copy.deepcopy(_overrides["base"])
    _overrides["impuritysupercellconv"]["pre_relax"]["base_relax"] = copy.deepcopy(
        _overrides["impuritysupercellconv"]["pre_relax"]["base"]
    )
        
    return _overrides, start_mg_dict, structure, magmom


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

    #QE inputs validation:
    keys = ["tot_charge","nspin","occupations","smearing"]
    
    wrong_inputs_relax = []
    wrong_inputs_pwscf = []
    
    impuritysupercellconv_inconsistency = ''
    if "impuritysupercellconv" in parameters:
        impuritysupercellconv_inconsistency = impuritysupercellconv_input_validator(parameters["impuritysupercellconv"],None,caller="FindMuonWorkchain")
    
    if impuritysupercellconv_inconsistency: inconsistency_sentence += impuritysupercellconv_inconsistency

    # Support both the new "base_relax" key and the legacy "base" key.
    if "base_relax" in parameters["relax"]:
        base_key = "base_relax"
    elif "base" in parameters["relax"]:
        base_key = "base"
    else:
        raise ValueError("Neither 'relax.base_relax' nor 'relax.base' found in inputs.")

    if parameters["relax"][base_key]["pw"]["parameters"].get_dict()["CONTROL"]["calculation"] != 'relax':
        inconsistency_sentence+=f'Checking inputs.relax.{base_key}.pw.parameters.CONTROL.calculation: can be only "relax". No cell relaxation should be performed.'
    
    
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
        value_input_relax = iterdict(parameters["relax"][base_key]["pw"]["parameters"].get_dict(),key)
        value_overrides = iterdict(_overrides,key)
        #print(value_input_relax,value_input_pwscf,value_overrides)
        if value_input_relax != value_overrides:
            if value_input_relax in [0, None] and value_overrides in [0, None]:
                continue # 0 is None and viceversa
            wrong_inputs_relax.append(key)
            inconsistency_sentence += f'Checking inputs.relax.{base_key}.pw.parameters input: "{key}" is not correct. You provided the value "{value_input_relax}", but only "{value_overrides}" is consistent with your settings.\n'
        
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
