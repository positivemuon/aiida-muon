"""
WorkChain for active learning.

The idea is to perform iterations of fine tuning in such a way to improve the model. 
If only one iteration is performed, then this is just a fine tuning workflow.

1. The user can provide trajectories or structures, and the workflow will perform the necessary calculations to obtain the forces and energies... 
2. ... which are then used to fine tune the model.
3. Then, there will be a validation step, where the model is used to predict forces and energies for a set of structures, and the predictions are compared to reference calculations.
4. If the predictions are not good enough, then the workflow will select new structures to be added to the training set, and the process will be repeated. Otherwise, the workflow will terminate.

Ideal structures/trajectories depends on the purpose:
- just pre-clustering: unit cells with muon should be ok
- for final clustering and predictions: supercells with muon should be used

Pre-processing of the structures can be done, like rattling the muon/atoms.

This workflow can be used as standalone or within the FindMuonWorkChain.

If too many frames in a trajectory or too many structures, the user can specify
how many to select within a dedicated optional input (default is 3).

The user can also decide which QE calculation to do, singlepoint (PwBaseWorkChain) or relaxation (PwRelaxWorkChain). 
The relax case is particularly useful if only one structure is provided.

for now, the finetuning will be performed using the FineTuningWorkChain.
"""

import os
from aiida import orm
from aiida.engine import WorkChain, ToContext, calcfunction, if_, while_
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_pythonjob import PythonJob

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')

from aiida_muon.workflows.finetuning import FineTuningWorkChain
from aiida_muon.pythonjobs.score_frames import prepare_score_frames_pythonjob_inputs
from aiida_muon.utils.trajectory import atoms_list_to_trajectory_data

# ─────────────────────────────────────────────────────────────────────────────
# Helper calcfunction: write ASE-readable .xyz training file from DFT outputs
# ─────────────────────────────────────────────────────────────────────────────

@calcfunction
def collect_dft_outputs_to_xyz(
    structures: orm.List,
    energies: orm.List,
    forces: orm.List,
    output_path: orm.Str,
) -> orm.Str:
    """
    Write a training .xyz file from DFT outputs.

    Parameters
    ----------
    structures : orm.List
        List of StructureData nodes (AiiDA ORM) in the order they were computed.
    energies : orm.List
        List of total energies (float, eV) in the same order.
    forces : orm.List
        List of force arrays (list-of-lists, eV/Å) in the same order.
    output_path : orm.Str
        Directory where `train_data.xyz` will be written.

    Returns
    -------
    orm.Str
        Full path to the written .xyz file.
    """
    from ase.io import write
    import numpy as np

    structs = structures.get_list()      # list of StructureData nodes
    ens     = energies.get_list()        # list of float
    fs      = forces.get_list()          # list of list-of-list

    atoms_list = []
    for struct, energy, force in zip(structs, ens, fs):
        # struct is already a StructureData node
        atoms = struct.get_ase()
        atoms.info['energy'] = float(energy)
        atoms.arrays['forces'] = np.array(force)
        atoms_list.append(atoms)

    out_dir = output_path.value
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'train_data.xyz')
    write(out_file, atoms_list, format='extxyz')

    return orm.Str(out_file)


# ─────────────────────────────────────────────────────────────────────────────
# ActiveLearningWorkChain
# ─────────────────────────────────────────────────────────────────────────────

class ActiveLearningWorkChain(ProtocolMixin, WorkChain):
    """
    WorkChain for (iterative) active learning of a MatterSim MLIP model.

    Outline
    -------
    1. setup
       Inspect inputs; decide whether DFT labelling is required
       (structures provided → DFT needed) or whether a training file is
       already available.

    2. [optional] run_dft_calculations
       Submit one PwBaseWorkChain (singlepoint) or PwRelaxWorkChain (relax)
       per structure in the input list.

    3. [optional] collect_dft_results
       Gather energies and forces from the finished DFT jobs and write a
       train_data.xyz file via the `collect_dft_outputs_to_xyz` calcfunction.

    4. while(should_iterate):
       a. run_finetuning          – submit FineTuningWorkChain
       b. collect_finetuning      – retrieve finetuned model path, store in ctx
       c. run_validation          – TODO: run finetuned model on test frames
       d. collect_validation      – TODO: compute score; decide whether to stop

    5. set_outputs

    Inputs
    ------
    structures (optional)
        List of StructureData nodes to be labelled with DFT.
        If omitted, `train_data_path` must be provided directly.
    train_data_path (optional)
        Absolute path to an already-prepared training .xyz/.pkl file.
        Required when `structures` is not given.
    test_data_path (optional)
        Absolute path to test frames used during validation.
    load_model_path
        Absolute path to the starting MatterSim checkpoint (.pth).
    save_path
        Base directory where finetuned models will be saved.
        Each iteration saves to `<save_path>/iter_<N>/`.
    max_iterations
        Maximum number of active-learning iterations (default 1).
    run_relax (optional, default False)
        If True, use PwRelaxWorkChain instead of PwBaseWorkChain for DFT.
    pseudo_family
        Pseudo-potential family label (required when structures are given).
    pythonjob_code
        PythonJob code node for the finetuning step.
    finetuning.*
        All FineTuningWorkChain hyper-parameter inputs, forwarded directly.
    relax.* / pwscf.*
        Exposed PwRelaxWorkChain / PwBaseWorkChain namespaces for DFT options.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ── Structures / training data ────────────────────────────────────────
        spec.input_namespace(
            'structures',
            valid_type=(orm.StructureData,),
            required=False,
            dynamic=True,
            help='Dict of StructureData nodes to be labelled with DFT.',
        )
        spec.input(
            'train_data_path',
            valid_type=orm.Str,
            required=False,
            help='Path to an existing training data file (.xyz or .pkl). '
                 'Required when `structures` is not provided.',
        )
        spec.input(
            'test_data_path',
            valid_type=orm.Str,
            required=False,
            help='Path to test frames used for validation.',
        )

        # ── Model paths ───────────────────────────────────────────────────────
        spec.input(
            'load_model_path',
            valid_type=orm.Str,
            help='Absolute path to the starting MatterSim checkpoint (.pth).',
        )
        spec.input(
            'save_path',
            valid_type=orm.Str,
            default=lambda: orm.Str('./finetuned_model'),
            required=False,
            help='Base directory for saving finetuned models.',
        )

        # ── Loop control ──────────────────────────────────────────────────────
        spec.input(
            'max_iterations',
            valid_type=orm.Int,
            default=lambda: orm.Int(1),
            required=False,
            help='Maximum number of active-learning iterations.',
        )

        # ── DFT options ───────────────────────────────────────────────────────
        spec.input(
            'run_relax',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help='If True, use PwRelaxWorkChain; otherwise PwBaseWorkChain (singlepoint).',
        )
        spec.input(
            'pseudo_family',
            valid_type=orm.Str,
            default=lambda: orm.Str('SSSP/1.3/PBE/efficiency'),
            required=False,
            help='Pseudo-potential family label (needed when structures are provided).',
        )

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace='relax',
            exclude=('structure', 'base_final_scf'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for PwRelaxWorkChain (used when run_relax=True).',
            },
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='pwscf',
            exclude=('pw.structure', 'kpoints'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for PwBaseWorkChain singlepoint.',
            },
        )

        # ── PythonJob code + finetuning hyper-params (forwarded) ─────────────
        spec.input(
            'pythonjob_code',
            valid_type=orm.AbstractCode,
            help='PythonJob code node for the finetuning PythonJob.',
        )
        spec.expose_inputs(
            FineTuningWorkChain,
            namespace='finetuning',
            exclude=('train_data_path', 'atoms_list', 'load_model_path', 'save_path', 'pythonjob_code'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Hyper-parameter inputs forwarded to FineTuningWorkChain.',
            },
        )

        # ── Score-based frame selection (optional) ─────────────────────────────
        spec.input(
            'score_callback_calculator',
            non_db=True,
            required=False,
            help='ASE-compatible MLIP callable used for frame scoring. '
                 'When provided (together with DFT structures), the workflow runs '
                 'a score-calculation step before fine tuning and selects only the '
                 'most informative frames as training data. '
                 'Omit to skip the scoring step and train on all DFT frames.',
        )
        spec.input(
            'score_num_frames',
            valid_type=orm.Int,
            default=lambda: orm.Int(5),
            required=False,
            help='Maximum number of training frames to select via the score filter.',
        )
        spec.input(
            'score_w_E',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.2),
            required=False,
            help='Weight for per-atom energy disagreement in the composite score.',
        )
        spec.input(
            'score_w_F',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.8),
            required=False,
            help='Weight for force RMSE in the composite score.',
        )
        spec.input(
            'score_similarity_thr',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.01),
            required=False,
            help='Structural RMSE threshold (\u00c5) below which two frames are '
                 'considered too similar during the similarity filter.',
        )
        spec.input(
            'score_energy_shift',
            valid_type=orm.Float,
            required=False,
            help='Cohesive-energy offset (eV) to subtract from DFT energies '
                 'before scoring (use to align DFT and MLIP energy references).',
        )

        # ── Outline ──────────────────────────────────────────────────────────
        spec.outline(
            cls.setup,
            if_(cls.should_run_dft)(
                cls.run_dft_calculations,
                cls.collect_dft_results,
            ),
            if_(cls.should_run_score_filtering)(
                cls.run_score_calculation,
                cls.collect_score_results,
            ),
            while_(cls.should_iterate)(
                cls.run_finetuning,
                cls.collect_finetuning,
                cls.run_validation,       # TODO: implement validation
                cls.collect_validation,   # TODO: implement score + stopping criterion
            ),
            cls.set_outputs,
        )

        # ── Exit codes ────────────────────────────────────────────────────────
        spec.exit_code(420, 'ERROR_NO_TRAINING_DATA',
                       message='Neither structures nor train_data_path were provided.')
        spec.exit_code(421, 'ERROR_DFT_FAILED',
                       message='One or more DFT calculations failed.')
        spec.exit_code(422, 'ERROR_FINETUNING_FAILED',
                       message='The FineTuningWorkChain failed.')
        spec.exit_code(423, 'ERROR_SCORE_CALCULATION_FAILED',
                       message='The score-calculation PythonJob failed.')

        # ── Outputs ───────────────────────────────────────────────────────────
        spec.output('finetuned_model_path', valid_type=orm.Str,
                    help='Path to the final best-model checkpoint.')
        spec.output('train_data_path_used', valid_type=orm.Str, required=False,
                    help='Path to the training data file that was used (when DFT path taken).')
        spec.output('iterations_performed', valid_type=orm.Int,
                    help='Number of active-learning iterations performed.')
        spec.output('score_reliability', valid_type=orm.Dict, required=False,
                    help='Model reliability summary from the score calculation step.')
        spec.output('score_selected_indices', valid_type=orm.List, required=False,
                    help='Indices of the frames selected from the DFT trajectory for training.')

    @classmethod
    def get_builder_from_protocol(
        cls,
        pythonjob_code: orm.AbstractCode,
        load_model_path: str,
        train_data_path: str = None,
        structures: dict = None,
        test_data_path: str = None,
        save_path: str = './finetuned_model',
        max_iterations: int = 1,
        run_relax: bool = False,
        pseudo_family: str = 'SSSP/1.3/PBE/efficiency',
        pw_code: orm.AbstractCode = None,
        protocol: str = None,
        overrides: dict = None,
        # FineTuningWorkChain hyper-params
        epochs: int = 100,
        batch_size: int = 4,
        lr: float = 2e-4,
        device: str = 'cpu',
        include_forces: bool = True,
        include_stresses: bool = False,
        force_loss_ratio: float = 1.0,
        stress_loss_ratio: float = 0.1,
        seed: int = 42,
        pythonjob_metadata: dict = None,
        options: dict = None,
    ):
        """
        Return a builder prepopulated with inputs for the ActiveLearningWorkChain.

        Parameters
        ----------
        pythonjob_code : orm.AbstractCode
            PythonJob code node for the MatterSim finetuning step.
        load_model_path : str
            Absolute path to the starting MatterSim checkpoint (.pth).
        train_data_path : str, optional
            Path to a pre-existing training file (.xyz or .pkl).
            Either this or `structures` must be provided.
        structures : dict, optional
            Dict mapping string labels to StructureData nodes that will be
            labelled with DFT before fine tuning.
            Either this or `train_data_path` must be provided.
        test_data_path : str, optional
            Path to test frames used during validation.
        save_path : str
            Base directory for saving finetuned models.
            Each iteration writes to ``<save_path>/iter_NN/``.
        max_iterations : int
            Maximum number of active-learning iterations (default 1).
        run_relax : bool
            If True, use PwRelaxWorkChain for DFT; otherwise PwBaseWorkChain.
        pseudo_family : str
            Pseudo-potential family label (needed when structures are provided).
        pw_code : orm.AbstractCode, optional
            QE pw.x code node.  Required when `structures` is provided.
        protocol : str, optional
            AiiDA-QE protocol name (e.g. 'moderate').  Used to build the DFT
            inputs when `structures` is provided.
        overrides : dict, optional
            Protocol overrides forwarded to PwRelaxWorkChain / PwBaseWorkChain.
        epochs : int
            Number of training epochs.
        batch_size : int
            Training batch size.
        lr : float
            Learning rate.
        device : str
            Device for training: 'cpu' or 'cuda'.
        include_forces : bool
            Include forces in the training loss.
        include_stresses : bool
            Include stresses in the training loss.
        force_loss_ratio : float
            Weight for the force loss.
        stress_loss_ratio : float
            Weight for the stress loss.
        seed : int
            Random seed.
        pythonjob_metadata : dict, optional
            Metadata forwarded to the PythonJob (resources, wall time, …).
        options : dict, optional
            Scheduler options applied to the DFT CalcJobs
            (e.g. ``{'resources': {'num_machines': 1}, 'max_wallclock_seconds': 3600}``).

        Returns
        -------
        ProcessBuilder
            A fully populated builder ready for submission.
        """
        if train_data_path is None and not structures:
            raise ValueError(
                'Either train_data_path or structures must be provided.'
            )

        builder = cls.get_builder()

        # ── Required / scalar inputs ──────────────────────────────────────────
        builder.pythonjob_code  = pythonjob_code
        builder.load_model_path = orm.Str(load_model_path)
        builder.save_path       = orm.Str(save_path)
        builder.max_iterations  = orm.Int(max_iterations)
        builder.run_relax       = orm.Bool(run_relax)
        builder.pseudo_family   = orm.Str(pseudo_family)

        if train_data_path is not None:
            builder.train_data_path = orm.Str(train_data_path)

        if test_data_path is not None:
            builder.test_data_path = orm.Str(test_data_path)

        # ── Structures (DFT path) ─────────────────────────────────────────────
        if structures:
            for label, structure in structures.items():
                builder.structures[label] = structure

        # ── DFT sub-workchain inputs ──────────────────────────────────────────
        if structures and pw_code is not None:
            # Use any structure as a representative for building the protocol
            representative = next(iter(structures.values()))
            overrides = overrides or {}

            if options:
                overrides.setdefault('base', {}).setdefault('pw', {}).setdefault(
                    'metadata', {}
                )['options'] = options

            if run_relax:
                from aiida_quantumespresso.common.types import RelaxType
                dft_builder = PwRelaxWorkChain.get_builder_from_protocol(
                    code=pw_code,
                    structure=representative,
                    pseudo_family=pseudo_family,
                    protocol=protocol,
                    overrides=overrides,
                    relax_type=RelaxType.POSITIONS,
                )
                dft_builder.pop('structure', None)
                dft_builder.pop('base_final_scf', None)
                builder.relax = dft_builder
            else:
                dft_builder = PwBaseWorkChain.get_builder_from_protocol(
                    code=pw_code,
                    structure=representative,
                    pseudo_family=pseudo_family,
                    protocol=protocol,
                    overrides=overrides.get('base', overrides),
                )
                dft_builder['pw'].pop('structure', None)
                dft_builder.pop('kpoints_distance', None)
                builder.pwscf = dft_builder

        # ── Finetuning hyper-parameters ───────────────────────────────────────
        builder.finetuning.epochs            = orm.Int(epochs)
        builder.finetuning.batch_size        = orm.Int(batch_size)
        builder.finetuning.lr               = orm.Float(lr)
        builder.finetuning.device           = orm.Str(device)
        builder.finetuning.include_forces   = orm.Bool(include_forces)
        builder.finetuning.include_stresses = orm.Bool(include_stresses)
        builder.finetuning.force_loss_ratio = orm.Float(force_loss_ratio)
        builder.finetuning.stress_loss_ratio= orm.Float(stress_loss_ratio)
        builder.finetuning.seed             = orm.Int(seed)

        if pythonjob_metadata is not None:
            builder.finetuning.pythonjob.metadata = pythonjob_metadata

        return builder

    # ── Step 1: setup ─────────────────────────────────────────────────────────

    def setup(self):
        """
        Initialise context variables.

        Decides:
          - whether DFT labelling is needed (ctx.run_dft)
          - what the starting model path is (ctx.current_model_path)
          - the iteration counter (ctx.iteration)
        """
        self.ctx.iteration = 0
        self.ctx.current_model_path = self.inputs.load_model_path.value
        self.ctx.finetuned_model_path = None
        self.ctx.selected_atoms = None   # populated by collect_score_results
        self.ctx.score_reliability = None

        # Decide if DFT is needed
        if 'structures' in self.inputs and len(self.inputs.structures) > 0:
            self.ctx.run_dft = True
            self.ctx.structure_keys = list(self.inputs.structures.keys())
            self.report(
                f'Found {len(self.ctx.structure_keys)} input structures. '
                'DFT labelling will be performed.'
            )
        elif 'train_data_path' in self.inputs:
            self.ctx.run_dft = False
            self.ctx.train_data_path = self.inputs.train_data_path.value
            self.report(
                f'Using pre-existing training data: {self.ctx.train_data_path}. '
                'Skipping DFT step.'
            )
        else:
            self.report('ERROR: neither structures nor train_data_path provided.')
            return self.exit_codes.ERROR_NO_TRAINING_DATA

    # ── Condition: run DFT? ───────────────────────────────────────────────────

    def should_run_dft(self):
        return self.ctx.run_dft
    # ── Condition: run score filtering? ───────────────────────────────────────────

    def should_run_score_filtering(self):
        """
        Run score-based filtering when:
        - DFT was performed (we have raw trajectory frames in ctx), AND
        - a ``score_callback_calculator`` was provided by the user.
        """
        has_calculator = 'score_callback_calculator' in self.inputs
        if self.ctx.run_dft and has_calculator:
            self.report('Score-based frame filtering will be performed.')
            return True
        if self.ctx.run_dft and not has_calculator:
            self.report(
                'No score_callback_calculator provided — '
                'skipping score filtering; all DFT frames will be used for training.'
            )
        return False
    # ── Step 2: run DFT ───────────────────────────────────────────────────────

    def run_dft_calculations(self):
        """
        Submit one PwBaseWorkChain (singlepoint) or PwRelaxWorkChain (relax)
        per structure in self.inputs.structures.
        """
        from aiida_muon.workflows.find_muon import get_pseudos  # reuse find_muon helper if available

        do_relax = self.inputs.run_relax.value

        for key in self.ctx.structure_keys:
            structure = self.inputs.structures[key]

            if do_relax:
                inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
                inputs.structure = structure
                inputs.base.pw.pseudos = get_pseudos(structure, self.inputs.pseudo_family.value)
                inputs.metadata.call_link_label = f'dft_{key}'
                future = self.submit(PwRelaxWorkChain, **inputs)
                self.report(f'Submitted PwRelaxWorkChain (PK={future.pk}) for structure {key}')
            else:
                inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='pwscf'))
                inputs.pw.structure = structure
                inputs.pw.pseudos = get_pseudos(structure, self.inputs.pseudo_family.value)
                inputs.metadata.call_link_label = f'dft_{key}'
                future = self.submit(PwBaseWorkChain, **inputs)
                self.report(f'Submitted PwBaseWorkChain (PK={future.pk}) for structure {key}')

            self.to_context(**{f'dft_{key}': future})

    # ── Step 3: collect DFT results ───────────────────────────────────────────

    def collect_dft_results(self):
        """
        Gather energies and forces from DFT jobs; write train_data.xyz.
        """
        do_relax = self.inputs.run_relax.value

        structures_out = []
        energies_out   = []
        forces_out     = []
        n_failed       = 0

        for key in self.ctx.structure_keys:
            wc = self.ctx[f'dft_{key}']

            if not wc.is_finished_ok:
                self.report(f'DFT calculation for structure {key} failed (status {wc.exit_status}). Skipping.')
                n_failed += 1
                continue

            if do_relax:
                params  = wc.outputs.output_parameters.get_dict()
                energy  = params['energy']                   # eV
                struct  = wc.outputs.output_structure
                # forces from the last ionic step – stored in output_trajectory
                forces  = wc.outputs.output_trajectory.get_array('forces')[-1].tolist()
            else:
                params  = wc.outputs.output_parameters.get_dict()
                energy  = params['energy']
                struct  = wc.inputs.pw.structure             # input structure (singlepoint)
                forces  = wc.outputs.output_parameters.get_dict().get('forces', [])

            structures_out.append(struct)
            energies_out.append(energy)
            forces_out.append(forces)

        if n_failed == len(self.ctx.structure_keys):
            return self.exit_codes.ERROR_DFT_FAILED

        if n_failed > 0:
            self.report(f'{n_failed} DFT jobs failed; continuing with the rest.')

        # Write training file via calcfunction (provenance-tracked)
        train_file = collect_dft_outputs_to_xyz(
            orm.List(list=structures_out),
            orm.List(list=energies_out),
            orm.List(list=forces_out),
            orm.Str(self.inputs.save_path.value),
        )
        self.ctx.train_data_path = train_file.value
        self.report(f'Training data written to {self.ctx.train_data_path}')

        # Also store raw lists for the optional score-filtering step
        self.ctx.dft_structures = structures_out    # list of StructureData nodes
        self.ctx.dft_energies   = energies_out      # list of float
        self.ctx.dft_forces     = forces_out        # list of list-of-list
        self.report(
            f'Stored {len(structures_out)} DFT frames for optional score filtering.'
        )
    # ── Step 3b: score calculation ───────────────────────────────────────────────

    def run_score_calculation(self):
        """
        Submit a PythonJob that:
        1. Receives the full DFT trajectory (as a pickled list of ASE Atoms built
           from ``ctx.dft_structures``, ``ctx.dft_energies``, ``ctx.dft_forces``).
        2. Evaluates the MLIP (``score_callback_calculator``) on each frame.
        3. Computes per-frame disagreement scores.
        4. Returns the most informative subset of frames (pickled list of Atoms).

        All data flows as pickled Python objects through aiida-pythonjob — no
        intermediate files are written to any shared filesystem.
        """
        import numpy as np
        from ase.calculators.singlepoint import SinglePointCalculator

        self.report('Building DFT Atoms list for score calculation')

        # Reconstruct ASE Atoms with embedded DFT results from the raw lists
        # stored in ctx by collect_dft_results.
        dft_atoms_list = []
        for struct, energy, forces in zip(
            self.ctx.dft_structures,
            self.ctx.dft_energies,
            self.ctx.dft_forces,
        ):
            atoms = struct.get_ase()
            forces_arr = np.array(forces)
            atoms.calc = SinglePointCalculator(
                atoms, energy=float(energy), forces=forces_arr
            )
            dft_atoms_list.append(atoms)

        energy_shift = (
            self.inputs.score_energy_shift.value
            if 'score_energy_shift' in self.inputs
            else None
        )

        # Convert ASE Atoms list → TrajectoryData (proper AiiDA node, no pickle)
        dft_trajectory = atoms_list_to_trajectory_data(dft_atoms_list)

        pythonjob_inputs = prepare_score_frames_pythonjob_inputs(
            dft_trajectory=dft_trajectory,
            callback_calculator=self.inputs.score_callback_calculator,
            pythonjob_code=self.inputs.pythonjob_code,
            num_frames=self.inputs.score_num_frames.value,
            w_E=self.inputs.score_w_E.value,
            w_F=self.inputs.score_w_F.value,
            similarity_thr=self.inputs.score_similarity_thr.value,
            energy_shift=energy_shift,
        )

        future = self.submit(PythonJob, **pythonjob_inputs)
        self.report(
            f'Submitted score-calculation PythonJob (PK={future.pk}) '
            f'on {len(dft_atoms_list)} DFT frames.'
        )
        return ToContext(score_job=future)

    def collect_score_results(self):
        """
        Retrieve the selected frames from the score PythonJob and store them
        in ``ctx.selected_atoms`` for direct forwarding to the finetuning job.
        """
        job = self.ctx.score_job

        if not job.is_finished_ok:
            self.report(
                f'Score PythonJob failed (status {job.exit_status}). '
                'Falling back to full DFT trajectory for training.'
            )
            # Soft failure: do not abort, just skip the filtering.
            self.ctx.selected_atoms = None
            return

        # selected_atoms is a pickled list of ASE Atoms (PickledData node)
        self.ctx.selected_atoms          = job.outputs.selected_atoms
        self.ctx.score_selected_indices  = list(job.outputs.selected_indices)
        reliability                      = dict(job.outputs.reliability)
        self.ctx.score_reliability       = reliability
        selected_indices = self.ctx.score_selected_indices

        self.report(
            f'Score filtering done: selected {len(selected_indices)} frames '
            f'(indices={selected_indices}). '
            f'Mean score={reliability.get("mean_score", "n/a"):.4f}, '
            f'Mean force RMSE={reliability.get("mean_deltaF_rmse", "n/a"):.4f} eV/\u00c5.'
        )
    # ── Loop condition ─────────────────────────────────────────────────────────

    def should_iterate(self):
        """Continue if we have not yet reached max_iterations and no stopping signal."""
        keep_going = self.ctx.iteration < self.inputs.max_iterations.value
        if not keep_going:
            self.report(
                f'Reached max_iterations={self.inputs.max_iterations.value}. Stopping.'
            )
        return keep_going

    # ── Step 4a: run finetuning ────────────────────────────────────────────────

    def run_finetuning(self):
        """Submit FineTuningWorkChain for the current iteration."""

        self.ctx.iteration += 1
        iter_save_path = os.path.join(
            self.inputs.save_path.value, f'iter_{self.ctx.iteration:02d}'
        )
        self.report(
            f'Starting fine-tuning iteration {self.ctx.iteration} '
            f'(save_path={iter_save_path})'
        )

        inputs = AttributeDict(self.exposed_inputs(FineTuningWorkChain, namespace='finetuning'))
        inputs.load_model_path  = orm.Str(self.ctx.current_model_path)
        inputs.save_path        = orm.Str(iter_save_path)
        inputs.pythonjob_code   = self.inputs.pythonjob_code

        if self.ctx.selected_atoms is not None:
            # ctx.selected_atoms is the PickledData node returned by the score pythonjob.
            # .value unpacks it to a plain list of ASE Atoms (energy in atoms.info,
            # forces in atoms.arrays, calculator stripped).
            inputs.atoms_list = self.ctx.selected_atoms.value
            self.report(
                'Using score-filtered frames (atoms_list) as training data.'
            )
        else:
            # Fall back to the file-based path written by collect_dft_results
            # or provided directly by the user.
            inputs.train_data_path = orm.Str(self.ctx.train_data_path)
            self.report(
                f'Using full DFT trajectory at {self.ctx.train_data_path} as training data.'
            )

        future = self.submit(FineTuningWorkChain, **inputs)
        self.report(f'Submitted FineTuningWorkChain (PK={future.pk})')
        return ToContext(finetuning_wc=future)

    # ── Step 4b: collect finetuning ───────────────────────────────────────────

    def collect_finetuning(self):
        """Retrieve the finetuned model path; update ctx.current_model_path."""

        wc = self.ctx.finetuning_wc

        if not wc.is_finished_ok:
            self.report(f'FineTuningWorkChain failed (status {wc.exit_status})')
            return self.exit_codes.ERROR_FINETUNING_FAILED

        self.ctx.current_model_path  = wc.outputs.model_path.value
        self.ctx.finetuned_model_path = self.ctx.current_model_path
        self.report(
            f'Iteration {self.ctx.iteration} finetuning done. '
            f'Model path: {self.ctx.current_model_path}'
        )

    # ── Step 4c: run validation ───────────────────────────────────────────────

    def run_validation(self):
        """
        Run the finetuned model on the test frames and collect a validation score.

        TODO: implement — submit a PythonJob that evaluates the model on
              self.inputs.test_data_path using ctx.current_model_path and
              returns a scalar score (e.g. MAE on forces).
        """
        self.report(
            'Validation step is not yet implemented. '
            'Continuing to next iteration (or terminating if max_iterations reached).'
        )

    # ── Step 4d: collect validation ───────────────────────────────────────────

    def collect_validation(self):
        """
        Inspect the validation score and decide whether to stop early.

        TODO: implement — read the score from ctx, compare against a
              user-supplied threshold, and set ctx.validation_passed = True/False
              to allow early exit from the while_ loop.
        """
        # Placeholder: always accept the model and continue the loop
        self.ctx.validation_passed = True

    # ── Step 5: set outputs ───────────────────────────────────────────────────

    def set_outputs(self):
        """Expose final outputs."""

        if self.ctx.finetuned_model_path is None:
            self.ctx.finetuned_model_path = self.ctx.current_model_path

        self.out('finetuned_model_path', orm.Str(self.ctx.finetuned_model_path).store())
        self.out('iterations_performed', orm.Int(self.ctx.iteration).store())

        if hasattr(self.ctx, 'train_data_path'):
            self.out('train_data_path_used', orm.Str(self.ctx.train_data_path).store())

        if self.ctx.score_reliability is not None:
            self.out('score_reliability', orm.Dict(self.ctx.score_reliability).store())

        if self.ctx.selected_atoms is not None and hasattr(self.ctx, 'score_selected_indices'):
            self.out('score_selected_indices',
                     orm.List(list=self.ctx.score_selected_indices).store())

        self.report(
            f'ActiveLearningWorkChain finished after {self.ctx.iteration} iteration(s). '
            f'Final model: {self.ctx.finetuned_model_path}'
        )
