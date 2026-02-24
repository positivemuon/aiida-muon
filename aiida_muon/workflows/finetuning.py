"""
WorkChain single shot finetuning.

for now, the finetuning will be performed using the pythonjob for mattersim (see ../pythonjobs/finetuning_mattersim.py)... but in the future it can be extended to other models and pythonjobs.

TODO: atoms_list should somehow be in the provenance, for now we pass it as non_db.
"""

from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida_pythonjob import PythonJob
from aiida.common import AttributeDict
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_muon.pythonjobs.finetuning_mattersim import prepare_mattersim_finetuning_inputs


class FineTuningWorkChain(ProtocolMixin, WorkChain):
    """
    WorkChain for one-shot fine tuning of a MatterSim MLIP model.

    It submits a single PythonJob that performs the fine tuning using the
    helper function in aiida_muon.pythonjobs.finetuning_mattersim and
    exposes the resulting model path as an output.

    This workchain is designed to be embedded in a larger active-learning
    loop: the caller is responsible for producing the training data and for
    any subsequent validation / iteration logic.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ── Exposed PythonJob namespace (metadata, computer, …) ──────────────
        spec.expose_inputs(
            PythonJob,
            namespace='pythonjob',
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Extra inputs forwarded directly to the PythonJob.',
            },
        )

        # ── Training data / model ─────────────────────────────────────────────
        spec.input(
            'train_data_path',
            valid_type=orm.Str,
            required=False,
            help='Absolute path to the training data file (.xyz or .pkl). '
                 'Either this or ``atoms_list`` must be provided.',
        )
        spec.input(
            'atoms_list',
            non_db=True,
            required=False,
            help='Pre-built list of ASE Atoms with DFT results embedded '
                 '(``atoms.info["energy"]`` / ``atoms.arrays["forces"]``). '
                 'Passed directly to the pythonjob — no filesystem required. '
                 'Either this or ``train_data_path`` must be provided.',
        )
        spec.input(
            'load_model_path',
            valid_type=orm.Str,
            help='Absolute path to the pretrained MatterSim checkpoint (.pth).',
        )
        spec.input(
            'save_path',
            valid_type=orm.Str,
            default=lambda: orm.Str('./finetuned_model'),
            required=False,
            help='Directory in which the finetuned model will be saved.',
        )

        # ── Hyper-parameters ─────────────────────────────────────────────────
        spec.input(
            'epochs',
            valid_type=orm.Int,
            default=lambda: orm.Int(100),
            required=False,
            help='Number of training epochs.',
        )
        spec.input(
            'batch_size',
            valid_type=orm.Int,
            default=lambda: orm.Int(4),
            required=False,
            help='Training batch size.',
        )
        spec.input(
            'lr',
            valid_type=orm.Float,
            default=lambda: orm.Float(2e-4),
            required=False,
            help='Learning rate.',
        )
        spec.input(
            'device',
            valid_type=orm.Str,
            default=lambda: orm.Str('cpu'),
            required=False,
            help="Device for training: 'cpu' or 'cuda'.",
        )
        spec.input(
            'include_forces',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
            help='Include forces in the training loss.',
        )
        spec.input(
            'include_stresses',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help='Include stresses in the training loss.',
        )
        spec.input(
            'force_loss_ratio',
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0),
            required=False,
            help='Weight for the force loss contribution.',
        )
        spec.input(
            'stress_loss_ratio',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.1),
            required=False,
            help='Weight for the stress loss contribution.',
        )
        spec.input(
            'seed',
            valid_type=orm.Int,
            default=lambda: orm.Int(42),
            required=False,
            help='Random seed for reproducibility.',
        )

        # ── Code ─────────────────────────────────────────────────────────────
        spec.input(
            'pythonjob_code',
            valid_type=orm.AbstractCode,
            help='The PythonJob code to use for the finetuning calculation.',
        )

        # ── Outline ──────────────────────────────────────────────────────────
        spec.outline(
            cls.run_finetuning,
            cls.collect_results,
        )

        # ── Exit codes ───────────────────────────────────────────────────────
        spec.exit_code(
            410,
            'ERROR_FINETUNING_FAILED',
            message='The PythonJob finetuning calculation failed.',
        )

        # ── Outputs ──────────────────────────────────────────────────────────
        spec.output(
            'model_path',
            valid_type=orm.Str,
            help='Path to the best finetuned model checkpoint.',
        )
        spec.output(
            'num_training_samples',
            valid_type=orm.Int,
            help='Number of structures used for training.',
        )
        spec.output(
            'num_epochs',
            valid_type=orm.Int,
            help='Number of epochs the model was trained for.',
        )
        spec.output(
            'status',
            valid_type=orm.Str,
            help="Status string returned by the finetuning function (e.g. 'completed').",
        )

    @classmethod
    def get_builder_from_protocol(
        cls,
        pythonjob_code: orm.AbstractCode,
        train_data_path: str,
        load_model_path: str,
        save_path: str = './finetuned_model',
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
    ):
        """
        Return a builder prepopulated with inputs for a one-shot MatterSim fine tuning.

        Parameters
        ----------
        pythonjob_code : orm.AbstractCode
            The PythonJob code node to use for the finetuning calculation.
        train_data_path : str
            Absolute path to the training data file (.xyz or .pkl).
        load_model_path : str
            Absolute path to the pretrained MatterSim checkpoint (.pth).
        save_path : str
            Directory where the finetuned model will be saved.
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
            Weight for the force loss contribution.
        stress_loss_ratio : float
            Weight for the stress loss contribution.
        seed : int
            Random seed for reproducibility.
        pythonjob_metadata : dict, optional
            Metadata dict forwarded to the PythonJob (resources, wall time, …).
            If None, a sensible default (1 machine, 2 h) is used.

        Returns
        -------
        ProcessBuilder
            A builder instance with all inputs set, ready for submission.
        """
        builder = cls.get_builder()

        builder.pythonjob_code   = pythonjob_code
        builder.train_data_path  = orm.Str(train_data_path)
        builder.load_model_path  = orm.Str(load_model_path)
        builder.save_path        = orm.Str(save_path)
        builder.epochs           = orm.Int(epochs)
        builder.batch_size       = orm.Int(batch_size)
        builder.lr               = orm.Float(lr)
        builder.device           = orm.Str(device)
        builder.include_forces   = orm.Bool(include_forces)
        builder.include_stresses = orm.Bool(include_stresses)
        builder.force_loss_ratio = orm.Float(force_loss_ratio)
        builder.stress_loss_ratio= orm.Float(stress_loss_ratio)
        builder.seed             = orm.Int(seed)

        if pythonjob_metadata is not None:
            builder.pythonjob.metadata = pythonjob_metadata

        return builder

    # ── Step 1: submit ───────────────────────────────────────────────────────

    def run_finetuning(self):
        """Prepare inputs and submit the MatterSim finetuning PythonJob."""

        self.report('Preparing MatterSim finetuning inputs')

        # Collect optional pythonjob metadata if exposed
        pythonjob_metadata = None
        if 'pythonjob' in self.inputs and 'metadata' in self.inputs.pythonjob:
            pythonjob_metadata = self.inputs.pythonjob.metadata

        if 'train_data_path' not in self.inputs and 'atoms_list' not in self.inputs:
            self.report('Neither train_data_path nor atoms_list provided — cannot run finetuning')
            return self.exit_codes.ERROR_FINETUNING_FAILED
        elif 'train_data_path' in self.inputs and 'atoms_list' in self.inputs:
            self.report('Both train_data_path and atoms_list provided — using train_data_path and ignoring atoms_list')
            return self.exit_codes.ERROR_FINETUNING_FAILED

        pythonjob_inputs = prepare_mattersim_finetuning_inputs(
            load_model_path=self.inputs.load_model_path.value,
            pythonjob_code=self.inputs.pythonjob_code,
            train_data_path=self.inputs.train_data_path.value if 'train_data_path' in self.inputs else None,
            atoms_list=self.inputs.atoms_list if 'atoms_list' in self.inputs else None,
            save_path=self.inputs.save_path.value,
            epochs=self.inputs.epochs.value,
            batch_size=self.inputs.batch_size.value,
            lr=self.inputs.lr.value,
            device=self.inputs.device.value,
            include_forces=self.inputs.include_forces.value,
            include_stresses=self.inputs.include_stresses.value,
            force_loss_ratio=self.inputs.force_loss_ratio.value,
            stress_loss_ratio=self.inputs.stress_loss_ratio.value,
            seed=self.inputs.seed.value,
            pythonjob_metadata=pythonjob_metadata,
        )

        # Merge any extra pythonjob inputs exposed by the user
        if 'pythonjob' in self.inputs:
            extra = AttributeDict(self.exposed_inputs(PythonJob, namespace='pythonjob'))
            # metadata may already be set above; only update keys not yet present
            for key, value in extra.items():
                if key not in pythonjob_inputs:
                    pythonjob_inputs[key] = value

        future = self.submit(PythonJob, **pythonjob_inputs)
        self.report(f'Submitted finetuning PythonJob (PK={future.pk})')
        return ToContext(finetuning_job=future)

    # ── Step 2: collect ──────────────────────────────────────────────────────

    def collect_results(self):
        """Retrieve outputs from the finished finetuning PythonJob."""

        job = self.ctx.finetuning_job

        if not job.is_finished_ok:
            self.report(
                f'Finetuning PythonJob failed with exit status {job.exit_status}'
            )
            return self.exit_codes.ERROR_FINETUNING_FAILED

        self.report('Finetuning completed successfully — collecting outputs')

        self.out('model_path',          orm.Str(job.outputs.model_path.value).store())
        self.out('num_training_samples', orm.Int(job.outputs.num_training_samples.value).store())
        self.out('num_epochs',           orm.Int(job.outputs.num_epochs.value).store())
        self.out('status',               orm.Str(job.outputs.status.value).store())
