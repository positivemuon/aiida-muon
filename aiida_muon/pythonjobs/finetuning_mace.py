"""MACE finetuning via aiida-pythonjob (CLI approach).

The remote function runs::

    mace_run_train --name NAME --foundation_model /path/to/pretrained.model \\
        --train_file train_data.xyz ...

Training data can be supplied as:

* a local XYZ file  (``local_train_xyz``)
* an AiiDA ``TrajectoryData`` node  (``train_trajectorydata``)
* a raw trajectory dict  (``train_trajectory_dict``)

Energy and forces are written using the ``REF_energy`` / ``REF_forces`` keys
(safe with ASE ≥ 3.23.0b1) and matched by ``--energy_key`` / ``--forces_key``.
"""

# ── helper shared with the module ────────────────────────────────────────────

def _atoms_to_ref_xyz_bytes(atoms_list):
    """Serialise *atoms_list* to an extXYZ byte string using safe keys.

    Writes ``info['REF_energy']`` and ``arrays['REF_forces']`` directly instead
    of relying on a ``SinglePointCalculator``, which avoids the ASE ≥ 3.23 warning
    about ``energy`` / ``forces`` namespace collisions with MACE.
    """
    import io
    import numpy as np
    import ase.io
    from ase import Atoms

    safe = []
    for atoms in atoms_list:
        a = atoms.copy()
        # strip calculator to avoid reserved-key collision
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        a.calc = None
        a.info['REF_energy'] = float(e)
        a.arrays['REF_forces'] = np.array(f)
        safe.append(a)

    buf = io.StringIO()
    ase.io.write(buf, safe, format='extxyz')
    return buf.getvalue().encode()


def _trajectorydata_to_ref_xyz_bytes(trajectorydata):
    """Convert an AiiDA ``TrajectoryData`` to a REF_energy/REF_forces XYZ byte string."""
    import io
    import numpy as np
    import ase.io
    from ase import Atoms
    from aiida_muon.utils.trajectory import trajectory_data_to_trajectory_dict

    td = trajectory_data_to_trajectory_dict(trajectorydata)
    positions = np.array(td['positions'])
    cells     = np.array(td['cells'])
    symbols   = td['symbols']
    energies  = np.array(td['energies'])
    forces    = np.array(td['forces'])
    stresses  = np.array(td['stresses']) if 'stresses' in td else None
    pbc       = np.array(td['pbc'], dtype=bool) if 'pbc' in td else True

    atoms_list = []
    for i in range(len(positions)):
        a = Atoms(symbols=symbols, positions=positions[i], cell=cells[i], pbc=pbc)
        a.info['REF_energy'] = float(energies[i])
        a.arrays['REF_forces'] = forces[i]
        if stresses is not None:
            a.info['REF_stress'] = stresses[i]
        atoms_list.append(a)

    buf = io.StringIO()
    ase.io.write(buf, atoms_list, format='extxyz')
    return buf.getvalue().encode()


def _traj_dict_to_ref_xyz_bytes(trajectory_dict):
    """Convert a raw trajectory dict to a REF_energy/REF_forces XYZ byte string."""
    import io
    import numpy as np
    import ase.io
    from ase import Atoms

    positions = np.array(trajectory_dict['positions'])
    cells     = np.array(trajectory_dict['cells'])
    symbols   = trajectory_dict['symbols']
    energies  = np.array(trajectory_dict['energies'])
    forces    = np.array(trajectory_dict['forces'])
    stresses  = np.array(trajectory_dict['stresses']) if 'stresses' in trajectory_dict else None
    pbc       = np.array(trajectory_dict['pbc'], dtype=bool) if 'pbc' in trajectory_dict else True

    atoms_list = []
    for i in range(len(positions)):
        a = Atoms(symbols=symbols, positions=positions[i], cell=cells[i], pbc=pbc)
        a.info['REF_energy'] = float(energies[i])
        a.arrays['REF_forces'] = forces[i]
        if stresses is not None:
            a.info['REF_stress'] = stresses[i]
        atoms_list.append(a)

    buf = io.StringIO()
    ase.io.write(buf, atoms_list, format='extxyz')
    return buf.getvalue().encode()


# ── main public function ──────────────────────────────────────────────────────

def prepare_mace_finetuning_inputs(
    pythonjob_code,
    foundation_model,
    # ── training data (one of these three) ───────────────────────────────────
    local_train_xyz=None,
    train_trajectorydata=None,
    train_trajectory_dict=None,
    # ── optional validation data ──────────────────────────────────────────────
    local_valid_xyz=None,
    valid_trajectorydata=None,
    valid_trajectory_dict=None,
    valid_fraction=0.1,           # used when no validation data is supplied
    # ── model / job identity ──────────────────────────────────────────────────
    name='MACE_finetune',
    model='MACE',
    seed=3,
    # ── loss / weighting ─────────────────────────────────────────────────────
    loss='universal',
    energy_weight=1.0,
    forces_weight=10.0,
    compute_stress=False,
    stress_weight=1.0,
    E0s='average',
    # ── data keys ────────────────────────────────────────────────────────────
    energy_key='REF_energy',
    forces_key='REF_forces',
    stress_key='REF_stress',
    # ── optimiser / training schedule ────────────────────────────────────────
    lr=0.005,
    scaling='rms_forces_scaling',
    batch_size=4,
    valid_batch_size=None,
    max_num_epochs=200,
    patience=2048,
    ema=True,
    ema_decay=0.99,
    amsgrad=True,
    # ── architecture ─────────────────────────────────────────────────────────
    num_interactions=2,
    num_channels=128,
    max_L=1,
    r_max=None,                   # None = use foundation model default
    # ── output / checkpointing ───────────────────────────────────────────────
    default_dtype='float32',
    clip_grad=10,
    device='cuda',
    error_table='PerAtomRMSE',
    keep_checkpoints=False,
    save_all_checkpoints=False,
    restart_latest=False,
    checkpoints_dir='checkpoints',
    model_dir='.',
    save_cpu=False,
    # ── distributed ──────────────────────────────────────────────────────────
    distributed=False,
    # ── epoch-milestone snapshots ────────────────────────────────────────────
    model_epochs=None,
    # ── aiida-pythonjob ───────────────────────────────────────────────────────
    pythonjob_metadata=None,
):
    """Prepare inputs for MACE finetuning via aiida-pythonjob (CLI approach).

    The remote function assembles and executes a ``mace_run_train`` command with
    all supplied arguments.  Training data is uploaded as ``train_data.xyz``
    (and optionally``valid_data.xyz``) using the safe ``REF_energy`` /
    ``REF_forces`` extXYZ keys to avoid the ASE ≥ 3.23 namespace warning.

    Parameters
    ----------
    pythonjob_code : orm.Code
        The AiiDA PythonJob code to use on the compute side.
    foundation_model : str
        Path to the pretrained MACE ``.model`` file **on the compute node**.
    local_train_xyz : str, optional
        Absolute local path to a training extXYZ file.  The file is uploaded
        as ``train_data.xyz``.  Keys ``REF_energy`` / ``REF_forces`` are
        expected; if the file uses the legacy ``energy`` / ``forces`` keys you
        should override ``energy_key`` / ``forces_key`` accordingly.
    train_trajectorydata : orm.TrajectoryData, optional
        AiiDA TrajectoryData node — serialised to XYZ (with REF keys) on the
        submission side.
    train_trajectory_dict : dict, optional
        Raw trajectory dict (output of a relax PythonJob).
    local_valid_xyz : str, optional
        Absolute local path to a validation extXYZ file.
    valid_trajectorydata : orm.TrajectoryData, optional
        Validation frames as an AiiDA TrajectoryData node.
    valid_trajectory_dict : dict, optional
        Validation frames as a raw trajectory dict.
    valid_fraction : float
        Fraction of training data used for validation when no explicit
        validation set is provided.  Ignored when validation data is present.
    name : str
        ``--name`` passed to ``mace_run_train``.  Also forms part of the
        output model filename (``{name}_run-{seed}.model``).
    model : str
        MACE model class (e.g. ``'MACE'``, ``'ScaleShiftMACE'``).
    seed : int
        Random seed (also determines the output filename suffix).
    loss : str
        Loss function (``'universal'``, ``'weighted'``, ``'stress'``, …).
    energy_weight : float
        Weight for energy loss.
    forces_weight : float
        Weight for forces loss.
    compute_stress : bool
        Whether to compute and include stress in the loss.
    stress_weight : float
        Weight for stress loss (only used when ``compute_stress=True``).
    E0s : str or dict
        Atomic reference energies.  ``'average'`` computes them from the data.
    energy_key : str
        Key used for energies in the XYZ file (default ``'REF_energy'``).
    forces_key : str
        Key used for forces in the XYZ file (default ``'REF_forces'``).
    stress_key : str
        Key used for stresses in the XYZ file (default ``'REF_stress'``).
    lr : float
        Learning rate.
    scaling : str
        Output scaling (``'rms_forces_scaling'``, ``'std_scaling'``, …).
    batch_size : int
        Training batch size.
    valid_batch_size : int, optional
        Validation batch size.  Defaults to ``batch_size``.
    max_num_epochs : int
        Maximum number of training epochs.
    patience : int
        Early-stopping patience (epochs without improvement).
    ema : bool
        Use Exponential Moving Average.
    ema_decay : float
        EMA decay parameter.
    amsgrad : bool
        Use AMSGrad variant of Adam.
    num_interactions : int
        Number of interaction layers.
    num_channels : int
        Number of embedding channels.
    max_L : int
        Maximum spherical harmonic order.
    r_max : float, optional
        Radial cutoff in Å.  When *None* the foundation-model value is kept.
    default_dtype : str
        ``'float32'`` or ``'float64'``.
    clip_grad : float
        Gradient clipping value.
    device : str
        ``'cpu'`` or ``'cuda'``.
    error_table : str
        Type of error table printed during training.
    keep_checkpoints : bool
        Keep **all** checkpoint files (not just the latest best).
    save_all_checkpoints : bool
        Save a checkpoint after **every** epoch.
    restart_latest : bool
        Restart training from the latest checkpoint in ``checkpoints_dir``.
    checkpoints_dir : str
        Remote directory for checkpoint ``.pt`` files.
    model_dir : str
        Remote directory for the final ``.model`` file.
    save_cpu : bool
        Save the final model in CPU-compatible format.
    distributed : bool
        Run in multi-GPU data-parallel mode.
    model_epochs : list of int, optional
        When provided, training is run *iteratively* using this list of epoch
        milestones (e.g. ``[10, 50, 100, 500]``).  After each milestone the
        compiled model is copied to
        ``{checkpoints_dir}/{name}_run-{seed}_epoch-{N}.model`` and training
        resumes from the latest checkpoint.  ``max_num_epochs`` is ignored
        in this mode — use the last value in ``model_epochs`` instead.
        Mirrors the pattern from ``mace_train_example.py``.
    pythonjob_metadata : dict, optional
        PythonJob metadata (resources, walltime, …).

    Returns
    -------
    dict
        Dictionary with PythonJob inputs ready to pass to ``submit(PythonJob, **inputs)``.
    """
    import os
    import tempfile
    from aiida_pythonjob import prepare_pythonjob_inputs, spec
    from typing import Any

    # ── validate inputs ───────────────────────────────────────────────────────
    if (local_train_xyz is None
            and train_trajectorydata is None
            and train_trajectory_dict is None):
        raise ValueError(
            'Provide one of: local_train_xyz, train_trajectorydata, or '
            'train_trajectory_dict.'
        )

    if pythonjob_metadata is None:
        pythonjob_metadata = {
            'options': {
                'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
                'max_wallclock_seconds': 12 * 3600,
            }
        }

    # ── build upload_files ────────────────────────────────────────────────────
    upload_files = {}
    _tmp_files = []  # temp files to clean up after prepare_pythonjob_inputs

    def _register(remote_name, local_path=None, content_bytes=None):
        if local_path is not None:
            upload_files[remote_name] = os.path.abspath(local_path)
        elif content_bytes is not None:
            tmp = tempfile.NamedTemporaryFile(
                suffix='.xyz', delete=False, prefix=f'mace_{remote_name}_'
            )
            tmp.write(content_bytes)
            tmp.flush()
            tmp.close()
            upload_files[remote_name] = tmp.name
            _tmp_files.append(tmp.name)

    # Training data
    if local_train_xyz is not None:
        _register('train_data.xyz', local_path=local_train_xyz)
    elif train_trajectorydata is not None:
        _register('train_data.xyz',
                  content_bytes=_trajectorydata_to_ref_xyz_bytes(train_trajectorydata))
    else:
        _register('train_data.xyz',
                  content_bytes=_traj_dict_to_ref_xyz_bytes(train_trajectory_dict))

    # Validation data (optional)
    has_valid = False
    if local_valid_xyz is not None:
        _register('valid_data.xyz', local_path=local_valid_xyz)
        has_valid = True
    elif valid_trajectorydata is not None:
        _register('valid_data.xyz',
                  content_bytes=_trajectorydata_to_ref_xyz_bytes(valid_trajectorydata))
        has_valid = True
    elif valid_trajectory_dict is not None:
        _register('valid_data.xyz',
                  content_bytes=_traj_dict_to_ref_xyz_bytes(valid_trajectory_dict))
        has_valid = True

    # ── build function_inputs dict ────────────────────────────────────────────
    if valid_batch_size is None:
        valid_batch_size = batch_size

    function_inputs = dict(
        foundation_model=foundation_model,
        name=name,
        model=model,
        seed=seed,
        loss=loss,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        compute_stress=compute_stress,
        stress_weight=stress_weight,
        E0s=E0s,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        lr=lr,
        scaling=scaling,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        max_num_epochs=max_num_epochs,
        patience=patience,
        ema=ema,
        ema_decay=ema_decay,
        amsgrad=amsgrad,
        num_interactions=num_interactions,
        num_channels=num_channels,
        max_L=max_L,
        r_max=r_max,
        default_dtype=default_dtype,
        clip_grad=clip_grad,
        device=device,
        error_table=error_table,
        keep_checkpoints=keep_checkpoints,
        save_all_checkpoints=save_all_checkpoints,
        restart_latest=restart_latest,
        checkpoints_dir=checkpoints_dir,
        model_dir=model_dir,
        save_cpu=save_cpu,
        distributed=distributed,
        has_valid=has_valid,
        valid_fraction=valid_fraction,
        model_epochs=model_epochs,
    )

    # ── inner function — runs on the compute node ─────────────────────────────
    def finetune_function(
        foundation_model,
        name,
        model,
        seed,
        loss,
        energy_weight,
        forces_weight,
        compute_stress,
        stress_weight,
        E0s,
        energy_key,
        forces_key,
        stress_key,
        lr,
        scaling,
        batch_size,
        valid_batch_size,
        max_num_epochs,
        patience,
        ema,
        ema_decay,
        amsgrad,
        num_interactions,
        num_channels,
        max_L,
        r_max,
        default_dtype,
        clip_grad,
        device,
        error_table,
        keep_checkpoints,
        save_all_checkpoints,
        restart_latest,
        checkpoints_dir,
        model_dir,
        save_cpu,
        distributed,
        has_valid,
        valid_fraction,
        model_epochs,
    ):
        """Run ``mace_run_train`` on the remote node."""
        import os
        import subprocess
        import sys

        cmd = [
            'mace_run_train',
            f'--name={name}',
            f'--foundation_model={foundation_model}',
            f'--model={model}',
            '--train_file=train_data.xyz',
            f'--energy_key={energy_key}',
            f'--forces_key={forces_key}',
            f'--E0s={E0s}',
            f'--loss={loss}',
            f'--energy_weight={energy_weight}',
            f'--forces_weight={forces_weight}',
            f'--lr={lr}',
            f'--scaling={scaling}',
            f'--batch_size={batch_size}',
            f'--valid_batch_size={valid_batch_size}',
            f'--max_num_epochs={max_num_epochs}',
            f'--patience={patience}',
            f'--ema_decay={ema_decay}',
            f'--default_dtype={default_dtype}',
            f'--clip_grad={clip_grad}',
            f'--device={device}',
            f'--error_table={error_table}',
            f'--seed={seed}',
            f'--checkpoints_dir={checkpoints_dir}',
            f'--model_dir={model_dir}',
            f'--num_interactions={num_interactions}',
            f'--num_channels={num_channels}',
            f'--max_L={max_L}',
        ]

        # Boolean flags
        if ema:
            cmd.append('--ema')
        if amsgrad:
            cmd.append('--amsgrad')
        if compute_stress:
            cmd += [
                '--compute_stress=True',
                f'--stress_weight={stress_weight}',
                f'--stress_key={stress_key}',
            ]
        if keep_checkpoints:
            cmd.append('--keep_checkpoints')
        if save_all_checkpoints:
            cmd.append('--save_all_checkpoints')
        if restart_latest:
            cmd.append('--restart_latest')
        if save_cpu:
            cmd.append('--save_cpu')
        if distributed:
            cmd.append('--distributed')

        if r_max is not None:
            cmd.append(f'--r_max={r_max}')

        # Validation data
        if has_valid:
            cmd.append('--valid_file=valid_data.xyz')
        else:
            cmd.append(f'--valid_fraction={valid_fraction}')

        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        model_filename = f'{name}_run-{seed}.model'
        model_path = os.path.abspath(os.path.join(model_dir, model_filename))

        def _run(epochs):
            """Run mace_run_train for a given max_num_epochs."""
            epoch_cmd = cmd.copy()
            # replace the max_num_epochs flag with the current milestone
            epoch_cmd = [c for c in epoch_cmd if not c.startswith('--max_num_epochs=')]
            epoch_cmd.append(f'--max_num_epochs={epochs}')
            sys.stdout.flush()
            result = subprocess.run(epoch_cmd, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f'mace_run_train failed at epoch {epochs} (exit {result.returncode}):\n'
                    f'STDERR:\n{result.stderr}'
                )

        if model_epochs:
            # ── iterative milestone mode ──────────────────────────────────
            # Always restart from latest checkpoint after the first run.
            # The first run uses --restart_latest only if the caller asked for it.
            first_run = True
            for epoch_target in sorted(model_epochs):
                if not first_run and '--restart_latest' not in cmd:
                    cmd.append('--restart_latest')
                _run(epoch_target)
                first_run = False
                # Copy compiled model to a milestone snapshot
                snapshot = os.path.join(
                    checkpoints_dir,
                    f'{name}_run-{seed}_epoch-{epoch_target}.model',
                )
                import shutil
                if os.path.isfile(model_path):
                    shutil.copy2(model_path, snapshot)
                    print(f'Saved snapshot: {snapshot}')
        else:
            # ── single run mode (original behaviour) ─────────────────────
            _run(max_num_epochs)

        return {
            'model_path': model_path,
            'checkpoints_dir': os.path.abspath(checkpoints_dir),
            'status': 'completed',
        }

    outputs_spec = spec.namespace(
        model_path=Any,
        checkpoints_dir=Any,
        status=Any,
    )

    pythonjob_inputs = prepare_pythonjob_inputs(
        function=finetune_function,
        function_inputs=function_inputs,
        outputs_spec=outputs_spec,
        register_pickle_by_value=True,
        code=pythonjob_code,
        metadata=pythonjob_metadata,
        upload_files=upload_files,
    )

    # ── cleanup local temp files ──────────────────────────────────────────────
    import shutil
    for f in _tmp_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    return pythonjob_inputs
