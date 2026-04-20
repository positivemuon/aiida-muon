def prepare_metatrain_finetuning_inputs(
    pythonjob_code,
    options,
    train_data_path=None,
    local_xyz_path=None,
    trajectory_dict=None,
    trajectorydata=None,
    output="model.pt",
    checkpoint_dir=".",
    extensions="extensions/",
    restart_from=None,
    pythonjob_metadata=None,
):
    """
    Prepare inputs for metatrain finetuning via aiida-pythonjob.

    Metatrain is configured entirely through an ``options`` dict (equivalent to a
    YAML options file).  The ``architecture.training.finetune.read_from`` key inside
    ``options`` must point to the pretrained checkpoint on the compute node.

    Training data can be supplied in three ways:

    | Option | Parameter            | When to use                                          |
    |--------|----------------------|------------------------------------------------------|
    | A      | ``train_data_path``  | XYZ / ASE-readable file already on the compute node |
    | B      | ``trajectory_dict``  | Raw dict from a relax PythonJob output               |
    | C      | ``trajectorydata``   | AiiDA ``TrajectoryData`` node                        |

    For options B and C the training frames are serialised into an XYZ file on the
    fly on the remote node.  If ``options`` does not yet contain
    ``training_set[0].systems.read_from`` / ``targets.energy.read_from`` they will
    be set to the temporary file path automatically.

    Parameters
    ----------
    pythonjob_code : orm.Code
        The PythonJob code to use.
    options : dict
        metatrain options dictionary (equivalent to ``options.yaml``).  Must contain
        at minimum ``architecture.name``.  For finetuning also set
        ``architecture.training.finetune.read_from`` to the checkpoint path.
    train_data_path : str, optional
        Path to training data on the compute node (.xyz or any ASE-readable format).
    local_xyz_path : str, optional
        Absolute path to a local XYZ/extXYZ file to upload to the remote
        working directory via ``upload_files``.  When provided the file is
        uploaded as ``"train_data.xyz"`` and ``train_data_path`` is
        automatically set to that name on the remote side.
    trajectory_dict : dict, optional
        Raw trajectory dict (keys: positions, cells, symbols, energies, forces, …).
        Serialised to a temporary XYZ file on the remote node.
    trajectorydata : orm.TrajectoryData, optional
        AiiDA TrajectoryData node; converted server-side before being written.
    output : str
        Filename (on the compute node) for the exported ``.pt`` model.
    checkpoint_dir : str
        Directory for intermediate checkpoints on the compute node.
    extensions : str
        Directory for model extensions on the compute node.
    restart_from : str, optional
        Path to a ``.ckpt`` file to restart interrupted training.
    pythonjob_metadata : dict, optional
        Metadata for PythonJob (resources, walltime, etc.).

    Returns
    -------
    dict
        Dictionary with PythonJob inputs ready for submission.
    """
    from aiida_pythonjob import prepare_pythonjob_inputs, spec
    from typing import Any

    # Default metadata
    if pythonjob_metadata is None:
        pythonjob_metadata = {
            "options": {
                "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
                "max_wallclock_seconds": 7200,
            }
        }

    if local_xyz_path is not None:
        import os
        train_data_path = 'train_data.xyz'
        local_xyz_path = os.path.abspath(local_xyz_path)

    if train_data_path is None and trajectory_dict is None and trajectorydata is None:
        raise ValueError(
            "Either train_data_path, local_xyz_path, trajectory_dict, or trajectorydata must be "
            "provided to prepare_metatrain_finetuning_inputs."
        )

    if trajectorydata is not None:
        from aiida_muon.utils.trajectory import trajectory_data_to_trajectory_dict
        trajectory_dict = trajectory_data_to_trajectory_dict(trajectorydata)

    function_inputs = {
        "options": options,
        "train_data_path": train_data_path,
        "trajectory_dict": trajectory_dict,
        "output": output,
        "checkpoint_dir": checkpoint_dir,
        "extensions": extensions,
        "restart_from": restart_from,
    }

    if train_data_path is None:
        function_inputs.pop("train_data_path", None)
    if trajectory_dict is None:
        function_inputs.pop("trajectory_dict", None)

    # ── Inner function — runs on the compute node ─────────────────────────────
    def finetune_function(
        options,
        output,
        checkpoint_dir,
        extensions,
        restart_from,
        train_data_path=None,
        trajectory_dict=None,
    ):
        """Run metatrain `train_model` on the remote node.

        Training data is accepted either as a path (``train_data_path``, must be
        accessible on the compute node) or as a raw trajectory dict
        (``trajectory_dict``), which is written to a temporary XYZ file before
        training starts.
        """
        import os
        import tempfile

        import numpy as np
        from omegaconf import OmegaConf

        # ── Resolve training data path ─────────────────────────────────────
        _tmp_dir = None
        if trajectory_dict is not None:
            # Inline version of trajectory_dict_to_atoms_list to avoid aiida_muon
            # on the remote node (only ase / numpy needed).
            from ase import Atoms
            from ase.calculators.singlepoint import SinglePointCalculator
            import ase.io

            _positions = np.array(trajectory_dict["positions"])
            _cells     = np.array(trajectory_dict["cells"])
            _symbols   = trajectory_dict["symbols"]
            _energies  = np.array(trajectory_dict["energies"])
            _forces    = np.array(trajectory_dict["forces"])
            _stresses  = (
                np.array(trajectory_dict["stresses"])
                if "stresses" in trajectory_dict
                else None
            )
            _pbc = (
                np.array(trajectory_dict["pbc"], dtype=bool)
                if "pbc" in trajectory_dict
                else True
            )

            atoms_list = []
            for _i in range(len(_positions)):
                _atoms = Atoms(
                    symbols=_symbols,
                    positions=_positions[_i],
                    cell=_cells[_i],
                    pbc=_pbc,
                )
                _calc_kw = {"energy": float(_energies[_i]), "forces": _forces[_i]}
                if _stresses is not None:
                    _calc_kw["stress"] = _stresses[_i]
                _atoms.calc = SinglePointCalculator(_atoms, **_calc_kw)
                atoms_list.append(_atoms)

            _tmp_dir = tempfile.mkdtemp(prefix="metatrain_train_")
            train_data_path = os.path.join(_tmp_dir, "train_data.xyz")
            ase.io.write(train_data_path, atoms_list)

        # ── Patch options with training data path if not already set ──────
        cfg = OmegaConf.create(options)

        # Auto-fill training_set if the user left a placeholder or omitted it
        training_set = OmegaConf.to_container(cfg.get("training_set", []), resolve=True)
        if not training_set:
            training_set = [
                {
                    "systems": {
                        "read_from": train_data_path,
                        "length_unit": "angstrom",
                    },
                    "targets": {
                        "energy": {
                            "read_from": train_data_path,
                            "key": "energy",
                            "unit": "eV",
                            "forces": {
                                "read_from": train_data_path,
                                "key": "forces",
                            },
                        }
                    },
                }
            ]
            cfg = OmegaConf.merge(cfg, {"training_set": training_set})
        elif train_data_path is not None:
            # Replace any placeholder "__TRAIN_DATA__" strings
            cfg_str = OmegaConf.to_yaml(cfg).replace("__TRAIN_DATA__", train_data_path)
            cfg = OmegaConf.create(cfg_str)

        # ── Run metatrain ─────────────────────────────────────────────────
        import metatrain.cli.train as _mt_train
        from metatrain.utils.io import load_model  # noqa: F401 – side-effect import

        _mt_train.train_model(
            options=cfg,
            output=output,
            extensions=extensions,
            checkpoint_dir=checkpoint_dir,
            restart_from=restart_from,
        )

        # ── Cleanup ───────────────────────────────────────────────────────
        if _tmp_dir is not None:
            import shutil
            shutil.rmtree(_tmp_dir, ignore_errors=True)

        return {
            "model_path": os.path.abspath(output),
            "checkpoint_dir": os.path.abspath(checkpoint_dir),
            "status": "completed",
        }

    outputs_spec = spec.namespace(
        model_path=Any,
        checkpoint_dir=Any,
        status=Any,
    )

    upload_files = {}
    if local_xyz_path is not None:
        upload_files['train_data.xyz'] = local_xyz_path

    pythonjob_inputs = prepare_pythonjob_inputs(
        function=finetune_function,
        function_inputs=function_inputs,
        outputs_spec=outputs_spec,
        register_pickle_by_value=True,
        code=pythonjob_code,
        metadata=pythonjob_metadata,
        **(dict(upload_files=upload_files) if upload_files else {}),
    )

    return pythonjob_inputs


# ── YAML-template / CLI-based approach ────────────────────────────────────────

_METATRAIN_YAML_TEMPLATE = """\
seed: {seed}

device: "{device}"

architecture:
  name: "{arch_name}"
  training:
    batch_size: {batch_size}
    num_epochs: {num_epochs}
    learning_rate: {learning_rate}
    finetune:
      method: "{finetune_method}"
      read_from: {checkpoint}
{config_extensions}training_set:
  systems:
    read_from: "train_data.xyz"
    length_unit: angstrom
  targets:
    energy:
      key: "energy"
      unit: "eV"
      forces:
        read_from: "train_data.xyz"
        reader: ase
        key: forces
{validation_block}{test_block}"""

_VALIDATION_BLOCK = """\

validation_set:
  systems:
    read_from: "validation_data.xyz"
    length_unit: angstrom
  targets:
    energy:
      key: "energy"
      unit: "eV"
      forces:
        read_from: "validation_data.xyz"
        reader: ase
        key: forces
"""

_TEST_BLOCK = """\

test_set:
  systems:
    read_from: "test_data.xyz"
    length_unit: angstrom
  targets:
    energy:
      key: "energy"
      unit: "eV"
      forces:
        read_from: "test_data.xyz"
        reader: ase
        key: forces
"""


def _trajectorydata_to_xyz_bytes(trajectorydata):
    """Convert an AiiDA TrajectoryData node to an in-memory XYZ byte string."""
    import io
    import numpy as np
    import ase.io
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from aiida_muon.utils.trajectory import trajectory_data_to_trajectory_dict

    td = trajectory_data_to_trajectory_dict(trajectorydata)
    positions = np.array(td["positions"])
    cells     = np.array(td["cells"])
    symbols   = td["symbols"]
    energies  = np.array(td["energies"])
    forces    = np.array(td["forces"])
    stresses  = np.array(td["stresses"]) if "stresses" in td else None
    pbc       = np.array(td["pbc"], dtype=bool) if "pbc" in td else True

    atoms_list = []
    for i in range(len(positions)):
        atoms = Atoms(symbols=symbols, positions=positions[i], cell=cells[i], pbc=pbc)
        calc_kw = {"energy": float(energies[i]), "forces": forces[i]}
        if stresses is not None:
            calc_kw["stress"] = stresses[i]
        atoms.calc = SinglePointCalculator(atoms, **calc_kw)
        atoms_list.append(atoms)

    buf = io.StringIO()
    ase.io.write(buf, atoms_list, format="extxyz")
    return buf.getvalue().encode()


def prepare_metatrain_finetuning_inputs_cli(
    pythonjob_code,
    checkpoint,
    train_trajectorydata=None,
    validation_trajectorydata=None,
    test_trajectorydata=None,
    local_train_xyz=None,
    local_validation_xyz=None,
    local_test_xyz=None,
    arch_name="pet",
    num_epochs=1000,
    batch_size=5,
    learning_rate=2e-5,
    finetune_method="full",
    finetune_config=None,
    device="cpu",
    seed=42,
    output="finetuned",
    pythonjob_metadata=None,
):
    """
    Prepare inputs for metatrain finetuning via ``mtt train options.yaml -o <output>``.

    A ``options.yaml`` is generated from the supplied parameters and uploaded to
    the remote working directory together with the training (and optionally
    validation / test) XYZ files.  The remote function simply calls::

        mtt train options.yaml -o <output>

    Training data can be supplied as:

    * an AiiDA ``TrajectoryData`` node (``*_trajectorydata`` parameters) — the
      node is serialised to an XYZ byte string on the **submission side** and
      uploaded, or
    * a local path to an existing XYZ file (``local_*_xyz`` parameters).

    NB if not validation / test data is provided, the corresponding blocks are automatically
    set to 0.1 (i.e. we take from the training data) and the YAML is patched accordingly.

    Parameters
    ----------
    pythonjob_code : orm.Code
        The PythonJob code to use.
    checkpoint : str
        Path to the pretrained checkpoint on the **compute node**
        (``architecture.training.finetune.read_from``).
    train_trajectorydata : orm.TrajectoryData, optional
        Training frames as AiiDA TrajectoryData.
    validation_trajectorydata : orm.TrajectoryData, optional
        Validation frames as AiiDA TrajectoryData.
    test_trajectorydata : orm.TrajectoryData, optional
        Test frames as AiiDA TrajectoryData.
    local_train_xyz : str, optional
        Absolute local path to a training XYZ file (alternative to TrajectoryData).
    local_validation_xyz : str, optional
        Absolute local path to a validation XYZ file.
    local_test_xyz : str, optional
        Absolute local path to a test XYZ file.
    arch_name : str
        metatrain architecture name (e.g. ``"pet"``, ``"soap-bpnn"``).
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    learning_rate : float
        Learning rate.
    finetune_method : str
        Finetuning strategy (e.g. ``"full"``, ``"heads"``).
    finetune_config : dict, optional
        Extra keys to place under ``architecture.training.finetune.config``
        in the generated YAML.  For example::

            finetune_config={
                "head_modules": ["node_heads", "edge_heads"],
                "last_layer_modules": ["node_last_layers", "edge_last_layers"],
            }
    device : str
        ``"cpu"`` or ``"cuda"``.
    seed : int
        Random seed.
    output : str
        Output name passed to ``mtt train … -o <output>``.
    pythonjob_metadata : dict, optional
        Metadata for PythonJob (resources, walltime, …).

    Returns
    -------
    dict
        Dictionary with PythonJob inputs ready for submission.
    """
    import io
    import os
    import tempfile
    from aiida_pythonjob import prepare_pythonjob_inputs, spec
    from typing import Any

    if train_trajectorydata is None and local_train_xyz is None:
        raise ValueError(
            "Either train_trajectorydata or local_train_xyz must be provided."
        )

    if pythonjob_metadata is None:
        pythonjob_metadata = {
            "options": {
                "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
                "max_wallclock_seconds": 7200,
            }
        }

    # ── Build upload_files dict ───────────────────────────────────────────────
    upload_files = {}
    _tmp_files = []  # local temp files to clean up after prepare_pythonjob_inputs

    def _add_xyz(label, trajectorydata, local_xyz):
        """Write data to a temp file and register for upload."""
        if local_xyz is not None:
            upload_files[label] = os.path.abspath(local_xyz)
        elif trajectorydata is not None:
            content = _trajectorydata_to_xyz_bytes(trajectorydata)
            tmp = tempfile.NamedTemporaryFile(
                suffix=".xyz", delete=False, prefix=f"mtt_{label}_"
            )
            tmp.write(content)
            tmp.flush()
            tmp.close()
            upload_files[label] = tmp.name
            _tmp_files.append(tmp.name)

    _add_xyz("train_data.xyz", train_trajectorydata, local_train_xyz)
    
    if validation_trajectorydata and not local_validation_xyz:
        _add_xyz("validation_data.xyz", validation_trajectorydata, local_validation_xyz)
    
    if test_trajectorydata and not local_test_xyz:
         _add_xyz("test_data.xyz", test_trajectorydata, local_test_xyz)

    # ── Build options.yaml from template ─────────────────────────────────────
    validation_block = _VALIDATION_BLOCK if "validation_data.xyz" in upload_files else "validation_set: 0.1\n"
    test_block = _TEST_BLOCK if "test_data.xyz" in upload_files else "test_set: 0.1\n"

    # ── Build config extensions block (under finetune) ────────────────────
    if finetune_config:
        import yaml as _yaml
        _cfg_yaml = _yaml.dump({"config": finetune_config}, default_flow_style=None)
        config_extensions = "\n".join(
            "      " + line for line in _cfg_yaml.splitlines()
        ) + "\n"
    else:
        config_extensions = ""

    yaml_content = _METATRAIN_YAML_TEMPLATE.format(
        seed=seed,
        device=device,
        arch_name=arch_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        finetune_method=finetune_method,
        checkpoint=checkpoint,
        config_extensions=config_extensions,
        validation_block=validation_block,
        test_block=test_block,
    )

    _tmp_yaml_dir = tempfile.mkdtemp(prefix="mtt_yaml_")
    _tmp_yaml_path = os.path.join(_tmp_yaml_dir, "options.yaml")
    with open(_tmp_yaml_path, "w") as _f:
        _f.write(yaml_content)
    upload_files["options.yaml"] = _tmp_yaml_path
    _tmp_files.append(_tmp_yaml_dir)

    # ── Inner function — runs on the compute node ─────────────────────────────
    def finetune_function_cli(output):
        """Run ``mtt train options.yaml -o <output>`` on the remote node."""
        import os
        import subprocess
        import sys

        # Flush Python stdout so prior output appears before mtt's output
        sys.stdout.flush()

        # Let stdout flow directly to the parent process (aiida.out)
        # while capturing stderr for error reporting.
        result = subprocess.run(
            ["mtt", "train", "options.yaml", "-o", output],
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mtt train failed (exit {result.returncode}):\n"
                f"STDERR:\n{result.stderr}"
            )

        return {
            "model_path": os.path.abspath(f"{output}.pt"),
            "checkpoint_dir": os.path.abspath(output),
            "status": "completed",
        }

    outputs_spec = spec.namespace(
        model_path=Any,
        checkpoint_dir=Any,
        status=Any,
    )

    pythonjob_inputs = prepare_pythonjob_inputs(
        function=finetune_function_cli,
        function_inputs={"output": output},
        outputs_spec=outputs_spec,
        register_pickle_by_value=True,
        code=pythonjob_code,
        metadata=pythonjob_metadata,
        upload_files=upload_files,
    )

    # ── Cleanup local temp files ──────────────────────────────────────────────
    import shutil as _shutil
    for f in _tmp_files:
        try:
            if os.path.isdir(f):
                _shutil.rmtree(f, ignore_errors=True)
            else:
                os.unlink(f)
        except OSError:
            pass

    return pythonjob_inputs
