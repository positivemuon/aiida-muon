def prepare_mattersim_finetuning_inputs(
    load_model_path,
    pythonjob_code,
    train_data_path=None,
    trajectory_dict=None,
    trajectorydata=None,
    save_path="./finetuned_model",
    epochs=100,
    batch_size=4,
    lr=2e-4,
    device="cpu",
    include_forces=True,
    include_stresses=False,
    force_loss_ratio=1.0,
    stress_loss_ratio=0.1,
    seed=42,
    pythonjob_metadata=None,
):
    """
    Prepare inputs for MatterSim finetuning via aiida-pythonjob.
    
    This follows the pattern from:
    - aiida_muon.pythonjobs.relax.prepare_ase_pythonjob_relaxation_inputs
    - aiida_impuritysupercellconv.pythonjobs.forces.prepare_ase_pythonjob_forces_inputs
    
    Parameters
    ----------
    load_model_path : str
        Path to pretrained MatterSim checkpoint
    pythonjob_code : orm.Code
        The PythonJob code to use (listed here for clarity; also a positional arg)
    train_data_path : str, optional
        Path to training data file (.xyz or .pkl).  Either this or ``trajectory``
        must be provided.
    trajectory : list of ASE Atoms, optional
        Training frames passed directly — typically the ``trajectory`` output
        of a ``relax.py`` PythonJob (``node.outputs.trajectory.value``).
        Each ``Atoms`` must have a ``SinglePointCalculator`` attached with
        ``energy`` and ``forces`` in ``atoms.calc.results``.
        Either this or ``train_data_path`` must be provided.
    pythonjob_code : orm.Code
        The PythonJob code to use
    save_path : str
        Directory to save finetuned model
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    lr : float
        Learning rate
    device : str
        'cpu' or 'cuda'
    include_forces : bool
        Include forces in loss
    include_stresses : bool
        Include stresses in loss
    force_loss_ratio : float
        Weight for force loss
    stress_loss_ratio : float
        Weight for stress loss
    seed : int
        Random seed
    pythonjob_metadata : dict, optional
        Metadata for PythonJob
    
    Returns
    -------
    dict
        Dictionary with PythonJob inputs ready for submission
    """
    from aiida_pythonjob import prepare_pythonjob_inputs, spec
    from typing import Any
    
    # Default metadata
    if pythonjob_metadata is None:
        pythonjob_metadata = {
            'options': {
                'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
                'max_wallclock_seconds': 7200,  # 2 hours
            }
        }
    
    if train_data_path is None and trajectory_dict is None and trajectorydata is None:
        raise ValueError(
            'Either train_data_path or trajectory or trajectorydata must be provided to '
            'prepare_mattersim_finetuning_inputs.'
        )

    if trajectorydata is not None:
        from aiida_muon.utils.trajectory import trajectory_data_to_trajectory_dict
        trajectory_dict = trajectory_data_to_trajectory_dict(trajectorydata)

    # Prepare function inputs
    function_inputs = {
        'train_data_path': train_data_path,
        'trajectory_dict': trajectory_dict,
        'load_model_path': load_model_path,
        'save_path': save_path,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'device': device,
        'include_forces': include_forces,
        'include_stresses': include_stresses,
        'force_loss_ratio': force_loss_ratio,
        'stress_loss_ratio': stress_loss_ratio,
        'seed': seed,
    }

    if train_data_path is None:
        function_inputs.pop("train_data_path", None)
    if trajectory_dict is None:
        function_inputs.pop("trajectory_dict", None)
    
    # Define the finetuning function (must be defined here to be pickled)
    def finetune_function(
        load_model_path,
        save_path,
        epochs,
        batch_size,
        lr,
        device,
        include_forces,
        include_stresses,
        force_loss_ratio,
        stress_loss_ratio,
        seed,
        train_data_path=None,
        trajectory_dict=None,    
    ):
        """Wrapper function for MatterSim finetuning.

        Accepts training data either as a file path (``train_data_path``) or
        as the ``trajectory`` list of ASE ``Atoms`` objects (typically the
        ``trajectory`` output of a ``relax.py`` PythonJob).  Each frame must
        have a ``SinglePointCalculator`` attached with energy and forces.
        """
        import os
        import random
        import pickle as pkl
        import sys
        
        import numpy as np
        import torch
        from ase.units import GPa
        
        # Workaround for mattersim.__version__
        import types
        version_module = types.ModuleType('mattersim.__version__')
        version_module.__version__ = "1.0.0"
        sys.modules['mattersim.__version__'] = version_module
        
        from mattersim.datasets.utils.build import build_dataloader
        from mattersim.forcefield.potential import Potential
        from mattersim.utils.atoms_utils import AtomsAdaptor
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # ── Load training data ────────────────────────────────────────────────
        if trajectory_dict is not None:
            from aiida_muon.utils.trajectory import trajectory_dict_to_atoms_list
            atoms_train = trajectory_dict_to_atoms_list(trajectory_dict)
        else:
            atoms_train = AtomsAdaptor.from_file(filename=train_data_path)
        
        # Extract labels
        energies = [atoms.get_potential_energy() for atoms in atoms_train]
        forces = [atoms.get_forces() for atoms in atoms_train] if include_forces else None
        stresses = [atoms.get_stress(voigt=False) / GPa for atoms in atoms_train] if include_stresses else None
        
        # Build dataloader
        dataloader = build_dataloader(
            atoms_train, energies, forces, stresses,
            batch_size=batch_size, shuffle=True,
            pin_memory=(device == "cuda"), is_distributed=False,
        )
        
        # Load and finetune model
        potential = Potential.from_checkpoint(
            load_path=load_model_path,
            load_training_state=False,
            device=device,
        )
        
        potential.train_model(
            dataloader, val_dataloader=None,
            loss=torch.nn.HuberLoss(delta=0.01),
            is_distributed=False,
            epochs=epochs, lr=lr,
            force_loss_ratio=force_loss_ratio,
            stress_loss_ratio=stress_loss_ratio,
            save_path=save_path,
            save_checkpoint=True,
            device=device,
        )
        
        return {
            "model_path": os.path.join(save_path, "best_model.pth"),
            "num_training_samples": len(atoms_train),
            "num_epochs": epochs,
            "status": "completed",
        }
    
    # Prepare outputs specification
    outputs_spec = spec.namespace(
        model_path=Any, num_training_samples=Any,
        num_epochs=Any, status=Any,
    )
    
    # Prepare complete pythonjob inputs
    pythonjob_inputs = prepare_pythonjob_inputs(
        function=finetune_function,
        function_inputs=function_inputs,
        outputs_spec=outputs_spec,
        register_pickle_by_value=True,
        code=pythonjob_code,
        metadata=pythonjob_metadata,
    )
    
    return pythonjob_inputs


'''Example usage:

# Complete workflow example
def run_finetuning_workflow():
    """Complete example of MatterSim finetuning with PythonJob."""
    
    # 1. Prepare training data
    train_file = "/home/jovyan/bind_mount/work/MLIPs_PROJECT/notebooks_clustering_ft_pythonjobs/train_data.xyz"
    
    # 2. Load pythonjob code
    pythonjob_code = orm.load_code('python3_mattersim_p311@localhost')  # Adjust to your code label
    
    # 3. Prepare inputs using helper function
    inputs = prepare_mattersim_finetuning_inputs(
        train_data_path=train_file,
        load_model_path="/home/jovyan/bind_mount/codes/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth",
        pythonjob_code=pythonjob_code,
        save_path="/home/jovyan/bind_mount/work/MLIPs_PROJECT/notebooks_clustering_ft_pythonjobs/finetuned_model",
        epochs=10,
        batch_size=2,
        lr=2e-4,
        device='cpu',
        include_forces=True,
        include_stresses=False,
    )
    
    # 4. Submit job
    node = submit(PythonJob, **inputs)
    print(f"Submitted finetuning job: PK={node.pk}")
    
    return node

# Uncomment to run:
# node = run_finetuning_workflow()
'''