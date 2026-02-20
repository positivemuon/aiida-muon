# -*- coding: utf-8 -*-
"""Functions for relaxation using ASE optimizers and calculators via aiida-pythonjob."""

from typing import Callable, Union
from ase import Atoms
from aiida import orm

def optimize_structure(atoms, calculator, fmax=1e-4, optimizer='BFGS', trajectory=None, 
                      optimizer_kwargs=None, fix_symmetry=False):
    """Optimize an ASE Atoms structure using the specified calculator and optimizer.
    
    This function works with any ASE-compatible calculator (MLIPs, EMT, GPAW, etc.)
    and supports various optimization algorithms.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to optimize.
    calculator : ase.calculators.calculator.Calculator or callable
        The ASE calculator to use for forces and energy, or a function that returns one.
    fmax : float, optional
        Force convergence criterion in Ry/bohr units (converted internally to eV/Å).
        Default is 1e-4 Ry/bohr (the value used in QE moderate protocol for muons).
    optimizer : str, optional
        The optimizer to use. Options: 'BFGS', 'LBFGS', 'FIRE', 'MDMin', 'BFGS_LineSearch'.
        Default is 'BFGS'.
    trajectory : str, optional
        Path to save the trajectory file. If None, no trajectory is saved.
    optimizer_kwargs : dict, optional
        Additional keyword arguments to pass to the optimizer.
    fix_symmetry : bool, optional
        If True, apply FixSymmetry constraint to preserve space group symmetry.
        Default is False.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'structure': Optimized ASE Atoms object (calculator removed for pickling)
        - 'energy': Final energy in eV
        - 'forces': Final forces array in eV/Å
        - 'nsteps': Number of optimization steps
    """
    from ase.optimize import BFGS, LBFGS, FIRE, MDMin
    from ase.optimize.bfgslinesearch import BFGSLineSearch
    from ase.constraints import FixSymmetry
    from ase import units

    
    # QE forces are in atomic units (Ry/bohr), convert to eV/Å
    QE_to_ASE_force_units = units.Ry / units.Bohr
    fmax_ase = fmax * QE_to_ASE_force_units
    
    # Handle calculator (could be a callable or instance)
    if callable(calculator) and not hasattr(calculator, 'calculate'):
        calculator = calculator()
    
    # Set the calculator
    atoms.calc = calculator
    
    # Apply symmetry constraint if requested
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))
    
    # Map optimizer names to classes
    optimizer_map = {
        'BFGS': BFGS,
        'LBFGS': LBFGS,
        'FIRE': FIRE,
        'MDMIN': MDMin,
        'BFGS_LINESEARCH': BFGSLineSearch,
    }
    
    optimizer_class = optimizer_map.get(optimizer.upper())
    if optimizer_class is None:
        available = ', '.join(optimizer_map.keys())
        raise ValueError(
            f"Optimizer '{optimizer}' not supported. "
            f"Available options: {available}"
        )
    
    # Prepare optimizer kwargs
    opt_kwargs = {'trajectory': trajectory} if trajectory else {}
    if optimizer_kwargs:
        opt_kwargs.update(optimizer_kwargs)
    
    # Initialize optimizer
    dyn = optimizer_class(atoms, **opt_kwargs)
    
    # Run optimization
    dyn.run(fmax=fmax_ase)
    
    # Extract results before removing calculator
    final_energy = atoms.get_potential_energy()
    final_forces = atoms.get_forces()
    nsteps = dyn.get_number_of_steps()
    
    # Remove calculator from atoms to allow pickling (some calculators aren't picklable)
    atoms.calc = None
    
    # Collect results
    result = {
        'structure': atoms,
        'energy': final_energy,
        'forces': final_forces,
        'nsteps': nsteps,
    }
    
    return result

def relax_with_mace(atoms, fmax=1e-4, optimizer='BFGS', trajectory=None, 
                   optimizer_kwargs=None, fix_symmetry=False):
    """Convenience function for ASE-based relaxation that returns the optimized structure.
    
    This is the main function to be called from aiida-pythonjob for structure relaxations
    using any ASE calculator (MLIPs, DFT codes, empirical potentials, etc.).
    """

    from mace.calculators import mace_mp
    mace_mp(
            model="medium",      # Options: small, medium, large
            device="cpu",        # Use 'cuda' if GPU available
            default_dtype="float64",
            dispersion=False     # Set True to include dispersion corrections
        ) 

    result = optimize_structure(
        atoms, calculator=mace_mp, fmax=fmax, optimizer=optimizer, 
        trajectory=trajectory, optimizer_kwargs=optimizer_kwargs,
        fix_symmetry=fix_symmetry
    )
    return {
        "structure": result['structure'], 
        "energy": result['energy'], 
        "forces": result['forces'], 
        "nsteps": result['nsteps']
    }


def prepare_ase_pythonjob_relaxation_inputs(
    structure: Union[orm.StructureData, Atoms],
    relax_function=relax_with_mace,
    fmax=1e-4,
    optimizer='BFGS',
    fix_symmetry=True,
    pythonjob_inputs=None,
    trajectory=None,
):
    """Prepare inputs for ASE-based structure relaxation using aiida-pythonjob.

    NB: this is used in the prepare_pythonjob_inputs that follows in this file
        which is the one that should be used!
    
    This function creates the necessary inputs to submit a PythonJob that will
    perform structure relaxation using ASE calculators and optimizers.
    
    Parameters
    ----------
    structure : ase.Atoms or orm.StructureData
        The atomic structure to optimize.
    calculator : callable or ASE calculator instance
        The ASE calculator to use (e.g., MACE, CHGNet, M3GNet).
        Can be a function that returns a calculator or a calculator instance.
    fmax : float, optional
        Force convergence criterion in Ry/bohr units.
    optimizer : str, optional
        Optimization algorithm. Options: 'BFGS', 'LBFGS', 'FIRE', 'MDMin', 'BFGS_LineSearch'.
    trajectory : str, optional
        Path to save the trajectory file.
    optimizer_kwargs : dict, optional
        Additional optimizer-specific parameters.
    fix_symmetry : bool, optional
        Apply FixSymmetry constraint to preserve space group symmetry.
    
    Returns
    -------
    dict
        Dictionary with prepared PythonJob inputs ready for submission.
    
    Example
    -------
    >>> from aiida import orm
    >>> from aiida_pythonjob import prepare_pythonjob_inputs
    >>> from mace.calculators import mace_mp
    >>> 
    >>> structure = orm.StructureData(...)
    >>> calculator = mace_mp(model="medium", device="cpu")
    >>> 
    >>> inputs = prepare_ase_relaxation_inputs(
    ...     structure=structure,
    ...     calculator=calculator,
    ...     fmax=1e-4,
    ...     pythonjob_inputs={'code': orm.load_code('pythonjob@localhost')}
    ... )
    >>> 
    >>> # Submit the job
    >>> from aiida_pythonjob import PythonJob
    >>> future = submit(PythonJob, **inputs)
    """
    from typing import Any
    from aiida_pythonjob import prepare_pythonjob_inputs, spec
    
    # Prepare default pythonjob inputs if not provided
    if pythonjob_inputs is None:
        pythonjob_inputs = {}
    
    # Prepare the function inputs
    function_inputs = {
        'atoms': structure,
        # 'calculator': calculator,
        'optimizer': optimizer,
        'fmax': fmax,
    }
    
    if trajectory is not None:
        function_inputs['trajectory'] = trajectory
    
    # Prepare the complete pythonjob inputs
    pythonjob_dict = prepare_pythonjob_inputs(
        function=relax_function,
        function_inputs=function_inputs,
        outputs_spec=spec.namespace(structure=orm.StructureData, energy=Any, forces=Any, nsteps=Any),
        register_pickle_by_value=True,
        # deserializers={
        #     "aiida.orm.nodes.data.structure.StructureData": "aiida_pythonjob.data.deserializer.structure_data_to_atoms",
        # },
        # override the default `AtomsData`
        serializers={
            "ase.atoms.Atoms": "aiida_pythonjob.data.serializer.atoms_to_structure_data"
        },
        **pythonjob_inputs
    )
    
    return pythonjob_dict

# Prepare pythonjob inputs in a complete way for the submission.
def prepare_pythonjob_inputs(
    structure, 
    pythonjob_code,
    callback_calculator,
    pythonjob_metadata = {'options': {'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 1800}},
    optimizer='BFGS',
    trajectory='ase.traj',
    fix_symmetry=True,
    fmax=1e-4,
    custom_relax_function:Callable=None):

    """Prepare inputs for ASE-based relaxation using aiida-pythonjob.
    
    The custom_relax_function input allows users to provide their own 
    relaxation function, which can be useful for advanced users who want 
    to modify the default relaxation behavior. If not provided, 
    it defaults to the standard optimize_structure function from aiida_muon.

    The important point is that the optimize_function is imported and defined outside the definition of the function, as 
    done here. Implcitly, it will be defined in the custom_relax_function accessing it from outer scope with
    respect to the one of the function. It is fine. Note that the same happens for the callback_calculator here.
    In principle we could also put the optimize_as argument of this prepare function, but we prefer to leave it 
    to be implicitly defined in the custom_relax_function and also we assume that the optimize function we provide here
    as default is used in the majority of the case.
    """
    
    # we need to import and then define it here, so it can be pickled without having the need of importing 
    # from the aiida_muon package on the remote side (which may not be installed if using a custom PythonJob code)
    from aiida_muon.pythonjobs.relax import optimize_structure
    optimize_function = optimize_structure

    def relax_function(atoms, fmax=1e-4, optimizer='BFGS', trajectory=None, 
                   fix_symmetry=False, optimizer_kwargs=None):
        """Convenience function for ASE-based relaxation that returns the optimized structure.
        
        This is the main function to be called from aiida-pythonjob for structure relaxations
        using any ASE calculator (MLIPs, DFT codes, empirical potentials, etc.).
        """

        result = optimize_function(
            atoms, calculator=callback_calculator, fmax=fmax, optimizer=optimizer, 
            trajectory=trajectory, optimizer_kwargs=optimizer_kwargs,
            fix_symmetry=fix_symmetry
        )
        return {
            "structure": result['structure'], 
            "energy": result['energy'], 
            "forces": result['forces'], 
            "nsteps": result['nsteps']
        }

    # override the relax function. Please make sure that the inputs and outputs are 
    # the same as the default relax_function, so it can be used in 
    # the prepare_ase_pythonjob_relaxation_inputs function.
    if custom_relax_function is not None:
        relax_function = custom_relax_function
        
    pythonjob_inputs = prepare_ase_pythonjob_relaxation_inputs(
        relax_function=relax_function,
        structure=structure,
        fmax=fmax,  # Force convergence in Ry/bohr (QE units)
        optimizer=optimizer,
        fix_symmetry=fix_symmetry,  # Set True to keep symmetry constraints
        pythonjob_inputs={
            'code': pythonjob_code,
            'metadata': pythonjob_metadata,
        },
        trajectory=trajectory
    )
    return pythonjob_inputs