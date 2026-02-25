# -*- coding: utf-8 -*-
"""Functions for relaxation using ASE optimizers and calculators via aiida-pythonjob."""

from typing import Callable, Union
from ase import Atoms
from aiida import orm

"""
Problem is that we cannot define the optimize_structure and relax_function outside of 
the prepare_ase_pythonjob_relaxation_inputs function because they need to be pickled 
and sent to the remote computer. If we define them outside, the pickling will try to 
import them from aiida_muon.pythonjobs.relax, which will not work.
"""


def prepare_ase_pythonjob_relaxation_inputs(
    structure: Union[orm.StructureData, Atoms],
    callback_calculator: Callable,
    pythonjob_code: orm.Code,
    pythonjob_metadata = {'options': {'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 1800}},
    charged_supercell=False,
    fmax=1e-4,
    optimizer='BFGS',
    fix_symmetry=True,
    pythonjob_inputs=None,
    trajectory="trajectory.traj",
    custom_relax_function:Callable=None
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
    pythonjob_inputs_dict = {
        'code': pythonjob_code,
        'metadata': pythonjob_metadata,
    }
    
    # Prepare the function inputs
    function_inputs = {
        'atoms': structure,
        'optimizer': optimizer,
        'fix_symmetry': fix_symmetry,
        'fmax': fmax,
        'charged_supercell': charged_supercell,
    }
    
    if trajectory is not None:
        function_inputs['trajectory'] = trajectory

    class TrajectoryDataSpec(dict):
        """Specification for the TrajectoryData output of the relaxation function.

        NOTE: it is currently not used. we just return the trajectory as a dict.
        
        The TrajectoryData will contain the following arrays:
        
        - ``positions`` — 3-D float array, shape ``(n_steps, n_atoms, 3)``, in Å.
        - ``energies`` — 1-D float array, shape ``(n_steps,)``, in eV.
        - ``forces`` — 3-D float array, shape ``(n_steps, n_atoms, 3)``, in eV/Å.
        - ``stresses`` — 2-D float array, shape ``(n_steps, 6)``, Voigt order, in eV/Å³ (optional; only stored when
        """

        def __init__(self, iterable):
            super().__init__(iterable)


    def optimize_structure(atoms, calculator, fmax=1e-4, optimizer='BFGS', trajectory="trajectory.traj",
                      optimizer_kwargs=None, fix_symmetry=False, charged_supercell=False):
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
        from ase.io.trajectory import Trajectory

        
        # QE forces are in atomic units (Ry/bohr), convert to eV/Å
        QE_to_ASE_force_units = units.Ry / units.Bohr
        fmax_ase = fmax * QE_to_ASE_force_units
        
        # Handle calculator (could be a callable or instance)
        if callable(calculator) and not hasattr(calculator, 'calculate'):
            calculator = calculator()
        
        # Set the calculator
        atoms.calc = calculator

        if charged_supercell: # but still not working properly...
            atoms.info["charge"] = 1
            atoms.info["spin"] = 0.5
        
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
        
        # Read trajectory frames into a plain list and close the file immediately.
        # Returning a Trajectory object would fail because it holds an open
        # _io.BufferedReader file handle that cloudpickle cannot serialise.
        # We reattach energies/forces via SinglePointCalculator so the frames
        # carry full DFT/MLIP results while remaining picklable.
        from ase.calculators.singlepoint import SinglePointCalculator as SPC
        traj_atoms = {
            "positions": [],
            "cells": [],
            "energies": [],
            "forces": [],
            "symbols": atoms.get_chemical_symbols(),
            "stresses": [],
        }
        if trajectory:
            _traj = Trajectory(trajectory)
            for _at in _traj:
                traj_atoms["positions"].append(_at.get_positions())
                traj_atoms["cells"].append(_at.get_cell())
                traj_atoms["energies"].append(_at.get_potential_energy())
                traj_atoms["forces"].append(_at.get_forces())
                try:
                    traj_atoms["stresses"].append(_at.get_stress(voigt=False))
                except Exception:
                    # traj_atoms["stresses"].append(None)
                    pass
            _traj.close()
        
        # Collect results
        result = {
            'structure': atoms,
            'energy': final_energy,
            'forces': final_forces,
            'nsteps': nsteps,
            'trajectory': traj_atoms   # dict with lists of positions, energies, forces for each frame
        }
        
        return result

    def relax_function(atoms, fmax=1e-4, optimizer='BFGS', trajectory="trajectory.traj", 
                   fix_symmetry=False, optimizer_kwargs=None, charged_supercell=False):
        """Convenience function for ASE-based relaxation that returns the optimized structure.
        
        This is the main function to be called from aiida-pythonjob for structure relaxations
        using any ASE calculator (MLIPs, DFT codes, empirical potentials, etc.).
        """

        result = optimize_structure(
            atoms, calculator=callback_calculator, fmax=fmax, optimizer=optimizer, 
            trajectory=trajectory, optimizer_kwargs=optimizer_kwargs,
            fix_symmetry=fix_symmetry, charged_supercell=charged_supercell
        )

        return {
            "structure": result['structure'], 
            "energy": result['energy'], 
            "forces": result['forces'], 
            "nsteps": result['nsteps'],
            "trajectory": result['trajectory']
        }

    
    # Prepare the complete pythonjob inputs
    pythonjob_inputs = prepare_pythonjob_inputs(
        function=relax_function,
        function_inputs=function_inputs,
        outputs_spec=spec.namespace(structure=orm.StructureData, energy=Any, forces=Any, nsteps=Any, trajectory=Any),
        register_pickle_by_value=True,
        # deserializers={
        #     "aiida.orm.nodes.data.structure.StructureData": "aiida_pythonjob.data.deserializer.structure_data_to_atoms",
        # },
        # override the default `AtomsData`, which is the default serializer for Atoms. The above commented deserializer is not needed because the StructureData is, by default, deserialized to Atoms by aiida-pythonjob.
        serializers={
            "ase.atoms.Atoms": "aiida_pythonjob.data.serializer.atoms_to_structure_data",
        },
        **pythonjob_inputs_dict
    )
    
    return pythonjob_inputs