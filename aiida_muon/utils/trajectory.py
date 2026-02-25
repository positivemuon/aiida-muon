# -*- coding: utf-8 -*-
"""
Utility helpers for converting between a list of ASE ``Atoms`` objects (with
DFT results attached via ``SinglePointCalculator``) and AiiDA's
``TrajectoryData`` node.

Using ``TrajectoryData`` instead of ``PickledData`` gives:

* A proper, queryable AiiDA-native node stored in the database.
* JSON-serialisable metadata (no opaque pickle blobs).
* Arrays that can be inspected via ``verdi data trajectory show`` or the
  AiiDA REST API without deserialising the whole trajectory.
* Easy interoperability with other AiiDA plugins (e.g. aiida-phonopy).

Conventions
-----------
* ``energies``  — 1-D float array, shape ``(n_steps,)``, units eV (total).
* ``forces``    — 3-D float array, shape ``(n_steps, n_atoms, 3)``, units eV/Å.
* ``stresses``  — 2-D float array, shape ``(n_steps, 6)``, Voigt order, units
  eV/Å³ (optional; only stored when present in the calculator results).

All arrays are stored via ``TrajectoryData.set_array``.
"""
from __future__ import annotations

__all__ = ['atoms_list_to_trajectory_data', 'trajectory_data_to_atoms_list']

import numpy as np

def trajectory_dict_to_trajectory_data(traj_dict):
    """
    Convert a trajectory dictionary (e.g. as returned by the relax pythonjob)
    to a TrajectoryData node.

    The input dict should have the same format as the 'trajectory' entry in the relax pythonjob result, i.e. a dict with lists of positions, energies, forces for each frame.
    """


    from aiida.orm import TrajectoryData

    # Build a TrajectoryData node using the data in traj_dict
    traj = TrajectoryData()
    traj.set_trajectory(
        stepids   = np.arange(len(traj_dict['energies']), dtype=int),
        cells     = np.array(traj_dict['cells']),
        symbols   = traj_dict['symbols'],
        positions = np.array(traj_dict['positions']),
    )
    traj.set_array('energies', np.array(traj_dict['energies']))
    traj.set_array('forces', np.array(traj_dict['forces']))
    if 'stresses' in traj_dict:
        traj.set_array('stresses', np.array(traj_dict['stresses']))

    return traj

def trajectory_data_to_trajectory_dict(traj_data):
    """
    Convert a TrajectoryData node to a trajectory dictionary (e.g. as returned by the relax pythonjob).

    The output dict has the same format as the 'trajectory' entry in the relax pythonjob result, i.e. a dict with lists of positions, energies, forces for each frame.
    """
    traj_dict = {
        'positions': traj_data.get_array('positions').tolist(),
        'cells': traj_data.get_array('cells').tolist(),
        'symbols': list(traj_data.symbols),
        'energies': traj_data.get_array('energies').tolist(),
        'forces': traj_data.get_array('forces').tolist(),
    }
    if 'stresses' in traj_data.get_arraynames():
        traj_dict['stresses'] = traj_data.get_array('stresses').tolist()

    return traj_dict


def atoms_list_to_trajectory_data(atoms_list, store_stresses: bool = True):
    """
    Convert a list of ASE ``Atoms`` objects to an AiiDA ``TrajectoryData``.

    Each ``Atoms`` object **must** have DFT results attached via an ASE
    ``SinglePointCalculator`` (or any calculator whose ``.results`` dict
    contains at least ``'energy'`` (float, eV) and ``'forces'`` (array,
    eV/Å)).

    Parameters
    ----------
    atoms_list : list of ase.Atoms
        Trajectory frames.  All frames must have the same species in the
        same order. Can also be an ASE Trajectory object (e.g. read from a .traj file).
    store_stresses : bool
        If ``True`` (default) and stress is present in the first frame's
        calculator results, the Voigt stress tensor is stored as well.

    Returns
    -------
    aiida.orm.TrajectoryData
        A new (unstored) ``TrajectoryData`` node.

    Raises
    ------
    ValueError
        If any frame is missing energy or forces in its calculator results.
    """
    from aiida.orm import TrajectoryData, List
    from ase.atoms import Atoms

    n_steps = len(atoms_list)
    if n_steps == 0:
        raise ValueError('atoms_list is empty')

    if False in [isinstance(atoms, Atoms) for atoms in atoms_list]:
        raise ValueError('All items in atoms_list must be ASE Atoms objects')

    # --- validate ----------------------------------------------------------
    for i, atoms in enumerate(atoms_list):
        if atoms.calc is None or not atoms.calc.results:
            # fall back to atoms.info / atoms.arrays
            if 'energy' not in atoms.info or 'forces' not in atoms.arrays:
                raise ValueError(
                    f'Frame {i} has no DFT results: attach a SinglePointCalculator '
                    'or store energy in atoms.info["energy"] and forces in '
                    'atoms.arrays["forces"].'
                )

    symbols = atoms_list[0].get_chemical_symbols()
    n_atoms = len(symbols)

    positions = np.zeros((n_steps, n_atoms, 3), dtype=float)
    cells     = np.zeros((n_steps, 3, 3),       dtype=float)
    energies  = np.zeros(n_steps,               dtype=float)
    forces    = np.zeros((n_steps, n_atoms, 3), dtype=float)

    has_stress = False
    stresses   = np.zeros((n_steps, 6),          dtype=float)

    for i, atoms in enumerate(atoms_list):
        positions[i] = atoms.get_positions()
        cells[i]     = np.array(atoms.get_cell())

        # prefer calculator results, fall back to atoms.info / atoms.arrays
        if atoms.calc is not None and atoms.calc.results:
            energies[i] = atoms.calc.results['energy']
            forces[i]   = atoms.calc.results['forces']
            if store_stresses and 'stress' in atoms.calc.results:
                stresses[i] = atoms.calc.results['stress']
                if i == 0:
                    has_stress = True
        else:
            energies[i] = atoms.info['energy']
            forces[i]   = atoms.arrays['forces']

    # --- build TrajectoryData ----------------------------------------------
    traj = TrajectoryData()
    traj.set_trajectory(
        stepids   = np.arange(n_steps, dtype=int),
        cells     = cells,
        symbols   = symbols,
        positions = positions,
    )
    traj.set_array('energies', energies)
    traj.set_array('forces',   forces)
    if has_stress:
        traj.set_array('stresses', stresses)

    # store PBC (take from first frame)
    traj.set_array('pbc', np.array(atoms_list[0].get_pbc(), dtype=bool))

    return traj

def trajectory_dict_to_atoms_list(traj_dict):
    """
    Convert a trajectory dictionary (e.g. as returned by the relax pythonjob)
    to a list of ASE Atoms objects with SinglePointCalculator attached.

    The input dict should have the same format as the 'trajectory' entry in the relax pythonjob result, i.e. a dict with lists of positions, energies, forces for each frame.
    """
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    positions = np.array(traj_dict['positions'])   # (n_steps, n_atoms, 3)
    cells     = np.array(traj_dict['cells'])       # (n_steps, 3, 3)
    symbols   = traj_dict['symbols']               # list[str]
    energies  = np.array(traj_dict['energies'])    # (n_steps,)
    forces    = np.array(traj_dict['forces'])      # (n_steps, n_atoms, 3)

    try:
        stresses = np.array(traj_dict['stresses']) # (n_steps, 6)
    except KeyError:
        stresses = None

    try:
        pbc = np.array(traj_dict['pbc'], dtype=bool) # (3,) bool
    except KeyError:
        pbc = True

    atoms_list = []
    for i in range(len(positions)):
        atoms = Atoms(
            symbols   = symbols,
            positions = positions[i],
            cell      = cells[i],
            pbc       = pbc,
        )
        calc_kwargs = {
            'energy': float(energies[i]),
            'forces': forces[i],
        }
        if stresses is not None:
            calc_kwargs['stress'] = stresses[i]

        atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
        atoms_list.append(atoms)

    return atoms_list

def trajectory_data_to_atoms_list(traj_data):
    """
    Convert an AiiDA ``TrajectoryData`` to a list of ASE ``Atoms`` objects
    with DFT results reattached via ``SinglePointCalculator``.
    This list is basically the analogous of an ASE Trajectory object (e.g. read from a .traj file).

    This is the inverse of :func:`atoms_list_to_trajectory_data`.

    Parameters
    ----------
    traj_data : aiida.orm.TrajectoryData
        Input node.  Must contain ``'energies'`` and ``'forces'`` custom
        arrays (as stored by :func:`atoms_list_to_trajectory_data`).

    Returns
    -------
    list of ase.Atoms
        Each ``Atoms`` has an attached ``SinglePointCalculator`` with
        ``energy`` and ``forces`` (and ``stress`` if present in the node).
    """
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    positions = traj_data.get_array('positions')   # (n_steps, n_atoms, 3)
    cells     = traj_data.get_array('cells')       # (n_steps, 3, 3)
    energies  = traj_data.get_array('energies')    # (n_steps,)
    forces    = traj_data.get_array('forces')      # (n_steps, n_atoms, 3)
    symbols   = list(traj_data.symbols)            # list[str]

    try:
        stresses = traj_data.get_array('stresses') # (n_steps, 6)
    except KeyError:
        stresses = None

    try:
        pbc = traj_data.get_array('pbc')           # (3,) bool
    except KeyError:
        pbc = True

    atoms_list = []
    for i in range(len(positions)):
        atoms = Atoms(
            symbols   = symbols,
            positions = positions[i],
            cell      = cells[i],
            pbc       = pbc,
        )
        calc_kwargs = {
            'energy': float(energies[i]),
            'forces': forces[i],
        }
        if stresses is not None:
            calc_kwargs['stress'] = stresses[i]

        atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
        atoms_list.append(atoms)

    return atoms_list
