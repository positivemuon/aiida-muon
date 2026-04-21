# -*- coding: utf-8 -*-
"""
PythonJob helper: score DFT trajectory frames with an MLIP and return the
most informative subset as training data.

The ScoreCalculator logic is inlined inside the job function (same pattern as
relax.py / forces.py) so that the function can be pickled with
``register_pickle_by_value=True`` and executed on a remote machine that does
**not** need to have ``aiida_muon`` installed.

Data transfer strategy
----------------------
Trajectory data is passed as an AiiDA ``TrajectoryData`` node, **not** as a
``PickledData`` blob.  This gives a proper, queryable DB record:

* **Input** ``dft_trajectory`` — ``TrajectoryData`` node built with
  :func:`aiida_muon.utils.trajectory.atoms_list_to_trajectory_data`.
  Positions, cells, energies and forces are stored as plain numpy arrays.
* **Output** ``selected_atoms`` — ``PickledData`` wrapping a list of ASE ``Atoms``
  (energy in ``atoms.info['energy']``, forces in ``atoms.arrays['forces']``,
  calculator stripped).  Use ``.value`` to unpack, or convert to
  ``TrajectoryData`` with :func:`aiida_muon.utils.trajectory.atoms_list_to_trajectory_data`.
"""

from typing import Callable
from aiida import orm
from aiida.orm import TrajectoryData


def prepare_score_frames_pythonjob_inputs(
    dft_trajectory: TrajectoryData,
    callback_calculator: Callable,
    pythonjob_code: orm.AbstractCode,
    num_frames: int = 5,
    w_E: float = 0.2,
    w_F: float = 0.8,
    similarity_thr: float = 0.01,
    energy_shift: float = None,
    pythonjob_metadata: dict = None,
):
    """
    Prepare inputs for a PythonJob that scores a DFT trajectory and selects
    the most informative frames for fine tuning.

    Parameters
    ----------
    dft_trajectory : aiida.orm.TrajectoryData
        The DFT trajectory stored as an AiiDA ``TrajectoryData`` node.  Must
        contain ``'energies'`` (eV) and ``'forces'`` (eV/Å) custom arrays as
        produced by :func:`aiida_muon.utils.trajectory.atoms_list_to_trajectory_data`.
    callback_calculator : callable or ASE Calculator
        The MLIP to evaluate on the trajectory.  Can be a callable (0-arg
        factory) or a live calculator instance.
    pythonjob_code : orm.AbstractCode
        The PythonJob code node to use.
    num_frames : int
        Maximum number of training frames to select (default 5).
    w_E : float
        Weight for the per-atom energy disagreement in the score (default 0.2).
    w_F : float
        Weight for the force RMSE in the score (default 0.8).
    similarity_thr : float
        Structural-RMSE threshold (Å) below which two frames are considered
        too similar (default 0.01 Å).
    energy_shift : float, optional
        Cohesive-energy offset (eV) to subtract from all DFT energies before
        computing the score.  Use this when the MLIP and DFT energy references
        differ.
    pythonjob_metadata : dict, optional
        Scheduler metadata forwarded to the PythonJob.

    Returns
    -------
    dict
        PythonJob inputs ready for ``submit(PythonJob, **inputs)``.

    Outputs of the submitted job
    ----------------------------
    selected_atoms : PickledData (list of ASE Atoms)
        The selected training frames.  Energy is stored in
        ``atoms.info['energy']`` (eV) and forces in ``atoms.arrays['forces']``
        (eV/Å); the calculator is stripped so the list is compact and
        picklable.  Use ``.value`` to unpack, or convert to a proper
        ``TrajectoryData`` with
        :func:`aiida_muon.utils.trajectory.atoms_list_to_trajectory_data`.
    selected_indices : list of int
        Indices of selected frames in the original trajectory.
    reliability : dict
        Model reliability summary (mean/max score, force RMSE, …).
    score_values : list of float
        Per-frame composite score for reference / logging.
    """
    from typing import Any
    from aiida_pythonjob import prepare_pythonjob_inputs, spec

    if pythonjob_metadata is None:
        pythonjob_metadata = {
            'options': {
                'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
                'max_wallclock_seconds': 3600,
            }
        }

    function_inputs = {
        'dft_trajectory': dft_trajectory,   # proper AiiDA TrajectoryData node
        'num_frames':     num_frames,
        'w_E':            w_E,
        'w_F':            w_F,
        'similarity_thr': similarity_thr,
        'energy_shift':   energy_shift,
    }

    # ── Inner function (everything inlined so it can be pickled by value) ────

    def score_and_select_frames(
        dft_trajectory,
        num_frames,
        w_E,
        w_F,
        similarity_thr,
        energy_shift,
    ):
        """
        Score a DFT trajectory (TrajectoryData) with the MLIP and return
        the most informative frames as a new TrajectoryData node.

        The entire ScoreCalculator logic is reproduced here so that the
        function is self-contained and can be pickled with
        ``register_pickle_by_value=True``.
        """
        import copy
        import numpy as np
        from scipy.signal import find_peaks

        # aiida-pythonjob has already called trajectory_data_to_atoms_list() on
        # the TrajectoryData input (via the custom deserializer), so dft_trajectory
        # here is already a plain list of ASE Atoms with SinglePointCalculator.
        dft_atoms_list = dft_trajectory

        # ── 1. Extract DFT energies, forces, positions ────────────────────────
        def _extract_dft(traj):
            energies, forces, positions = [], [], []
            for atoms in traj:
                energies.append(atoms.calc.results['energy'])
                forces.append(atoms.calc.results['forces'])
                positions.append(atoms.positions.copy())
            return np.array(energies), np.array(forces), np.array(positions)

        def _extract_mlip(traj, calc):
            energies, forces, positions = [], [], []
            for frame in traj:
                atoms = copy.deepcopy(frame)
                if callable(calc) and not hasattr(calc, 'calculate'):
                    calc = calc()
                atoms.calc = calc
                e = atoms.get_total_energy()
                f = atoms.calc.results['forces']
                energies.append(e)
                forces.append(f)
                positions.append(atoms.positions.copy())
            return np.array(energies), np.array(forces), np.array(positions)

        n_frames = len(dft_atoms_list)
        n_atoms = len(dft_atoms_list[0])

        dft_e, dft_f, dft_r = _extract_dft(dft_atoms_list)
        if energy_shift is not None:
            dft_e = dft_e - energy_shift

        mlip_e, mlip_f, _ = _extract_mlip(dft_atoms_list, callback_calculator)

        # ── 2. Per-frame metrics ──────────────────────────────────────────────
        delta_e = np.abs(mlip_e - dft_e) / n_atoms
        delta_f = np.array([
            np.sqrt(np.mean((mlip_f[i] - dft_f[i]) ** 2))
            for i in range(n_frames)
        ])
        delta_r = np.array([
            np.sqrt(np.mean((dft_r[i] - dft_r[i - 1]) ** 2))
            for i in range(1, n_frames)
        ])

        # ── 3. Composite score ────────────────────────────────────────────────
        score = w_E * delta_e + w_F * delta_f

        # Local score maxima (candidate frames)
        peaks, _ = find_peaks(score)
        if len(score) > 1:
            if score[0] > score[1]:
                peaks = np.insert(peaks, 0, 0)
            if score[-1] > score[-2]:
                peaks = np.append(peaks, n_frames - 1)
        else:
            peaks = np.array([0])

        # Local minima of delta_r (diversity frames)
        if len(delta_r) > 1:
            min_dr, _ = find_peaks(-delta_r)
            if delta_r[0] < delta_r[1]:
                min_dr = np.insert(min_dr, 0, 0)
        else:
            min_dr = np.array([0])

        # ── 4. Frame selection ────────────────────────────────────────────────
        log_similarity = []

        def remove_similar(frames, positions, score, thr):
            frames = np.array(frames)
            keep = np.ones(len(frames), dtype=bool)
            log = f'Similarity filter (thr={thr:.3f} Å): {frames.tolist()}\n'
            for i in range(len(frames)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(frames)):
                    if not keep[j]:
                        continue
                    fi, fj = int(frames[i]), int(frames[j])
                    rmse = np.sqrt(np.mean((positions[fi] - positions[fj]) ** 2))
                    if rmse < thr:
                        log += f'  Frames {fi} and {fj} similar (RMSE={rmse:.4f} Å)'
                        if score[fi] >= score[fj]:
                            keep[j] = False
                            log += f' → drop {fj}\n'
                        else:
                            keep[i] = False
                            log += f' → drop {fi}\n'
                            break
            log_similarity.append(log)
            return frames[keep]

        top = peaks[np.argsort(score[peaks])[::-1]][:num_frames]
        top = remove_similar(top, dft_r, score, similarity_thr)

        remaining = num_frames - len(top)
        if remaining > 0 and len(min_dr) > 0:
            diversity = 1 + min_dr[np.argsort(delta_r[min_dr])[::-1]][:remaining]
            top = np.append(top, diversity)
        top = remove_similar(top, dft_r, score, similarity_thr)

        selected_indices = [int(i) for i in top]

        # ── 5. Build output: stripped Atoms list (PickledData-safe) ──────────
        # Return a plain Python list so aiida-pythonjob wraps it as PickledData.
        # PickledData *outputs* are fine (no JSON validation); the caller can
        # convert to TrajectoryData if needed via atoms_list_to_trajectory_data.
        selected_atoms_out = []
        for idx in selected_indices:
            at = copy.deepcopy(dft_atoms_list[idx])
            at.info['energy']   = float(dft_e[idx])
            at.arrays['forces'] = np.array(dft_f[idx])
            at.calc = None   # strip calculator so pickle is small and clean
            selected_atoms_out.append(at)

        # ── 6. Reliability summary ────────────────────────────────────────────
        reliability = {
            'mean_score':       float(np.mean(score)),
            'max_score':        float(np.max(score)),
            'std_score':        float(np.std(score)),
            'mean_deltaF_rmse': float(np.mean(delta_f)),
            'max_deltaF_rmse':  float(np.max(delta_f)),
            'mean_deltaE':      float(np.mean(delta_e)),
            'max_deltaE':       float(np.max(delta_e)),
            'similarity_log':   log_similarity,
        }

        return {
            'selected_atoms':   selected_atoms_out,  # list[ASE Atoms] → PickledData
            'selected_indices': selected_indices,
            'score_values':     score.tolist(),
            'reliability':      reliability,
        }

    # ── Assemble pythonjob inputs ─────────────────────────────────────────────
    # Custom deserializer so aiida-pythonjob can unpack TrajectoryData into
    # a list of ASE Atoms before calling the remote function.
    # The key must match the fully-qualified class path that aiida-pythonjob
    # builds as: f"{data_type.__module__}.{data_type.__name__}"
    _deserializers = {
        'aiida.orm.nodes.data.array.trajectory.TrajectoryData':
            'aiida_muon.utils.trajectory.trajectory_data_to_atoms_list',
    }

    pythonjob_inputs = prepare_pythonjob_inputs(
        function=score_and_select_frames,
        function_inputs=function_inputs,
        outputs_spec=spec.namespace(
            selected_atoms=Any,     # list[ASE Atoms] → PickledData (output, safe)
            selected_indices=Any,
            score_values=Any,
            reliability=Any,
        ),
        deserializers=_deserializers,
        register_pickle_by_value=True,
        code=pythonjob_code,
        metadata=pythonjob_metadata,
    )

    return pythonjob_inputs


# make Any importable at module level (needed for the return type annotation above)
try:
    from typing import Any
except ImportError:
    pass
