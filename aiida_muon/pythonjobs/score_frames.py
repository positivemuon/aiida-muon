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
All trajectory data is passed as Python objects that get pickled by
aiida-pythonjob:

* **Input** ``dft_atoms_list``  — list of ASE ``Atoms`` that already carry
  DFT results in ``atoms.calc.results`` (populated from QE outputs upstream).
* **Output** ``selected_atoms`` — list of ASE ``Atoms`` (subset, same format).

These pickled lists flow as ``PickledData`` nodes through AiiDA provenance and
can be forwarded directly as inputs to downstream pythonjobs (e.g.
``FineTuningWorkChain``) without any filesystem intermediate.
"""

from typing import Callable, List
from ase import Atoms
from aiida import orm


def prepare_score_frames_pythonjob_inputs(
    dft_atoms_list: list,
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
    dft_atoms_list : list of ASE Atoms
        The DFT trajectory frames.  Each ``Atoms`` must have its DFT results
        stored under ``atoms.calc.results`` as a dict with at least keys
        ``'energy'`` (float, eV) and ``'forces'`` (array, eV/Å).
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
    selected_atoms : list of ASE Atoms
        The selected training frames (DFT results embedded in
        ``atoms.info['energy']`` and ``atoms.arrays['forces']``; calculator
        stripped for pickling).
    selected_indices : list of int
        Indices of selected frames in the original trajectory.
    reliability : dict
        Model reliability summary from ``ScoreCalculator.model_reliability()``.
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
        'dft_atoms_list': dft_atoms_list,
        'num_frames':     num_frames,
        'w_E':            w_E,
        'w_F':            w_F,
        'similarity_thr': similarity_thr,
        'energy_shift':   energy_shift,
    }

    # ── Inner function (everything inlined so it can be pickled by value) ────

    def score_and_select_frames(
        dft_atoms_list,
        num_frames,
        w_E,
        w_F,
        similarity_thr,
        energy_shift,
    ):
        """
        Score a DFT trajectory with the MLIP and return the most informative
        frames for fine tuning.

        The entire ScoreCalculator logic is reproduced here so that the
        function is self-contained and can be pickled with
        ``register_pickle_by_value=True``.
        """
        import copy
        import numpy as np
        from scipy.signal import find_peaks

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

        # ── 5. Build output Atoms (DFT results embedded, calculator stripped) ─
        selected_atoms = []
        for idx in selected_indices:
            atoms = copy.deepcopy(dft_atoms_list[idx])
            atoms.info['energy'] = float(dft_e[idx])
            # ensure forces array is on the atoms so ASE's extxyz writer can use it
            atoms.arrays['forces'] = np.array(dft_f[idx])
            atoms.calc = None   # strip calculator → picklable
            selected_atoms.append(atoms)

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
            'selected_atoms':   selected_atoms,
            'selected_indices': selected_indices,
            'score_values':     score.tolist(),
            'reliability':      reliability,
        }

    # ── Assemble pythonjob inputs ─────────────────────────────────────────────
    pythonjob_inputs = prepare_pythonjob_inputs(
        function=score_and_select_frames,
        function_inputs=function_inputs,
        outputs_spec=spec.namespace(
            selected_atoms=Any,
            selected_indices=Any,
            score_values=Any,
            reliability=Any,
        ),
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
