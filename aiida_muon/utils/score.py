"""
score.py — Frame-selection and reliability scoring for active-learning fine tuning.

Given a DFT reference trajectory and an ASE-compatible MLIP calculator, this
module computes per-frame disagreement scores that can be used to:

  1. Select the most informative frames from a DFT trajectory as training data
     for fine tuning (high score → large MLIP error → high training value).
  2. Assess the reliability of a (fine-)tuned model on a test trajectory
     (low score over all frames → model is trustworthy).

Typical usage
-------------
::

    from ase.io.trajectory import Trajectory
    from aiida_muon.utils.score import ScoreCalculator

    # 1. Build or load an ASE calculator for your MLIP
    from mattersim.forcefield import MatterSimCalculator
    calc = MatterSimCalculator(load_path="best_model.pth", device="cpu")

    # 2. Load DFT trajectory (.traj, .xyz, …)
    traj = Trajectory("relax_QE.traj")

    # 3. Run scorer
    scorer = ScoreCalculator(calculator=calc, w_E=0.2, w_F=0.8)
    scorer.add_dft_trajectory(traj, shift=None)   # shift: cohesive-energy offset
    scorer.evaluate_on_dft_trajectory()
    scorer.compute_energy_distance()
    scorer.compute_force_rmse()
    scorer.compute_structural_rmse()
    scorer.compute_scores()
    scorer.find_frames(num_frames=5, similarity_thr=0.01)

    # 4. Inspect selected frames and print logs
    print("Selected frames:", scorer.frames)
    for log in scorer.log_similarity_analysis:
        print(log)

    # 5. Check overall model reliability
    reliability = scorer.model_reliability()
    print(f"Mean score: {reliability['mean_score']:.4f},  "
          f"Max score: {reliability['max_score']:.4f},  "
          f"Reliable: {reliability['reliable']}")
"""

import copy
import numpy as np


class ScoreCalculator:
    """
    Compute per-frame disagreement scores between a DFT trajectory and an MLIP.

    The score for frame *i* is::

        score_i = w_E * deltaE_i + w_F * deltaF_rmse_i

    where

    * ``deltaE_i = |E_mlip_i - E_dft_i| / N_atoms`` is the per-atom energy
      disagreement (eV/atom).
    * ``deltaF_rmse_i = sqrt(mean((F_mlip_i - F_dft_i)^2))`` is the RMSE of
      the Cartesian force components (eV/Å).

    Parameters
    ----------
    calculator : ASE Calculator
        An ASE-compatible calculator wrapping the MLIP to be evaluated.
    w_E : float
        Weight for the energy disagreement term (default 0.2).
    w_F : float
        Weight for the force RMSE term (default 0.8).
    """

    def __init__(self, calculator, w_E: float = 0.2, w_F: float = 0.8):
        self.calculator = calculator
        self.w_E = w_E
        self.w_F = w_F

        # --- DFT reference arrays ---
        self.dft_trajectory = None
        self.shift = None
        self.dft_energies = None
        self.dft_forces = None
        self.dft_positions = None
        self.num_frames = None

        # --- MLIP prediction arrays ---
        self.energies = None
        self.forces = None
        self.positions = None

        # --- Derived quantities ---
        self.deltaE = None
        self.deltaF_rmse = None
        self.deltaR_rmse = None
        self.score = None
        self.peaks = None
        self.minima_deltaR_rmse = None

        # --- Frame selection ---
        self.frames = None
        self.full_score_frames = None
        self.full_deltaR_frames = None
        self.curated_score_frames = None

        # --- Logging ---
        self.log_similarity_analysis = None

    # ──────────────────────────────────────────────────────────────────────────
    # Data ingestion
    # ──────────────────────────────────────────────────────────────────────────

    def add_dft_trajectory(self, dft_trajectory, shift: float = None):
        """
        Store the DFT reference trajectory and extract energies, forces and
        positions.

        Parameters
        ----------
        dft_trajectory : iterable of ASE Atoms
            Any iterable of ``Atoms`` objects that already carry DFT results
            (``atoms.calc.results`` must contain ``'energy'`` and ``'forces'``).
        shift : float, optional
            Cohesive-energy offset to subtract from the DFT energies (eV).
            Useful when the MLIP reference energy differs from the DFT one.
        """
        self.dft_trajectory = dft_trajectory
        self.shift = shift
        self.dft_energies, self.dft_forces, self.dft_positions = \
            self.extract_E_F_R(trajectory=dft_trajectory)
        if shift is not None:
            self.dft_energies = self.dft_energies - shift
        self.num_frames = len(self.dft_energies)

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_on_dft_trajectory(self):
        """
        Run the MLIP calculator on each frame of the stored DFT trajectory and
        collect energies, forces and positions.

        Requires :meth:`add_dft_trajectory` to have been called first.
        """
        if self.dft_trajectory is None:
            raise ValueError(
                "DFT trajectory not set.  Call add_dft_trajectory() first."
            )
        self.energies, self.forces, self.positions = self.extract_E_F_R(
            trajectory=self.dft_trajectory, calculator=self.calculator
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Per-frame metrics
    # ──────────────────────────────────────────────────────────────────────────

    def compute_energy_distance(self):
        """
        Compute per-atom absolute energy difference for each frame.

        Sets ``self.deltaE`` (shape ``(num_frames,)``, units eV/atom).
        Requires :meth:`evaluate_on_dft_trajectory` to have been called first.
        """
        if self.energies is None or self.dft_energies is None:
            raise ValueError(
                "Energies not available.  Call evaluate_on_dft_trajectory() first."
            )
        n_atoms = len(self.dft_trajectory[0])
        self.deltaE = np.abs(self.energies - self.dft_energies) / n_atoms

    def compute_force_rmse(self):
        """
        Compute the per-frame RMSE of Cartesian force components.

        Sets ``self.deltaF_rmse`` (shape ``(num_frames,)``, units eV/Å).
        Requires :meth:`evaluate_on_dft_trajectory` to have been called first.
        """
        if self.forces is None or self.dft_forces is None:
            raise ValueError(
                "Forces not available.  Call evaluate_on_dft_trajectory() first."
            )
        rmse = [
            np.sqrt(np.mean((self.forces[i] - self.dft_forces[i]) ** 2))
            for i in range(self.num_frames)
        ]
        self.deltaF_rmse = np.array(rmse)

    def compute_structural_rmse(self):
        """
        Compute the RMSE of atomic displacements between consecutive DFT frames.

        Sets ``self.deltaR_rmse`` (shape ``(num_frames - 1,)``, units Å).
        This measures how much the structure changes from one step to the next,
        which is used as a *diversity* criterion to avoid selecting correlated
        frames.
        """
        if self.dft_positions is None:
            raise ValueError(
                "Positions not available.  Call evaluate_on_dft_trajectory() first."
            )
        rmse = [
            np.sqrt(np.mean(
                (self.dft_positions[i] - self.dft_positions[i - 1]) ** 2
            ))
            for i in range(1, self.num_frames)
        ]
        self.deltaR_rmse = np.array(rmse)

    # ──────────────────────────────────────────────────────────────────────────
    # Score + frame identification
    # ──────────────────────────────────────────────────────────────────────────

    def compute_scores(self):
        """
        Compute the composite disagreement score for each frame and identify
        local score maxima (candidate training frames) and structural-change
        minima (diversity frames).

        Sets:

        * ``self.score`` — composite score array (eV/Å weighted).
        * ``self.peaks`` — indices of local score maxima (incl. edge cases).
        * ``self.minima_deltaR_rmse`` — indices of local minima in
          ``deltaR_rmse`` (i.e. frames that are structurally *least similar*
          to their neighbour, good for diversity).

        Requires :meth:`compute_energy_distance`, :meth:`compute_force_rmse`
        and :meth:`compute_structural_rmse` to have been called first.
        """
        from scipy.signal import find_peaks as _find_peaks

        if self.deltaE is None or self.deltaF_rmse is None:
            raise ValueError(
                "Score components not computed. "
                "Call compute_energy_distance() and compute_force_rmse() first."
            )

        self.score = self.w_E * self.deltaE + self.w_F * self.deltaF_rmse

        # Local score maxima
        self.peaks, _ = _find_peaks(self.score)
        if len(self.score) > 1:
            if self.score[0] > self.score[1]:
                self.peaks = np.insert(self.peaks, 0, 0)
            if self.score[-1] > self.score[-2]:
                self.peaks = np.append(self.peaks, len(self.score) - 1)
        else:
            self.peaks = np.array([0])

        # Local minima of structural RMSE (diversity criterion)
        if self.deltaR_rmse is not None and len(self.deltaR_rmse) > 1:
            self.minima_deltaR_rmse, _ = _find_peaks(-self.deltaR_rmse)
            if self.deltaR_rmse[0] < self.deltaR_rmse[1]:
                self.minima_deltaR_rmse = np.insert(self.minima_deltaR_rmse, 0, 0)
        else:
            self.minima_deltaR_rmse = np.array([0])

    def remove_similar(self, frames: np.ndarray, thr: float = 0.01) -> np.ndarray:
        """
        Remove near-duplicate frames based on structural RMSE.

        When two frames are closer than *thr* Å (RMSE of atomic positions),
        the one with the *lower* score is dropped.

        Parameters
        ----------
        frames : array-like of int
            Frame indices to filter.
        thr : float
            Structural-RMSE threshold (Å) below which two frames are
            considered similar (default 0.01 Å).

        Returns
        -------
        np.ndarray
            Filtered frame indices.
        """
        frames = np.array(frames)
        to_keep = np.ones(len(frames), dtype=bool)

        if self.log_similarity_analysis is None:
            self.log_similarity_analysis = []

        log = (
            f"Similarity Analysis Log:\n"
            f"Frames: {frames}\n"
            f"Threshold: {thr} Å\n"
        )

        for i in range(len(frames)):
            if not to_keep[i]:
                continue
            for j in range(i + 1, len(frames)):
                if not to_keep[j]:
                    continue
                fi, fj = int(frames[i]), int(frames[j])
                rmse = np.sqrt(np.mean(
                    (self.dft_positions[fi] - self.dft_positions[fj]) ** 2
                ))
                if rmse < thr:
                    log += f"\nFrames {fi} and {fj} are similar (RMSE: {rmse:.4f} Å)"
                    if self.score[fi] >= self.score[fj]:
                        to_keep[j] = False
                    else:
                        to_keep[i] = False
                        break

        self.log_similarity_analysis.append(log)
        return frames[to_keep]

    def find_frames(self, num_frames: int = 5, similarity_thr: float = 0.01):
        """
        Select the most informative training frames from the trajectory.

        Strategy
        --------
        1. Rank all local score-peak frames by descending score.
        2. Take the top *num_frames* candidates and remove near-duplicates
           (structural-RMSE below *similarity_thr*).
        3. If fewer than *num_frames* remain, fill the remainder with the
           frames that exhibit the *largest* structural change relative to
           the previous step (local minima of ``deltaR_rmse`` sorted
           descending).
        4. Apply a final similarity-removal pass over the combined set.

        Parameters
        ----------
        num_frames : int
            Desired number of training frames to select (default 5).
        similarity_thr : float
            Structural-RMSE threshold (Å) for the similarity filter
            (default 0.01 Å).

        Sets
        ----
        self.frames              : final selected frame indices.
        self.full_score_frames   : all peak frames ranked by score.
        self.full_deltaR_frames  : all diversity frames ranked by deltaR.
        self.curated_score_frames: score frames after first similarity pass.
        """
        if self.score is None or self.peaks is None:
            raise ValueError("Call compute_scores() first.")

        # ── Primary selection: highest-score peaks ─────────────────────────
        self.full_score_frames = self.peaks[
            np.argsort(self.score[self.peaks])[::-1]
        ]
        candidates = self.full_score_frames[:num_frames]
        candidates = self.remove_similar(candidates, similarity_thr)
        self.curated_score_frames = copy.deepcopy(candidates)

        # ── Secondary selection: diversity frames (minima of deltaR) ───────
        # +1 because deltaR_rmse[i] = dist(frame i+1, frame i)
        self.full_deltaR_frames = (
            1 + self.minima_deltaR_rmse[
                np.argsort(self.deltaR_rmse[self.minima_deltaR_rmse])[::-1]
            ]
        )
        remaining = num_frames - len(candidates)
        if remaining > 0:
            diversity = self.full_deltaR_frames[:remaining]
            candidates = np.append(candidates, diversity)

        # ── Final similarity pass ───────────────────────────────────────────
        self.frames = self.remove_similar(candidates, similarity_thr)

    # ──────────────────────────────────────────────────────────────────────────
    # Model reliability
    # ──────────────────────────────────────────────────────────────────────────

    def model_reliability(
        self,
        force_thr: float = 0.1,
        energy_thr: float = 0.05,
    ) -> dict:
        """
        Return a summary dict assessing how reliable the MLIP is on the
        test trajectory.

        A model is considered *reliable* when:

        * ``mean_deltaF_rmse < force_thr`` (eV/Å)
        * ``mean_deltaE < energy_thr`` (eV/atom)

        Parameters
        ----------
        force_thr : float
            Force-RMSE threshold (eV/Å) for reliability (default 0.1).
        energy_thr : float
            Energy-per-atom threshold (eV/atom) for reliability (default 0.05).

        Returns
        -------
        dict with keys:
            mean_score, max_score, std_score,
            mean_deltaF_rmse, max_deltaF_rmse,
            mean_deltaE, max_deltaE,
            reliable (bool)
        """
        if self.score is None:
            raise ValueError("Call compute_scores() first.")

        mean_dF = float(np.mean(self.deltaF_rmse))
        max_dF  = float(np.max(self.deltaF_rmse))
        mean_dE = float(np.mean(self.deltaE))
        max_dE  = float(np.max(self.deltaE))

        return {
            "mean_score":       float(np.mean(self.score)),
            "max_score":        float(np.max(self.score)),
            "std_score":        float(np.std(self.score)),
            "mean_deltaF_rmse": mean_dF,
            "max_deltaF_rmse":  max_dF,
            "mean_deltaE":      mean_dE,
            "max_deltaE":       max_dE,
            "reliable":         (mean_dF < force_thr) and (mean_dE < energy_thr),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Static helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_E_F_R(
        trajectory,
        calculator=None,
    ):
        """
        Extract energies, forces and positions from an ASE trajectory.

        Parameters
        ----------
        trajectory : iterable of ASE Atoms
            Frames to process.
        calculator : ASE Calculator, optional
            If given, attach it to each frame before extracting results
            (MLIP evaluation path).  If ``None``, results are read from the
            existing ``atoms.calc.results`` (DFT path).

        Returns
        -------
        energies  : np.ndarray, shape (N,)        — total energy in eV.
        forces    : np.ndarray, shape (N, natoms, 3) — forces in eV/Å.
        positions : np.ndarray, shape (N, natoms, 3) — positions in Å.
        """
        energies  = []
        forces    = []
        positions = []

        for frame in trajectory:
            atoms = copy.deepcopy(frame)
            if calculator is not None:
                atoms.calc = calculator
                e = atoms.get_total_energy()
                f = atoms.calc.results['forces']
            else:
                e = atoms.calc.results['energy']
                f = atoms.calc.results['forces']

            energies.append(e)
            forces.append(f)
            positions.append(atoms.positions.copy())

        return (
            np.array(energies),
            np.array(forces),
            np.array(positions),
        )
