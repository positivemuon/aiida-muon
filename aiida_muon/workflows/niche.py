# -*- coding: utf-8 -*-
"""
Niche - a program that finds interstitial spaces in crystals.
"""
import argparse
import sys

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_shortest_vectors


class Niche:
    @staticmethod
    def find_equivalent_positions(frac_coords, host_lattice, atol=1e-3):
        """
        Creates a list of symmetry equivalent positions for the input structure.
        The output is the same as spg.get_symmetry_dataset()['equivalent_atoms']
        >>> from pymatgen.util.testing import PymatgenTest
        >>> from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        >>> st = PymatgenTest.TEST_STRUCTURES['Li10GeP2S12']
        >>> Niche.find_equivalent_positions(st.frac_coords,st) == SpacegroupAnalyzer(st).get_symmetry_dataset()['equivalent_atoms'] # doctest: +NORMALIZE_WHITESPACE
        array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True])
        """

        lattice = host_lattice.lattice

        # Bring to unit cell
        frac_coords %= 1

        # prepare list of equivalent atoms. -1 mean "not yet analyzed".
        eq_list = np.zeros(len(frac_coords), dtype=np.int32) - 1

        spg = SpacegroupAnalyzer(host_lattice, symprec=atol)

        ops = spg.get_symmetry_operations()

        # This hosts all the equivalent positions obtained for each of the
        # lattice points using all symmetry operations.
        eq_pos = np.zeros([len(ops), len(frac_coords), 3])

        for i, op in enumerate(ops):
            eq_pos[i] = op.operate_multi(frac_coords) % 1

        # Compute equivalence list
        for i in range(len(frac_coords)):
            if eq_list[i] >= 0:
                continue

            for j in range(i, len(frac_coords)):
                diff = pbc_shortest_vectors(
                    lattice, eq_pos[:, j, :], frac_coords[i]
                ).squeeze()
                if (np.linalg.norm(diff, axis=1) < atol).any():
                    eq_list[j] = i

        return eq_list

    @staticmethod
    def prune_atoms_too_close(grid_coords, host_lattice, min_distance):
        """
        List interstitial positions too close to the lattice points of the
        hosting compounds. Threshold is set by min_distance.
        Parameters
        ----------
        grid_coords:
            An array that contains interstitial points in scaled coordinates.
        atoms: pymatgen.core.Structure
            A pymatgen structure.
        min_distance : list
            Minimum spacing in $A$ between interstitial points and lattice points.
        Returns
        -------
        mask : np.ndarray
            List of interstitial positions not too close to host_lattice lattice points.
        """

        all_differences = pbc_shortest_vectors(
            host_lattice.lattice, grid_coords, host_lattice.frac_coords
        )

        mask = np.ones(len(grid_coords), dtype=bool)
        for i, p in enumerate(all_differences):
            if any(np.linalg.norm(p, axis=1) < min_distance):
                mask[i] = False
        return mask

    @staticmethod
    def prune_atoms_covalent_radius(
        grid_coords, host_lattice, interstitial_atom, scaling_factor=0.9
    ):
        """
        List interstitial positions too close to the lattice points of the
        hosting compounds. Threshold is the sum of host_lattice covalent radius
        for each atomic specie and interstitial_atom covalent radius.
        Parameters
        ----------
        grid_coords: np.ndarray
            An array that contains interstitial points in scaled coordinates.
        host_lattice: pymatgen.core.Structure
            A pymatgen structure.
        interstitial_atom : str
            The atom used to compute the covalent radius when calculating bond distance.
        scaling_factor : float
            A scaling factor for the bond lenght.
        Returns
        -------
        mask : np.ndarray
            List of interstitial positions not too close to host_lattice lattice points.
        """
        from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
        from pymatgen.core.lattice import get_points_in_spheres

        interstitial_cart_coords = host_lattice.lattice.get_cartesian_coords(
            grid_coords
        )

        h_r = CovalentRadius.radius[interstitial_atom]

        orig_species = [x.name for x in host_lattice.species]

        mask = np.ones(len(grid_coords), dtype=bool)
        for specie in set(orig_species):
            sp_r = CovalentRadius.radius[specie]
            cart_coords = host_lattice.cart_coords[np.array(orig_species) == specie]

            res = get_points_in_spheres(
                cart_coords,
                interstitial_cart_coords,
                r=(h_r + sp_r) * scaling_factor,
                lattice=st.lattice,
            )
            for i, r in enumerate(res):
                if r != []:
                    mask[i] = False

        return mask

    @staticmethod
    def spacing_to_grid(host_lattice, spacing, calculated_spacing=None):
        """
        Calculate the kpoint mesh that is equivalent to the given spacing
        in reciprocal space (units Angstrom^-1). The number of kpoints is each
        dimension is rounded up.
        Parameters
        ----------
        host_lattice: pymatgen.core.Structure
            A structure that has host_lattice.lattice.matrix.
        spacing: float
            Minimum grid point spacing in $A$.
        calculated_spacing : list
            If a three item list (or similar mutable sequence) is given the
            members will be replaced with the actual calculated spacing in
            $A$.
        Returns
        -------
        interstitial_grid : [int, int, int]
            Grid specification to give the required spacing.
        >>> from pymatgen.util.testing import PymatgenTest
        >>> Niche.spacing_to_grid(PymatgenTest.TEST_STRUCTURES['He_BCC'],1)
        [3, 3, 3]
        """

        # Lattice parameters lenght
        r_x, r_y, r_z = np.linalg.norm(host_lattice.lattice.matrix, axis=1)

        inters_grid = [
            int(r_x / spacing) + 1,
            int(r_y / spacing) + 1,
            int(r_z / spacing) + 1,
        ]

        if calculated_spacing is not None:
            calculated_spacing[:] = [
                r_x / inters_grid[0],
                r_y / inters_grid[1],
                r_z / inters_grid[2],
            ]

        return inters_grid

    @staticmethod
    def build_supercell(host_lattice, atom, position, sc_matrix):
        """
        Builds a supercell assuming that a single impurity atom
        is presented as last item.
        """
        stc = host_lattice.copy()
        stc.append(atom, position)
        stc.make_supercell(sc_matrix)

        nimp = int(np.linalg.det(sc_matrix))

        # Remove additional impurities in supercell
        for _ in range(nimp - 1):
            stc.pop(-1)

        return stc

    def __init__(self, st, atom, cov_atom=""):
        """
        st: pytmatgen Structure describing the hosting material
        atom: str The atom to be used as impurity
        cov_atom: str The atom used to compute covalent radiuses (optional)
        """
        self.st = st
        self.atom = atom
        self.cov_atom = cov_atom

    def apply(self, spacing, distance):
        """
        Places the impurities inside the hosting material.
        Parameters
        ----------
        spacing: float
            Distance among grid points.
        distance: float
            distance from hosting atoms.
        calculated_spacing : list
            If a three item list (or similar mutable sequence) is given the
            members will be replaced with the actual calculated spacing in
            $A$.
        Returns
        -------
        stc : pymatgen.core.Structure
            A copy of the original structure with all impurities of the
            symmetry reduced grid.
        """
        st = self.st

        # Compute density along 3 lattice parameters
        nx, ny, nz = self.spacing_to_grid(self.st, spacing)

        # generate grid of interstitial positions accordingly
        X, Y, Z = np.meshgrid(
            np.linspace(0, 1, nx, endpoint=False),
            np.linspace(0, 1, ny, endpoint=False),
            np.linspace(0, 1, nz, endpoint=False),
        )

        pos = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T

        # remove positions too close to host compound atoms
        if self.cov_atom == "":
            good = self.prune_atoms_too_close(pos, st, distance)
        else:
            good = self.prune_atoms_covalent_radius(pos, st, self.cov_atom)
        pos = pos[good]

        # find equivalent positions among grid points
        good = self.find_equivalent_positions(pos, st, atol=1e-3)
        pos = pos[good == np.arange(len(pos))]

        # Add positions to original structure
        stc = st.copy()
        for p in pos:
            stc.append(self.atom, p)
        return stc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute interstitial positions")
    parser.add_argument(
        "--spacing",
        metavar="S",
        type=float,
        help="Maximum grid point spacing (Ang.)",
        default=1.0,
    )
    parser.add_argument(
        "--distance",
        metavar="D",
        type=float,
        default=1.0,
        help="Distance between interstitial grid points and lattice points (Ang.)",
    )
    parser.add_argument(
        "--atom",
        metavar="A",
        type=str,
        default="H",
        help="Atom to be inserted in structures",
    )
    parser.add_argument(
        "--covalent-atom",
        metavar="C",
        type=str,
        default="",
        help="Atom symbol to be cosidered for covalence radius calc.",
    )
    parser.add_argument(
        "--supercell",
        metavar="SC",
        type=str,
        default="1 0 0  0 1 0  0 0 1",
        help="Supercell to be built",
    )

    parser.add_argument("input_structure")

    args = parser.parse_args()
    spacing = args.spacing
    distance = args.distance
    atom = args.atom
    cov_atom = args.covalent_atom
    sc_matrix = (
        np.array([float(x) for x in args.supercell.split()]).reshape(-1, 3).squeeze()
    )

    # load structure with pymatgen
    st = Structure.from_file(args.input_structure)

    nc = Niche(st, atom)
    nc.apply(spacing, distance).to(filename="positions.cif".format(i))
