# -*- coding: utf-8 -*-
"""Test of the dipolar field computed with `pylocfield`, against the values of the `muesr` implementation it replaced."""
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

pytest.importorskip("pylocfield")

from aiida_muon.utils.magnetism import compute_dip_field  # noqa: E402


def test_ferromagnetic_bcc_iron():
    """bcc Fe, 2x2x2 supercell, moments of 2.2 mu_B along z, the muon at (1/2, 1/4, 0), and a contact field of 0.1 T.

    The reference B_dip+B_Lorentz (T) is the one computed by the former implementation, which used `muesr`
    (-0.201151 T along z). The two methods differ by < 0.1 %, as the sums are done differently (Ewald here).
    """
    unit_cell = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    sc_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    supercell = unit_cell.copy()
    supercell.make_supercell(sc_matrix)
    supercell.append("H", [0.5, 0.25, 0.0])  # the muon, in the relaxed supercell

    b_dip_lor, b_total, b_total_norm = compute_dip_field(unit_cell.copy(), [2.2, 2.2], sc_matrix, supercell, 0.1)

    assert np.allclose(b_dip_lor, [0.0, 0.0, -0.201151], atol=5e-4)
    # the contact field is added along the direction of the moments (z): the total is B_dip+B_L + 0.1 T
    assert np.allclose(b_total, np.asarray(b_dip_lor) + [0.0, 0.0, 0.1], atol=1e-8)
    assert np.isclose(b_total_norm, np.linalg.norm(b_total))


def test_antiferromagnetic_mno():
    """MnO (rocksalt, conventional cell), 2x2x2 supercell, antiferromagnetic moments of 4.6 mu_B along z.

    The reference is again the former implementation with `muesr`: B_dip+B_Lorentz = [-0.58805, 1.13174, 2.61817] T.
    It has moments of both signs and a muon that is not on a symmetry axis, so all the components are non zero.
    With the default cutoffs of `pylocfield` it takes ~25 s, with the ones of `get_ewald_cutoffs` ~1 s.
    """
    unit_cell = Structure(
        Lattice.cubic(4.45),
        ["Mn"] * 4 + ["O"] * 4,
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    )
    sc_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    supercell = unit_cell.copy()
    supercell.make_supercell(sc_matrix)
    rng = np.random.default_rng(0)
    for i in range(len(supercell)):  # a small relaxation, as the one induced by the muon
        supercell.translate_sites([i], 0.02 * rng.normal(size=3), frac_coords=False, to_unit_cell=True)
    supercell.append("H", [0.55, 0.12, 0.3])

    b_dip_lor, _, _ = compute_dip_field(unit_cell.copy(), [4.6, -4.6, -4.6, 4.6, 0, 0, 0, 0], sc_matrix, supercell, 0.1)

    assert np.allclose(b_dip_lor, [-0.58805, 1.13174, 2.61817], atol=5e-4)


@pytest.mark.parametrize("cell_type", ["cubic", "oblique"])
def test_the_cutoffs_do_not_change_the_result(monkeypatch, cell_type):
    """The cutoffs of `get_ewald_cutoffs` give the same field as the (huge) default ones of `pylocfield`.

    A small cubic cell (the muon has many periodic images within the cutoff) and an oblique one (the diagonal is longer
    than the lattice vectors): the cases in which a too small real-space cutoff would miss atoms.
    """
    import aiida_muon.utils.magnetism as magnetism

    if cell_type == "cubic":
        unit_cell = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        magm, sc_matrix, muon = [2.2, 2.2], [[2, 0, 0], [0, 2, 0], [0, 0, 1]], [0.5, 0.25, 0.0]
    else:
        unit_cell = Structure(Lattice.hexagonal(4.0, 5.5), ["Mn", "Mn", "O"], [[0, 0, 0], [1 / 3, 2 / 3, 0.5], [0.3, 0.1, 0.25]])
        magm, sc_matrix, muon = [[1.0, 0.5, 2.0], [-1.0, -0.5, -2.0], [0, 0, 0]], [[2, 0, 0], [0, 2, 0], [0, 0, 1]], [0.1, 0.45, 0.4]

    def field():
        supercell = unit_cell.copy()
        supercell.make_supercell(sc_matrix)
        supercell.append("H", muon)
        return np.asarray(compute_dip_field(unit_cell.copy(), magm, sc_matrix, supercell, 0.1)[0], dtype=float)

    with_cutoffs = field()

    original = magnetism.compute_dipolar_tensors
    monkeypatch.setattr(
        magnetism, "compute_dipolar_tensors", lambda atoms, r_c=None, g_c=None, **kw: original(atoms, r_c=12.0, g_c=12.0, **kw)
    )
    with_defaults = field()

    assert np.allclose(with_cutoffs, with_defaults, atol=1e-10)


def test_ewald_cutoffs():
    """The cutoffs follow from the Ewald parameter: see the comment at the top of `aiida_muon.utils.magnetism`."""
    from aiida_muon.utils.magnetism import EWALD_RHO, get_ewald_cutoffs

    r_c, g_c = get_ewald_cutoffs(np.eye(3) * 8.61)
    # g_c = 1.5 * 2 rho sqrt(ln 1e10) / (2 pi) = 4.12 (1/Angstrom, without the 2 pi); it does not depend on the cell
    assert np.isclose(g_c, 1.5 * 2 * EWALD_RHO * np.sqrt(np.log(1e10)) / (2 * np.pi))
    assert np.isclose(g_c, get_ewald_cutoffs(np.eye(3) * 30.0)[1])
    # r_c = 1.5 sqrt(ln 1e10) / rho + half the longest diagonal of the cell (the muon is at its centre)
    assert np.isclose(r_c, 1.5 * np.sqrt(np.log(1e10)) / EWALD_RHO + 0.5 * np.sqrt(3) * 8.61)
    # an oblique cell: the longest of the four diagonals a1 +- a2 +- a3 is the one to use
    cell = np.array([[4.0, 0.0, 0.0], [-2.0, 3.4641016, 0.0], [0.0, 0.0, 5.5]])
    diagonals = [np.linalg.norm(cell[0] + s2 * cell[1] + s3 * cell[2]) for s2 in (1, -1) for s3 in (1, -1)]
    assert np.isclose(get_ewald_cutoffs(cell)[0] - 1.5 * np.sqrt(np.log(1e10)) / EWALD_RHO, 0.5 * max(diagonals))
