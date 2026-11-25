"""Tests of the identifiers of the muon sites in `aiida_muon.utils.export_findmuon`.

The `FindMuonWorkChain` node is replaced by a light fake, so no calculation is needed. The fake deliberately lists the
sites in different orders in `all_index_uuid` and in `all_sites`, with a two-digit supercell index, and with the
clustering removing sites: the cases in which the position of a site in a dictionary is not its index.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from aiida_muon.utils import export_findmuon as ef

N_HOST = 8  # atoms of the (unit cell of the) host
# supercell index -> total energy (eV). Not in the order of the indexes.
ENERGIES = {"5": -10.00, "0": -10.50, "3": -9.90, "1": -10.49, "10": -9.00, "2": -9.89}
ALL_SITES_ORDER = ["5", "0", "3", "1", "10", "2"]  # the order of the sites in the `all_sites` output
ALL_INDEX_UUID_ORDER = ["0", "1", "2", "3", "5", "10"]  # deliberately a different one
# clustering: 1-based position in ALL_SITES_ORDER of the representative of each site. The sites "1" and "2" are
# in the cluster of "0" and "3", respectively, so the unique sites are {"0", "3", "5", "10"}.
MAPPING = [1, 2, 3, 2, 5, 3]
UNIQUE = {"5", "0", "3", "10"}


class FakeDict:
    def __init__(self, data):
        self._data = data

    def get_dict(self):
        return self._data


def site_entry(index):
    """The `[structure, energy]` entry of `all_sites`: a host atom and the muon, as a pymatgen dictionary."""
    structure = Structure(Lattice.cubic(5.0), ["Si", "H"], [[0, 0, 0], [0.05 * (int(index) + 1), 0.2, 0.3]])
    return [structure.as_dict(), ENERGIES[index]]


@pytest.fixture
def host_structure():
    """The unit cell of the host: `N_HOST` Si atoms."""
    coords = [[x / 2, y / 2, z / 2] for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    return Structure(Lattice.cubic(5.0), ["Si"] * N_HOST, coords)


@pytest.fixture
def fake_findmuon(host_structure):
    caller = SimpleNamespace(
        inputs=SimpleNamespace(
            structure=SimpleNamespace(sites=list(host_structure), get_pymatgen=lambda: host_structure),
            sc_matrix=SimpleNamespace(get_list=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )
    )
    all_index_uuid = FakeDict({key: f"uuid-{key}" for key in ALL_INDEX_UUID_ORDER})
    all_index_uuid.creator = SimpleNamespace(caller=caller)

    class FakeFindMuon:
        def __init__(self):
            self.all_index_uuid = all_index_uuid
            self.all_sites = FakeDict({key: site_entry(key) for key in ALL_SITES_ORDER})
            self.unique_sites = FakeDict({key: site_entry(key) for key in ALL_SITES_ORDER if key in UNIQUE})

        def __contains__(self, name):  # the magnetic fields are only there if a test adds them
            return name in ("unique_sites_dipolar", "unique_sites_hyperfine") and hasattr(self, name)

    return FakeFindMuon()


@pytest.fixture
def patched_export(monkeypatch):
    """Replace what needs the database or a calculation: the relaxation nodes, the distortions and the clustering."""

    def load_node(uuid):
        key = uuid.replace("uuid-", "")
        structure = SimpleNamespace(pk=1000 + int(key), get_ase=lambda: None)
        return SimpleNamespace(
            is_finished_ok=True,
            outputs=SimpleNamespace(
                output_parameters=SimpleNamespace(get_dict=lambda: {"energy": ENERGIES[key]}),
                output_structure=structure,
            ),
            inputs=SimpleNamespace(structure=SimpleNamespace(get_ase=lambda: None)),
        )

    monkeypatch.setattr(ef.orm, "load_node", load_node)
    monkeypatch.setattr(ef, "get_distortions", lambda *args, **kwargs: {})
    monkeypatch.setattr(ef, "get_clustering_after_run", lambda node: {"mapping": np.array(MAPPING)})


def test_numeric_order_of_the_site_keys(fake_findmuon):
    assert ef.get_ordered_site_keys(fake_findmuon, before_clustering=True) == ["0", "1", "2", "3", "5", "10"]
    assert ef.get_ordered_site_keys(fake_findmuon) == ["0", "3", "5", "10"]


def test_unit_cell_atom_indices_are_zero_based_after_the_host():
    assert ef.get_unit_cell_atom_indices(8, ["0", "3", "10"]) == {"0": 8, "3": 9, "10": 10}


def test_atom_indices_of_the_tables(fake_findmuon, patched_export):
    df, df_all, _ = ef.produce_muonic_dataframe(fake_findmuon)

    # the unique sites: indexes in `unit_cell`, which only has these muons.
    unique_by_supercell = {int(row.supercell_index): row.atom_index_in_unit_cell for _, row in df.iterrows()}
    assert unique_by_supercell == {0: N_HOST, 3: N_HOST + 1, 5: N_HOST + 2, 10: N_HOST + 3}

    # all the sites: indexes in `unit_cell_all`.
    all_by_supercell = {int(row.supercell_index): row.atom_index_in_unit_cell for _, row in df_all.iterrows()}
    assert all_by_supercell == {0: 8, 1: 9, 2: 10, 3: 11, 5: 12, 10: 13}

    # the index of the table is the supercell index, and it is named
    assert df.index.name == df_all.index.name == "supercell_index"
    assert list(df.index) == list(df["supercell_index"])
    assert list(df_all.index) == list(df_all["supercell_index"])


def test_clustering_follows_the_order_of_all_sites(fake_findmuon, patched_export):
    """The mapping is listed in the order of `all_sites`, not of `all_index_uuid`."""
    _, df_all, _ = ef.produce_muonic_dataframe(fake_findmuon)
    # representative -> its own group; "1" is in the group of "0", "2" in the one of "3".
    groups = {int(idx): group for idx, group in df_all["cluster group"].items()}
    assert groups[1] == groups[0]
    assert groups[2] == groups[3]
    assert len({groups[0], groups[3], groups[5], groups[10]}) == 4
    assert sorted(set(groups.values())) == [1, 2, 3, 4]


def test_deprecated_columns(fake_findmuon, patched_export):
    df, df_all, _ = ef.produce_muonic_dataframe(fake_findmuon)
    for table in (df, df_all):
        # the old names are kept, with the old format: the index as a string, and the atom index as 1-based.
        assert list(table["muon_index"]) == [str(i) for i in table["supercell_index"]]
        assert list(table["muon_index_global_unitcell"]) == list(table["atom_index_in_unit_cell"] + 1)

    df, df_all, _ = ef.produce_muonic_dataframe(fake_findmuon, include_deprecated_columns=False)
    for table in (df, df_all):
        assert "muon_index" not in table.columns
        assert "muon_index_global_unitcell" not in table.columns


@pytest.mark.parametrize("before_clustering", [False, True])
def test_the_atom_index_is_the_muon_of_that_site(fake_findmuon, patched_export, aiida_profile, before_clustering):
    """The invariant behind the column: the atom at `atom_index_in_unit_cell` is the muon of that site."""
    df, df_all, _ = ef.produce_muonic_dataframe(fake_findmuon)
    table = df_all if before_clustering else df
    unit_cell = ef.produce_collective_unit_cell(fake_findmuon, before_clustering=before_clustering)

    assert len(unit_cell.sites) == N_HOST + len(table)
    for _, row in table.iterrows():
        atom = row.atom_index_in_unit_cell
        assert unit_cell.tags[atom] == str(row.supercell_index)
        # (the kind name is `H<index>` plus what pymatgen makes of it, e.g. an oxidation state: only check the element)
        assert unit_cell.get_kind(unit_cell.sites[atom].kind_name).symbol == "H"
        # and it is where the muon of that site was relaxed (crystal coordinates of the unit cell, sc_matrix = 1)
        assert np.allclose(unit_cell.sites[atom].position, np.array(row.muon_position_cc) * 5.0, atol=1e-2)


def test_positions_and_fields_are_lists_of_python_floats(fake_findmuon, patched_export):
    """A list of `np.float64` is printed as `[np.float64(0.1), ...]` (numpy 2), e.g. in the csv files of the app."""
    df, df_all, _ = ef.produce_muonic_dataframe(fake_findmuon)
    for table in (df, df_all):
        for position in table["muon_position_cc"]:
            assert isinstance(position, list)
            assert all(type(coordinate) is float for coordinate in position)
    assert "np.float64" not in df.to_csv() + df_all.to_csv()


def test_fields_are_lists_of_python_floats(fake_findmuon, patched_export):
    fake_findmuon.unique_sites_dipolar = SimpleNamespace(
        get_list=lambda: [
            {"idx": int(key), "B_T": np.array([0.1234567, 0.0, 0.2]), "Bdip": np.array([0.01, 0.02, 0.03])}
            for key in UNIQUE
        ]
    )
    fake_findmuon.unique_sites_hyperfine = SimpleNamespace(get_dict=lambda: {key: [0.5, 0.25] for key in UNIQUE})
    df, _, _ = ef.produce_muonic_dataframe(fake_findmuon)
    for column in ("B_T", "Bdip"):
        for value in df[column]:
            assert isinstance(value, list)
            assert all(type(component) is float for component in value)
    assert list(df["B_hf_norm"]) == [0.25] * len(df)  # the last element: in Tesla
    assert "np.float64" not in df.to_csv()
