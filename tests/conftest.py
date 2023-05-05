# -*- coding: utf-8 -*-
"""Initialise a text database and profile for pytest."""
import os

import pytest
from aiida.common.folders import SandboxFolder
from aiida.orm import Code, StructureData

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]  # pylint: disable=invalid-name


@pytest.fixture(scope="session")
def filepath_tests():
    """Return the absolute filepath of the `tests` folder.
    .. warning:: if this file moves with respect to the `tests` folder,
    the implementation should change.
    :return: absolute filepath of `tests` folder which is the basepath
     for all test resources.
    """
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the directory containing the file `fixtures`."""
    return os.path.join(filepath_tests, "fixtures")


@pytest.fixture(scope="function")
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(fixture_localhost):
    """Return a `Code` instance configured to run calculations of given entry point
    on localhost `Computer`.
    """

    def _fixture_code(entry_point_name):
        return Code(
            input_plugin_name=entry_point_name,
            remote_computer_exec=[fixture_localhost, "/bin/true"],
        )

    return _fixture_code


@pytest.fixture
def generate_structure():
    """Return a ``StructureData`` representing either bulk silicon or a water molecule."""

    def _generate_structure(structure_id="Si"):
        """Return a ``StructureData`` representing bulk silicon or a snapshot
        of a single water molecule dynamics.
        :param structure_id: identifies the ``StructureData`` you want to
        generate. Either 'Si' or 'H2O' or 'GaAs'.
        """
        if structure_id == "Si":
            param = 5.43
            cell = [
                [param / 2.0, param / 2.0, 0],
                [param / 2.0, 0, param / 2.0],
                [0, param / 2.0, param / 2.0],
            ]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
            structure.append_atom(
                position=(param / 4.0, param / 4.0, param / 4.0),
                symbols="Si",
                name="Si",
            )
        elif structure_id == "H2O":
            structure = StructureData(
                cell=[
                    [5.29177209, 0.0, 0.0],
                    [0.0, 5.29177209, 0.0],
                    [0.0, 0.0, 5.29177209],
                ]
            )
            structure.append_atom(
                position=[12.73464656, 16.7741411, 24.35076238], symbols="H", name="H"
            )
            structure.append_atom(
                position=[-29.3865565, 9.51707929, -4.02515904], symbols="H", name="H"
            )
            structure.append_atom(
                position=[1.04074437, -1.64320127, -1.27035021], symbols="O", name="O"
            )
        elif structure_id == "GaAs":
            structure = StructureData(
                cell=[
                    [0.0, 2.8400940897, 2.8400940897],
                    [2.8400940897, 0.0, 2.8400940897],
                    [2.8400940897, 2.8400940897, 0.0],
                ]
            )
            structure.append_atom(position=[0.0, 0.0, 0.0], symbols="Ga", name="Ga")
            structure.append_atom(
                position=[1.42004704485, 1.42004704485, 4.26014113455],
                symbols="As",
                name="As",
            )
        elif structure_id == "BaTiO3":
            structure = StructureData(
                cell=[
                    [3.93848606, 0.0, 0.0],
                    [0.0, 3.93848606, 0.0],
                    [0.0, 0.0, 3.93848606],
                ]
            )
            structure.append_atom(position=[0.0, 0.0, 0.0], symbols="Ba", name="Ba")
            structure.append_atom(
                position=[1.969243028987539, 1.969243028987539, 1.969243028987539],
                symbols="Ti",
                name="Ti",
            )
            structure.append_atom(
                position=[0.0, 1.969243028987539, 1.969243028987539],
                symbols="O",
                name="O",
            )
            structure.append_atom(
                position=[1.969243028987539, 1.969243028987539, 0.0],
                symbols="O",
                name="O",
            )
            structure.append_atom(
                position=[1.969243028987539, 0.0, 1.969243028987539],
                symbols="O",
                name="O",
            )
        else:
            raise KeyError(f"Unknown structure_id='{structure_id}'")
        return structure

    return _generate_structure
