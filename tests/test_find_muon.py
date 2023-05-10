# -*- coding: utf-8 -*-
"""Tests for the `FindMuonWorkChain` class."""
import pytest
from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_muon.workflows.find_muon import FindMuonWorkChain


@pytest.fixture
def generate_builder(generate_structure, fixture_code):
    """Generate default inputs for `FindMuonWorkChain`"""

    def _get_builder():
        """Generate default builder for `FindMuonWorkChain`"""

        inputstructure = generate_structure("Si")
        scmat_node = orm.List(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ]
        )
        code = fixture_code("quantumespresso.pw")

        builder = FindMuonWorkChain.get_builder()
        builder.structure = inputstructure
        builder.qe.pw_code = code
        builder.sc_matrix = scmat_node

        paramters = {
            "CONTROL": {
                "max_seconds": 45000,
                "forc_conv_thr": 0.1,
                "etot_conv_thr": 0.1,
            },
            "SYSTEM": {
                "ecutwfc": 30.0,
                "ecutrho": 240.0,
            },
            "ELECTRONS": {
                "conv_thr": 1.0e-4,
                "electron_maxstep": 100,
            },
        }

        pw_metadata = {
            "description": "Si test",
            #'dry_run' : True,
            "options": {
                "max_wallclock_seconds": 50000,
                "resources": {"num_machines": 1},
            },
            "label": "Si Fe MnO  relax test",
        }

        pw_settings = {"ONLY_INITIALIZATION": True}

        builder.qe.parameters = orm.Dict(dict=paramters)
        builder.qe.metadata = orm.Dict(dict=pw_metadata)
        # builder.qe.settings =orm.Dict(dict=pw_settings)

        return builder

    return _get_builder


@pytest.fixture
def generate_workchain(generate_builder):
    """Generate an instance of FindMuonWorkChain"""

    def _generate_workchain(exit_code=None):
        builder = generate_builder()
        runner = get_manager().get_runner()
        process = instantiate_process(runner, builder)

        # if exit_code is not None:
        #    node = generate_calc_job_node(
        #    entry_point_calc_job, fixture_localhost, test_name, inputs["FindMuonWorkChain"]
        #    )
        #    node.set_process_state(ProcessState.FINISHED)
        #    node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain


def test_getmuon(aiida_profile, generate_workchain):
    """
    Test `FindMuonWorkChain.initialization`.
    This checks that we can create the workchain successfully,
     and that it is initialised into the correct state.
    """
    process = generate_workchain()
    assert process.get_initial_muon_sites() is None
    # assert process.ctx.n.value == 0
    assert isinstance(process.ctx.mu_lst, orm.List)


def test_gethubbardu(aiida_profile, generate_workchain):
    """Test"""
    process = generate_workchain()
    assert process.setup_magnetic_hubbardu_dict() is None
    assert process.ctx.hubbardu_dict is None
