# -*- coding: utf-8 -*-
"""Tests for the `FindMuonWorkChain` class."""
import inspect

import pytest
from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_muon.workflows.find_muon import FindMuonWorkChain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generate_builder(generate_structure, fixture_code):
    """Return a factory that produces a minimal valid FindMuonWorkChain builder."""

    def _get_builder():
        structure = generate_structure("Si")
        sc_matrix = orm.List([[[2, 0, 0], [0, 2, 0], [0, 0, 2]]])
        code = fixture_code("quantumespresso.pw")

        builder = FindMuonWorkChain.get_builder()
        builder.structure = structure
        builder.sc_matrix = sc_matrix

        pw_params = orm.Dict({
            "CONTROL": {"calculation": "relax", "forc_conv_thr": 1.0e-3},
            "SYSTEM": {"ecutwfc": 30.0, "ecutrho": 240.0},
            "ELECTRONS": {"conv_thr": 1.0e-4},
        })
        builder.relax.base.pw.code = code
        builder.relax.base.pw.parameters = pw_params
        builder.relax.base.pw.metadata.options.resources = {"num_machines": 1}

        return builder

    return _get_builder


@pytest.fixture
def generate_workchain(generate_builder):
    """Return a factory that instantiates FindMuonWorkChain from the default builder."""

    def _generate_workchain():
        builder = generate_builder()
        runner = get_manager().get_runner()
        return instantiate_process(runner, builder)

    return _generate_workchain


# ---------------------------------------------------------------------------
# Spec tests – class-level introspection, no calculation launched
# ---------------------------------------------------------------------------

def test_spec_exit_codes():
    """All expected error exit codes must be registered on the spec."""
    exit_codes = FindMuonWorkChain.spec().exit_codes
    assert exit_codes.ERROR_MUSCONV_CALC_FAILED.status == 404
    assert exit_codes.ERROR_RELAX_CALC_FAILED.status == 405
    assert exit_codes.ERROR_BASE_CALC_FAILED.status == 406
    assert exit_codes.ERROR_PP_CALC_FAILED.status == 407
    assert exit_codes.ERROR_NO_SUPERCELLS.status == 408


def test_spec_required_outputs():
    """Required outputs must be present in the spec."""
    outputs = FindMuonWorkChain.spec().outputs
    for name in ("all_sites", "unique_sites", "all_index_uuid"):
        assert name in outputs, f"Output '{name}' missing from spec"


def test_spec_optional_outputs():
    """Optional magnetic outputs must be registered with required=False."""
    outputs = FindMuonWorkChain.spec().outputs
    assert "unique_sites_hyperfine" in outputs
    assert "unique_sites_dipolar" in outputs
    assert outputs["unique_sites_hyperfine"].required is False
    assert outputs["unique_sites_dipolar"].required is False


def test_spec_default_mu_spacing(aiida_profile):
    """mu_spacing input should default to 1.0 Å."""
    default = FindMuonWorkChain.spec().inputs["mu_spacing"].default()
    assert abs(default.value - 1.0) < 1e-9


def test_spec_default_spin_pol_dft(aiida_profile):
    """spin_pol_dft input should default to True."""
    assert FindMuonWorkChain.spec().inputs["spin_pol_dft"].default().value is True


def test_spec_default_niche_atom(aiida_profile):
    """niche_atom input should default to 'H'."""
    assert FindMuonWorkChain.spec().inputs["niche_atom"].default().value == "H"


def test_spec_default_pseudo_family(aiida_profile):
    """pseudo_family input should default to SSSP/1.3/PBE/efficiency."""
    assert FindMuonWorkChain.spec().inputs["pseudo_family"].default().value == "SSSP/1.3/PBE/efficiency"


def test_spec_default_ml_pre_relax(aiida_profile):
    """ML_pre_relax should be False by default."""
    assert FindMuonWorkChain.spec().inputs["ML_pre_relax"].default().value is False


def test_spec_default_gamma_pre_relax(aiida_profile):
    """gamma_pre_relax should be False by default."""
    assert FindMuonWorkChain.spec().inputs["gamma_pre_relax"].default().value is False


def test_spec_default_full_dft_relax(aiida_profile):
    """full_dft_relax should be False by default."""
    assert FindMuonWorkChain.spec().inputs["full_dft_relax"].default().value is False


# ---------------------------------------------------------------------------
# Workflow condition methods
# ---------------------------------------------------------------------------

def test_check_converge_supercell_size_false_when_sc_matrix_given(aiida_profile, generate_workchain):
    """check_converge_supercell_size returns False when sc_matrix is provided as input."""
    process = generate_workchain()
    assert process.check_converge_supercell_size() is False


def test_structure_is_magnetic_false_without_magmom(aiida_profile, generate_workchain):
    """structure_is_magnetic returns False when no magmom input is provided."""
    process = generate_workchain()
    assert process.structure_is_magnetic() is False


def test_spin_polarized_dft_true_by_default(aiida_profile, generate_workchain):
    """spin_polarized_dft returns True because spin_pol_dft defaults to True."""
    process = generate_workchain()
    assert process.spin_polarized_dft() is True


def test_should_run_mlip_relaxation_false_by_default(aiida_profile, generate_workchain):
    """should_run_mlip_relaxation returns False because ML_pre_relax defaults to False."""
    process = generate_workchain()
    assert process.should_run_mlip_relaxation() is False


def test_should_run_full_relaxations_false_by_default(aiida_profile, generate_workchain):
    """should_run_full_relaxations returns False because full_dft_relax defaults to False."""
    process = generate_workchain()
    assert process.should_run_full_relaxations() is False


# ---------------------------------------------------------------------------
# Muon site generation
# ---------------------------------------------------------------------------

def test_get_initial_muon_sites_populates_mu_list(aiida_profile, generate_workchain):
    """get_initial_muon_sites should populate ctx.mu_lst for a Si structure."""
    process = generate_workchain()
    # Run prerequisite steps that set up ctx without launching any calculations
    process.pre_check_structure_data_compatibility()
    process.check_converge_supercell_size()
    process.setup()
    result = process.get_initial_muon_sites()
    # Expect at least one muon site; no early-exit ExitCode should be returned
    assert result is None
    assert isinstance(process.ctx.mu_lst, list)
    assert len(process.ctx.mu_lst) > 0


# ---------------------------------------------------------------------------
# Builder protocol – signature inspection only (no pseudo family required)
# ---------------------------------------------------------------------------

def test_get_builder_from_protocol_signature():
    """get_builder_from_protocol must accept the key public parameters."""
    sig = inspect.signature(FindMuonWorkChain.get_builder_from_protocol)
    params = sig.parameters
    for name in ("pw_code", "structure", "mu_spacing", "sc_matrix", "magmom",
                 "spin_pol_dft", "ML_pre_relax", "gamma_pre_relax", "full_dft_relax"):
        assert name in params, f"Parameter '{name}' missing from get_builder_from_protocol"