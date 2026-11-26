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
        builder.relax.base_relax.pw.code = code
        builder.relax.base_relax.pw.parameters = pw_params
        builder.relax.base_relax.pw.metadata.options.resources = {"num_machines": 1}

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
    """full_dft_relax should be True by default."""
    assert FindMuonWorkChain.spec().inputs["full_dft_relax"].default().value is True


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
    assert process.should_run_full_relaxations() is True


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

# ---------------------------------------------------------------------------
# No relaxation is done twice: the cluster analysis after a skipped Gamma pre-relaxation
# ---------------------------------------------------------------------------

class _FakeWorkChain:
    """Just what `run_cluster_analysis` and `new_struct_after_analyze` use, to follow the control flow of the outline.

    `analyze_relaxed_structures` is replaced by a recorder: the 'pre-clustering' analysis keeps one (relaxed) structure
    per cluster, the 'full' one only the magnetically inequivalent sites (none here), as the real ones do.
    """

    should_pre_clustering = FindMuonWorkChain.should_pre_clustering
    run_cluster_analysis = FindMuonWorkChain.run_cluster_analysis
    new_struct_after_analyze = FindMuonWorkChain.new_struct_after_analyze

    def __init__(self, run_type, pre_clustering, gamma_pre_relax, pre_clustering_done=False):
        from types import SimpleNamespace

        self.inputs = SimpleNamespace(
            pre_clustering=SimpleNamespace(value=pre_clustering), gamma_pre_relax=SimpleNamespace(value=gamma_pre_relax)
        )
        self.ctx = SimpleNamespace(
            run_type=run_type,
            supc_list=["s0", "s1", "s2"],
            relaxed_outputs=[1, 2, 3],
            has_magnetic_inequivalent=False,
            pre_clustering_done=pre_clustering_done,
        )
        self.modes = []

    def report(self, message):
        pass

    def analyze_relaxed_structures(self, mode="pre-clustering", d_tol=0.5):
        self.modes.append(mode)
        self.ctx.supc_list = ["s0"] if mode == "pre-clustering" else []


@pytest.mark.parametrize("pre_clustering", [True, False])
@pytest.mark.parametrize("gamma_pre_relax", [True, False])
def test_full_relaxation_is_followed_by_the_full_analysis_and_no_second_round(pre_clustering, gamma_pre_relax):
    """After the full relaxation the full analysis is done, whatever happened before.

    In particular when the Gamma pre-relaxation was skipped (the mesh is 1x1x1) with `pre_clustering` and
    `gamma_pre_relax` both True: the pre-clustering must not be done on the fully relaxed structures, because then
    `new_struct_after_analyze` would send the cluster representatives to a second full relaxation.
    """
    workchain = _FakeWorkChain("full", pre_clustering, gamma_pre_relax)
    workchain.run_cluster_analysis()
    assert workchain.modes == ["full"]
    assert workchain.new_struct_after_analyze() is False  # nothing magnetically inequivalent: no second round


def test_pre_clustering_after_the_gamma_pre_relaxation_is_unchanged():
    """After the Gamma pre-relaxation, with `pre_clustering`, the pre-clustering is done (once), and a new round follows."""
    workchain = _FakeWorkChain("gamma", pre_clustering=True, gamma_pre_relax=True)
    workchain.run_cluster_analysis()
    assert workchain.modes == ["pre-clustering"]
    assert workchain.ctx.pre_clustering_done is True
    assert workchain.new_struct_after_analyze() is True  # the representatives are relaxed with the full mesh

    # a second pre-relaxation (e.g. Gamma after an MLIP one) does not repeat it
    workchain = _FakeWorkChain("gamma", True, True, pre_clustering_done=True)
    workchain.run_cluster_analysis()
    assert workchain.modes == []


def test_no_pre_clustering_after_a_pre_relaxation_when_not_asked():
    workchain = _FakeWorkChain("gamma", pre_clustering=False, gamma_pre_relax=True)
    workchain.run_cluster_analysis()
    assert workchain.modes == []


def test_gamma_pre_relaxation_is_skipped_for_a_1x1x1_mesh(aiida_profile, generate_builder):
    """With a mesh of one k-point, Gamma is the only sampled point: no pre-relaxation, and the full ones use Gamma only."""
    builder = generate_builder()
    builder.gamma_pre_relax = orm.Bool(True)
    builder.kpoints_distance = orm.Float(100.0)  # a huge spacing: the mesh is 1x1x1
    process = instantiate_process(get_manager().get_runner(), builder)
    process.ctx.supc_list = [builder.structure]
    process.ctx.non_collinear = False
    process.ctx.set_gamma_only = False

    assert process.should_run_gamma_relaxations() is False
    assert process.ctx.set_gamma_only is True
    assert process.should_run_full_relaxations() is True  # the (only) relaxation


def test_gamma_pre_relaxation_is_run_for_a_denser_mesh(aiida_profile, generate_builder):
    builder = generate_builder()
    builder.gamma_pre_relax = orm.Bool(True)
    builder.kpoints_distance = orm.Float(0.1)  # a dense mesh
    process = instantiate_process(get_manager().get_runner(), builder)
    process.ctx.supc_list = [builder.structure]
    process.ctx.non_collinear = False
    process.ctx.set_gamma_only = False

    assert process.should_run_gamma_relaxations() is True
    assert process.ctx.run_type == "gamma"
