import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_core_internal_operator_decomposition_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_core_internal", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_manifest_order_exposes_kernel_boundary_before_core_output():
    runner = load_runner()

    ids = [stage["stage_id"] for stage in runner.build_stage_manifest()]

    assert ids.index("STAGE_02_KDA_KERNEL_BOUNDARY") < ids.index("STAGE_03_CORE_OUTPUT")
    assert "STAGE_02A_KERNEL_DECAY_UPDATE_READOUT_INTERNAL" in ids


def test_identity_status_uses_prespecified_tolerances():
    runner = load_runner()

    assert runner.classify_identity_error(1e-7, 1e-8) == "IDENTITY_PASS"
    assert runner.classify_identity_error(5e-4, 5e-7) == "IDENTITY_APPROXIMATE"
    assert runner.classify_identity_error(0.37, 1e-4) == "IDENTITY_FAIL"
    assert runner.classify_identity_error(None, None, theoretical_valid=False) == "NOT_APPLICABLE_NONLINEAR"


def test_earliest_stage_helpers_select_first_matching_stage():
    runner = load_runner()
    rows = [
        {"stage_id": "STAGE_00", "identity_status": "IDENTITY_PASS", "N_SIGN_CONCORDANT": 10, "layer22_reverse": "NO"},
        {"stage_id": "STAGE_01", "identity_status": "IDENTITY_FAIL", "N_SIGN_CONCORDANT": 17, "layer22_reverse": "NO"},
        {"stage_id": "STAGE_02", "identity_status": "IDENTITY_FAIL", "N_SIGN_CONCORDANT": 18, "layer22_reverse": "YES"},
    ]

    assert runner.earliest_identity_failure(rows) == "STAGE_01"
    assert runner.earliest_sign_closure(rows, target=18) == "STAGE_02"
    assert runner.earliest_sign_closure(rows, target=17) == "STAGE_01"
    assert runner.earliest_layer22_reverse(rows) == "STAGE_02"


def test_layer_sign_concordance_preserves_layer22_reverse_control():
    runner = load_runner()
    layer_rows = [
        {"stage_id": "STAGE_03_CORE_OUTPUT", "layer_id": 0, "delta_G": 0.2},
        {"stage_id": "STAGE_03_CORE_OUTPUT", "layer_id": 22, "delta_G": -0.1},
        {"stage_id": "STAGE_03_CORE_OUTPUT", "layer_id": 21, "delta_G": -0.05},
    ]
    prior = {0: "C_SENSITIVE", 21: "C_SENSITIVE", 22: "R_SENSITIVE"}

    rows = runner.stage_layer_sign_concordance(layer_rows, prior)

    assert rows[0]["N_SIGN_CONCORDANT"] == 2
    assert rows[0]["DISCORDANT_LAYER_IDS"] == "21"
    assert rows[0]["LAYER22_REVERSE_CONTROL"] == "PASS"


def test_kernel_internal_classification_when_boundary_is_unresolved_and_core_output_closes():
    runner = load_runner()
    master = [
        {"stage_id": "STAGE_02_KDA_KERNEL_BOUNDARY", "identity_status": "NOT_APPLICABLE_BOUNDARY", "N_SIGN_CONCORDANT": 0, "layer22_reverse": "UNRESOLVED"},
        {"stage_id": "STAGE_03_CORE_OUTPUT", "identity_status": "IDENTITY_FAIL", "N_SIGN_CONCORDANT": 18, "layer22_reverse": "YES"},
    ]

    out = runner.classify_localization(master, first_unresolved="STAGE_02_KDA_KERNEL_BOUNDARY")

    assert out["OPERATOR_LOCALIZATION_STATUS"] == "KERNEL_INTERNAL"
    assert out["FINAL_CLASSIFICATION"] == "KDA_LOCAL_DIRECTIONAL_GEOMETRY_LOCALIZED_INSIDE_KDA_KERNEL"
    assert out["KERNEL_INTERNAL_DECOMPOSITION_REQUIRED"] == "YES"


def test_master_table_keeps_unobserved_stage_signs_unresolved():
    runner = load_runner()
    identity_rows = [
        {"stage_id": "STAGE_02A_KERNEL_DECAY_UPDATE_READOUT_INTERNAL", "identity_status": "UNRESOLVED_KERNEL_INTERNAL", "median_identity_rel_error": None},
        {"stage_id": "STAGE_03_CORE_OUTPUT", "identity_status": "IDENTITY_FAIL", "median_identity_rel_error": 0.37},
    ]
    sign_rows = [{"stage_id": "STAGE_03_CORE_OUTPUT", "N_SIGN_CONCORDANT": 18, "CONCORDANCE_RATE": 1.0}]
    l22_rows = [{"stage_id": "STAGE_03_CORE_OUTPUT", "layer22_reverse": "YES"}]

    rows = runner.operator_decomposition_master_table(identity_rows, sign_rows, l22_rows)

    assert rows[0]["N_SIGN_CONCORDANT"] == "UNRESOLVED"
    assert rows[0]["CONCORDANCE_RATE"] == "UNRESOLVED"


def test_prior_label_gate_validates_expected_layer_mapping():
    runner = load_runner()
    rows = [
        {"layer_id": str(layer), "layer_classification": "R_SENSITIVE" if layer == 22 else "C_SENSITIVE"}
        for layer in runner.KDA_LAYERS
    ]
    bad_rows = list(rows)
    bad_rows[0] = {"layer_id": "0", "layer_classification": "R_SENSITIVE"}

    assert runner.prior_layer_label_reuse_gate(rows) == "PASS"
    assert runner.prior_layer_label_reuse_gate(bad_rows) == "FAIL"
