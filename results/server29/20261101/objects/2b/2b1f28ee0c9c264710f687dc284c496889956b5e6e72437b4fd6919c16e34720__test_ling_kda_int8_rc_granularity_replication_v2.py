import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_replication_v2.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ling_kda_replication_v2", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_declares_two_panel_grid_and_counts():
    runner = load_runner()

    assert runner.TASK == "LING_KDA_INT8_RC_GRANULARITY_PHASE_DIAGRAM_REPLICATION_V2"
    assert runner.PANELS == ["exact", "independent"]
    assert runner.CONFIGS == ["R128", "R64", "R32", "R16", "C128", "C64", "C32", "C16"]
    assert runner.HISTORY_LENGTHS == [1, 4, 8, 16, 32, 64]
    assert runner.expected_panel_runs() == 864
    assert runner.expected_total_runs() == 1728
    assert runner.REPLICATION_MANIFEST_SEED == 20260910


def test_manifest_construction_is_prompt_disjoint_and_fixed_seed():
    runner = load_runner()
    original = [
        {"unit_id": "60|64", "problem_id": "60", "problem_index": 0, "t0": 64, "teacher_forced_token_ids": [1, 2]},
        {"unit_id": "60|128", "problem_id": "60", "problem_index": 0, "t0": 128, "teacher_forced_token_ids": [2, 3]},
        {"unit_id": "61|64", "problem_id": "61", "problem_index": 1, "t0": 64, "teacher_forced_token_ids": [3, 4]},
    ]
    rows_by_pid = {str(i): {"problem_id": str(i), "problem_index": i} for i in range(60, 90)}
    teacher_tokens = {str(i): list(range(400 + i)) for i in range(60, 90)}

    manifest = runner.construct_independent_manifest(
        original, rows_by_pid, teacher_tokens, n_prompts=3, t0_values=[64, 128, 256], seed=20260910
    )
    audit = runner.independence_audit(original, manifest["units"])

    assert manifest["seed"] == 20260910
    assert len(manifest["units"]) == 9
    assert audit["REPLICATION_INDEPENDENCE_LEVEL"] == "PROMPT_DISJOINT"
    assert audit["PROMPT_OVERLAP_COUNT"] == 0
    assert audit["UNIT_KEY_OVERLAP_COUNT"] == 0
    assert len({u["problem_id"] for u in manifest["units"]}) == 3


def test_c_minus_r_cell_and_group_statistics_use_units_not_unit_history_rows():
    runner = load_runner()
    rows = []
    for unit in ["u0", "u1"]:
        for history in runner.HISTORY_LENGTHS:
            rows.append({"unit_id": unit, "group_size": 64, "history_length": history, "orientation": "R", "future_kl_auc": 0.2})
            rows.append({"unit_id": unit, "group_size": 64, "history_length": history, "orientation": "C", "future_kl_auc": 0.1})

    cell = runner.panel_analysis(rows)

    assert cell["C_MINUS_R_GAP_MATRIX"]["64"]["1"] == pytest.approx(-0.1)
    assert cell["CLASS_MATRIX"]["64"]["1"] == "C_PREFERRED"
    assert cell["GROUP_STATS"]["64"]["n"] == 2
    assert cell["GROUP_PATTERN"] == "TIE/C/TIE/TIE"


def test_replication_classifiers_follow_predeclared_g64_rules():
    runner = load_runner()
    strong = {
        "GROUP_STATS": {
            "128": {"median": 0.2, "bootstrap_CI": [0.1, 0.3]},
            "64": {"median": -0.2, "bootstrap_CI": [-0.3, -0.1]},
            "32": {"median": 0.2, "bootstrap_CI": [0.1, 0.3]},
            "16": {"median": 0.2, "bootstrap_CI": [0.1, 0.3]},
        },
        "CELL_STATS": {
            "64": {str(L): {"median": -0.1, "bootstrap_CI": [-0.2, -0.01]} for L in runner.HISTORY_LENGTHS}
        },
        "GROUP_PATTERN": "R/C/R/R",
    }

    assert runner.classify_g64_replication(strong) == "STRONG"
    assert runner.classify_full_pattern_replication(strong) == "STRONG"

    partial = dict(strong)
    partial["GROUP_STATS"] = dict(strong["GROUP_STATS"])
    partial["GROUP_STATS"]["64"] = {"median": -0.01, "bootstrap_CI": [-0.1, 0.1]}
    partial["CELL_STATS"] = {"64": {str(L): {"median": -0.1 if L in [1, 4, 8] else 0.1, "bootstrap_CI": [-0.2, 0.2]} for L in runner.HISTORY_LENGTHS}}
    partial["GROUP_PATTERN"] = "R/TIE/R/R"
    assert runner.classify_g64_replication(partial) == "PARTIAL"
    assert runner.classify_full_pattern_replication(partial) == "PARTIAL"


def test_final_console_summary_contains_required_keys():
    runner = load_runner()
    summary = {key: "PASS" for key in runner.REQUIRED_CONSOLE_KEYS}
    summary.update({"TASK": runner.TASK, "METHOD_DESIGN_READY": "NO"})

    text = runner.render_console_summary(summary)

    for key in runner.REQUIRED_CONSOLE_KEYS:
        assert f"{key} =" in text
    assert text.rstrip().endswith("STOP.")
