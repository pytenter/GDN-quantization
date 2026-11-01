"""Calibration-only pair selection; never opens any long-horizon label or trace."""
import math
import os
import statistics
import time
from objective import digest
from run_matched_l7 import ROOT, publish
from phase_b_calibration_common_v2 import RUNS, CONFIG, FREEZE, read, checked_json, verify, audit_b
from analyze_phase_a import (METRICS, restored_fp_identity, content_sha, finite_tree,
                             matched_tables, opposite_pairs)


def validate_completed(cfg):
    runs, identities = {}, {}
    selected = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json")
    expected_tokens = {x["document_id"]: x["token_file"]["sha256"] for x in selected["selected_documents"]}
    for doc in cfg["documents"]:
        gpu = cfg["document_gpu_map"][doc]
        path = RUNS / f"formal_{doc}_gpu{gpu}.json"
        run = checked_json(path)
        assert run["valid"] and finite_tree(run) and run["mode"] == "formal"
        assert run["document"] == doc and run["gpu_uuid"] == cfg["gpu_uuids"][str(gpu)]
        assert run["config_sha256"] == digest(CONFIG) and run["freeze_sha256"] == digest(FREEZE)
        assert run["runner_sha256"] == digest(ROOT / "run_phase_b_calibration_v2.py")
        assert run["token_file"]["sha256"] == expected_tokens[doc]
        assert not run["prior_score_set_joined"] and not run["gradients_or_optimizer_used"]
        assert not run["long_horizon_outcome_generated_or_used"]
        assert [x["condition"] for x in run["results"]] == cfg["conditions"]
        assert run["fp_reference"]["tokens"] == 1152 and run["fp_reference"]["replay_parity"]["passed"]
        assert content_sha(restored_fp_identity(run["fp_reference"]["tensor_identities"])) == run["fp_reference"]["content_sha256"]
        checks = run["fp_reference"]["helper_real_forward_checks"]
        assert len(checks) == 72 and all(x["bitwise_equal"] and x["identity_loss"] == 0 for x in checks)
        for item in run["results"]:
            gates = item["mechanical_gates"]
            assert gates["passed"] and gates["every_token_layer_qdq_count"] == 1152 * 18
            assert gates["actual_next_kernel_equal_post_qdq_count"] == gates["expected_consumption_count"] == 1151 * 18
            assert [x["start"] for x in item["windows"]] == cfg["windows"]
            assert item["l7_scored_token_count"] == 256 and not item["historical_training_sampling_protocol_replayed"]
            for metric in METRICS:
                assert math.isclose(item["aggregate"][metric], statistics.mean(x[metric] for x in item["windows"]), rel_tol=1e-14)
            for scores in item["windows"] + [item["aggregate"]]:
                assert math.isclose(scores["L7_TOTAL_ORIGINAL"], scores["L7_FUNCTIONAL_ORIGINAL"] +
                                    0.1 * scores["L7_STATE_ORIGINAL"], rel_tol=2e-6, abs_tol=0)
        runs[doc] = run
        identities[doc] = dict(canonical_path=str(path), canonical_sha256=digest(path),
            sidecar_sha256=digest(path.with_suffix(".sha256.json")), input_sha256=expected_tokens[doc],
            fp_reference_content_sha256=run["fp_reference"]["content_sha256"], gpu_uuid=run["gpu_uuid"])
    return runs, identities


def winner(conditions, values, pair):
    a, b = pair["condition_a"], pair["condition_b"]
    va, vb = values[conditions.index(a)], values[conditions.index(b)]
    return a if va < vb else b if vb < va else None


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "PAIR_SELECTION_MUST_BE_CPU_ONLY"
    methods = checked_json(ROOT / "PHASE_B_PAIR_METHODS_FREEZE.json")
    assert methods["selector_sha256"] == digest(__file__)
    assert methods["analysis_helpers_sha256"] == digest(ROOT / "analyze_phase_a.py")
    cfg = verify()
    assert cfg["route"] == "A", "PRIMARY_COMPARATOR_MUST_NOT_CHANGE"
    ledger = audit_b()
    assert len(ledger["phase_b_calibration_runs"]) == 4
    assert all(x["status"] == "COMPLETE_MECHANICAL_PASS" for x in ledger["phase_b_calibration_runs"])
    runs, identities = validate_completed(cfg)
    tables, panels = matched_tables(runs, cfg)
    table_path = ROOT / "PHASE_B_CALIBRATION_MATCHED_TABLES.json"
    publish(table_path, tables)
    publish(table_path.with_suffix(".sha256.json"), dict(path=str(table_path), sha256=digest(table_path)))
    selections = []
    for doc in cfg["documents"]:
        values = panels[(doc, "MEAN_256_1024")]
        candidates = opposite_pairs(cfg["conditions"], values["P1"], values["L7_TOTAL_ORIGINAL"], 3)
        pair = candidates[0] if candidates else None
        selections.append(dict(document=doc, gpu=cfg["document_gpu_map"][doc],
            status="CALIBRATION_PAIR_SELECTED" if pair else "NO_DISCRIMINATIVE_PAIR_FOR_DOCUMENT",
            pair=pair, eligible_pairs_in_frozen_order=candidates,
            predicted_winners={metric: winner(cfg["conditions"], values[metric], pair) for metric in METRICS} if pair else {},
            pair_scores={metric: {condition: values[metric][cfg["conditions"].index(condition)]
                for condition in (pair["condition_a"], pair["condition_b"])} for metric in METRICS} if pair else {},
            calibration_identity=identities[doc]))
    informative = all(x["pair"] is not None for x in selections)
    path = ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json"
    publish(path, dict(status="BOTH_DOCUMENTS_DISCRIMINATIVE_PAIRS_FROZEN" if informative else
        "NO_DISCRIMINATIVE_PAIR_FOR_DOCUMENT", epoch=time.time(), route="A", documents=selections,
        selection_rule="opposite sign, both complete-roster midrank gaps>=3, descending min-gap then ordered scientific-key lexicographic tie-break",
        comparator=["P1", "L7_TOTAL_ORIGINAL"], scoring_windows=[256, 1024], horizon=128,
        aggregation="equal mean of the two frozen windows", lower_score_is_safer=True,
        primary="mean full-vocabulary FP32 KL(FP||student) on token positions [8192,32768)",
        secondary="mean same-trajectory KL(FP||student) on [2048,32768)",
        signed_primary_gap="KL(P1_predicted_winner)-KL(L7_TOTAL_predicted_winner); negative supports P1",
        independent_document_count=2, max_formal_32k_units=4, max_gpu_hours=16,
        selection_code_sha256=digest(__file__), calibration_code_sha256=digest(ROOT / "run_phase_b_calibration_v2.py"),
        calibration_freeze_sha256=digest(FREEZE), methods_freeze_sha256=digest(ROOT / "PHASE_B_PAIR_METHODS_FREEZE.json"),
        document_selection_sha256=digest(ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json"),
        model_and_rotation_identities=read(ROOT / "SOURCE_IDENTITIES.json"),
        calibration_tables_sha256=digest(table_path), completed_gpu_hours_upper_bound=ledger["total_gpu_hours_upper_bound"],
        new_long_outcomes_generated_or_read=False, long_runner_qualified=False,
        long_run_authorized=False, require_separate_persistent_correctness_and_complete_total_cost_gate=True,
        no_rule_relaxation_no_replacement_document_no_partial_validation=True))
    publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print(path.name, "BOTH_PAIRS_FROZEN_NOT_LONG_RUN_AUTHORIZATION" if informative else
          "NO_DISCRIMINATIVE_PAIR_FOR_DOCUMENT", digest(path))


if __name__ == "__main__":
    main()
