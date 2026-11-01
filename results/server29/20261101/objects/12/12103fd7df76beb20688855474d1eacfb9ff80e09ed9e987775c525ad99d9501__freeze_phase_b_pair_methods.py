"""Synthetic CPU tests and immutable selection method receipt before full16 outcomes."""
import os
import time
from objective import digest
from run_matched_l7 import ROOT, publish
from phase_b_calibration_common_v2 import RUNS, read, verify
from analyze_phase_a import opposite_pairs, midranks
from select_phase_b_pairs import winner


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cfg = verify()
    assert not list(RUNS.glob("formal_*.LAUNCH_AUTHORIZATION.json")), "FREEZE_BEFORE_FORMAL_CALIBRATION"
    assert digest(ROOT / "analyze_phase_a.py") == "e8f260576e022cbfbe2a60d02b10b04cb6f6d6b0f59b50929137cd060f40408a"
    names = [f"R{i:02}" for i in range(16)]
    ascending, descending = list(range(16)), list(reversed(range(16)))
    pairs = opposite_pairs(names, ascending, descending, 3)
    assert pairs[0]["condition_a"] == "R00" and pairs[0]["condition_b"] == "R15"
    assert pairs[0]["minimum_rank_gap"] == 15
    assert opposite_pairs(names, ascending, ascending, 3) == []
    assert opposite_pairs(names, [0] * 16, descending, 3) == []
    assert opposite_pairs(["b", "a", "d", "c"], [0, 1, 2, 3], [3, 2, 1, 0], 3)[0]["condition_a"] == "b"
    assert midranks([1, 2, 2, 4]) == [1, 2.5, 2.5, 4]
    assert winner(names, ascending, pairs[0]) == "R00"
    assert winner(names, descending, pairs[0]) == "R15"
    assert winner(names, [0] * 16, pairs[0]) is None
    assert opposite_pairs(list(reversed(names)), descending, ascending, 3) == pairs
    path = ROOT / "PHASE_B_PAIR_METHODS_FREEZE.json"
    publish(path, dict(status="SYNTHETIC_CPU_SELECTION_TESTS_PASS_NOT_SCIENTIFIC_RESULT", epoch=time.time(),
        selector_sha256=digest(ROOT / "select_phase_b_pairs.py"),
        analysis_helpers_sha256=digest(ROOT / "analyze_phase_a.py"),
        freeze_code_sha256=digest(__file__), route="A", minimum_gap_both=3,
        rule="opposite score signs; exact-equality midranks; descending minimum gap then ordered scientific-key lexicographic tie-break",
        no_informative_pair_behavior="retain evidence and do not launch any32K; no document replacement or threshold relaxation",
        scoring_basis="equal mean windows256/1024 H128 all18 KDA/all16heads, full16 roster",
        new_formal_calibration_outcomes_generated=False, future_outcomes_read=False, gpu_hours=0))
    publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print("B_PAIR_METHODS_CPU_TESTED_AND_FROZEN", digest(path))


if __name__ == "__main__":
    main()
