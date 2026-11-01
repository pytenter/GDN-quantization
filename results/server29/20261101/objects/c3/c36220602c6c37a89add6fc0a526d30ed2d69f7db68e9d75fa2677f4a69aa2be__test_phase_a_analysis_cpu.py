"""Synthetic-only statistics/selection tests; not GPU or scientific results."""
import json
import math
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from analyze_phase_a import (content_sha, restored_fp_identity, midranks, pearson,
    spearman, pairwise_concordance, partial_spearman, opposite_pairs, choose_route, loo)


def main():
    assert midranks([2, 1, 2, 4]) == [2.5, 1, 2.5, 4]
    assert spearman([1, 2, 3], [3, 2, 1]) == -1
    assert spearman([1, 1], [2, 3]) is None
    assert pairwise_concordance([1, 1, 2], [2, 1, 3]) == dict(
        concordant=2, discordant=0, tied_a_only=1, tied_b_only=0, tied_both=0,
        eligible_untied_pairs=2, concordance=1)
    assert partial_spearman([1, 2, 3], [3, 1, 2], [1, 2, 3])["partial_spearman"] is None
    assert partial_spearman([1, 2, 3], [3, 2, 1], [1, 1, 1])["partial_spearman"] == -1
    rng = random.Random(0)
    checked = 0
    for _ in range(100):
        a, b, z = [[rng.randrange(8) for i in range(16)] for j in range(3)]
        ranks = [pd.Series(values).rank(method="average").to_numpy() for values in [a, b, z]]
        assert midranks(a) == ranks[0].tolist()
        reference = np.corrcoef(ranks[0], ranks[1])[0, 1]
        assert math.isclose(spearman(a, b), reference, abs_tol=1e-12)
        design = np.column_stack([np.ones(16), ranks[2]])
        residuals = [x-design@np.linalg.lstsq(design, x, rcond=None)[0] for x in ranks[:2]]
        reference_partial = np.corrcoef(*residuals)[0, 1]
        actual_partial = partial_spearman(a, b, z)["partial_spearman"]
        assert math.isclose(actual_partial, reference_partial, abs_tol=1e-12)
        checked += 1
    names = [f"R{i:02}" for i in range(16)]
    ascending, descending = list(range(16)), list(reversed(range(16)))
    pairs = opposite_pairs(names, ascending, descending, 3)
    assert pairs[0]["condition_a"] == "R00" and pairs[0]["condition_b"] == "R15"
    assert pairs[0]["minimum_rank_gap"] == 15
    assert all(x["rank_gap_a"] >= 3 and x["rank_gap_b"] >= 3 for x in pairs)
    assert opposite_pairs(names, ascending, ascending, 3) == []
    assert opposite_pairs(names, [0]*16, ascending, 3) == []
    assert opposite_pairs(["a", "b", "c", "d"], [0, 1, 2, 3], [3, 2, 1, 0], 3)[0]["minimum_rank_gap"] == 3
    assert opposite_pairs(["a", "b", "c"], [0, 1, 2], [2, 1, 0], 3) == []
    cfg = dict(documents=["d1", "d2"], conditions=names,
               route_rule=dict(minimum_midrank_gap_both_metrics=3))
    def panel(p1, total, b0):
        return dict(P1=p1, L7_TOTAL_ORIGINAL=total, B0=b0)
    assert choose_route({(d, "MEAN_256_1024"): panel(ascending, descending, ascending)
                         for d in cfg["documents"]}, cfg)["route"] == "A"
    assert choose_route({(d, "MEAN_256_1024"): panel(ascending, ascending, descending)
                         for d in cfg["documents"]}, cfg)["route"] == "B"
    assert choose_route({(d, "MEAN_256_1024"): panel(ascending, ascending, ascending)
                         for d in cfg["documents"]}, cfg)["route"] == "C"
    assert len(loo(ascending, descending, names)) == 16
    assert all(x["spearman_rho"] == -1 for x in loo(ascending, descending, names))
    root = Path(__file__).resolve().parent
    checked_smoke_identities = []
    for path in (root / "evidence_20261011_smoke").glob("smoke_*.json"):
        if path.name.endswith(".sha256.json"):
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        identity = restored_fp_identity(run["fp_reference"]["tensor_identities"])
        assert content_sha(identity) == run["fp_reference"]["content_sha256"]
        checked_smoke_identities.append(run["document"])
    print(json.dumps(dict(status="CPU_ANALYSIS_TESTS_PASS", random_reference_checks=checked,
        reference="bundled pandas midrank and NumPy corrcoef/lstsq", seed=0,
        verified_real_smoke_fp_identity_roundtrips=checked_smoke_identities,
        gpu_or_scientific_score_qualification=False, gpu_hours=0)))


if __name__ == "__main__":
    main()
