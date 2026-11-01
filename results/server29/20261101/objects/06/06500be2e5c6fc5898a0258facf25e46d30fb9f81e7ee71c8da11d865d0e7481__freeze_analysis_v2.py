import time
from run_short_horizon_proxy_v2 import ROOT, sha256, atomic_json
from finalize_surrogate import tests

tests()
assert (ROOT / "ANALYZER_FREEZE.json").exists()
assert not (ROOT / "PROXY_SCORE_FREEZE.json").exists()
atomic_json(ROOT / "ANALYZER_FREEZE_v2.json", {
    "frozen_epoch": time.time(), "source_sha256": sha256(ROOT / "finalize_surrogate.py"),
    "statistics_tests": "PASS", "futurekl_join_not_started": True,
    "rule_priority": "stable and incremental before cost preference; no candidate selected if all fail",
    "tie_rules": "midranks; exclude pairwise ties; Pearson residual ranks controlling rank(JPROP)",
    "cost_planning": "two documents/four conditions/one anchor; no execution",
    "engineering_only_change": "undefined partial Spearman (zero residual variance) cannot count as positive; preserve NA instead of TypeError",
    "acceptance_thresholds_unchanged": True,
})
print("ANALYZER_FROZEN_BEFORE_SCORE_TARGET_JOIN")
