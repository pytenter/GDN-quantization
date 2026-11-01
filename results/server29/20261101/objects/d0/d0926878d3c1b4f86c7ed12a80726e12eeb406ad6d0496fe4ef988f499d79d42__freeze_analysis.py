import time
from run_short_horizon_proxy_v2 import ROOT, sha256, atomic_json
from finalize_surrogate import tests

tests()
atomic_json(ROOT / "ANALYZER_FREEZE.json", {
    "frozen_epoch": time.time(), "source_sha256": sha256(ROOT / "finalize_surrogate.py"),
    "statistics_tests": "PASS", "futurekl_join_not_started": True,
    "rule_priority": "stable and incremental before cost preference; no candidate selected if all fail",
    "tie_rules": "midranks; exclude pairwise ties; Pearson residual ranks controlling rank(JPROP)",
    "cost_planning": "two documents/four conditions/one anchor; no execution",
})
print("ANALYZER_FROZEN_BEFORE_SCORE_TARGET_JOIN")
