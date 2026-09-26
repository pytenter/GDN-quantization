# LING_DETERMINISTIC_16DOC_REPLICATION_V2 reproducibility report

- Fresh processes bitwise identical at audited trajectory/checkpoint hashes: **YES**
- Teacher-forced trajectory statistics identical: **YES**
- AUC variance/range zero: **YES**
- All four conditions stable: **YES**
- AUC noise p50/p95/max: `0` / `0` / `0`

Each document-condition cell was evaluated in five independent Python processes. Full tensors were not retained; compact SHA-256 evidence covers token IDs, reference horizons, condition checkpoints, logits, caches, and scalar horizon metrics.
