# Memory scaling — NOT_DETERMINED

The only measured GPU memory values are stage-local load and teacher-forward snapshots in `analysis/stage_load_rank*.json` and `analysis/pp2_teacher_rank*.json`. They are not BPTT memory measurements. H1/H4/H8/H16/H32, the five-update H32 lifetime test, activation slope, leak test, and 5%/10% headroom gates were not run because the forward gate failed. `H32_FEASIBILITY=NOT_DETERMINED`; `memory_leak=UNKNOWN` for this PP2 runtime.
