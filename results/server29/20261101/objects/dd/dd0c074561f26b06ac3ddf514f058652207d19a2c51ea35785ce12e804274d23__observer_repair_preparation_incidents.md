# Observer-repair preparation incidents (before GPU model loading)

Parent: QWEN_GDN_C6R_C7_4X3090_DEPLOYMENT_CLOSURE_V1

Subphase: QWEN_GDN_3090_OBSERVER_LIFETIME_REPAIR_SMOKE_V1

1. The initial CPU git preflight incorrectly required an entirely empty untracked status after the two new preparation files had been uploaded. Read-only inspection showed exactly scripts/observer_repair_audit.py and scripts/observer_repair_recipe.json, both authored for this subphase. The failed CPU log and original preparation source were preserved. The guard was prospectively changed to permit only those two exact untracked paths; tracked diffs remained forbidden. No model was loaded and no scientific artifact gate failed.

2. The initial prospective manifest included the still-open console log of its own freeze process at the empty-file SHA. The completion line subsequently changed that log. This was detected before invoking the GPU runner. Every other original entry was verified unchanged, the initial manifest was preserved byte-for-byte, and only the completed console-log hash was updated, with an explicit metadata amendment record. The corrected manifest was immediately reverified. The repaired runner, condition computations, input, rotation, model, allocator, thresholds, deterministic settings and cleanup operations did not change.

Evidence:

- logs/observer_repair_preflight_incident_001.json
- logs/observer_repair_audit_001.log
- logs/observer_repair_audit_failed_preflight_001.txt
- manifests/observer_repair_pre_gpu_sha256.initial_console_incident_001.txt
- analysis/observer_repair_manifest_amendment_001.json

Both were preparation bookkeeping corrections before the one authorized GPU model attempt, not GPU/model retries. No additional repair was applied after that attempt began.
