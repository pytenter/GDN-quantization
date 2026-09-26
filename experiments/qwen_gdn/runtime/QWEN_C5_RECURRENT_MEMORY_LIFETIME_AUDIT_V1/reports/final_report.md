# QWEN_C5_RECURRENT_MEMORY_LIFETIME_AUDIT_V1

## Final verdict

**NO_LEAK_H32_TOO_LARGE.** C5's H=32 BPTT is not sustained resource-feasible under the audited 24GB implementation. The recurrent training method has passed its semantic and numerical gates; this result is a resource feasibility limit.

```text
{
  "C5_TRAINING_STATUS": "BLOCKED_MEMORY_FEASIBILITY",
  "C5_formal_AIME": "NOT_STARTED",
  "C6_formal_AIME": "NOT_STARTED",
  "FINAL_VERDICT": "NO_LEAK_H32_TOO_LARGE",
  "HOOK_LIFETIME_GATE": "PASS",
  "LOSS_GRAPH_RETENTION_GATE": "PASS",
  "MEMORY_BASELINE_GROWTH": "NO",
  "MEMORY_LIFETIME_STABILITY_GATE": "PASS",
  "NUMERICAL_TRAINING_SEMANTICS_PARITY": "NOT_RUN",
  "PRIMARY_RETENTION_SOURCE": "NONE_FOUND",
  "QDQ_AUTOGRAD_LIFETIME_GATE": "PASS",
  "QWEN_C5_MEMORY_LIFETIME_AUDIT": "PASS",
  "RECURRENT_CACHE_LIFETIME_GATE": "PASS",
  "STUDENT_STATE_DETACH_GATE": "PASS",
  "SUSTAINED_HORIZON_FEASIBILITY": "FAIL",
  "VALIDATION_MEMORY_RETURN_GATE": "PASS"
}
```

## Evidence

The audit completed **11 optimizer updates / 88 target exposures**, then OOM occurred during the normal recurrent rollout of update 12 (`wikitext2raw-train-036`, 1024 tokens). The third failed formal attempt reached the same 11-update boundary without the audit instrumentation. At OOM the audit measured 23592.3 MiB allocated, 23896.0 MiB reserved, and 3.7 MiB device free; the failed request was 2 MiB.

The A12 allocated baseline was **17125.6 MiB on every one of the 11 completed updates** (bytewise identical). This includes update 7, which ran the full 16-document validation panel. Validation entered and exited with 17125.6 MiB allocated (delta 0 bytes). Thus validation residuals are not the sustained cause.

All 11 recurrent cache records had zero graph-bearing tensors after each BPTT detach boundary and at end-of-rollout detach. Per-layer rows are in `analysis/per_layer_cache_lifetime.csv`. The loss weakref was dead at A12. The corrected probe found all watched loss, old student state, QDQ/post-state, and cache-entry objects dead at A12. QDQ uses a plain STE expression with no custom autograd Function, `ctx` storage, or `save_for_backward` in the audited source. Hooks are registered once per context and removed at context exit; no per-update registration occurs. Historical metric rows contain Python numbers rather than graph-bearing tensors.

The first weakref implementation reported live student/QDQ/cache objects because the audit script itself kept `state_tensor` and `cache_tensor` local variables alive. Their lifetime was constant across updates and the allocated baseline did not grow. The corrected one-update diagnostic probe removes those local references; its weakrefs are all dead at A12. The original probe data and the diagnostic diff are preserved.

## Ten required answers

1. End-of-update baseline growth: **NO**, across 11 completed updates.
2. Validation contribution: validation returned exactly to its entry baseline; it is not the sustained cause in this run.
3. Recurrent cache old graph retention: **not observed** after BPTT detach (0 graph tensors).
4. QDQ/STE cross-update retention: **not observed** in the corrected weakref probe.
5. Python list/dict/loss/history graph retention: **not observed** at update boundaries; the original weakref alarm came from the diagnostic harness itself.
6. Patch parity: no training-runtime patch was applied, so formal training semantics parity after patch is **NOT_RUN**. A one-update diagnostic probe comparison is in `analysis/diagnostic_probe_parity.json`.
7. 20–30 update sustained stability: **not achieved**; OOM stopped the run at update 12. The 11 observed A12 baselines were identical.
8. H=32 sustained 24GB feasibility: **FAIL** under the current audited implementation.
9. C5 formal restart: **not allowed** under the current frozen H=32 configuration.
10. Horizon amendment: a prospective H=16/H=8 proposal is recorded; no smaller horizon was tested or selected here.

## Action boundary

C5 formal training remains `BLOCKED_MEMORY_FEASIBILITY`. C6 and formal AIME were not started. A future horizon change requires a separately reviewed prospective amendment and new semantic, numerical, and sustained resource gates.

## Student state boundary probe

A separate H=32 step0 probe inspected all 24 GDN recurrent states before and after detach. Every post-detach state had `requires_grad=False` and `grad_fn=None`; byte hashes of numerical values were unchanged. Per-layer device, dtype, shape, and leaf status are recorded in `analysis/student_state_boundary_probe.json`.

## Diagnostic probe comparison

The corrected one-update probe reproduced the first update's loss exactly. Gradient norm was 0.312733322 in the initial audit and 0.311492085 in the corrected probe; therefore the separate diagnostic runs do not establish exact gradient parity. No training-runtime change was made, so the requested post-runtime-patch parity gate is NOT_RUN.
