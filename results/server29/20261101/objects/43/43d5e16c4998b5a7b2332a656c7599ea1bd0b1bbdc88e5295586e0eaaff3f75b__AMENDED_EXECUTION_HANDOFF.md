# Amended offline clean-room execution

The original registration, qualified scientific commit, training engine and canonical endpoint sources remain frozen. The amendment was recorded before any optimizer update. Update-0 subsequently completed naturally, with three-arm full Shared16 exact parity PASS. Only the stopped original launch shell was retired after its Python child finished; no GPU worker was interrupted.

Training continues unchanged: C5R, C6R, C7; original readiness gates, 125 updates/1000 exposures, selection cadence and metrics. New control files are separately hashed. The original scheduler and waiting posttraining/inventory/endpoint launchers are superseded, with their evidence preserved.

A800 tmux: `qwen_a800_amended`. Coordinator: `scripts/amended_a800_controller.py`. It performs fresh original readiness processes, then original training processes, then fixed Shared16, non-AIME C5R reproduction routing and full scientific inventory. Logs use `amended_` names, not the superseded exit codes.

3090 tmux: `qwen_a800_amended_endpoint`. Coordinator: `scripts/amended_canonical3090_launcher.py`. It waits for `amended_inventory.exitcode=0`, verifies original and amended hashes, copies evidence and only routed selected R bytes, waits for all four GPUs to be idle, then runs unchanged canonical non-AIME smoke and formal conditions sequentially. Within each condition four fixed workers have 15 sample/seed pairs each. Disk must stay above 15GiB to start each new sample. The existing independent SSH ControlMaster is used; closing Codex does not stop tmux.

Primary endpoint conditions are C6R then C7 (120 records). C5R is additional only if the frozen non-AIME reproduction gate justifies MATERIAL_SHIFT. CLOSE skips; INCONCLUSIVE skips by default. The conservative automatic gate records all eight required evidence dimensions without inventing a historical repeatability tolerance or weighted similarity score. A justified qualitative review, if available before routing, is recorded in `analysis/C5R_NON_AIME_REPRODUCTION_REVIEW.json` and must cite the exact evidence hashes. No AIME results enter this decision. A skipped C5R has `aime26/C5R_NOT_RUN.json`, no fake empty scores.

C5R-specific failure requires explicit evidence review of shared model/teacher/runtime/quantizer/data integrity. The controller preserves evidence and waits at `C5R_FAILURE_ATTRIBUTION_REQUIRED.json`; it does not infer isolation or retry C5R. Shared or C6R/C7 failure stops downstream execution. No failure is automatically downgraded.

All routed outputs must be frozen and verified before new scoring. Frozen V4 is applied once; five historical baselines are reused. Six fixed mandatory comparisons use Holm; two conditional C5R comparisons use a separate auxiliary Holm family. The primary comparison is C7 vs C6R. Question-cluster uncertainty and exact paired IDs follow archived project helpers. Directional verdict labels follow amendment cases; statistical support is reported separately. C6R historical shift remains UNKNOWN absent multiplicity-controlled evidence, never automatically attributed to old corruption.

Final artifacts remain in the existing experiment root: `FINAL_REPORT.md`, `NEXT_EXPERIMENT_DECISION.md`, `analysis/final_status.json`, non-AIME evidence, all scheduled candidates, endpoint files and SHA256SUMS. No follow-on experiment starts automatically.

Validation: Python compilation and CPU-only synthetic route/default-skip/final-verdict tests. A CPU prepare namespace guard collision occurred before any model load or new endpoint; the original marker and incident record are preserved, and only the new control namespace was corrected before its source freeze.
