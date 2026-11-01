# Zero-output-anchor causal test

The anchored scalar was exactly `C5 + 0.0 * final_captured_token_model_logits.float().sum()`. The ordinary logits were a finite, grad-connected `[1,1,248320]` tensor, retained by an experiment-local helper that makes the same model call and cache writeback as frozen V1. Original and anchored conditions used that *same* helper. No epsilon, detach, sanitization, or production source edit was used.

In one single-GPU full-Qwen H1 process, original and anchored C5 loss hashes matched exactly; all 24 rotation gradient hashes, all 24 one-step Adam theta hashes, final-token QDQ hashes and recurrent writeback checks matched. The exact gate **PASS** is in `analysis/zero_anchor_single_semantics.json`.

The frozen parent H4 failure was reused, not rerun: both ranks failed with `setStorage [8192,1,1,4] ... storage size 0` during C5 backward. One and only one two-rank H4 anchor intervention was run. Both ranks again failed within the same zero-storage error family, but the failing shape was **`[1024,4096]`**, a different tensor shape. The protocol's different-error stop rule classifies this as `NEW_FAILURE_MODE`, not a rescue. At the failure each rank had peak allocated **14,201,228,288 bytes** and peak reserved **17,190,354,944 bytes**; no OOM occurred. See both raw error JSONs and `analysis/h4_zero_anchor_result.json`.

The changed error shape shows altered execution, not identification of either failed storage owner or proof of a missing FSDP hook. The protocol required stopping H4 intervention at this new failure; no anchor variants or repeat runs were attempted. `ZERO_ANCHOR_H4_RESCUE=NO`, `H4_COMPATIBILITY_GATE=BLOCKED`.
