# C5 loss graph audit — static phase

Frozen source shows `forward_student()` calls the full Qwen model but retains only `past_key_values`; ordinary returned logits are discarded. At the captured token, `RecurrentExperimentPatch._wrap()` creates `state_losses` from the internal recurrent `initial` state, its inverse rotation, and a detached teacher target. The `out_proj` forward hook separately creates `functional_losses`. `captured_losses()` returns **state alone as C5** and `functional + 0.1 × state` as C6. The parent FSDP script calls `c5_loss.backward()`.

Therefore `C5_LOSS_STANDARD_OUTPUT_DEPENDENCY=PARTIAL`: no explicit C5 path through final model logits or a GDN block's own returned residual/MLP output, while downstream GDN states can depend on preceding decoder-block outputs. PyTorch 2.5.1 FSDP2 registers its pre-backward hook on tensors requiring gradients in each wrapped module's ordinary output (`_fsdp_state.py:316-323`). This structural mismatch makes output-hook bypass testable, not proven. The exact runtime dependency will be checked without changing frozen C5 semantics.

Sources are the frozen `run_recurrent_dense.py` (SHA256 `48bd12d...49e516e`), frozen FSDP smoke script (SHA256 `f1cec3...829cef4`), and the installed Qwen/PyTorch source. Parent `FAILED_STORAGE_OWNER` remains **UNKNOWN**.
