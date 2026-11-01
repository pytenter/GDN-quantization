# Training-objective audit

## Verified implementation semantics

The inspectable recurrent trainer computes two distinct objectives on real, persistent student cache state.  `recover_state` maps the rotated cache back by applying learned `Delta` and then `H128` on the key axis.  The state objective is squared error normalized by the FP state's squared norm, averaged across heads and captured GDN layers.  The functional objective is relative MSE at each GDN layer's complete `linear_attn.out_proj` output, averaged across the 24 GDN layers.  C5 uses the state term alone; C6 uses `functional + 0.1 * state`.

Numerical state persists across the document.  Autograd history is detached at the BPTT boundary, but the numerical cache is not reset.  Canonical C128 QDQ is written back after every recurrent decode step.  Each 1024-token training document has eight capture points and one optimizer step.  C5 and C6 both start independently at theta=0, i.e. Hadamard; C6 is not warm-started from C5.

## Final artifact identity and deployment

- Selected C5 is identified by the governing frozen record as update 063 / exposure 0504.
- Selected C6 is identified as update 125 / exposure 1000.
- Deployment loads the saved FP32 per-layer Delta matrices directly.  It does not reconstruct matrices from theta.
- C5 rotation SHA256: `0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9`.
- C6 rotation SHA256: `67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c`.
- C5 source checkpoint SHA256: `54416ccaf8b02cfa8d0ce5235f0b12c7db888c2ab8f07d44565eecc4c45c40d4`.
- C6 source checkpoint SHA256: `8312d59e593181e8b36de01058e492171434e3d0d9b0fb992aa6a979bdd151520`.

All 48 deployed matrices are finite and nearly orthogonal.  Across layers the maximum absolute orthogonality residual is about 2e-6.  Distance from Hadamard is large but purely descriptive: mean Frobenius distance is 16.0012 for C5 and 15.9990 for C6.  Coordinate distance is not a performance metric.

## Recovered formal H64 training evidence

The user supplied the original training server after the first diagnosis was frozen.  Read-only inspection located the exact formal experiment at `/root/autodl-tmp/GDN-formal-c5-c6-h64-v1/experiments/qwen_gdn/recurrent_training/QWEN_GDN_RECURRENT_C5_C6_H64_FORMAL_V1`.  Small configs, summaries, selection records, reports, and their original hash manifest are preserved under `training_provenance/`; checkpoints were not copied.  The recorded source branch is `exp/qwen-gdn-recurrent-c5-c6-h64-formal-v1`, with preregistration-time Git head `31a1a63c54ec86fa4a9e03a4cd4463fd0ff386ca`.  Original file hashes are verified in `analysis/h64_training_provenance_audit.json`.

Both formal runs completed 125 updates / 1000 target exposures with zero retries, no NaN/Inf, unchanged model sentinels, and `memory_creep = NO`.  Each has 20 validation candidates evaluated on the same frozen 16-document / 128-capture held-out panel.  Lower is better for every metric below.

| Condition / checkpoint | State relative-MSE | Local out-proj relative-MSE | Selection role |
|---|---:|---:|---|
| C5 first saved candidate, update 7 / exposure 56 | 0.008787584 | 0.052710483 | earliest retained candidate, not initialization |
| C5 selected, update 63 / exposure 504 | **0.007916185** | 0.049509294 | minimum C5 state primary |
| C5 best functional candidate, update 113 / exposure 904 | 0.008100550 | **0.046988553** | not selected by C5 state rule |
| C6 first saved candidate, update 7 / exposure 56 | 0.008776775 | 0.050523607 | earliest retained candidate, not initialization |
| C6 selected, update 125 / exposure 1000 | 0.008251136 | **0.049016486** | minimum C6 functional primary |
| C6 best state candidate, update 119 / exposure 952 | **0.008231279** | 0.052344892 | not selected by C6 functional rule |

This closes the former cross-objective gap for saved candidates.  On the shared panel, selected C5 has 4.06% lower state error than selected C6, while selected C6 has 1.00% lower functional error than selected C5.  The specialization direction therefore matches the objectives, but the selected-checkpoint separation is small.  More importantly, C5's functionally best saved candidate is 5.09% better on functional error than selected C5 while being only 2.33% worse on C5's state primary.  This is direct evidence that the C5 selection rule leaves function-space fidelity on the table; it is not evidence that update 113 would improve AIME.

Relative to each condition's first *saved* candidate, selected C5 improves state by 9.92% and functional error by 6.07%; selected C6 improves functional error by 2.98% and state by 5.99%.  The archive contains no explicit update-0 held-out validation row, so improvement over the exact theta=0 Hadamard initializer still cannot be quantified from the retained validation records.  Training minibatch losses are document-dependent and are not substituted for that missing baseline.

## Audit conclusion

The objectives are recurrent-aware and implementation-consistent with persistent QDQ/writeback.  Final H64 provenance, candidate-level dual metrics, selection rules, and artifact identity are now verified.  The evidence shows objective-aligned specialization and a concrete C5 state-selection versus function-space tradeoff.  It still does not establish improvement over the exact Hadamard initializer because update-0 validation was not saved, and it does not make validation fidelity equivalent to downstream correctness.
