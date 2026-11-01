# Cost and trainability audit

Conservative total: **1.095953 GPU-hours / 4.0**. Includes four retained engineering-failed probes, repeated micro, model loading, current-runtime FP teacher, actual short prefixes, real quantized forward, metric reductions and shutdown. GPU-hours sum both cards, not elapsed wall time. GPU_COST_LEDGER.json is the budget authority. Maximum measured forward VRAM: 15.770 GiB.

Model loads: [9.394, 10.512] seconds; FP references: [97.209, 87.4] seconds. Mean actual condition step: 0.087447 seconds. Each formal rotation computes one 1152-token prefix and both 128-token windows; H32 is a subset of H128, so these integrated measurements do not separately benchmark a dedicated H32-only trainer.

Marginal cost estimates from measured reductions/forward throughput:

```json
{
  "P1_local_readout_seconds_upper_bound": 0.0002966180530954432,
  "P2_local_contraction_seconds": 0.00453172753157105,
  "P3_H32_rollout_seconds_estimated_from_actual_average_step": 2.798306955655379,
  "P3_H128_rollout_seconds_estimated_from_actual_average_step": 11.193227822621516,
  "B2_KL_reduction_seconds_per_token": 0.00021162524198903299,
  "all_panel_measurements_share_forward_cost": true,
  "backward_memory_measured": false,
  "persistent_state_lower_bound_H128_FP32_GiB": 2.25
}
```

P1 timing includes state-MSE and auxiliary reductions and is an upper bound for P1 alone. P2 timing excludes common FP history acquisition/covariance formation. Dedicated train-time cost requires those shared teacher/prefix costs. PREDICTIVE_BENEFIT_PER_GPU_HOUR.csv reports both the signed rho increment over B0 and the descriptive marginal-cost ratio; a large ratio for a tiny negative/unstable effect is not evidence of utility. Every candidate shares the actual panel acquisition cost; costs must not be summed repeatedly across CSV window/aggregate rows.

## Static trainability findings (no backward or optimizer run)

Original Cayley code constructs A(theta) skew-symmetric and R=(I-A)(I+A)^-1 with torch.linalg.solve; 8128 parameters/layer, shared across heads. Original exact-forward STE is value+(dequant-value).detach(). It permits surrogate gradients through the real forward QDQ, but is a biased identity backward and does not differentiate code/scale discontinuities faithfully.

P1/P2 mathematics can connect physical E through rotation, QDQ and inverse rotation to Cayley parameters with an explicit STE. However the reused historical K3 helper **detaches E and q**, and this evaluator runs inference/no-grad: it is **not trainable as written**. A future implementation must preserve teacher detachment but remove the error-path detach, verify STE forward identity and finite nonzero rotation gradients. No readiness PASS is claimed here.

P3 additionally needs H32/H128 truncated recurrent BPTT, with numerical student state persisting across detached prefixes. FP teacher/query/covariance can remain detached. The state-tensor-only H128 storage lower bound is 2.250 GiB; this excludes kernel intermediates, query/value tensors, model activations, logits and autograd workspaces. Forward peak VRAM does not establish backward feasibility; checkpointing or shorter H32 may be necessary.

Historical training has a BF16 quant boundary/dequant state and local gated o_norm/o_proj relative-MSE functional objective (+0.1 state loss). Our inherited deployed Phase2 uses FP32 physical recurrent-cache dequant, and B2 is full-logit KL. Therefore B2 is a short functional-family baseline, **not an exact reproduction of the historical L7 training loss**, nor proof that a new loss improves L7 optimization. These distinctions must be resolved in any later gradient-readiness probe.

Prediction and trainability remain separate. No optimizer.step, backward GPU experiment, new learned checkpoint or L9 was performed. A same-runtime prospective pilot would use two held-out documents, four calibration-selected natural rotations per document, one Primary8192-to-32768 target per document and same-UUID FP pairing. The measured-throughput extrapolation is 10.200 GPU-hours: ten 32K-equivalent passes (two teachers + eight conditions), multiplied by 1.25, plus 0.25 hours of load allowance. This is not a full-context benchmark or authorization; a second anchor adds continuation cost, and attention/cache cost can change with context length. This round does not run it.
