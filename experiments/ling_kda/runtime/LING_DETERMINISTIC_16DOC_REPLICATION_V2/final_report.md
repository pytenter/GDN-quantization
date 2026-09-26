# LING_DETERMINISTIC_16DOC_REPLICATION_V2

1. `LING_DETERMINISTIC_16DOC_GATE = PASS`

2. Fresh-process repeatability: **PASS**

3. AUC runtime noise: p50 = `0`, p95 = `0`, max = `0`

4. Native AUC: mean `0.0317437357717`, median `0.0165501295445`, std `0.038228663016`, min `0.00613670038559`, max `0.148624287872`

5. Hadamard AUC: mean `0.0188733338775`, median `0.009968616854`, std `0.0236622799318`, min `0.00308151902309`, max `0.083417398644`

6. Dense-State AUC: mean `0.0281163782818`, median `0.00997816834388`, std `0.0347886817759`, min `0.00120582049435`, max `0.127485930257`

7. Dense-Functional AUC: mean `0.0162689810637`, median `0.00992365956923`, std `0.0134046864413`, min `0.00198319663677`, max `0.0461076612724`

8. Hadamard vs Native:

   - paired mean delta: `0.0128704018942`
   - paired median delta: `0.00596519030038`
   - bootstrap 95% CI (median delta): `[0.00218974908685, 0.0155721807576]`
   - win/tie/loss: `12/0/4`

9. Dense State vs Hadamard:

   - paired mean delta: `-0.00924304440436`
   - paired median delta: `-0.00212010311082`
   - bootstrap 95% CI (median delta): `[-0.00468166660211, 0.000393427422055]`
   - win/tie/loss: `4/0/12`

10. Dense Functional vs Hadamard:

   - paired mean delta: `0.00260435281373`
   - paired median delta: `-0.00273658423825`
   - bootstrap 95% CI (median delta): `[-0.00415577839946, 0.0037953723656]`
   - win/tie/loss: `6/0/10`

11. Credible Dense persistent headroom over Hadamard: **NO**. Dense-State: NO; Dense-Functional: NO.

12. Previous large local Dense gains reflected persistently: **NO; the deterministic persistent panel does not establish translation of the large local gains**.

13. Fixed-config path ready on this RTX3090 environment: **YES**. Exact runtime dependency hashes are preserved.

14. 64-document confirmation scientifically justified: **YES**; whether it should be run next based on effect size: **NO**. It was not started by this task.

15. RTX3090 execution environment versus canonical Ling RTX4090:

   - host: `nlpg-SYS-4029GP-TRT` vs `lthpc1`
   - selected GPUs: `2 x NVIDIA GeForce RTX 3090` (physical 2,3 of 8) vs `2 x NVIDIA GeForce RTX 4090`
   - driver: `580.173.02` vs `535.113.01`
   - Python/PyTorch/CUDA/Triton/Transformers/FLA: `3.11.16` / `2.7.1+cu128` / `12.8` / `3.3.1` / `4.57.6` / `0.5.2`; canonical `3.11.16` / `2.7.1+cu128` / `12.8` / `3.3.1` / `4.57.6` / `0.5.2`
   - frozen panel, checkpoints, audited evaluator dependencies, generation protocol, and kernel configuration files are SHA-256 identical to V1.

Positive paired delta means the candidate has lower Future-KL AUC. Historical pre-freeze AUC values are not used as targets or direct comparators.

## GPU execution amendment

The original immutable runtime manifest was captured for physical GPUs 2 and 3. After 28 raw fresh-process results, both workers were quiesced at process boundaries. GPU2 subsequently became occupied by another user, so the main run resumed on the two free physical GPUs 3 and 7. A brief user-authorized GPU4/GPU5 helper trial produced one complete GPU4 result before being stopped because host CPU contention slowed the main workers. Later, a foreign process on GPU3 triggered the audited watchdog at 281 preserved results; the final continuation again used GPUs 3 and 7. Existing result identities were skipped and never overwritten. Observed physical GPU indices across the complete run: `[2, 3, 4, 7]`; up to four Ling evaluator processes ran concurrently only during the brief helper trial, and the main execution otherwise used two GPUs. All are RTX3090 devices under the same driver, software environment, fixed kernel configuration, checkpoints, panel, seed, and evaluator. The original runtime manifest hash remains attached to every record; this amendment documents orchestration only. Fresh-process equality is evaluated across all repeats regardless of physical GPU.
