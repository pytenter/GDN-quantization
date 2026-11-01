# Current authorized work — 2026-10-27

The full request is HUMAN_REQUEST.txt. The latest human question asks whether the numerical gate was too strict and asks for an explanation. This steers the existing task; it does not cancel it or authorize silently relaxing tolerances. No formal AIME has started.

Local root: /Users/pytenter/Documents/临时项目1/PACE48_LEAP_UNROLL_DP8_DP16_V1

55 development root: /data/zypan/analyses/pace48_leap_unroll_dp8_dp16_v1

29 intended archive: /data/zypan/qwen_gdn_pace48_archive/PACE48_LEAP_UNROLL_DP8_DP16_V1

55 Python: /data/ydai/miniconda3/envs/bitdecode/bin/python. operations/transport.py uses an existing SSH master at /tmp/qwen_gdn_replication_20261013.sock;55→29 existing master /tmp/qwen_replication_archive29.sock. Credentials are not stored in this capsule. 55/usr/bin/python3 does not exist;29/usr/bin/python3 does exist. Use empty CVD for all remote CPU operations. Only physical55 GPUs0..3, original UUIDs, TP1 independent workers are authorized. Never signal foreign processes.29 is CPU-only.

## Source identity

All4220 files329792662bytes of the old frozen rebalance source were freshly verified against d70a1b7a7d54a59d12bba3ad33b60b495af787295629744990543a899f947267. Original artifacts were not modified. baseline/ contains the original small files; src/pace_buffer.py,src/pace_quantizer.py,src/pace_runtime.py and vendor/protocol files are exact copies. Quantizer89f4801aac1498eb68d285fb40ea3eb922b52645782853367e25558042798807 is unchanged. New files: packed_direct_read.py, lq_unroll_buffer.py, unroll_runtime.py. New generation body differs only in adapter factory, period selection, and configuration/task labels. Prefill delegates to original PACEBuffer, final preserve is unchanged.

The production Triton kernel fuses tiled packed mixed-base decode,2matvecs, pending vector readouts, u and output. It emits no dense state tensor. Pending normalized keys/FP32 updates are temporarily stacked each call; they are not persistent high-precision keys or dense state. This preparation is a significant remaining dispatch cost. The original state codec is invoked only at the original prompt boundaries and new decode8/16 boundaries. Boundary reconstruction uses the actual u records, never hidden V/beta. No compensator/smoothing.

## Actual completed experiments (all non-AIME)

- CPU_PROTOTYPE_V1:34 synthetic cases.
- CUDA_KERNEL_V1:custom kernel compile failed due to a Python global `tl` lookup. True owning exit1 and release saved. Fixed without math changes. Do not hide this failure.
- CUDA_KERNEL_V2:38 cases, actual kernel.
- CPU_FULL_V3 and CUDA_HOLDOUT_V3:40 cases each, including257-step weak-decay holdout and257-token prompt boundary, real32-head case.
- CUDA_FRESH_V4:second fresh actual CUDA run, GPU3. Worst metrics identical to V3.
- MODEL_SMOKE_GPU0..3_V1:each passed actual Qwen3.5-9B non-AIME smoke with both periods; exact owning exit0 and release saved. Driver UUID/CVD/24threads/deterministicFalse checked.
- SAME_STATE_ORACLE_V1:300 independent NumPy/CPU FP64 comparisons of actual GPU packed bases/keys/u/decays, all24 real masks, both periods, optional exponents. Independent packed decoder does not call production dequantize. Only metrics were downloaded, not raw state tensors.
- KERNEL_MICROBENCH_V1:GPU2, readout CUDA-event graph medians19.098us(dP8),27.954us(dP16); ordinary Python submit ~173/174us. Single boundary-step synchronized observations15.09/23.05ms (not stable medians).
- MATCHED_THROUGHPUT_V1:GPU1, one model, same non-AIME teacher-forced inputs and original sampler/checks, prompt16,warmup32,measured128,3rotated repeats. Medians replay8=4.925996,direct8=5.308182(+7.76%),direct16=4.420670(-10.26%)TPS. Performance negative for16 must be preserved. No AIME length confound, model loading excluded.

All engineering sessions have finished and actual owning wait/release receipts are in runtime/. No current local exec waiter. Original master sockets are connection helpers, not duplicated delivery readers. No formal SOURCE manifest or permit has yet been generated.

## Numerical interpretation and unresolved admission review

configs/NUMERICAL_ADMISSION_PREREGISTRATION.json sets cross-replay maxNRMSE0.005 and same-actual-delta-state readout0.001, before holdout and before AIME. These are assistant-chosen engineering screens, not values mandated by the user/paper or a theorem. They have not been loosened. Do not silently turn a failed screen into PASS.

CPU prototype cross-replay worst output2.42456%,delta1.54353%,boundary1.00454% in dP8 weak-decay257. At the worst token255 only1/512 BF16 outputs differs(maxabs0.00634766,reference RMS0.0115703,cosine0.999707); its output equals the readout from its own actual delta state exactly. First quantized split is token80:oneFP16 scale ULP changes an INT4 code1vs2 although that particular preencode element is equal; the group scale derives from other extrema. Later differences inherit already different packed bases.

Actual CUDA cross-replay worst output0.0981302%,delta0.0987156%,boundary0.0665270%; same-delta-state output0.0560762%. First weak-decay8 quantized split is token136: scaled coordinate16.5vs16.5000014382 crosses unchanged RNE (16vs17), preencode state absdifference1.86e-9. This is a real new FP32 numerical trajectory, not old PACE bit identity. The local `explain_rounding` function labels some subsequent differences UNEXPLAINED because it only checks adjacent local rounding and fails to track inherited earlier quantized-base divergence. Those labels require an honest causal review; don't just delete them or assume all are harmless.

Same-state FP64 oracle worst uNRMSE0.258169%,BF16 output vs unroundedFP640.175924%(includes BF16 rounding),BF16 output vs likewise BF16-rounded oracle0.0191457%. It supports correctness of the target GPU operator, independent of cross-trajectory drift. It does not establish80K equivalence or AIME accuracy.

The human was told that treating the CPU prototype cross-trajectory screen as proof of CUDA mathematical failure is overly conservative. Formal admission remains false while reviewing the proper comparison objects and quantization-lineage differences. Do not merely raise0.5% to force a pass. Initial authorization explicitly permits a documented independent FP32 implementation rather than bit equality, but disallows temporary tolerance loosening. Resolve this coherently and transparently; preserve every original gate result.

## Remaining authorized work

1. Complete the scientific admission review; if it cannot pass within the actual user constraints, deliver honest STOP/PARTIAL artifacts without AIME. Otherwise freeze one exact common implementation and period-only configs, then60 genuinely new trajectories, no old-result reuse.
2. Draft execution_common.py, archive_cpu.py, formal_worker.py, formal_controller.py now exist locally, NOT deployed/tested/frozen and NOT launched. Review/test before use! Need build canonical60 job manifest, GPU permit/source manifest, CPU integration tests, scoring/parent finisher and delivery monitor. Source identity and all invariants must actually pass. Controller supports explicit same-source completed-trajectory resume; don't automatically rerun failed GPU units.
3. 55/data had ~7.0GB free, /tmp5.2GB, /dev/shm120.99GB. New code/data capsule is ~2MB. /dev/shm/pace48_leap_unroll_dp8_dp16_v1/triton holds this task's compile cache. No foreign cleanup. Preserve4GiB output/8GiB archive/2GiB scratch safety; use actual anticipated storage rather than pretending GPU-active hours are known.29 had~885GB free.
4. The old model17 SHA manifest is copied to analysis/MODEL_TOKENIZER_FRESH_SHA.json. Need fresh full remote model17/source/actual mapped-file/library identity before formal. Do not download model weights or compiled binaries to Mac.
5. Complete all user-requested reports, memory inventory, runtime/true exit accounting, paired statistics only for real complete outputs, manifest and independent55/29/Mac SHA closure. Do not create fake or empty scientific result CSVs when no formal trajectories exist.
6. Final allowed statuses are in HUMAN_REQUEST.txt. On final completion/terminal diagnostic delivery stop and remove this task's continuation automation only; retain other tasks. No new periods, calibration, seed, rotation or experiments.

Use WORK_STATE.json for current active sessions and status. Source deployment helper refuses changes once FORMAL_SOURCE_MANIFEST.json exists. Don't deploy changes over active production. Monitoring should not restart an owning chain or reuse a claimed audit/transfer. Preserve both failed and passed artifacts.

+
+## Latest fast-path steering and actual progress (supersedes earlier prepare-cost description)
+
+Human latest: “你使用那个高效的路径来跑啊，不要用那个低效的方法”. Formal must use the restored high-throughput base runtime plus the new actual packed Direct Read; no original replay formal arm. Original replay8 remains only the explicitly requested engineering performance reference.
+
+The inefficient per-record key unpack/normalization has now been replaced, in the unfrozen new candidate only, by batching the EXACT original PackedAffineGroups decoder and native BF16 normalization across an outer token/batch dimension. K4 fields/dtype/cast/repeat/norm mathematics are unchanged, not replaced by a new approximate kernel. No full state is reconstructed. All key batches are validation-checked against the original per-record _token_keys byte values. verify_batched_keys=True exists solely for validation/smoke; formal defaultFalse removes the duplicate comparison work, not the actual math or original input/finite checks.
+
+FAST_BATCH_KEYS_V5 completed40 full CUDA cases with exact key checks. FAST_MODEL_SMOKE_GPU0..3_V2 each passed both periods on the real model with real24 head masks and exact batched-key comparison, true owning exit0/release. These artifacts and receipts are now downloaded locally. Full production mathematics unchanged beyond the independently developed direct readout; new batched preparation should produce the same numerical results as V4, check it explicitly.
+
+FAST_MATCHED_THROUGHPUT_V2 is currently running alone on GPU1, local exec session15557, remote runtime/FAST_MATCHED_THROUGHPUT_V2/stdout.log;3rotated repeats of all3 engineering paths,prompt16,warmup32,measured128. Do not duplicate it or overwrite result. Fetch and verify real owning wait/release before interpreting results. Previous medians4.926/5.308/4.421 describe the older per-record preparation, not the optimized candidate. Preserve both.
+
+All draft formal code (execution_common/archive_cpu/formal_worker/formal_controller) has now been deployed as development source, but remains untested/unfrozen and MUST NOT launch before admission and integration checks. No FORMAL_SOURCE_MANIFEST/GPU_PERMIT/FORMAL_STARTED exists yet. Formal controller's resume flag only supports same-source verified complete trajectories, moving task-owned incomplete prefixes aside with zero credit. Review its edge cases and resource/closure before use. Need implement scoring/parent/monitor/report delivery, canonical60 manifest, actual source/model identity and CPUcontrol tests.
+
+Continuation heartbeat was actually created successfully: automation_id=pace-lqunroll, ACTIVE. Earlier invalid create attempts created nothing. Use this existing automation; do not create a duplicate. Complete the full authorized task and delete it only after actual final/terminal diagnostic delivery closure.
+