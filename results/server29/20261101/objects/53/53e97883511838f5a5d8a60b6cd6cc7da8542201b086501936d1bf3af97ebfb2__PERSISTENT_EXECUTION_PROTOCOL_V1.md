# Phase B persistent execution protocol — pre-GPU freeze

The document/pair selection and predicted winners remain immutable. Route A is
P1 versus original L7_TOTAL, not a new comparator. The two pairs are 860:
H_to_L6_a50 versus H_to_L7_a25 on GPU6, and 5956: H_to_L6_a50 versus
H_to_L7_a50 on GPU3. No third document, partial roster, new rotation or training.

One worker per document keeps one fresh same-GPU native FP cache and two
independent student caches. All three execute the unchanged full model
teacher-forced single-token step, including all 18 KDA, 6 MLA and convolution
caches. Each student uses the original symmetric rowwise value-dimension
INT8-R128 rotate/QDQ/inverse/writeback after every token, from document start
through token 32767. The FP32 cache carrier contains dequantized INT8 values,
not an unquantized persistent shadow. Diagnostic clones never enter the model.

Qualification: actual 512 tokens twice on each fixed GPU; exact FP recurrence
replay, full-logit numerical identity, genuine codes/inverse/physical writeback,
actual next-kernel consumption, independent storage, native hybrid inventories,
finite values and repeat equality. The same engine is used for an 8448-token
microbenchmark, which crosses both 2048/8192 boundaries and checks continued
QDQ. These are mechanical/cost evidence, not confirmatory primary outcomes.
No GPU PASS is inferred from AST checks or synthetic CPU tests.

Qualification reserves at most 900 seconds per GPU, micro 3600 seconds per GPU.
Both complete qualification records are needed before either micro launch.
Both micro records are needed before the once-published complete cost gate.
Full cost uses actual entire-task A+B+qualification+micro device elapsed times.
Per-document full-stream prediction is 1.5 times actual micro wall cost plus
remaining tokens times the maximum post-2048 measured block rate times
32768/8448, plus 30 seconds. Loading, probes, FP and both student streams are
included. Native cache growth is extrapolated conservatively to 32768 and must
fit 85% of physical device memory. At least 2 GiB storage remains. No KL magnitude
or winner outcome is used for timing, memory, or launch decisions. If the sum for
all four student units and both necessary FP references exceeds 16 GPU-hours,
or capacity/continuity fails, stop ENGINEERING_OR_BUDGET_STOP; no partial sample.

Formal primary is mean full-vocabulary FP32 KL(FP||student) on [8192,32768).
Secondary [2048,32768), mean/AUC/p50/p99 and state/scales diagnostics use the
same continuous trajectory. Full FP and student logit byte hashes identify the
references and results; scalar per-token KL traces are canonically retained.
No top-k, reversed KL, pulse-only surrogate or anchor-off quantization.

All code, sources, weights freshness, rotations, tokenizer/token inputs,
prediction/metric, document-to-UUID mapping and gates are bound before GPU work.
No repeat after STARTED/receipt/result/STOP. Any measurement anomaly produces
an exclusive STOP and shared fail-closed latch; workers check at safe token
boundaries without signals. Device holders and physical UUIDs are checked before
launch and monitored. Busy external devices are waited for naturally. CPU
engineering failures v1/v2 are retained separately and do not count as GPU FAIL.

This protocol is an implementation contract, not a completed correctness report
or a 32K launch authorization. Phase C remains proposal only. No optimizer step.
