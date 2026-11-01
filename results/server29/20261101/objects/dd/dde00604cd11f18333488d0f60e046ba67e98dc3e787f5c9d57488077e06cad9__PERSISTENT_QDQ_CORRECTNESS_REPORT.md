# Persistent-QDQ correctness report

- Gate: **PASS**
- Runtime: 2.7.1+cu128 / CUDA 12.8 / driver 580.178.04
- Correctness smoke: FP plus IDENTITY (Native), H, L6 and L7 through 512 tokens on GPU6.
- GPU UUID: GPU-de53f243-b271-521c-d875-d96611ed8574
- Deterministic algorithms: enabled; TF32: disabled; CUBLAS workspace: `:4096:8`.
- Each branch executed 18 R128 QDQ/writebacks per token, and instrumented the next kernel to verify that its input exactly equaled the preceding post-QDQ state.
- FP and branches were advanced in one process on one GPU using identical teacher-forced token positions.
- Scorer: full-vocabulary `KL(FP logits || persistent-QDQ logits)` reduced online. Logit position t reflects QDQ/writeback through t-1; state at t is immediately QDQ-written for t+1.
- No pulse, no `NO_FURTHER_QDQ`, no high-precision recurrent carry, no optimizer, no free generation.
