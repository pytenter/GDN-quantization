# Orchestration-only extension

No historical evaluator file is edited. `run_one.py` imports the audited
`run_ling_repeatability.py`, changes only `SELECTED_INDICES` from `(0, 7, 15)`
to `tuple(range(16))`, and invokes its existing `main()`. Model loading, warmup,
rotation, INT8_R128 quantization, prefill, recurrent decode, Future-KL, AUC, and
checkpoint hashing therefore remain implemented by the audited code. Smoke parity
against yesterday's doc0 Hadamard and Dense-State fixed-config artifacts is mandatory.
