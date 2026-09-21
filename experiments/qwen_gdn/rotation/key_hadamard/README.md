# Qwen/GDN Key-Hadamard canonical path

The canonical executable implementation is integrated in
`experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py` and comes from
the `qwen3090` audit branch.

Contract:

- quantization is INT8_C128 over the K axis (`dim=-2`, group size 128);
- normalized H128 is applied on the Key side after q/k normalization and before the GDN core;
- prefill and recurrent decode use the same rotated Key basis;
- the kernel naturally maintains recurrent state in that Key-rotated basis;
- q uses the same basis; Value is not rotated;
- Qwen uses the manual Hugging Face runtime, not SGLang-only flags.

Supporting operator-equivalence, FP-parity/math, precision, and formal AIME26 evidence is under
`results/aime26/81920/qwen_gdn/evidence/reference_closure/` and the frozen formal result directory.

The sibling `prototypes/` directory is historical failed/closed work. It is not a canonical learnable-rotation result.
