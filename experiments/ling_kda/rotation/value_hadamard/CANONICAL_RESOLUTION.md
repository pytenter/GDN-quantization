# Ling/KDA Value-Hadamard canonical resolution

The selected files in this directory are byte-for-byte identical between the
`ling4090` and `new4090` audit branches for all four implementation scripts and
all four regression tests. The compact corrected post-fix result files also
match byte-for-byte.

- Canonical code provenance: the later `new4090` / `nlpg-SYS-4029GP-TRT` post-fix chain.
- Organized imported files: the equivalent `ling4090` copies.
- Formal AIME26 81,920 result provenance: `ling4090` / `lthpc1`.
- Semantics: `CORRECTED_PREFILL_ENDPOINT_V2`.
- `REDUNDANT_PREFILL_ENDPOINT_ROTATION`: `NO`.
- Prefill kernel-return -> cache -> first-decode state-basis continuity: `PASS`.

Historical double endpoint rotation is `INVALID_OLD_ROTATION_SEMANTICS` and is retained only in clearly labelled reinterpretation/audit evidence. It is not part of the selected implementation.
