# KDA_ROTATION_PREFILL_WRITEBACK_FINAL_LOCALIZATION_V1

PREFILL_DECODE_BOUNDARY_AUDIT = PASS
FULL_MINUS_L64_DOMINANT_REGION = PREFILL

FULL uses packed prefill plus token-by-token history decode.
L64 uses Native packed prefill and the same final 64 Rotated decode transitions.
At t0=64, FULL-L64 differs only in packed prefill and is positive in all 6 formal units.

- 60|64: prefill=152, decode=64, L64_start=152, full_end=215
- 61|128: prefill=121, decode=128, L64_start=185, full_end=248
- 64|256: prefill=81, decode=256, L64_start=273, full_end=336
- 69|64: prefill=138, decode=64, L64_start=138, full_end=201
- 76|128: prefill=66, decode=128, L64_start=130, full_end=193
- 60|128: prefill=152, decode=128, L64_start=216, full_end=279
- 61|256: prefill=121, decode=256, L64_start=313, full_end=376
- 68|64: prefill=118, decode=64, L64_start=118, full_end=181
- 69|128: prefill=138, decode=128, L64_start=202, full_end=265
- 76|256: prefill=66, decode=256, L64_start=258, full_end=321
- 60|256: prefill=152, decode=256, L64_start=344, full_end=407
- 64|64: prefill=81, decode=64, L64_start=81, full_end=144
- 68|128: prefill=118, decode=128, L64_start=182, full_end=245
- 69|256: prefill=138, decode=256, L64_start=330, full_end=393
- 61|64: prefill=121, decode=64, L64_start=121, full_end=184
- 64|128: prefill=81, decode=128, L64_start=145, full_end=208
- 68|256: prefill=118, decode=256, L64_start=310, full_end=373
- 76|64: prefill=66, decode=64, L64_start=66, full_end=129
