# Ling/KDA Value-Hadamard semantics

Canonical semantics version: `CORRECTED_PREFILL_ENDPOINT_V2`.

The recurrent state returned by the KDA prefill kernel is already in rotated Value coordinates. It is written directly to cache and consumed in the same basis by the first decode step. Do not apply an additional H or H-transpose at the prefill endpoint.

Only the KDA core/readout output is transformed back to native Value coordinates before downstream RMSNorm, learned scale/dynamic gate, head merge, and `o_proj`.

The scripts and tests in this directory localize the historical double-rotation bug and validate the corrected post-fix rebaseline. Historical outputs produced by the redundant endpoint rotation path are `INVALID_OLD_ROTATION_SEMANTICS` and must not be mixed with current formal results.
