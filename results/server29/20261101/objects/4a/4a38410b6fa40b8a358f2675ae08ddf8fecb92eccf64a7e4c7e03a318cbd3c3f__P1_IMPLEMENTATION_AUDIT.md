# P1 implementation audit

- P1 was implemented only after `STREAMING_REDUCER_ACCEL_P0_PARITY = PASS`.
- P0 histogram acceleration is retained unchanged.
- Per update, NumPy `lexsort` extracts the local Top-32 using descending value and ascending group identity.
- The selected local candidates are merged on CPU with the original frozen total-order key: descending value, ascending token index, ascending group index.
- Only local Top-32 candidates can affect the global Top-32, so discarded local events cannot change the result.
- Duplicate event identity checks and deterministic global merge checks are preserved.
- Top-42 tail computation remains the original frozen implementation.
- No model, formal path, rotation, quantizer, metric, writeback or loss logic is changed.

