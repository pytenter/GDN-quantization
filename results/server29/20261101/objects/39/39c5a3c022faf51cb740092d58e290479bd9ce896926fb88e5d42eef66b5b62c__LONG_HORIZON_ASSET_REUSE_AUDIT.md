# Long-horizon asset reuse audit

## Audited assets

### LING_ERROR_COHERENCE_DIAGNOSIS_V1

The frozen trace inventory contains 256 `.pt` chunks (64 per observability document, approximately 50 GiB total) for `37702`, `28988`, `5396`, and `26493`. Each document has 32,768 tokens in 512-token chunks, 18 KDA layer IDs, FP32 cache records, document-start all-zero state, token hashes, step hashes, and per-token records including `q`, `k`, `v`, `beta`, `log_decay`, and `fp_after`.

These traces are the strongest reusable asset: they preserve the frozen FP driver/readout inputs and can support deterministic trace-level replay without repeating the original language-model forward pass. Historical coherence result files themselves preserve scalar summaries and final-state hashes, not per-condition state tensors.

### Observability Bridge canonical and intermediates

The 80 canonical units contain verified scalar JPROP, POR, FutureKL, gate results, identities, and source hashes. They do not store the per-condition quantized state, FP state, or error vector at each anchor. The formal output set therefore cannot reconstruct a direction from scalar norms. A small postcommit `.pt` smoke artifact is not a substitute for the formal-unit state capture.

### LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1

The audited inventory has 102 JSON files and 273 `.pt` files (approximately 51 GiB), but its measured PG19 documents are `32761`, `33426`, `3754`, and `49078`, not the four observability documents. Matching 32K length and related rotation semantics do not make those per-condition measurements mergeable with the current panel. They are independent historical support for long-horizon propagation, not recoverable direction-level FutureKL evidence for these units.

### LING_KDA_QUERY_READOUT_CAUSAL_CLOSURE_V1

The retained result package contains CSV/JSON/Markdown summaries but no reusable state tensors. Its 36 common units and 1,296 local causal evaluations support the exact KDA query/readout mechanism: median native gap 0.00335486, query-equalized gap 0.00003301, median query rescue 0.00329849 (CI 0.00304159--0.00353883), and descriptive mediation fraction about 0.9909. It validates local query-visible versus query-null decomposition, not closed-loop long-horizon FutureKL causality.

## What can and cannot be recovered

- Can reuse: frozen tokens, FP trace driver, future query/readout inputs, rotation identities, original quantizer semantics, and all scalar outcomes.
- Can recompute by trace replay: per-condition quantized recurrent states and error vectors, provided the frozen replay implementation and dependencies are preserved. This avoids a full model forward, but may still require substantial tensor compute/memory and must be separately validated.
- Cannot recover from scalar JSON: an error direction, a norm-matched visible/null decomposition, or logits/FutureKL intermediates.
- Cannot directly merge: propagation-confirm per-condition measurements, because their documents differ.
- Cannot obtain full functional FutureKL from the archived traces alone unless the missing model/logit evaluation path is rerun; trace-only readout tests have a narrower local/trajectory-observability interpretation.

## Reuse verdict

The 80-unit descriptive analysis requires no new 32K rollout and is complete from canonical scalars. A direction-level causal diagnostic is feasible by replaying the frozen same-document traces to reconstruct states, which is much cheaper scientifically than regenerating all trajectories. However, no archived tensor snapshot permits a pure scalar-to-vector reconstruction, and full recurrent logit/FutureKL causality would still require a controlled model evaluation. Historical assets therefore support a staged intervention, not a zero-cost causal conclusion.
