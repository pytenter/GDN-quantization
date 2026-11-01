# Minimal GPU validation plan

## Decision

Do not repair the missing 48 and do not start a new 128-unit panel. The lowest-cost prospective test is four independent Primary units in a new pilot namespace: two preregistered rotation conditions on each unfinished document.

## Frozen contrast

Use `H_to_L7_a50` versus `L7R`, selected from the two complete documents using only input-side separation criteria: near JPROP magnitude with a materially different POR, with the same higher-POR member in both documents.

Historical selection diagnostics (reported for transparency, not used as prospective outcomes):

| complete document | relative JPROP gap | POR ratio (H_to_L7_a50 / L7R) | historical FutureKL difference |
|---|---:|---:|---:|
| 37702 Primary | 10.64% | 1.498 | +144.658 |
| 5396 Primary | 7.24% | 1.668 | +73.684 |

The strict historical match rule was 10%/1.50, so the 37702 pair narrowly misses both cutoffs. This pair is chosen for a prospective contrast because it is the best common near-match across both complete documents, not because it belongs to the 18 strict pairs.

## Four measured units

- GPU6 UUID `GPU-de53f243-b271-521c-d875-d96611ed8574`: `pg19-test-28988`, Primary anchor 8192, conditions `H_to_L7_a50` and `L7R`.
- GPU3 UUID `GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae`: `pg19-test-26493`, Primary anchor 8192, same two conditions.
- Endpoint 32768; original persistent every-token QDQ/writeback semantics; original rotation/checkpoint; no L9/new rotation/AIME/free generation/Qwen.
- Store in a separate pilot namespace. Never overwrite or reinterpret historical canonical, including the already existing old-runtime 28988/L7R unit.
- Within each document, both contrast members must be generated under the same runtime, same physical GPU UUID, and same-runtime FP reference. If a qualified reusable same-runtime FP reference does not already exist, generate one reference pass per document as explicit preparation; those passes are not silently counted as measured Primary units.

Primary endpoint: whether `H_to_L7_a50` has higher FutureKL than `L7R` in each untouched document while retaining the preregistered magnitude/observability contrast. Four units are a directional pilot, not a 16-condition ranking or a confirmatory test.

## Cost estimate

Attempt04 produced 70 new canonical units across two GPUs in 46.9277 hours wall time, equivalent to 93.855 GPU-hours or 1.3408 GPU-hours per unit on average.

| option | measured units | estimated total GPU-hours | estimated parallel wall time on GPU3+GPU6 | scientific role |
|---|---:|---:|---:|---|
| current analysis | 0 | 0 | 0 | completed descriptive triage |
| recommended pilot | 4 | 5.36 | 2.68 h | independent two-document directional replication |
| complete two Primary panels | 32 | 42.91 | 21.45 h | full new-doc rotation ranking |
| add two Secondary panels | +32 | +42.91 | +21.45 h | anchor replication |
| entirely new 128-unit runtime panel | 128 | 171.62 | 85.81 h | last-resort complete panel |

These are throughput-based estimates; per-anchor/unit variation and FP-reference preparation are excluded and must be measured separately. The pilot should stop after four measured units and be analyzed before expansion.

## Preregistration and stop rule

Before launch, freeze identities, runtime, same-GPU FP references, exact writer, thresholds, and the directional hypothesis. If either unit in a document fails a hardware/runtime/science gate, that document is not interpreted and no automatic retry occurs. If the two documents do not give the same direction, return to the causal direction intervention rather than expanding blindly to 32 or 128 units.

Execution status: `execution_authorized=false`; no GPU process was started by this task.
