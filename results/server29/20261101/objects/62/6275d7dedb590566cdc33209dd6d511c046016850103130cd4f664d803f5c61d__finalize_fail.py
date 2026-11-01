#!/usr/bin/env python3
"""Fail-closed report writer; performs no scientific recomputation."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V2_CLOSURE')

def pretty(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()

q=json.loads((ROOT/'validation/histogram_quantile_validation.json').read_text())
bad=[r for r in q['rows'] if r['gate']=='FAIL']
max_bad=max(bad,key=lambda r:r['relative_error'])
not_reached={
  'reason':'FAIL_CLOSED_AFTER_GLOBAL_HISTOGRAM_QUANTILE_GATE',
  'status':'NOT_REACHED',
}
for name, gate in [
 ('input_order_determinism.json','INPUT_ORDER_DETERMINISM_GATE'),
 ('four_shard_parity.json','FOUR_SHARD_PARITY'),
 ('merge_order_determinism.json','MERGE_ORDER_DETERMINISM_GATE'),
 ('resume_parity.json','RESUME_PARITY_GATE'),
 ('reducer_state_identity_tests.json','REDUCER_STATE_IDENTITY_GATE'),
 ('atomic_artifact_compatibility.json','ATOMIC_ARTIFACT_COMPATIBILITY_GATE')]:
    pretty(ROOT/'validation'/name,{**not_reached,gate:'NOT_REACHED'})

gates={
 'SOURCE_GATE':'PASS','RAW_SMOKE_IDENTITY_GATE':'PASS','GROUP_LINEAR_IDENTITY_GATE':'PASS',
 'TOPK_TOTAL_ORDER_FROZEN':True,'TOPK_VALUE_PARITY':'EXACT_PASS','TOPK_IDENTITY_PARITY':'EXACT_PASS',
 'TOPK_MERGE_PARITY':'EXACT_PASS','TOPK_SYNTHETIC_TEST_GATE':'PASS',
 'GLOBAL_HISTOGRAM_ALGORITHM_FROZEN':True,'QUANTILE_CONTRACT_V2_FROZEN':True,
 'GLOBAL_HISTOGRAM_QUANTILE_GATE':'FAIL','INPUT_ORDER_DETERMINISM_GATE':'NOT_REACHED',
 'FOUR_SHARD_PARITY':'NOT_REACHED','MERGE_ORDER_DETERMINISM_GATE':'NOT_REACHED',
 'RESUME_PARITY_GATE':'NOT_REACHED','REDUCER_STATE_IDENTITY_GATE':'NOT_REACHED',
 'ATOMIC_ARTIFACT_COMPATIBILITY_GATE':'NOT_REACHED',
 'STREAMING_REDUCER_V2_PARITY':'FAIL','FIRST_FAILED_GATE':'GLOBAL_HISTOGRAM_QUANTILE_GATE',
 'MODEL_LOADED':False,'CUDA_ALLOCATED':False,'GPU_STREAMING_SMOKE_STARTED':False,
 'FULL60_STARTED':False,'FULL60_RESOURCE_READINESS':'NOT_REACHED','RECURRENT_REPLAY_STARTED':False,
 'MULTI_HORIZON_STARTED':False,'PRE_DIVERGENCE_STARTED':False,'LAYER_INTERVENTION_STARTED':False,
 'C7_STARTED':False,'NEW_ROTATION_TRAINING_STARTED':False,'NEW_AIME_GENERATION_STARTED':False,
 'NEXT_RECOMMENDED_ACTION':'NEW_PROSPECTIVE_REDUCER_PROTOCOL_REQUIRED',
}
pretty(ROOT/'validation/final_gates.json',gates)
pretty(ROOT/'reference/exact_summary.json',{
 'raw_rows':393216,'topk_key_comparisons':384,'topk_merge_comparisons':1152,
 'histogram_required_combinations':len(q['rows']),'histogram_failed_combinations':len(bad),
 'maximum_relative_error':q['maximum_relative_error'],'maximum_failure':max_bad,
 'percentile_method':'linear','gates':gates,
})

spec="""# Reducer V2 specification

Schema: `stage_b_streaming_reducer_v2`. One independent state is used per sample; samples merge only in the frozen 60-sample manifest order.

Top-32 uses metric float64 descending, token index ascending, then canonical group-linear index ascending. Signed zero is canonicalized and nonfinite values fail closed.

Global secondary quantiles use an integer deterministic multiplicative histogram with `GAMMA=1.001`, an exact zero bucket, floor-log positive indexing with explicit boundary correction, and conservative bucket-upper-edge estimates. The prospective contract was frozen before raw validation: p95/p99/p99.9 relative error <=0.20%; exact-zero reference requires exact zero estimate.

V1 remains failed under its old algorithm and old 0.3% contract. V2 did not retroactively alter V1. V2 also failed its distinct prospective contract.
"""
(ROOT/'spec/REDUCER_V2_SPEC.md').write_text(spec,encoding='utf-8')

report=f"""# FINAL REDUCER V2 REPORT

## Verdict

`STREAMING_REDUCER_V2_PARITY = FAIL`

`FIRST_FAILED_GATE = GLOBAL_HISTOGRAM_QUANTILE_GATE`

The frozen histogram contract was evaluated on 1,200 required condition/layer-or-global/metric/percentile combinations. {len(bad)} failed. Maximum relative error was {100*q['maximum_relative_error']:.6f}% at `{max_bad['condition']}/{max_bad['layer']}/{max_bad['metric']}/{max_bad['percentile']}` (exact {max_bad['exact_value']:.17g}, estimate {max_bad['estimated_value']:.17g}). No tolerance, gamma, K, tie-break, metric, or observation was changed after seeing this result.

The key mechanism is that a conservative upper-order-statistic bucket estimate is not bounded relative to NumPy's linear-interpolation quantile merely by the 0.1% bucket width: a gap between adjacent order statistics can dominate. A new prospective protocol is required.

Q1. V1 Top-32 identity mismatch was caused by `argpartition` selecting an unspecified subset of large equal-value, especially zero, ties.

Q2. V2 total order is metric float64 descending, token index ascending, group-linear index ascending.

Q3. All-zero, tie-at-K, signed-zero, shuffled-input, sharded-merge, and duplicate-ID fail-fast synthetic tests passed.

Q4. Raw full-sort and streaming Top-32 values, identities, and ranks were exact across 384 keys.

Q5. One-, two-, and four-shard Top-K merges were exact across 1,152 comparisons.

Q6. The new global-tail algorithm is a deterministic integer multiplicative histogram with exact zero bucket and corrected positive-bin boundaries.

Q7. `GAMMA=1.001` was selected prospectively for about 0.1% multiplicative bin width.

Q8. The prospective contract was <=0.20% relative error for p95/p99/p99.9 (and exact zero for a zero reference), written before raw validation.

Q9. No. {len(bad)} of {len(q['rows'])} required combinations exceeded 0.20%.

Q10. NOT_REACHED because the task stopped at the first failed gate.

Q11. NOT_REACHED because the task stopped at the first failed gate.

Q12. NOT_REACHED because the task stopped at the first failed gate.

Q13. NOT_REACHED because the task stopped at the first failed gate.

Q14. NOT_REACHED because the task stopped at the first failed gate.

Q15. `STREAMING_REDUCER_V2_PARITY = FAIL`.

Q16. Was GPU run? **NO.**

Q17. Was full60 started? **NO.**

## Frozen stop state

```text
MODEL_LOADED = FALSE
CUDA_ALLOCATED = FALSE
GPU_STREAMING_SMOKE_STARTED = FALSE
FULL60_STARTED = FALSE
FULL60_RESOURCE_READINESS = NOT_REACHED
RECURRENT_REPLAY_STARTED = FALSE
MULTI_HORIZON_STARTED = FALSE
PRE_DIVERGENCE_STARTED = FALSE
LAYER_INTERVENTION_STARTED = FALSE
C7_STARTED = FALSE
NEW_ROTATION_TRAINING_STARTED = FALSE
NEW_AIME_GENERATION_STARTED = FALSE
NEXT_RECOMMENDED_ACTION = NEW_PROSPECTIVE_REDUCER_PROTOCOL_REQUIRED
```
"""
(ROOT/'reports/FINAL_REDUCER_V2_REPORT.md').write_text(report,encoding='utf-8')

hashfile=ROOT/'hashes/SHA256SUMS'
paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p!=hashfile)
hashfile.parent.mkdir(parents=True,exist_ok=True)
hashfile.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in paths),encoding='utf-8')
print(json.dumps({'status':'FAIL','first_failed_gate':'GLOBAL_HISTOGRAM_QUANTILE_GATE','failed_combinations':len(bad),'hashed_artifacts':len(paths)},sort_keys=True))
