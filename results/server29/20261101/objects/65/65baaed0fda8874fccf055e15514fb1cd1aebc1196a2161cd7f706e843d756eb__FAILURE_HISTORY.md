# Failure history

Historical executions01-05 and previous Clean-Room STOP are read-only. This GPU3-only execution has new explicit human authorization. Current first STOP:

```json
{
  "epoch": 1792118206.72933,
  "evidence_path": "/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2/FAILED_RESOURCE_SNAPSHOT.json",
  "evidence_sha256": "22cd8873acee4ed2f17732cca03bdb4bf612890bebc6f6e08ee05c67fa9d14a9",
  "forbiddenPIDs": [],
  "no_retry": true,
  "original_resource_error": {
    "repr": "AssertionError('MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID')",
    "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/resource_evidence.py\", line 114, in assess_with_evidence\n    return assess_fn(snapshot,uuid,owner,baseline)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/gpu_admission.py\", line 43, in assess\n    if m['used']>64*2**20:assert owner['pid'] in union,'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID'\n                                 ^^^^^^^^^^^^^^^^^^^^^\nAssertionError: MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\n",
    "type": "AssertionError"
  },
  "reason": "ResourceEvidenceError('RESOURCE_ASSESSMENT_FAILED_WITH_PINNED_EVIDENCE: {\\'type\\': \\'AssertionError\\', \\'repr\\': \"AssertionError(\\'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\\')\", \\'traceback\\': \\'Traceback (most recent call last):\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/resource_evidence.py\", line 114, in assess_with_evidence\\\\n    return assess_fn(snapshot,uuid,owner,baseline)\\\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-gpu3-single-v2/gpu_admission.py\", line 43, in assess\\\\n    if m[\\\\\\'used\\\\\\']>64*2**20:assert owner[\\\\\\'pid\\\\\\'] in union,\\\\\\'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\\\\\\'\\\\n                                 ^^^^^^^^^^^^^^^^^^^^^\\\\nAssertionError: MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID\\\\n\\'}')",
  "stage": "RUNNING_RESOURCE_SAFETY",
  "status": "STOP",
  "tag": "pg19-test-41603__H__parent",
  "task": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1"
}
```
No further attempt beyond this explicitly authorized GPU3-only execution is launched.
