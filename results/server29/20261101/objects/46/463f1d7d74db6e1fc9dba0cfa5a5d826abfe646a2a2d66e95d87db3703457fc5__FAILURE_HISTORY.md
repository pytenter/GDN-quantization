# Failure history

Historical executions01-05 are read-only, including their original STOPs. New Clean-Room first STOP:

```json
{
  "epoch": 1792043978.5036623,
  "no_retry": true,
  "reason": "AssertionError('UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS')",
  "stage": "DEPENDENCY_FAIL_CLOSED",
  "status": "STOP",
  "task": "LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1",
  "traceback": "Traceback (most recent call last):\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/cleanroom_run.py\", line 9, in protected\n    try:return fn(*args)\n               ^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/cleanroom_run.py\", line 33, in job\n    admitted=g.admit(uuid,tag);assert admitted['graphics_baseline']==read(D/'resource_baselines.json')[uuid]\n             ^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/gpu_admission.py\", line 64, in admit\n    result=assess_window(rows,uuid);result.update(epoch=time.time(),sample_path=str(path))\n           ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/gpu_admission.py\", line 48, in assess_window\n    values=[assess(s,uuid) for s in rows];base=values[0]['graphics']\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/gpu_admission.py\", line 48, in <listcomp>\n    values=[assess(s,uuid) for s in rows];base=values[0]['graphics']\n            ^^^^^^^^^^^^^^\n  File \"/data/zypan/worktrees/ling-kda-4doc-cleanroom-final-v1/gpu_admission.py\", line 40, in assess\n    assert not (union-allowed),'UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS'\n           ^^^^^^^^^^^^^^^^^^^\nAssertionError: UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS\n"
}
```
No second Clean-Room attempt launched.
