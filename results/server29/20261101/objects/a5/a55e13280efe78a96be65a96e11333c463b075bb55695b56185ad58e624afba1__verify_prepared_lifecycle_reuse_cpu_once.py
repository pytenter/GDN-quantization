import os,sys,json
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
import pace_control as C
from pace_four_gpu_results import verify_reuse,predecessor_inventory
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (r/'SESSION_STARTED.json').exists()
# Canonical prompts/math remain exact; normalize current CPU-staging assignments
# after local-only metadata sync, before any SOURCE freeze or dispatch.
plan=C.read(r/'FOUR_GPU_PLAN.json');reuse=C.read(r/'REUSE_ADMISSION.json');jobs=C.build_jobs(C.read(r/'AIME26_DATASET_FREEZE.json'))['jobs']
by={x['unit']:x for x in plan['jobs']};prior={x['unit']:x for x in reuse['units']}
for j in jobs:
 g=by[j['id']]['gpu'] if j['id'] in by else prior[j['id']]['origin_gpu'];j.update(gpu=g,gpu_uuid=C.UUIDS[g])
C.status(r/'FORMAL_JOB_MANIFEST.json',dict(task_id=C.TASK,count=60,jobs=jobs,execution_revision=C.REVISION))
v=verify_reuse(r,full_sources=True);proof=predecessor_inventory()
assert v['count']==25 and len(plan['jobs'])==35
value=dict(status='PASS_INDEPENDENT_CPU_ADMISSION_AFTER_INHERITED_RECEIPT_NAME_COLLISION',reuse=v,original_inspection=proof,mask=C.verify_head_mask(r),first_jobs=[next(x for x in plan['jobs'] if x['gpu']==g) for g in range(4)],GPU_calls=0,prior_preparation_outputs_preserved=True)
C.atomic_new(r/'operations/LIFECYCLE_CLOSED_REUSE_CPU_ADMISSION.json',value)
print(json.dumps(dict(status=value['status'],reused=25,new=35,first_jobs=value['first_jobs'])))
