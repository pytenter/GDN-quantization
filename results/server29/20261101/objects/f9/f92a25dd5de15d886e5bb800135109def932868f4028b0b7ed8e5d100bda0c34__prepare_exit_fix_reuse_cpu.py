"""Prepare fixed19 closed reuse and original-order41 four-card assignment; CPU only."""
import json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pace_control as C
from pace_four_gpu_results import predecessor_inventory,OLD,OLD_SOURCE
from pace_four_gpu_admission import partition_jobs
assert str(ROOT)==C.REMOTE and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (ROOT/'SESSION_STARTED.json').exists()
v=predecessor_inventory();assert not (ROOT/'REUSE_ADMISSION.json').exists()
for x in v['units']:
 source=Path(x['origin_target']);target=ROOT/'provenance/REUSED_FORMAL'/x['unit'];target.mkdir(parents=True,exist_ok=True)
 for p in source.iterdir():
  if not p.is_file() or p.suffix.lower() not in ('.json','.csv','.txt','.log','.md'):continue
  q=target/p.name
  if q.exists():assert C.sha(q)==C.sha(p)
  else:shutil.copyfile(p,q)
a=dict(status='PASS_CLOSED_PREDECESSOR_FIXED_PAYLOAD_REUSE',origin_root=OLD,origin_source_sha256=OLD_SOURCE,old_parent_controller_workers_finisher_gone=True,origin_terminal_controls=v['origin_terminal_controls'],units=v['units'],reused_ids=v['reused_ids'],interrupted_prefix_credit=0,scores_used_for_selection=False)
C.atomic_new(ROOT/'REUSE_ADMISSION.json',a)
jobs=C.read(Path(OLD)/'FORMAL_JOB_MANIFEST.json')['jobs'];plan=partition_jobs(jobs,a['reused_ids']);by={x['unit']:x for x in plan};prior={x['unit']:x for x in a['units']}
for j in jobs:
 g=by[j['id']]['gpu'] if j['id'] in by else prior[j['id']]['origin_gpu'];j.update(gpu=g,gpu_uuid=C.UUIDS[g])
C.status(ROOT/'FORMAL_JOB_MANIFEST.json',dict(task_id=C.TASK,count=60,jobs=jobs,execution_revision=C.REVISION))
C.atomic_new(ROOT/'FOUR_GPU_PLAN.json',dict(status='FROZEN_CANONICAL_REMAINING_ROUND_ROBIN_FOUR_TP1_QUEUES',reuse_admission_sha256=C.sha(ROOT/'REUSE_ADMISSION.json'),jobs=plan,reused_formal_count=19,new_formal_count=41,total_unique=60,TP=1,no_prefix_join=True,no_answer_based_assignment=True))
assert len(plan)==41
from pace_four_gpu_results import verify_reuse
p=verify_reuse(ROOT,full_sources=True);mask=C.verify_head_mask(ROOT);C.atomic_new(ROOT/'operations/CLOSED_19_REUSE_CPU_ADMISSION.json',dict(status='PASS',reuse=p,mask=mask,predecessor4_actual_CUDA_gate_proofs=v['gates'],first_jobs=[next(x for x in plan if x['gpu']==g) for g in range(4)],actual_new_GPU_dispatches=0))
print(json.dumps(dict(status='PASS_CLOSED19_CANONICAL41_PARTITION',first_jobs=[next(x for x in plan if x['gpu']==g) for g in range(4)])))
