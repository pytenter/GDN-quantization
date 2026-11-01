"""CPU-only admission of the closed54 and canonical remaining6; never CUDA."""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pace_control as C
from pace_four_gpu_results import predecessor_inventory, OLD, OLD_SOURCE, verify_reuse
from pace_four_gpu_admission import partition_jobs

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and str(ROOT) == C.REMOTE
assert not (ROOT / 'SESSION_STARTED.json').exists() and not (ROOT / 'source_manifest.json').exists()
assert not (ROOT / 'operations/REBALANCE_CLOSED_54_REUSE_CPU_ADMISSION.json').exists()
proof = predecessor_inventory()
assert len(proof['units']) == 54
saved = ROOT / 'provenance/REPLACED_INHERITED_CONTROL'
saved.mkdir(exist_ok=False)
for name in ('REUSE_ADMISSION.json', 'FOUR_GPU_PLAN.json', 'FORMAL_JOB_MANIFEST.json', 'operations/FOUR_GPU_CPU_RELEASE_ADMISSION.json', 'operations/FINAL_NUMERIC_SOURCE_IDENTITY_REVIEW.json'):
    shutil.move(str(ROOT / name), str(saved / Path(name).name))
for row in proof['units']:
    source = Path(row['origin_target'])
    target = ROOT / 'provenance/REUSED_FORMAL' / row['unit']
    target.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if not p.is_file() or p.is_symlink() or p.suffix.lower() not in ('.json', '.jsonl', '.csv', '.txt', '.log', '.md'):
            continue
        C.require(p.stat().st_size <= 32 * 1024**2, 'Unexpected large small-file reuse artifact: ' + str(p))
        q = target / p.name
        if q.exists():
            C.require(C.sha(q) == C.sha(p), 'Copied original reuse bytes differ')
        else:
            shutil.copyfile(p, q)
admission = dict(status='PASS_CLOSED_PREDECESSOR_FIXED_PAYLOAD_REUSE', origin_root=OLD,
                 origin_source_sha256=OLD_SOURCE, old_parent_controller_workers_finisher_gone=True,
                 origin_terminal_controls=proof['origin_terminal_controls'], units=proof['units'],
                 reused_ids=proof['reused_ids'], interrupted_prefix_credit=0, scores_used_for_selection=False)
C.atomic_new(ROOT / 'REUSE_ADMISSION.json', admission)
jobs = C.read(Path(OLD) / 'FORMAL_JOB_MANIFEST.json')['jobs']
plan = partition_jobs(jobs, admission['reused_ids'])
assert [x['unit'] for x in plan] == ['aime26_%02d_R2_seed1' % i for i in range(25, 31)]
by = {x['unit']: x for x in plan}
prior = {x['unit']: x for x in admission['units']}
for job in jobs:
    gpu = by[job['id']]['gpu'] if job['id'] in by else prior[job['id']]['origin_gpu']
    job.update(gpu=gpu, gpu_uuid=C.UUIDS[gpu])
C.atomic_new(ROOT / 'FORMAL_JOB_MANIFEST.json', dict(task_id=C.TASK, count=60, jobs=jobs, execution_revision=C.REVISION))
C.atomic_new(ROOT / 'FOUR_GPU_PLAN.json', dict(status='FROZEN_CANONICAL_REMAINING_ROUND_ROBIN_FOUR_TP1_QUEUES',
             reuse_admission_sha256=C.sha(ROOT / 'REUSE_ADMISSION.json'), jobs=plan,
             reused_formal_count=54, new_formal_count=6, total_unique=60, TP=1,
             no_prefix_join=True, no_answer_based_assignment=True))
metadata = ROOT / 'provenance/PREVIOUS_LIFECYCLE_TRUE_STOP_METADATA'
metadata.mkdir(exist_ok=False)
controls = list(proof['origin_terminal_controls']) + ['GLOBAL_STOP_REQUEST.json', 'FIRST_FAILURE.json']
for name in controls:
    shutil.copyfile(Path(OLD) / name, metadata / name)
verified = verify_reuse(ROOT, full_sources=True)
value = dict(status='PASS_54_ORIGINAL_CLOSED_PAYLOADS_CANONICAL6_FOUR_CARD_CPU_ADMISSION', reuse=verified,
             original_inspection=proof, mask=C.verify_head_mask(ROOT), jobs=plan,
             actual_new_GPU_dispatches=0, source_not_yet_frozen=True)
C.atomic_new(ROOT / 'operations/REBALANCE_CLOSED_54_REUSE_CPU_ADMISSION.json', value)
print(json.dumps(dict(status=value['status'], reused=54, remaining=6, jobs=plan)), flush=True)
