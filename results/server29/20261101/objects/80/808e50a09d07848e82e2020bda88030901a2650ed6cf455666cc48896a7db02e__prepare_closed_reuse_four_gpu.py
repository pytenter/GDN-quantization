"""CPU-only exact frozen25 admission. No GPU dispatch or answer scoring."""
import json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pace_control as C
from pace_four_gpu_results import OLD,OLD_SOURCE,predecessor_inventory,verify_reuse
from pace_four_gpu_admission import partition_jobs

def main():
    assert str(ROOT)==C.REMOTE and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (ROOT/'SESSION_STARTED.json').exists() and not (ROOT/'REUSE_ADMISSION.json').exists()
    value=predecessor_inventory();units=value['units'];assert len(units)==25
    for x in units:
        source=Path(x['origin_target']);dest=ROOT/'provenance/REUSED_FORMAL'/x['unit'];dest.mkdir(parents=True,exist_ok=True)
        for p in source.iterdir():
            if p.is_file() and p.suffix.lower() in ('.json','.csv','.txt','.md','.log'):
                assert not p.is_symlink() and p.stat().st_size<=16*1024**2
                q=dest/p.name
                if q.exists():assert C.sha(p)==C.sha(q)
                else:shutil.copyfile(p,q);assert C.sha(p)==C.sha(q)
    admission=dict(status='PASS_CLOSED_PREDECESSOR_FIXED_PAYLOAD_REUSE',origin_root=OLD,origin_source_sha256=OLD_SOURCE,old_parent_controller_workers_finisher_gone=True,origin_terminal_controls=value['origin_terminal_controls'],units=units,reused_ids=value['reused_ids'],copied_original_payloads_not_new_results=True,failed_or_interrupted_prefix_credit=0,scores_used_for_partition=False,unix=time.time())
    C.atomic_new(ROOT/'REUSE_ADMISSION.json',admission)
    jobs=C.read(Path(OLD)/'FORMAL_JOB_MANIFEST.json')['jobs'];plan=partition_jobs(jobs,admission['reused_ids']);by={x['unit']:x for x in plan};prior={x['unit']:x for x in units}
    for job in jobs:
        gpu=by[job['id']]['gpu'] if job['id'] in by else prior[job['id']]['origin_gpu'];job.update(gpu=gpu,gpu_uuid=C.UUIDS[gpu])
    C.status(ROOT/'FORMAL_JOB_MANIFEST.json',dict(task_id=C.TASK,count=60,jobs=jobs,execution_revision=C.REVISION))
    C.atomic_new(ROOT/'FOUR_GPU_PLAN.json',dict(status='FROZEN_CANONICAL_REMAINING_ROUND_ROBIN_FOUR_TP1_QUEUES',reuse_admission_sha256=C.sha(ROOT/'REUSE_ADMISSION.json'),jobs=plan,reused_formal_count=25,new_formal_count=35,total_unique=60,TP=1,no_prefix_join=True,no_answer_based_assignment=True))
    assert len(plan)==35
    proof=C.verify_head_mask(ROOT);result=verify_reuse(ROOT,full_sources=True)
    C.atomic_new(ROOT/'operations/LIFECYCLE_CLOSED_REUSE_CPU_ADMISSION.json',dict(status='PASS',reuse=result,mask=proof,original_chain_gone=True,original_inspection=value,first_jobs=[next(x for x in plan if x['gpu']==g) for g in range(4)],GPU_calls=0))
    print(json.dumps(dict(status='PASS_CLOSED_REUSE_AND_FIXED_FOUR_GPU_PARTITION',reused=25,new=35,first_jobs=[next(x for x in plan if x['gpu']==g) for g in range(4)])))
if __name__=='__main__':main()
