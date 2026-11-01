"""Bounded CPU-only preparation after genuine predecessor terminal closure."""
import json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pace_control as C
from pace_four_gpu_results import OLD,OLD_SOURCE,old_validators
from pace_four_gpu_admission import partition_jobs

def main():
    assert str(ROOT)==C.REMOTE and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (ROOT/'SESSION_STARTED.json').exists() and not (ROOT/'REUSE_ADMISSION.json').exists()
    old=Path(OLD);oldc,K=old_validators()
    controls=['source_manifest.json','GPU_PERMIT.json','SCIENCE_FREEZE.json','SESSION_RESULT.json','RESOURCE_LEDGER.json','final_verdict.json','DELIVERY_MANIFEST.json','ARCHIVE_VERIFICATION.json','DELIVERY_READY.json','OFFLINE_PARENT_STARTED.json','OFFLINE_PARENT_RESULT.json','PARENT_CONTROLLER_EXIT.json','PARENT_FINISHER_EXIT.json','FINISHER_STARTED.json','FINISHER_EXIT.json','GLOBAL_STOP_REQUEST.json']
    rows={n:dict(sha256=C.sha(old/n),bytes=(old/n).stat().st_size) for n in controls}
    assert rows['source_manifest.json']['sha256']==OLD_SOURCE
    parent=C.read(old/'OFFLINE_PARENT_RESULT.json');fin=C.read(old/'PARENT_FINISHER_EXIT.json');ctrl=C.read(old/'PARENT_CONTROLLER_EXIT.json')
    assert fin['actual_exited'] and fin['identity_gone'] and fin['returncode']==0
    assert ctrl['actual_exited'] and ctrl['identity_gone'] and ctrl['returncode']==1
    chain=dict(parent_OS_returncode='UNRESOLVED_NO_OWNING_WAIT_IN_PREDECESSOR',controller_owning_Popen_returncode=ctrl['returncode'],finisher_owning_Popen_returncode=fin['returncode'])
    ids=[parent['identity'],ctrl['identity'],fin['identity']]
    ledger=C.read(old/'RESOURCE_LEDGER.json');ids += [x['identity'] for x in ledger['intervals'] if x.get('identity')]
    assert all(not C.same_process(x) for x in ids)
    nvml=C.query_gpu();assert all(C.release_state(True,nvml,x['pid'],x) for x in ids)
    assert C.read(old/'DELIVERY_READY.json')['status']=='VERIFIED_COMPLETE_OR_REAL_STOP'
    bad=[];pairs=K.completed_rows(old,bad);assert len(pairs)==15 and len(bad)==1 and bad[0]['unit']=='aime26_08_R3_seed1' and bad[0]['partial_completion_credit']==0
    stopped=C.read(oldc.unit_target(old,'formal','aime26_08_R3_seed1')/'RESOURCE_INTERVAL.json');assert stopped['returncode']==-15 and stopped['released'] is True
    assert C.sha(old/'GLOBAL_STOP_REQUEST.json')=='bbee4aee49b2f7ab84fd2fbeff45822af378083012e53c47995520ac080afa6c'
    units=[]
    for job,wrapper,target in pairs:
        unit=job['id'];dest=ROOT/'provenance/REUSED_FORMAL'/unit;dest.mkdir(parents=True,exist_ok=False)
        for p in target.iterdir():
            if p.is_file() and p.suffix.lower() in ('.json','.csv','.txt','.md','.log'):
                assert not p.is_symlink() and p.stat().st_size<=16*1024**2
                shutil.copyfile(p,dest/p.name);assert C.sha(p)==C.sha(dest/p.name)
        interval=C.read(target/'RESOURCE_INTERVAL.json');assert interval['returncode']==0 and interval['released'] is True
        units.append(dict(unit=unit,payload_sha256=wrapper['payload_sha256'],result_sha256=C.sha(target/'RESULT.json'),interval_sha256=C.sha(target/'RESOURCE_INTERVAL.json'),origin_target=str(target),origin_source_sha256=wrapper['source_manifest_sha256']))
    admission=dict(status='PASS_CLOSED_PREDECESSOR_FIXED_PAYLOAD_REUSE',origin_root=OLD,origin_source_sha256=OLD_SOURCE,old_parent_controller_workers_finisher_gone=True,origin_terminal_controls=rows,units=units,reused_ids=[x['unit'] for x in units],copied_original_payloads_not_new_results=True,failed_or_interrupted_prefix_credit=0,scores_used_for_partition=False,identity_NVML=nvml,unix=time.time())
    C.atomic_new(ROOT/'REUSE_ADMISSION.json',admission)
    jobs=C.read(old/'FORMAL_JOB_MANIFEST.json')['jobs'];plan=partition_jobs(jobs,admission['reused_ids']);by={x['unit']:x for x in plan}
    for job in jobs:
        gpu=by[job['id']]['gpu'] if job['id'] in by else 0;job.update(gpu=gpu,gpu_uuid=C.UUIDS[gpu])
    C.status(ROOT/'FORMAL_JOB_MANIFEST.json',dict(task_id=C.TASK,count=60,jobs=jobs,execution_revision=C.REVISION))
    C.atomic_new(ROOT/'FOUR_GPU_PLAN.json',dict(status='FROZEN_CANONICAL_REMAINING_ROUND_ROBIN_FOUR_TP1_QUEUES',reuse_admission_sha256=C.sha(ROOT/'REUSE_ADMISSION.json'),jobs=plan,reused_formal_count=len(units),new_formal_count=len(plan),total_unique=60,TP=1,no_prefix_join=True,no_answer_based_assignment=True))
    assert len(plan)==45
    proof=C.verify_head_mask(ROOT);from pace_four_gpu_results import verify_reuse
    result=verify_reuse(ROOT,full_sources=True);C.atomic_new(ROOT/'operations/CLOSED_REUSE_CPU_ADMISSION.json',dict(status='PASS',reuse=result,mask=proof,original_chain_gone=True,old_owningwait=chain,first_jobs=[next(x for x in plan if x['gpu']==g) for g in range(4)],GPU_calls=0))
    print(json.dumps(dict(status='PASS_CLOSED_REUSE_AND_FIXED_FOUR_GPU_PARTITION',reused=15,new=45,first_jobs=[next(x for x in plan if x['gpu']==g) for g in range(4)])))

if __name__=='__main__':main()
