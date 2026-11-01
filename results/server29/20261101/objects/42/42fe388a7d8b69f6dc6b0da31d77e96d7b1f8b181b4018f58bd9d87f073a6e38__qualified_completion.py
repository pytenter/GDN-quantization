"""Engineering routing/reuse only. All scientific worker mathematics is frozen."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from recovery_state import atomic_json,lock,sha,complete
from gpu_admission import proc_identity
from prospective_common import OUT as D, WT as W, read, verify

FIELDS=['uninterrupted_suffix_full_logit_sha256','final_full_native_cache_sha256','audit_state_fingerprint','checkpoint_sha256']

def compare_exact(parent,fresh,uuid):
    assert parent['status']==fresh['status']=='PASS'
    assert parent['begin']==0 and fresh['begin']==8192 and parent['endpoint']==fresh['endpoint']==8256
    assert parent['observer_enabled'] is True and fresh['observer_enabled'] is False
    assert parent['document']==fresh['document'] and parent['condition']==fresh['condition']
    assert fresh['host']['gpu_uuid']==uuid,'WRONG_DEVICE_NUMERICAL_LABEL'
    assert all(parent[k]==fresh[k] for k in FIELDS),'EXACT_DEVICE_RESUME_PARITY_FAILED'
    for key in ['kernel','driver','torch','cuda_runtime','selector']:
        assert parent['host'][key]==fresh['host'][key],'RUNTIME_CHANGED'
    if parent['condition']!='FP':
        for r in [parent,fresh]:
            q=r['integrity'];assert q['passed'] and q['every_token_layer_qdq_count']==8256*18 and q['actual_next_kernel_equal_post_qdq_count']==8255*18
    return True

def reuse_parent(document,condition):
    bindings=read(D/'REUSED_PARENT_BINDINGS.json')['units']
    key=document+'__'+condition
    if key not in bindings:return None
    b=bindings[key];p=D/'runs'/(key+'__parent.json');cp=D/'runs'/(key+'__8192.pt')
    assert sha(p)==b['parent_sha256'] and sha(cp)==b['checkpoint_sha256']
    assert sha(b['original_manifest_path'])==b['original_manifest_sha256']
    parent=read(p);receipt=read(b['original_natural_exit_path'])
    assert sha(b['original_natural_exit_path'])==b['original_natural_exit_sha256']
    assert receipt['natural_exit'] and receipt['result_sha256']==sha(p) and parent['status']=='PASS'
    assert parent['document']==document and parent['condition']==condition and parent['begin']==0 and parent['endpoint']==8256
    assert parent['host']['gpu_uuid']==b['original_gpu_uuid'] and parent['checkpoint_sha256']==sha(cp)
    assert sha(D/'FROZEN_SPEC.json')==b['checkpoint_contract_sha256']
    from recovery_controller import event
    event('HISTORICAL_PARENT_REUSED_WITHOUT_RECOMPUTATION',document=document,rotation=condition,original_gpu_uuid=b['original_gpu_uuid'],target_gpu_uuid=os.environ['LING_TARGET_GPU_UUID'],new_device_qualification='PENDING_UNTIL_EXACT_FRESH64')
    return parent

def qualification_unit(u,uuid):
    from recovery_controller import run,event
    doc,rot=u['document'],u['rotation'];key=doc+'__'+rot
    parent=run(doc,rot,'parent');fresh=run(doc,rot,'resumed_parity')
    compare_exact(parent,fresh,uuid)
    inventory=parent['cache_inventory'];first=inventory['511'];last=inventory['8255']
    rate=(last['cache_bytes']-first['cache_bytes'])/(8256-512);assert rate==61440
    assert all(x['cache_bytes']==first['cache_bytes']+(int(k)+1-512)*rate for k,x in inventory.items())
    row=dict(document=doc,rotation=rot,status='PASS',parent_sha256=sha(D/'runs'/(key+'__parent.json')),resumed_sha256=sha(D/'runs'/(key+'__resumed_parity.json')),checkpoint_sha256=parent['checkpoint_sha256'],parent_gpu_uuid=parent['host']['gpu_uuid'],qualified_gpu_uuid=uuid,device_migration_exact=parent['host']['gpu_uuid']!=uuid,observer_ON_equals_OFF=True,own_rotation_history=True,fresh_process_exact_64_logits_cache_audit=True,native_cache_growth_bytes_per_token=rate,predicted_cache_bytes_32768=last['cache_bytes']+(32768-8256)*rate,checkpoint_source_spec_sha256=sha(D/'FROZEN_SPEC.json'))
    atomic_json(D/'qualification_units'/(key+'.json'),row)
    event('TARGET_DEVICE_CONDITION_QUALIFIED',document=doc,rotation=rot,uuid=uuid)
    return row

def lane_main():
    uuid=os.environ['LING_EXECUTION_LANE'];held=lock(D/('lane_'+uuid+'.lock'))
    cfg=verify();assert cfg['selected_gpu_uuid']==uuid
    routing=read(D/'EXECUTION_ROUTING.json');spec=read(D/'FROZEN_SPEC.json')
    units=[u for u in spec['new_units'] if routing['document_devices'][u['document']]==uuid]
    done=[qualification_unit(u,uuid) for u in units]
    atomic_json(D/('LANE_COMPLETE_'+uuid+'.json'),dict(status='PASS',uuid=uuid,units=done,count=len(done)))
    held.close()

def identity_live(identity):
    try:return proc_identity(identity['pid'])==identity
    except (OSError,ValueError):return False

def wait_lanes(children):
    while True:
        pending=False
        for uuid,identity,proc in children:
            complete_path=D/('LANE_COMPLETE_'+uuid+'.json')
            if proc is not None:
                rc=proc.poll()
                if rc is None:pending=True;continue
                assert rc==0 and complete_path.exists(),('LANE_FAILED',uuid,rc)
            else:
                if identity is not None and identity_live(identity):pending=True;continue
                assert complete_path.exists(),('LANE_LOST_NO_COMPLETE',uuid)
            report=read(complete_path);assert report['status']=='PASS' and report['uuid']==uuid
            for u in report['units']:assert read(D/'qualification_units'/(u['document']+'__'+u['rotation']+'.json'))==u
        if not pending:return
        if (D/'STOP.json').exists():raise RuntimeError('FIRST_FAILURE_IN_LANE')
        time.sleep(2)

def resource_plan():
    old=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')
    historical=[]
    for p in old.glob('**/*__formal.NATURAL_EXIT.json'):
        r=read(p)
        if r.get('natural_exit'):historical.append(dict(path=str(p),sha256=sha(p),wall_seconds=r['wall_seconds']))
    parents=[read(p)['elapsed_seconds'] for p in (D/'runs').glob('*__parent.json')]
    samples=[r['wall_seconds'] for r in historical]
    estimate=sum(samples)/len(samples) if samples else 3400.
    memories=[m['peak_reserved'] for p in (D/'runs').glob('*__parent.json') for m in read(p)['memory']]
    reserve=max(memories)+(2033025024-527007744)
    usage=shutil.disk_usage(D);assert usage.free>=64*1024**3
    # Original board capacity/physical allocation are independently rechecked
    # by every worker's original admission. This is a planning comparison.
    assert reserve<24576*2**20,'PROJECTED_PHYSICAL_MEMORY_INSUFFICIENT'
    report=dict(status='PASS_RESOURCE_PLANNING_NOT_RUNTIME_ADMISSION',epoch=time.time(),historical_formal_measurements=historical,qualification_parent_elapsed_seconds=parents,estimated_formal_worker_seconds=estimate,planned_formal_workers=12,estimated_formal_GPU_hours=12*estimate/3600,formal_wall_time_hours=12*estimate/3600+12/60,uncertainty='different physical RTX3090, shared host I/O and per-worker admission; actual measurements govern; estimate is not deadline',peak_measured_qualification_reserved=max(memories),conservative_reserved_plus_cache_growth=reserve,board_bytes=24576*2**20,full_native_cache_bytes=2033025024,checkpoint_projected_bytes=12*550000000,FP_full_vocabulary_payload_floor_bytes=4*384*157184*4,storage_free_bytes=usage.free,storage_minimum_original_bytes=64*1024**3,no_GPUh_cap=True,no_VRAM_percent_gate=True,resume_boundary='qualified own full native8192 checkpoint only; partial formal outputs never silently restarted',max_active_formal_caches=1)
    atomic_json(D/'PRE_FORMAL_RESOURCE_ESTIMATE.json',report)
    return report

def run_main():
    from recovery_controller import run,event,frozen_selection
    if os.environ.get('LING_EXECUTION_LANE'):return lane_main()
    held=lock(D/'controller.lock')
    assert not (D/'STOP.json').exists(),'FIRST_FAILURE_STOP'
    assert read(D/'PHASE0_HISTORICAL_ASSET_AUDIT.json')['status']=='PASS'
    routing=read(D/'EXECUTION_ROUTING.json');selection=frozen_selection();spec=read(D/'FROZEN_SPEC.json')
    assert sha(D/'FROZEN_SPEC.json')==routing['original_checkpoint_contract_sha256']
    children=[]
    if not (D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json').exists():
        for uuid,index in routing['authorized_devices'].items():
            done=D/('LANE_COMPLETE_'+uuid+'.json');lp=D/('LANE_LAUNCH_'+uuid+'.json')
            if done.exists():children.append((uuid,None,None));continue
            if lp.exists():
                identity=read(lp)['identity'];assert identity_live(identity),'LOST_LANE_NO_AUTOMATIC_RETRY'
                children.append((uuid,identity,None));continue
            env=dict(os.environ,LING_EXECUTION_LANE=uuid,LING_TARGET_GPU_UUID=uuid,CUDA_VISIBLE_DEVICES=str(index))
            with (D/('lane_'+uuid+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-B','-u',str(W/'recovery_controller.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=proc_identity(child.pid);atomic_json(lp,dict(identity=identity,uuid=uuid,epoch=time.time()))
            children.append((uuid,identity,child))
        bindings=read(D/'REUSED_PARENT_BINDINGS.json')['units']
        new_parents=len(spec['new_units'])-len(bindings)
        event('TWO_DEVICE_QUALIFICATION_RUNNING',devices=routing['authorized_devices'],new_qualification_workers_planned=len(spec['new_units'])+new_parents,new_parent_workers_planned=new_parents,historical_parents_reused=len(bindings))
        wait_lanes(children)
        units=[read(D/'qualification_units'/(u['document']+'__'+u['rotation']+'.json')) for u in spec['new_units']]
        assert len(units)==12 and all(u['status']=='PASS' and u['qualified_gpu_uuid']==routing['document_devices'][u['document']] for u in units)
        atomic_json(D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json',dict(status='PASS',units=units,count=12,same_GPU_runtime_for_FP_student_pair=True,device_migration_explicitly_qualified=True,no_tolerance_relaxation=True,checkpoint_restore_source_AST_unchanged=True,observer_no_mutation_PASS=True,INT8_dequantized_FP32_carrier_not_physical_INT8_storage=True,full32768_finite_status='PENDING_FORMAL',qualification_completed_epoch=time.time()))
    q=read(D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json');assert q['status']=='PASS' and q['count']==len(q['units'])==12
    assert [(u['document'],u['rotation']) for u in q['units']]==[(u['document'],u['rotation']) for u in spec['new_units']]
    for u in q['units']:
        key=u['document']+'__'+u['rotation']
        assert sha(D/'runs'/(key+'__parent.json'))==u['parent_sha256'] and sha(D/'runs'/(key+'__resumed_parity.json'))==u['resumed_sha256'] and sha(D/'runs'/(key+'__8192.pt'))==u['checkpoint_sha256']
        assert u['qualified_gpu_uuid']==routing['document_devices'][u['document']]
        assert u['checkpoint_sha256']==read(D/'runs'/(key+'__parent.json'))['checkpoint_sha256']
        compare_exact(read(D/'runs'/(key+'__parent.json')),read(D/'runs'/(key+'__resumed_parity.json')),u['qualified_gpu_uuid'])
    atomic_json(D/'LONG_HORIZON_QUALIFICATION.json',q)
    atomic_json(D/'QUALIFICATION_GATES_COMPLETED.json',dict(status='ALL_EIGHT_GATES_PASS',full32768_finite_status='PENDING_FORMAL',engineering_qualification_sha256=sha(D/'ENGINEERING_QUALIFICATION_PRE_FORMAL.json')))
    event('LONG_HORIZON_QUALIFIED',count=12)
    if not (D/'PRE_FORMAL_RESOURCE_ESTIMATE.json').exists():resource_plan()
    for d in selection['execution_order']:
        uuid=routing['document_devices'][d['document']];os.environ['LING_TARGET_GPU_UUID']=uuid;os.environ['CUDA_VISIBLE_DEVICES']=str(routing['authorized_devices'][uuid])
        for condition in d['conditions']:
            event('FULLMODEL_RUNNING',document=d['document'],rotation=condition,uuid=uuid);run(d['document'],condition,'formal')
    atomic_json(D/'PROSPECTIVE_COMPLETE.json',dict(status='COMPLETE',experiment=selection['experiment'],FP_complete=4,student_complete=8,planned_trajectories=12,endpoint=32768,epoch=time.time(),selection_sha256=sha(D/'FROZEN_4DOC_SELECTION.json'),original_formal_verdict='MIXED_PERMANENTLY_PRESERVED',training_steps=0))
    event('RESULTS_READY_FOR_CPU_VERIFICATION');held.close()
