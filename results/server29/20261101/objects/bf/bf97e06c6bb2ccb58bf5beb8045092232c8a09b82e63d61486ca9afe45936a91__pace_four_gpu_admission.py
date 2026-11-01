"""Four independent TP1 lanes. Numeric runtime, packed codec and sampler are unchanged."""
import os
from pathlib import Path

REVISION='QWEN_GDN_PACE48_AIME26_SEED1_60_V1_4GPU_EXIT_FIX_20261026'
REMOTE='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_exit_fix_20261026'
ARCHIVE='/data/zypan/qwen_gdn_pace48_archive/'+REVISION
UUIDS={0:'GPU-a78176ed-f97f-c54f-6201-5a01b570f422',1:'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3',
       2:'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31',3:'GPU-30b07aba-a728-7649-4899-e09d33f8bc75'}
METHOD='PACE48_FP16_EXPONENT_V1'
FORMAT='FP16_MANTISSA_PLUS_OPTIONAL_INT8_EXPONENT'


def partition_jobs(jobs,reused):
    ids=[j['id'] for j in jobs]
    if len(ids)!=60 or len(set(ids))!=60 or not set(reused)<=set(ids) or len(reused)!=len(set(reused)):
        raise ValueError('Exact60 unique canonical jobs and unique admitted reuse required')
    result=[]
    for j in jobs:
        if j['id'] in reused:continue
        gpu=len(result)%4
        result.append(dict(unit=j['id'],gpu=gpu,gpu_uuid=UUIDS[gpu],condition=j['condition']))
    return result


def check_permit(permit,grant,source_sha,grant_sha,plan_sha):
    require=lambda v,m: None if v else (_ for _ in ()).throw(RuntimeError(m))
    require(permit['ALLOW_GPU']=='YES' and permit['execution_root']==REMOTE and permit['execution_revision']==REVISION,'New4GPU permit missing')
    require(permit['physical_indices']==[0,1,2,3] and permit['devices']==list(UUIDS.values()),'Only exact55 fourUUIDs permitted')
    require(grant['status']=='EXPLICIT_HUMAN_PACE_FOUR_GPU_CONTINUATION' and grant['WSCALE_paused'] is True,'Latest human fourGPU grant required')
    require(grant['TP']==1 and grant['unique_formal_target']==60 and grant['server29_CPU_only'] is True,'TP1/fixed60/CPU archive scope differs')
    require(grant['authorized_GPUs']==[0,1,2,3] and grant['UUIDs']==list(UUIDS.values()) and grant['remote_root']==REMOTE and grant['archive_root']==ARCHIVE,'Exact fourGPU human scope differs')
    require(permit['source_manifest_sha256']==source_sha and permit['human_authorization_sha256']==grant_sha and permit['four_gpu_plan_sha256']==plan_sha,'Permit source/grant/plan binding differs')
    require(permit['seed']==1 and permit['max_new_tokens']==81920 and permit['method_variant']==METHOD and permit['state_scale_format']==FORMAT,'Original protocol/numeric variant differs')
    require(type(permit['formal_new_samples']) is int and type(permit['reused_formal_samples']) is int and permit['formal_new_samples']>=0 and permit['reused_formal_samples']>=0 and permit['formal_new_samples']+permit['reused_formal_samples']==60 and permit['no_automatic_GPU_retry'] is True,'Finite unique continuation scope differs')
    require(permit['watchdog_seconds']==1800 and permit['cleanup_seconds']==120 and all(permit.get(k) is None for k in ('GPUh_hard_cap','total_wall_cap','unit_elapsed_cap')),'Resource policy differs')
    return True


def install(C):
    C.REVISION=REVISION;C.REMOTE=REMOTE;C.ARCHIVE=ARCHIVE;C.UUIDS=UUIDS
    g=os.environ.get('PACE_PHYSICAL_GPU','0')
    C.require(g in ('0','1','2','3'),'Explicit physical GPU selector outside fourGPU grant')
    C.PHYSICAL_GPU=int(g);C.UUID=UUIDS[C.PHYSICAL_GPU]

    def authorize(root):
        root=Path(root).resolve();C.require(str(root)==REMOTE,'Exact new fourGPU execution root required')
        permit=C.read(root/'GPU_PERMIT.json');grant=C.read(root/'operations/HUMAN_4GPU_AUTHORIZATION.json')
        metadata=C.read(root/'ROOT.json')
        C.require(metadata['task_id']==permit['task_id']==C.TASK and metadata['execution_revision']==REVISION and metadata['remote_root']==REMOTE and metadata['archive_root']==ARCHIVE,'Exact execution metadata differs')
        check_permit(permit,grant,C.verify_sources(root),C.sha(root/'operations/HUMAN_4GPU_AUTHORIZATION.json'),C.sha(root/'FOUR_GPU_PLAN.json'))
        C.require(permit['method_spec_sha256']==C.sha(root/'method_spec.json'),'Original numeric method spec changed')
        spec=C.read(root/'method_spec.json')
        C.require(spec['common']['method_variant']==METHOD and spec['common']['state_scale_format']==FORMAT and spec['common']['no_rotation'] is True and spec['common']['group_size']==32,'Frozen method semantics differ')
        C.require(spec['configurations']['R2']['decode_period']==8 and spec['configurations']['R3']['decode_period']==1 and spec['generation']['deterministic_algorithms'] is False,'Frozen periods/determinism differ')
        validate_jobs(root)
        plan=C.read(root/'FOUR_GPU_PLAN.json');reuse=C.read(root/'REUSE_ADMISSION.json')
        C.require(permit['formal_new_samples']==len(plan['jobs']) and permit['reused_formal_samples']==len(reuse['reused_ids']),'Permit/actual unique partition counts differ')
        return permit

    def validate_jobs(root):
        root=Path(root);saved=C.read(root/'FORMAL_JOB_MANIFEST.json');data=C.read(root/'AIME26_DATASET_FREEZE.json')
        expected=[]
        for row in data['records']:
            for condition,period in C.CONDITIONS.items():
                expected.append(dict(id=row['problem_id']+'_'+condition+'_seed1',problem_id=row['problem_id'],condition=condition,seed=1,decode_period=period,input_ids_sha256=row['input_ids_sha256'],input_tokens=len(row['input_ids']),max_new_tokens=81920))
        C.require(len(expected)==60 and saved['count']==60 and len(saved['jobs'])==60,'Exact60 fixed roster required')
        plan=C.read(root/'FOUR_GPU_PLAN.json');reuse=C.read(root/'REUSE_ADMISSION.json')
        assignments=partition_jobs(expected,reuse['reused_ids'])
        C.require(plan['jobs']==assignments and plan['reuse_admission_sha256']==C.sha(root/'REUSE_ADMISSION.json'),'FourGPU partition/reuse binding differs')
        by={x['unit']:x for x in assignments}
        for job,e in zip(saved['jobs'],expected):
            C.require({k:job[k] for k in e}==e,'Canonical question/condition/seed/prompt/cap/order differs')
            prior={x['unit']:x for x in reuse['units']}
            gpu=by[job['id']]['gpu'] if job['id'] in by else prior[job['id']]['origin_gpu']
            C.require(job['gpu']==gpu and job['gpu_uuid']==UUIDS[gpu],'Frozen job/device assignment differs')
        return saved['jobs']

    def device(own_pid=None,minimum_free=23000):
        obs=C.query_gpu();gpu=C.PHYSICAL_GPU;uuid=UUIDS[gpu]
        rows=[x for x in obs['devices'] if x['physical_index']==gpu]
        C.require(len(rows)==1 and rows[0]['uuid']==uuid and '3090' in rows[0]['name'] and rows[0]['memory_total_MiB']>=24000,'Assigned physical GPU/UUID differs')
        C.require(not any(x['uuid']==uuid and x['pid']!=own_pid for x in obs['apps']),'Foreign process on assigned GPU; never preempt')
        if own_pid is None:C.require(rows[0]['memory_free_MiB']>=minimum_free,'Assigned GPU free VRAM insufficient')
        else:C.require(any(x['pid']==own_pid and x['uuid']==uuid for x in obs['apps']) and not any(x['pid']==own_pid and x['uuid']!=uuid for x in obs['apps']),'Own worker UUID differs')
        return obs

    original_unit_target=C.unit_target
    def unit_target(root,stage,unit):
        if stage=='scale_repair_gates':
            C.require(unit in {'SCALE_REPAIR_GATES_GPU'+str(g) for g in range(4)},'Exact four-device engineering phase required')
            return Path(root)/'execution'/unit
        return original_unit_target(root,stage,unit)

    def freeze_science(root):
        root=Path(root)
        names=[C.SOURCE,'method_spec.json','GPU_PERMIT.json','FOUR_GPU_PLAN.json','REUSE_ADMISSION.json','FORMAL_JOB_MANIFEST.json','AIME26_DATASET_FREEZE.json','protocol/calibration_windows.json','calibration/head_mask.json','calibration/head_scores.csv','calibration/calibration_manifest.json','protocol/sampler.py','protocol/common.py','protocol/aime26_scorer_v4.py','protocol/aime26_frozen.jsonl','operations/HUMAN_4GPU_AUTHORIZATION.json']
        value=dict(status='SCIENCE_FROZEN_AFTER_FOUR_DEVICE_GATES_AND_CLOSED_PAYLOAD_REUSE',task_id=C.TASK,execution_revision=REVISION,method_variant=METHOD,state_scale_format=FORMAT,mask=C.verify_head_mask(root),reused_formal_count=len(C.read(root/'REUSE_ADMISSION.json')['reused_ids']),current_formal_planned=len(C.read(root/'FOUR_GPU_PLAN.json')['jobs']),files=[dict(path=n,bytes=(root/n).stat().st_size,sha256=C.sha(root/n)) for n in names])
        C.atomic_new(root/'SCIENCE_FREEZE.json',value)
        return value

    def verify_science(root):
        root=Path(root);value=C.read(root/'SCIENCE_FREEZE.json')
        C.require(value['status']=='SCIENCE_FROZEN_AFTER_FOUR_DEVICE_GATES_AND_CLOSED_PAYLOAD_REUSE' and value['task_id']==C.TASK and value['execution_revision']==REVISION,'Exact fourGPU science freeze missing')
        for row in value['files']:
            p=C.relative(root,row['path']);C.require(p.stat().st_size==row['bytes'] and C.sha(p)==row['sha256'],'Frozen science changed: '+row['path'])
        return C.sha(root/'SCIENCE_FREEZE.json')

    C.authorize=authorize;C.validate_jobs=validate_jobs;C.unit_target=unit_target;C.device=device;C.freeze_science=freeze_science;C.verify_science=verify_science
