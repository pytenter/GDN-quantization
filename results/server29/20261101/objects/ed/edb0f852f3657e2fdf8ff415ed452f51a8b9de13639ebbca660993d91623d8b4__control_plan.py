"""Pure CPU finite-roster construction. Import never initializes model/CUDA."""
import hashlib
from collections import Counter

TASK='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
METHODS=('W','R','RW')
METHOD_LABELS={'W':'H+G32+Sym8+WScale','R':'H+G32+Sym8+RowBal','RW':'H+G32+Sym8+RowBal+WScale'}
RULE='SHA256(task_id+NUL+problem_id) rank; rank%3 selects ordered physical GPU1/2/3; method order cyclically rotates by rank//3%3'

def require(ok,message):
    if not ok:raise RuntimeError(message)

def records(dataset):
    rows=dataset['records']
    require(len(rows)==30 and len({r['problem_id'] for r in rows})==30,'Require exact30 unique frozen questions')
    require({r['problem_id'] for r in rows}=={'aime26_%02d'%i for i in range(1,31)},'Canonical30 question IDs differ')
    for r in rows:
        require(r['input_tokens']==len(r['input_ids']) and 0<r['input_tokens']<=373,'Prompt token count/range differs')
        require(isinstance(r['input_ids_sha256'],str) and len(r['input_ids_sha256'])==64,'Prompt SHA missing')
    return {r['problem_id']:r for r in rows}

def build_formal_plan(dataset,uuids,physical_indices=(1,2,3)):
    rows=records(dataset);require(tuple(physical_indices)==(1,2,3),'Only authorized55 GPU1/2/3')
    require(len({uuids[g] for g in physical_indices})==3,'Unique physical UUIDs required')
    ranked=sorted(rows,key=lambda pid:(hashlib.sha256((TASK+'\0'+pid).encode()).hexdigest(),pid))
    assignments=[];jobs=[]
    for rank,pid in enumerate(ranked):
        gpu=physical_indices[rank%3];offset=(rank//3)%3;order=METHODS[offset:]+METHODS[:offset]
        assignments.append(dict(problem_id=pid,hash_rank=rank,gpu=gpu,gpu_uuid=uuids[gpu],conditions=list(order),seed=1))
        for condition in order:
            row=rows[pid]
            jobs.append(dict(id=pid+'_'+condition+'_seed1',problem_id=pid,condition=condition,method_label=METHOD_LABELS[condition],seed=1,gpu=gpu,gpu_uuid=uuids[gpu],input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],kind='formal_free_generation',max_new_tokens=81920))
    assignment=dict(task_id=TASK,status='PLANNED_DETERMINISTIC_NO_DISPATCH',rule=RULE,devices=[dict(gpu=g,gpu_uuid=uuids[g]) for g in physical_indices],counts={str(g):10 for g in physical_indices},assignments=assignments)
    validate_formal_jobs(dataset,assignment,jobs,uuids,physical_indices)
    return assignment,dict(task_id=TASK,jobs=jobs,count=90,new_formal_generations=90,new_baseline_generations=0)

def validate_formal_jobs(dataset,assignment,jobs,uuids,physical_indices=(1,2,3)):
    rows=records(dataset);by={x['problem_id']:x for x in assignment['assignments']}
    require(len(jobs)==90 and len(by)==30,'Require90 jobs and30 question bundles')
    require(len({j['id'] for j in jobs})==90 and len({(j['problem_id'],j['condition'],j['seed']) for j in jobs})==90,'Repeated/missing formal jobs')
    require(set(by)==set(rows),'Assignment roster differs')
    require(Counter(x['gpu'] for x in by.values())==Counter({g:10 for g in physical_indices}),'Require10 questions per authorizedUUID')
    for pid,a in by.items():
        require(a['gpu'] in physical_indices and a['gpu_uuid']==uuids[a['gpu']],'Unauthorized assignment UUID')
        selected=[j for j in jobs if j['problem_id']==pid]
        require([j['condition'] for j in selected]==a['conditions'] and set(a['conditions'])==set(METHODS) and len(a['conditions'])==3,'Frozen within-question method order differs')
        for j in selected:
            require(j['gpu']==a['gpu'] and j['gpu_uuid']==a['gpu_uuid'] and j['seed']==1 and j['max_new_tokens']==81920,'Question/UUID/seed/length differs')
            require(j['input_ids_sha256']==rows[pid]['input_ids_sha256'] and j['input_tokens']==rows[pid]['input_tokens'],'Frozen input identity differs')
    # Independently regenerate exact ranking/order: a complete but rewritten
    # mapping cannot gain admission after freezing.
    expected=sorted(rows,key=lambda pid:(hashlib.sha256((TASK+'\0'+pid).encode()).hexdigest(),pid))
    require([x['problem_id'] for x in assignment['assignments']]==expected,'Deterministic question rank differs')
    for rank,pid in enumerate(expected):
        g=physical_indices[rank%3];o=(rank//3)%3
        require(by[pid]['gpu']==g and by[pid]['hash_rank']==rank and by[pid]['conditions']==list(METHODS[o:]+METHODS[:o]),'Frozen device/order rule differs')
    return True

def build_qualification_plan(uuids,physical_indices=(1,2,3)):
    """512 trace covers real32/128 checkpoints; no repeat of old qualification.

    Two fresh512 and64-token sampler leaves per method/device. One field leaf
    per method/device. Context pressure groups the three methods per device.
    Performance is four sequential matched cases (base,W,R,RW) on GPU1.
    """
    units=[]
    for gpu in physical_indices:
        for condition in METHODS:
            prefix='GPU%d_%s'%(gpu,condition)
            units.append(dict(id=prefix+'_FIELDS',gpu=gpu,condition=condition,kind='codec_fields',tokens=0))
            for kind,suffix,extra in [('native_persistent_fresh','PERSIST512',dict(tokens=512,checkpoints=[32,128,512])),('nonAIME_free_generation_fresh','FREE64',dict(max_new_tokens=64,legal_EOS_before_cap=True,problem_id='NON_AIME_CHECKSUM_EXPLANATION_FROZEN_V1'))]:
                for rep in (0,1):
                    units.append(dict(id=prefix+'_'+suffix+'_R'+str(rep),gpu=gpu,condition=condition,kind=kind,repeat=rep,reference_unit=prefix+'_'+suffix+'_R0',**extra))
        units.append(dict(id='GPU%d_CONTEXT82293'%gpu,gpu=gpu,kind='real_model_memory_pressure',tokens=16,conditions=list(METHODS),context=82293))
    # GPU1 may enter this new task only after its previously owned task exits.
    for condition in ('base',)+METHODS:
        units.append(dict(id='GPU1_'+condition+'_MATCHED4096',gpu=1,condition=condition,kind='matched_performance',tokens=4096,warmup_tokens=128,window_tokens=128,early_window=[128,256],late_window=[3968,4096],observer_in_windows=False,same_input=True))
    for u in units:u['gpu_uuid']=uuids[u['gpu']]
    require(len(units)==52 and len({u['id'] for u in units})==52,'Qualification roster construction error')
    return dict(task_id=TASK,status='PLANNED_FINITE_ROSTER_NO_GPU_EVIDENCE',qualification_units=units,qualification_count=52,formal_samples=90,qualification_reuse='Historical base is provenance only; every new method has actual new qualification',scientific_proxy_thresholds=[],formal_method_elimination_on_proxy=False)
