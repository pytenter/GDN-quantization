"""Independent final CPU audit/publication; no generation, grading or GPU launches."""
import ast
import base64
import collections
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics as S
import sys
import time

TASK=Path(__file__).resolve().parents[1]
BASE=TASK.parent
ROOT=TASK/'releases/FORMAL_V2'
OLD=BASE/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1_4GPU_REBALANCE_20261027'
PRIOR=BASE/'PACE_REBALANCE_FINAL_REVIEW_20261027'
OUT=TASK/'final_review/FINAL_EXTRA_CPU_REVIEW_20261029'
REMOTE='/data/zypan/analyses/pace48_leap_unroll_dp8_dp16_v1/releases/FORMAL_V2'
REMOTE_UNUSED='/data/zypan/analyses/pace48_leap_unroll_dp8_dp16_v1'
REMOTE_OLD='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_rebalance_20261027'
ARCHIVE='/data/zypan/qwen_gdn_pace48_archive/PACE48_LEAP_UNROLL_DP8_DP16_V1'
TARGET55=REMOTE+'_FINAL_EXTRA_CPU_REVIEW_20261029'
TARGET29=ARCHIVE+'.FINAL_EXTRA_CPU_REVIEW_20261029'
SOURCE='c07db17cbdf3f0dd3619ab17b34b97c255950221316a2a13fb97535c0a6a3f5e'
PERMIT='868f25026be1eadc0970adf20cffd7604ebbd4af3b6878aa1b4a16dffd9c50c4'
sys.path.insert(0,str(ROOT/'operations'))
from transport import run
import final_extra_templates as T

CACHE={}
def sha(p):
    p=Path(p);s=p.stat();key=str(p.resolve());finger=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if key in CACHE:
        assert CACHE[key][0]==finger,'Input changed: '+key
        return CACHE[key][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    z=p.stat();assert finger==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)
    CACHE[key]=(finger,h.hexdigest());return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def objsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def write(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def info(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify_rows(root,rows):
    total=0
    for x in rows:
        rel=Path(x['path']);assert not rel.is_absolute() and '..' not in rel.parts
        p=root/rel;assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p);total+=x['bytes']
    return dict(root=str(root),files=len(rows),bytes=total)
def constants(p,names):
    result={}
    for node in ast.parse(p.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            result[node.targets[0].id]=ast.literal_eval(node.value)
    assert set(result)==set(names)
    return result
def remote(code,payload,role='55'):
    code='P='+repr(payload)+'\n'+COMMON+code
    compile(code,'<independent-final-audit>','exec')
    return run(code,role=role,python='/usr/bin/python3' if role=='29' else '/data/ydai/miniconda3/envs/bitdecode/bin/python',timeout=None)

FRAGMENTS=constants(PRIOR/'review_once.py',['COMMON','AUDIT55','AUDIT29'])
COMMON=FRAGMENTS['COMMON']
SCORE_FRAGMENT=constants(PRIOR/'full_score_checks.py',['REMOTE'])['REMOTE']
LEGACY55=FRAGMENTS['AUDIT55'].replace('states=[]\n',SCORE_FRAGMENT+'\nstates=[]\n')

def inputs_and_plan():
    assert read(ROOT/'independent_delivery/RESULT.json')['status']=='PASS_INDEPENDENT55_29_MAC_TERMINAL_DELIVERY'
    primary=read(ROOT/'independent_delivery/LOCAL_TEXTUAL_ARTIFACT_REVIEW.json')['files']
    primary_inputs=[]
    for x in primary:
        p=Path(x.get('actual_local_path',str(ROOT/x['path'])))
        assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
        primary_inputs.append(info(p))
    verify_rows(ROOT,read(ROOT/'FORMAL_SOURCE_MANIFEST.json')['files'])
    assert sha(ROOT/'FORMAL_SOURCE_MANIFEST.json')==SOURCE and sha(ROOT/'GPU_PERMIT.json')==PERMIT
    lineage=read(OLD/'provenance/CLOSED_LINEAGE_PINS.json')
    old_delivery=read(OLD/'independent_delivery/RESULT.json')
    old_roots=lineage['roots']+[dict(remote=REMOTE_OLD,archive=read(OLD/'operations/CURRENT_4GPU_HANDOFF.json')['archive'],
            source=old_delivery['source_manifest_sha256'],permit=old_delivery['permit_sha256'],manifest=old_delivery['core_manifest_sha256'])]
    macproof=read(PRIOR/'MAC_AND_ORIGIN_VERIFICATION.json')
    mac=[];macfiles=[]
    for spec in macproof['roots']:
        root=Path(spec['root']);d=root/'independent_delivery'
        path=next((p for p in [d/'MAC_VERIFICATION.json',d/'MAC_SMALL_FILES_VERIFICATION.json'] if p.exists()),None)
        assert path is not None and sha(path)==spec['verification_sha256']
        z=read(path);late=z.get('late',z.get('late_receipts',[]))
        assert len(z['files'])==spec['small_count'] and len(late)==spec['late_count']
        a=verify_rows(d/'primary_archive_small_files',z['files'])
        b=verify_rows(d/'late_exit_receipts',late)
        mac.append(dict(root=str(root),small=a,late=b,verification_sha256=sha(path)))
        macfiles+=[info(path)]
        macfiles += [info(d/'primary_archive_small_files'/x['path']) for x in z['files']]
        macfiles += [info(d/'late_exit_receipts'/x['path']) for x in late]
    recovery=read(PRIOR/'RECOVERY46_VERIFICATION.json')
    recovery_rows=[]
    recovery_root=BASE/'PACE_EXIT_FIX_FINAL_REVIEW_20261026/CPU_SUPPLEMENT_PACKAGING_RECOVERY'
    for part in recovery['parts']:
        inv=recovery_root/('PART'+str(part['part'])+'_LOCAL_INVENTORY.json')
        assert sha(inv)==part['inventory_sha256']
        rows=read(inv)['files'];assert len(rows)==part['files']
        verify_rows(Path(part['local_root']),rows)
        recovery_rows += [info(Path(part['local_root'])/x['path']) for x in rows]
    assert len(recovery_rows)==recovery['total_files']==46
    for x in recovery['metadata']:
        p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
        recovery_rows.append(info(p))
    supplements={x['root']:x for x in lineage['supplements']+macproof['supplements']+recovery['supplements']}
    for p in PRIOR.glob('PART*_VERIFICATION.json'):
        z=read(p);supplements[z['archive_root']]=dict(root=z['archive_root'],manifest=z['manifest_sha256'])
    old_p=dict(roots=old_roots,current=REMOTE_OLD,source=old_roots[-1]['source'],permit=old_roots[-1]['permit'],
               supplements=list(supplements.values()),live_revisions=lineage['live_revisions']+[OLD.name])
    # Every declared local engineering file is pinned; never traverse new outputs/releases.
    engineering=[p for p in TASK.rglob('*') if p.is_file() and not any(s in p.relative_to(TASK).parts for s in ['releases','final_review','__pycache__','.git']) and p.suffix in {'.py','.json','.md','.log','.jsonl','.txt','.sh'}]
    pinned=primary_inputs+macfiles+recovery_rows+[info(p) for p in engineering]
    pinned+=[info(PRIOR/'review_once.py'),info(PRIOR/'full_score_checks.py'),info(PRIOR/'MAC_AND_ORIGIN_VERIFICATION.json'),info(PRIOR/'RECOVERY46_VERIFICATION.json'),info(PRIOR/'COMPLETION.json')]
    # Include actual Mac additions not in the original remote main seal.
    extra_local=[p for p in (ROOT/'operations').rglob('*') if p.is_file() and p.suffix in {'.json','.py','.md','.txt','.log'}]
    pinned += [info(p) for p in extra_local]
    copies=[]
    late=['runtime/BASH_WAIT.json','runtime/PARENT_DISPATCH.json','runtime/PARENT_COMPLETE.json','runtime/FINAL_ARCHIVE29.json','DELIVERY_READY.json','MANIFEST.sha256',
          'runtime/CPU_SCORE_FRESH1/OWNING_EXIT.json','runtime/CPU_SCORE_FRESH2/OWNING_EXIT.json',
          'runtime/CPU_REPORT_FINISHER/OWNING_EXIT.json','runtime/FORMAL_CONTROLLER/OWNING_EXIT.json']
    for n in late:copies.append(('late/'+n,ROOT/n))
    for p in (ROOT/'independent_delivery').glob('*.json'):copies.append(('primary_review/'+p.name,p))
    for p in extra_local:
        rel=p.relative_to(ROOT/'operations')
        if ('HUMAN_' in p.name or 'HUMAN_' in str(rel) or p.name in ['FORMAL_START_COMPLETION_REVIEW.json','INITIAL_FOUR_FORMAL_FRESH_CPU_REVIEW.json','INITIAL_FORMAL_SOURCE_BOOTSTRAP29_REVIEW.json','PRE_FORMAL_CPU_SCOPE_NOTES.md']):
            copies.append(('human_and_start/'+str(rel),p))
    for p in (ROOT/'runtime').glob('HUMAN_PARTIAL*/*'):
        if p.is_file():copies.append(('human_owning/'+str(p.relative_to(ROOT/'runtime')),p))
    copies += [('unused_V1/REJECTED_PRELAUNCH_RELEASE_V1.json',TASK/'operations/REJECTED_PRELAUNCH_RELEASE_V1.json'),
               ('unused_V1/CONTROL_RELEASE_V2.json',ROOT/'analysis/CONTROL_RELEASE_V2.json'),
               ('source/FORMAL_SOURCE_MANIFEST.json',ROOT/'FORMAL_SOURCE_MANIFEST.json'),
               ('source/GPU_PERMIT.json',ROOT/'GPU_PERMIT.json'),
               ('source/PUBLICATION_PREFLIGHT.json',TASK/'operations/FINAL_PUBLICATION_PREFLIGHT_20261029.json')]
    # Pin original start reviews and the full prior extra provenance without rerunning their claim.
    for name in ['LOCAL_EXTRA_EVIDENCE_VERIFICATION.json','MAC_AND_ORIGIN_VERIFICATION.json','RECOVERY46_VERIFICATION.json','COMPLETION.json']:
        copies.append(('prior_extra/'+name,PRIOR/name))
    copies=list({n:(n,p) for n,p in copies}.values())
    pinned += [info(p) for _,p in copies]
    pinned=list({x['path']:x for x in pinned}.values())
    outputs=['PLAN.json','CLAIM.json','LOCAL_REVIEW.json','MAC_ALL_ORIGINS_REVIEW.json',
             'CURRENT55_REVIEW.json','LEGACY55_REVIEW.json','CURRENT29_REVIEW.json','LEGACY29_REVIEW.json',
             'FINAL_MEMORY_RUNTIME_DATA.json','FINAL_AUDIT_REPORT.md','final_extra_review_once.py','final_extra_templates.py']
    outputs += [n for n,_ in copies]
    parts=[outputs[i:i+22] for i in range(0,len(outputs),22)]
    jobs=read(ROOT/'FORMAL_JOBS.json')['jobs']
    current_p=dict(current=REMOTE,archive=ARCHIVE,source=SOURCE,permit=PERMIT,
                   primary=read(ROOT/'independent_delivery/REMOTE55_REVIEW.json')['files'],
                   unused_root=REMOTE_UNUSED,unused_source='43c35ec14393758001185a2b7fed81f0da434bb405cb7205f2029d469afaa5b5',
                   unused_permit='167f09a8301605077f06c51c2fc4021eeeee5da24f38ad91a6f7fd401746f6c6',
                   unit_archives=[read(ROOT/'formal'/j['id']/'ARCHIVE29.json') for j in jobs],
                   main_archive=read(ROOT/'DELIVERY_READY.json')['archive'])
    plan=dict(status='FIXED_PLAN_BEFORE_SEPARATE_EXTRA_CLAIM',scope='All requested final independent CPU scopes only',
              source=SOURCE,permit=PERMIT,pinned_inputs=pinned,outputs=outputs,parts=parts,
              max_payload_files_per_part=22,max_files_including_metadata=24,max_bytes_per_part=32*1024*1024,
              target55=TARGET55,target29=TARGET29,legacy_payload=old_p,current_payload=current_p,
              originals_and_primary_seals_immutable=True,new_GPU_or_scoring_runs=0)
    return plan,copies,dict(status='PASS_ALL_EIGHT_MAC_ORIGIN_INVENTORIES_AND_ORIGINAL46_CPU_FILES',
                           roots=mac,original_recovery_safe_files=recovery_rows)

def local_review():
    jobs=read(ROOT/'FORMAL_JOBS.json')['jobs'];assert len(jobs)==60 and len({j['id'] for j in jobs})==60
    assert read(ROOT/'FORMAL_JOBS.json')['reuse']==0
    configs={p:read(ROOT/f'configs/DP{p}.json') for p in [8,16]}
    a=dict(configs[8]);b=dict(configs[16]);assert a.pop('decode_period')==8 and b.pop('decode_period')==16 and a==b
    score=read(ROOT/'runtime/SCORE_PASS1.json');assert score==read(ROOT/'runtime/SCORE_PASS2.json')==read(ROOT/'SCORED_RESULTS.json')
    scored={x['unit']:x for x in score['rows']};assert len(scored)==60
    events=[json.loads(s) for s in (ROOT/'runtime/CONTROLLER_EVENTS.jsonl').read_text().splitlines()]
    disp={x['unit']:x for x in events if x['phase']=='UNIT_DISPATCHED'}
    close={x['unit']:x for x in events if x['phase']=='UNIT_CLOSED_AND_ARCHIVED'}
    assert len(disp)==len(close)==60
    rows=[]
    for job in jobs:
        t=ROOT/'formal'/job['id'];wrap=read(t/'RESULT.json');payload=dict(wrap);digest=payload.pop('payload_sha256')
        assert objsha(payload)==digest and wrap['job']==job
        raw=wrap['result'];closed=read(t/'OWNING_CLOSED.json');dispatch=read(t/'DISPATCH.json')
        assert raw['seed']==1 and raw['status']=='COMPLETE' and not raw['nonfinite'] and raw['runtime_error'] is None
        assert wrap['source_manifest_sha256']==SOURCE and wrap['permit_sha256']==PERMIT
        assert closed['owning_Popen_exit']==0 and type(closed['owning_Popen_exit']) is int and closed['release_observed']
        assert closed['RESULT_sha256']==sha(t/'RESULT.json')==scored[job['id']]['RESULT_sha256']
        assert scored[job['id']]['raw_payload_sha256']==digest
        for key in ['pid','ppid','starttick','cmdline']:
            assert wrap['identity'][key]==closed['child'][key]==dispatch['child'][key]
        assert wrap['identity']['ppid']==dispatch['owner']['pid'] and dispatch['CVD']==job['gpu_uuid']
        assert wrap['runtime']['CVD']==wrap['runtime']['driver_uuid']==job['gpu_uuid']
        ids=raw['generated_token_ids'];n=len(ids);period=job['period']
        assert n==raw['generated_token_count']==scored[job['id']]['generated_token_count'] and objsha(ids)==raw['generated_token_ids_sha256']
        assert scored[job['id']]['finish_reason']==raw['finish_reason'] and scored[job['id']]['generation_wall_s']==raw['wall_seconds']
        assert raw['actual_model_forward_calls']==raw['input_tokens']+n-1
        audit=raw['quantizer_audit'];layers=audit['storage_by_layer'];assert len(layers)==24 and not audit['dense_recurrent_carrier']
        categories=collections.Counter();base=live=capacity=peak=vector=headmap=0
        for layer in layers.values():
            assert layer['decode_tokens']==layer['decode_direct_calls']==n-1
            assert layer['decode_dense_materializations']==(n-1)//period
            assert layer['pending_tokens']==(n-1)%period and layer['finalization_policy']=='preserve'
            assert layer['validation_dense_materializations']==0 and not layer['dense_state_persistent']
            assert layer['production_backend']=='TRITON_PACKED_DIRECT' and layer['key_preparation']=='BATCHED_ORIGINAL_CODEC_AND_NATIVE_BF16_NORM'
            assert layer['decode_buffer_capacity_nbytes']==17696*period
            for item in layer['inventory']:
                name=item['name'];kind=('base_codes' if name.startswith('base.') and name.endswith('.codes') else
                    'base_scale' if name.startswith('base.') and name.endswith('.scales') else
                    'base_zero' if name.startswith('base.') and name.endswith('.zero_points') else
                    'base_exponent' if name.startswith('base.') and name.endswith('.scale_exponents') else
                    'pending_key' if name.startswith('buffer.') and '.key.' in name else
                    'pending_u' if name.startswith('buffer.') and name.endswith('.u') else
                    'pending_log_decay' if name.startswith('buffer.') and name.endswith('.log_decay') else
                    'runtime_head_map' if name.startswith('runtime.head_map.') else 'other')
                categories[kind]+=item['nbytes']
                size={'torch.uint8':1,'torch.float16':2,'torch.float32':4,'torch.int8':1,'torch.int32':4}[item['dtype']]
                assert math.prod(item['shape'])*size==item['nbytes']
            base+=layer['base_nbytes'];live+=layer['buffer_nbytes'];capacity+=layer['decode_buffer_capacity_nbytes']
            peak+=layer['peak_decode_buffer_nbytes'];vector+=layer['vector_scratch_peak_payload_bytes'];headmap+=layer['runtime_head_map_bytes']
        assert base==sum(categories[k] for k in ['base_codes','base_scale','base_zero','base_exponent'])
        assert live==sum(categories[k] for k in ['pending_key','pending_u','pending_log_decay'])
        assert capacity==period*17696*24
        values=24*32*128*128;marker=audit['native_marker_tensor_nbytes']
        row=dict(unit=job['id'],problem_id=job['problem_id'],period=period,gpu=job['gpu'],correct=scored[job['id']]['correct'],
                 generated_tokens=n,decode_forwards=n-1,input_tokens=raw['input_tokens'],finish=raw['finish_reason'],
                 generation_s=raw['wall_seconds'],decode_s=raw['decode_and_sampling_seconds'],prefill_s=raw['prompt_seconds'],
                 model_load_s=wrap['model_load_wall_s'],dispatch_s=closed['dispatch_wall_s'],
                 categories=dict(categories),base_bytes=base,buffer_live_bytes=live,buffer_capacity_bytes=capacity,
                 per_layer_peak_buffer_sum_bytes=peak,vector_scratch_per_layer_peak_sum_bytes=vector,
                 marker_bytes=marker,head_map_bytes=headmap,terminal_exponent_bytes=categories['base_exponent'],
                 codes_only_bpv=categories['base_codes']*8/values,base_bpv=base*8/values,
                 base_plus_live_bpv=(base+live)*8/values,base_plus_capacity_bpv=(base+capacity)*8/values,
                 all_terminal_resident_bpv=(base+live+headmap+marker)*8/values,
                 allocated_peak=raw['peak_allocated_bytes'],reserved_peak=raw['peak_reserved_bytes'])
        assert math.isclose(row['generation_s'],row['decode_s']+row['prefill_s'],rel_tol=1e-9,abs_tol=1e-6)
        rows.append(row)
    paired=collections.Counter()
    for problem in sorted({j['problem_id'] for j in jobs}):
        a=next(x for x in rows if x['period']==8 and x['problem_id']==problem)
        b=next(x for x in rows if x['period']==16 and x['problem_id']==problem)
        paired['both_correct' if a['correct'] and b['correct'] else 'dp8_only' if a['correct'] else 'dp16_only' if b['correct'] else 'both_wrong']+=1
    assert dict(paired)==score['summary']['paired_counts']
    discord=paired['dp8_only']+paired['dp16_only'];pvalue=min(1.,2*sum(math.comb(discord,k) for k in range(min(paired['dp8_only'],paired['dp16_only'])+1))/2**discord)
    assert pvalue==score['summary']['exact_McNemar_pvalue']
    arms={}
    for period in [8,16]:
        z=[x for x in rows if x['period']==period];assert len(z)==30
        arms[str(period)]=dict(N=30,correct=sum(x['correct'] for x in z),accuracy=sum(x['correct'] for x in z)/30,
            EOS=sum(x['finish']=='eos' for x in z),cap=sum(x['finish']=='length' for x in z),
            pooled_generation_TPS=sum(x['generated_tokens'] for x in z)/sum(x['generation_s'] for x in z),
            pooled_decode_TPS=sum(x['decode_forwards'] for x in z)/sum(x['decode_s'] for x in z),
            sums_seconds={k:sum(x[k] for x in z) for k in ['generation_s','decode_s','prefill_s','model_load_s','dispatch_s']},
            max_allocated=max(x['allocated_peak'] for x in z),max_reserved=max(x['reserved_peak'] for x in z))
        saved=score['summary']['arms'][str(period)]
        assert saved['correct']==arms[str(period)]['correct'] and saved['accuracy']==arms[str(period)]['accuracy']
        assert math.isclose(saved['pooled_generation_TPS'],arms[str(period)]['pooled_generation_TPS'],rel_tol=1e-12)
    partial=[]
    for folder in sorted((ROOT/'operations').glob('HUMAN_PARTIAL_SCORE_*')):
        first=read(folder/'SCORE_PASS1.json');second=read(folder/'SCORE_PASS2.json')
        assert first==second and first['source_sha256']==SOURCE and first['permit_sha256']==PERMIT
        assert first['actual_GPU_calls']==first['new_model_generations']==0
        for index in [1,2]:
            x=read(ROOT/f"runtime/{folder.name}_PASS{index}/OWNING_EXIT.json")
            assert type(x['owning_Popen_exit']) is int and x['owning_Popen_exit']==0
        partial.append(dict(folder=folder.name,completed=first['completed_units'],arms=first['arms'],new_score_passes=0))
    evidence=dict(status='PASS_INDEPENDENT_ALL60_PAYLOAD_IDENTITY_MEMORY_AND_ORIGINAL_SCORE_AGGREGATES',
                  configurations_only_period_diff=True,new_GPU_or_score_passes=0,partial_snapshots_verified=partial,
                  both_original_fresh_scores_equal=True,paired_counts=dict(paired),McNemar_p=pvalue,
                  preclaim_CPU_scope_notes='Initial local AST extraction attempted to literal-evaluate a later runtime .replace assignment; corrected to extract only literal remote fragments, before any claim/remote call/GPU action. Original source/artifacts unchanged.',
                  negative_evidence_retained=['CPU prototype cross-trajectory screen FAIL','CUDA_KERNEL_V1 compile failure',
                  'CPU lifecycle/control prelaunch failures','slow per-record K dP16 throughput negative','full-model numerical non-equivalence'])
    return evidence,dict(rows=rows,arms=arms,paired_counts=dict(paired),McNemar_p=pvalue,
                         fixed=read(ROOT/'runtime_accounting.json')['fixed_workload_median_TPS'])

def report(data,current):
    lines=['# PACE48 LQUnroll dP8/dP16：最终额外独立审核','',
           '最终实验状态：COMPLETE_BOTH_CONFIGS。全部60条全新AIME26 seed1轨迹完成，没有旧轨迹复用。原两fresh FrozenV4 CPU同payload输出相同；本审核只复核已有正确性布尔与绑定、重算统计，不重新提取答案或评分。','',
           '|配置|正确数|正确率|EOS|长度上限|generation TPS|decode forwards/s|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for p,z in data['arms'].items():
        lines.append(f"|dP{p}|{z['correct']}/30|{z['accuracy']:.6%}|{z['EOS']}|{z['cap']}|{z['pooled_generation_TPS']:.6f}|{z['pooled_decode_TPS']:.6f}|")
    lines+=['',f"配对：{data['paired_counts']}；精确双侧McNemar p={data['McNemar_p']}。只涉及seed1 canonical30；不能据此宣称普遍或统计显著优势。旧PACE的21/30与0/30属于不同机制轨迹，不复用，也不当本次两配置控制变量比较。",'',
            '## 实际内存与有效bits','',
            'RUNTIME_VERIFIED：逐个正式RESULT检查24层tensor inventory，逐shape×element_size及持久dense=False；逐条u、packed K4、FP32 log-decay，保存的是更新而非完整FP32 State。周期末读出后一次临时重建State再原codec提交。未满窗口EOS/cap保持preserve，不补forward或提交；最后生成token未作为下一步input，所以实际decode forwards=generated−1。','',
            '单token/层：K codes1024 + FP16 K scales128 + packed zero32 + FP32 u16384 + FP32 log-decay128 =17696bytes；24层dP8容量3397632bytes、dP16容量6795264bytes。Python None slot没有tensor payload。完整State单层[1,32,128,128] FP32=2097152bytes，只在边界临时出现，非持久state。','',
            '|配置|codes bpv|base+metadata bpv范围|base+实际尾buffer bpv范围|base+容量 bpv范围|实际尾buffer bytes范围|尾部exponent bytes范围|',
            '|---|---:|---|---|---|---|---|']
    for p in [8,16]:
        z=[x for x in data['rows'] if x['period']==p]
        span=lambda k:f"{min(x[k] for x in z):.6f}..{max(x[k] for x in z):.6f}"
        lines.append(f"|dP{p}|{z[0]['codes_only_bpv']:.6f}|{span('base_bpv')}|{span('base_plus_live_bpv')}|{span('base_plus_capacity_bpv')}|{span('buffer_live_bytes')}|{span('terminal_exponent_bytes')}|")
    lines+=['','分母为24×32×128×128=12582912个state元素；base含mixed codes/scales/zeros/当时实际INT8 exponent，head-map3072bytes和native marker192bytes另外计入all_terminal_resident_bpv。e0 base为8896512bytes、5.65625bpv；不能用codes-only的5bpv掩盖元数据/缓存。exponent只给实际终态，整个轨迹指数分配峰值未跟踪。',
            'peak_decode_buffer和vector_scratch计数是每层各自峰值之和，不声称各层峰值同时发生；容量不是长期live。vector scratch计数包含临时keys/u stacks/alphas/key/query，FP32 State重建、原codec临时张量和allocator块生命周期未完整测量。CUDA allocated/reserved是整模型allocator峰值，不能充当state payload或实际GPU活跃时间。','',
            '|配置|整模型allocated峰值bytes|reserved峰值bytes|',
            '|---|---:|---:|']
    for p,z in data['arms'].items():lines.append(f"|dP{p}|{z['max_allocated']}|{z['max_reserved']}|")
    conf=current['model_config'];types=conf.get('layer_types',[])
    full=sum(x in ['full_attention','attention'] for x in types)
    kvheads=conf.get('num_key_value_heads');dim=conf.get('head_dim')
    if full and kvheads and dim:
        per=full*2*kvheads*dim*2
        lines+=['',f"INFERRED：实际模型config读取full-attention层{full}、KV heads={kvheads}、head_dim={dim}，若KV缓存为BF16且完整追加，理论K+V为每context token {per}bytes；每条终态context=input_tokens+generated−1。该公式不是实测KV张量峰值，实际dtype/容量/保留策略未单独hook，故KV实际峰值UNMEASURED。"]
    else:lines+=['','INFERRED/UNRESOLVED：模型config元数据见CURRENT55_REVIEW；没有逐KV张量hook，不能提供实际KV峰值。']
    lines+=['','模型权重17文件以真实字节SHA核对；磁盘safetensors bytes不是实际GPU参数payload。参数分配/allocator块生命周期未独立测量，保持UNMEASURED。','',
            '## 吞吐、长上下文与不同计时口径','',
            '短固定工作量同卡GPU1、非AIME16-token prompt、32 warmup/128计时forward、三轮中位：replay8=5.081395、Direct8=6.653195、Direct16=6.794332 TPS；Direct16对Direct8 +2.12%。它是快路径原同输入对照，不能与AIME不同答案长度的平均TPS混为同输入因果比较，也不是四GPU总吞吐。旧慢K版本4.926/5.308/4.421负结果保留。',
            '上表generation TPS=Σ生成tokens/Σwall_seconds，包含prefill；decode TPS=Σ(生成tokens−1)/Σdecode_and_sampling_seconds，含原sampler/checks与preserve/final adapter audit。dispatch包含加载、控制、CPU核验/归档等不同阶段；不能用dispatch−生成−加载直接当纯IO时间。GPU active/SM seconds没有测量。','',
            '|配置|generation总s|decode总s|prefill总s|模型加载总s|dispatch总s|',
            '|---|---:|---:|---:|---:|---:|']
    for p,z in data['arms'].items():
        a=z['sums_seconds'];lines.append('|dP'+p+'|'+'|'.join(f"{a[k]:.6f}" for k in ['generation_s','decode_s','prefill_s','model_load_s','dispatch_s'])+'|')
    lines+=['','同题同GPU等长区间的既有人类授权CPU比较：2048–8192有25对，6.28223/6.43443（dP16 +2.42%）；8192–16384有24对，6.19411/6.42275（+3.69%）。它来自同题不同自由生成token轨迹，仅长度/设备配对，仍不是同token teacher-forcing。更长区间只有10/2对，不能推广。完整原始phase点、摘要作用域与SHA已封入human_and_start证据，不新增测速。',
            '内核micro图计时19.10/27.95µs仅两fused kernels，不含全部K准备和完整模型；普通host提交约173/174µs另外计时。旧边界15.09/23.05ms只是单次观测，不是稳定中位。全模型延迟、SM active、kernel内部base/delta/pending排他时间不能从它们相减得到。','',
            '## 源码、真实生命周期及完整来源','',
            'CODE_VERIFIED：两config仅decode_period=8/16不同，原quantizer89f4801a…、mask09483fb4…、sampler/FrozenV4未改；正式实现为packed Direct Read和批量原codec/native BF16norm K，不走旧replay。完整argv/starttick/PPID、CVD/Driver UUID、真实Popen整数0/release、结果payload SHA、两fresh评分和Bashwait0逐一绑定SOURCE/permit。事件journal验证每张卡前单闭合并29durable后才派下一单；全部旧owning epoch gone，PID复用作为foreign处理，不发信号。',
            '实际LOADED与POST regular mapped动态并集逐SHA/bytes/device/inode，设备/匿名映射不当文件；模型17全SHA。Triton源码和运行时packed_backend/call counters提供实现/执行证据，当前编译缓存仅记录metadata和SHA，不下载二进制；没有逐specialization Driver加载trace，不把cache存在冒充加载。SM occupancy未测，不以NVML利用率替代。',
            '所有8个原科学root（包括REBALANCE4220及7个前驱）的55/29冻结SOURCE/permit/science全SHA、旧LIVE/bootstrap/unit和全部固定supplements、8套原Mac小文件/late清单以及EXIT_FIX原46份CPU恢复文件再次独立核验。当前SOURCE93、未启动V1冻结source/permit/拒绝原因、全部CPU负例与真实退出、原CUDA编译失败、CPU prototype FAIL、旧慢K负结果、300 FP64 oracle、lineage和full-model非bit等价均保留。',
            '三次人类明确部分评分固定30/42/51样本只复核原两CPU pass与真实owner收据，不重新评分，也不拿结果排序或选参；ETA/吞吐/GPU清单扩展全部作为描述性CPU证据。历史/proc未记录原因UNRESOLVED，没有把后来的CPU反例伪称历史生产实际原因。','',
            '## 审核与交付','',
            '主55/29/Mac交付已PASS；本独立审核不重复主claim，不改主MANIFEST、任何SOURCE或封存状态文件。晚到Bash/parent/finisher/scorer/29收据、主manifest与独立审查及以上补充按每part≤22 payload+2metadata、32MiB发布到新55/29 sibling，真实fsync+renameat2 NOREPLACE，另起只读reader逐SHA，Mac逐字节核对。这里只完成既有两配置任务，没有追加GPU实验或评分。',
            '源代码证据：src/lq_unroll_buffer.py 的 step/_delta_state/_batched_keys/storage_report；src/packed_direct_read.py 的 packed_readout/_kernel；src/pace_runtime.py 的 PACEAdapter；scripts/formal_worker.py 的 run；scripts/formal_controller.py 的真实owning和归档事件；scripts/score_cpu.py 的 run/summarize；scripts/build_reports.py 的各统计口径。冻结source snapshot是实验身份依据，没有编造git commit。','']
    return '\n'.join(lines)

def publish(parts):
    receipts=[]
    for index,names in enumerate(parts,1):
        part=f'part{index:02d}';local=OUT/part;local.mkdir()
        records=[]
        for name in names:
            b=(OUT/name).read_bytes();write(local/name,b)
            records.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        put(local/'PACKAGE.json',dict(source=SOURCE,permit=PERMIT,files=records,part=part))
        write(local/'MANIFEST.sha256',(''.join(x['sha256']+'  '+x['path']+'\n' for x in records)+sha(local/'PACKAGE.json')+'  PACKAGE.json\n').encode())
        records += [dict(path=n,bytes=(local/n).stat().st_size,sha256=sha(local/n)) for n in ['PACKAGE.json','MANIFEST.sha256']]
        assert len(records)<=24 and sum(x['bytes'] for x in records)<=32*1024*1024
        payload=[dict(x,base64=base64.b64encode((local/x['path']).read_bytes()).decode()) for x in records]
        item=dict(part=part)
        for role,dest in [('55',TARGET55),('29',TARGET29)]:
            code='DEST='+repr(dest)+'\nPART='+repr(part)+'\nFILES='+repr(payload)+'\n'+T.PUBLISH
            item[role]=remote(code,{},role);put(OUT/(part+'_'+role+'_RECEIPT.json'),item[role])
        receipts.append(item)
    for role in ['55','29']:
        code='RECEIPTS='+repr([x[role] for x in receipts])+'\n'+T.POST_READ
        put(OUT/('POST_PUBLICATION_'+role+'.json'),remote(code,{},role))
    for x in receipts:
        local=OUT/x['part'];assert sha(local/'MANIFEST.sha256')==x['55']['manifest_sha256']==x['29']['manifest_sha256']
        for line in (local/'MANIFEST.sha256').read_text().splitlines():
            h,n=line.split('  ',1);assert sha(local/n)==h
    return receipts

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'operations/DELIVERY_READER.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'CLAIM.json').exists(),'Already claimed; no automatic repeat'
    plan,copies,mac=inputs_and_plan()
    for payload,code in [(plan['current_payload'],T.CURRENT55),(plan['current_payload'],T.CURRENT29),
                         (plan['legacy_payload'],LEGACY55),(plan['legacy_payload'],FRAGMENTS['AUDIT29'])]:
        compile('P='+repr(payload)+'\n'+COMMON+code,'<final-review-exact-remote-syntax>','exec')
    evidence,data=local_review()
    if '--validate-plan-only' in sys.argv:
        print(json.dumps(dict(status='PASS_PRECLAIM_LOCAL_REVIEW_AND_EXACT_REMOTE_SYNTAX',
                              inputs=len(plan['pinned_inputs']),parts=[len(x) for x in plan['parts']],claim_created=False)))
        return
    put(OUT/'PLAN.json',plan)
    put(OUT/'CLAIM.json',dict(pid=os.getpid(),time=time.time(),plan_sha256=sha(OUT/'PLAN.json')))
    try:
        put(OUT/'LOCAL_REVIEW.json',evidence);put(OUT/'MAC_ALL_ORIGINS_REVIEW.json',mac)
        current=remote(T.CURRENT55,plan['current_payload']);put(OUT/'CURRENT55_REVIEW.json',current)
        old=remote(LEGACY55,plan['legacy_payload']);put(OUT/'LEGACY55_REVIEW.json',old)
        archive=remote(T.CURRENT29,plan['current_payload'],'29');put(OUT/'CURRENT29_REVIEW.json',archive)
        oldarchive=remote(FRAGMENTS['AUDIT29'],plan['legacy_payload'],'29');put(OUT/'LEGACY29_REVIEW.json',oldarchive)
        put(OUT/'FINAL_MEMORY_RUNTIME_DATA.json',data)
        write(OUT/'FINAL_AUDIT_REPORT.md',report(data,current).encode())
        write(OUT/'final_extra_review_once.py',Path(__file__).read_bytes())
        write(OUT/'final_extra_templates.py',(TASK/'operations/final_extra_templates.py').read_bytes())
        for name,p in copies:write(OUT/name,p.read_bytes())
        receipts=publish(plan['parts'])
        for x in plan['pinned_inputs']:
            p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
        result=dict(status='PASS_ALL60_LQUNROLL_FINAL_EXTRA_IDENTITY_LINEAGE_MEMORY_RUNTIME_AND_55_29_MAC_NOREPLACE',
                    source=SOURCE,permit=PERMIT,scientific_status='COMPLETE_BOTH_CONFIGS',extra_review_pending=False,
                    arms=data['arms'],parts=receipts,new_GPU_or_score_runs=0,
                    old_scientific_roots55=len(old['roots']),old_scientific_roots29=len(oldarchive['roots']),
                    all_primary_and_extra_seals_preserved=True)
        put(OUT/'RESULT.json',result)
        print(json.dumps(dict(status=result['status'],parts=len(receipts),extra_review_pending=False)),flush=True)
    except BaseException as e:
        put(OUT/'FIRST_FAILURE.json',dict(error=repr(e),no_automatic_repeat=True));raise
if __name__=='__main__':main()
