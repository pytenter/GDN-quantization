"""One independent, immutable CPU interpretation/audit/publication of existing results."""
import base64
import collections
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics as S
import sys
import time

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK / 'releases/ASSET_FIX_V2'
PARENT = TASK.parent / 'PACE48_LEAP_UNROLL_DP8_DP16_V1/releases/FORMAL_V2'
NAME = 'FINAL_CPU_REVIEW_20261029'
OUT = TASK / 'final_review' / NAME
REMOTE = '/data/zypan/analyses/pace48_lqunroll_dp32_throughput_v1/releases/ASSET_FIX_V2'
REMOTE_TASK = '/data/zypan/analyses/pace48_lqunroll_dp32_throughput_v1'
REMOTE_PARENT = '/data/zypan/analyses/pace48_leap_unroll_dp8_dp16_v1/releases/FORMAL_V2'
TARGET = REMOTE + '_' + NAME
SOURCE = 'f4a2224d206f9b33999b2352ceee0b87e3053d308cba3dd62a54a97453d26427'
sys.path.insert(0, str(ROOT / 'operations'))
from transport import run

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text())

def put(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def textfile(p, value):
    with p.open('x') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())

def info(p, remote=None):
    x = dict(local=str(p), bytes=p.stat().st_size, sha256=sha(p))
    if remote is not None:
        x['remote'] = str(remote)
    return x

def source_inputs(local, remote, manifest):
    p = local / manifest
    rows = [info(p, Path(remote) / manifest)]
    for row in read(p)['files']:
        q = local / row['path']
        assert q.stat().st_size == row['bytes'] and sha(q) == row['sha256'], str(q)
        rows.append(info(q, Path(remote) / row['path']))
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '_local/READER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'CLAIM.json').exists(), 'Already claimed; no automatic repeat'
    recovery = ROOT / '_local/CPU_TERMINAL_DELIVERY_RECOVERY_20261029'
    assert read(recovery / 'RESULT.json')['status'] == 'PASS_NEW_INDEPENDENT_CPU_RECOVERY_55_MAC_DELIVERY'
    assert read(ROOT / 'RESULT.json')['status'] == 'COMPLETE_NO_CLEAR_SPEEDUP'
    assert sha(ROOT / 'SOURCE_MANIFEST.json') == SOURCE
    sources = []
    for local, remote, manifest in [
        (ROOT, REMOTE, 'SOURCE_MANIFEST.json'),
        (TASK, REMOTE_TASK, 'SOURCE_MANIFEST.json'),
        (PARENT, REMOTE_PARENT, 'FORMAL_SOURCE_MANIFEST.json'),
    ]:
        sources += source_inputs(local, remote, manifest)
    pre_name = 'e6041397651e6f7842f1962caf462f6d872c97f3595461fb643314ec5dbf3bbb'
    pre = TASK / 'prelaunch_releases' / pre_name
    assert sha(pre / 'SOURCE_MANIFEST.json') == pre_name
    # The archived prelaunch snapshot is independently checked locally. V1/V2/parent
    # remote scientific identities are rehashed below; no never-deployed path invented.
    pre_inputs = source_inputs(pre, str(pre), 'SOURCE_MANIFEST.json')
    for x in pre_inputs:
        x.pop('remote')
    inputs = list(sources) + pre_inputs
    for line in (ROOT / 'MANIFEST.sha256').read_text().splitlines():
        digest, rel = line.split('  ', 1)
        p = ROOT / rel
        assert sha(p) == digest
        inputs.append(info(p, Path(REMOTE) / rel))
    inputs.append(info(ROOT / 'MANIFEST.sha256', Path(REMOTE) / 'MANIFEST.sha256'))
    original_recovery = read(TASK / '_local/CPU_RECOVERY_V1/REMOTE_REVIEW.json')
    for x in original_recovery['review']['files']:
        p = TASK / x['path']
        assert p.stat().st_size == x['bytes'] and sha(p) == x['sha256']
        inputs.append(info(p, Path(REMOTE_TASK) / x['path']))
    proof_names = [
        ('fast_verification.json', ROOT / '_local/HUMAN_FAST_PATH_VERIFICATION.json'),
        ('fast_environment.json', ROOT / '_local/HUMAN_FAST_PATH_ENVIRONMENT_CHECK.json'),
        ('cpu_cgroup.json', ROOT / '_local/HUMAN_FAST_PATH_CGROUP_CHECK.json'),
        ('slowdown_components.json', ROOT / '_local/HUMAN_SLOWDOWN_COMPONENTS.json'),
        ('V1_CPU_RECOVERY_RESULT.json', TASK / '_local/CPU_RECOVERY_V1/RESULT.json'),
        ('V1_CPU_RECOVERY_REVIEW.json', TASK / '_local/CPU_RECOVERY_V1/REMOTE_REVIEW.json'),
        ('V1_CPU_WRAPPER_FAILURE.txt', TASK / '_local/CPU_RECOVERY_V1/WRAPPER_FAILURE1.txt'),
        ('V2_FAILED_DELIVERY_CLAIM.json', ROOT / '_local/independent_delivery/CLAIM.json'),
        ('V2_FAILED_DELIVERY_FIRST_FAILURE.json', ROOT / '_local/independent_delivery/FIRST_FAILURE.json'),
        ('V2_CPU_RECOVERY_PLAN.json', recovery / 'PLAN.json'),
        ('V2_CPU_RECOVERY_RESULT.json', recovery / 'RESULT.json'),
        ('V2_CPU_RECOVERY_REVIEW.json', recovery / 'REMOTE_REVIEW.json'),
        ('V2_CPU_RECOVERY_LOCAL_SHA.json', recovery / 'LOCAL_TEXT_SHA.json'),
        ('UNIQUE_LAUNCH_RECEIPT.json', ROOT / '_local/UNIQUE_LAUNCH_RECEIPT.json'),
        ('EXTRA_REQUIREMENTS.json', ROOT / '_local/FINAL_EXTRA_REVIEW_REQUIRED.json'),
    ]
    inputs += [info(p) for _, p in proof_names]
    inputs += [info(Path(__file__)), info(ROOT / '_local/recover_terminal_delivery_cpu.py')]
    # Fixed output roster and <=22 payload files/part leave room for two metadata files.
    outputs = ['PLAN.json', 'CLAIM.json', 'INPUTS.json', 'LOCAL_REVIEW.json',
               'REMOTE_FRESH_REVIEW.json', 'COMPARISON_DATA.json', 'FINAL_COMPARISON.md',
               'final_extra_review_cpu.py', 'recover_terminal_delivery_cpu.py']
    outputs += ['evidence/' + name for name, _ in proof_names]
    parts = [outputs[i:i+22] for i in range(0, len(outputs), 22)]
    plan = dict(scope='Independent CPU-only final review, no new GPU/profile/accuracy runs',
                source_sha256=SOURCE, scientific_status='COMPLETE_NO_CLEAR_SPEEDUP',
                remote_target=TARGET, input_files=inputs, outputs=outputs, planned_parts=parts,
                max_payload_files_per_part=22, hard_max_files_including_metadata=24,
                hard_max_payload_bytes_per_part=32*1024*1024,
                original_main_seal_sha256=sha(ROOT/'MANIFEST.sha256'),
                original_failed_claim_preserved=True, prior_recovery_not_repeated=True)
    remote_code = r'''
import json,sys,os,stat,time
from pathlib import Path
r=Path(REMOTE)
sys.path.insert(0,str(r/'src'))
import execution_common as C
checked=[]
for x in INPUTS:
 if 'remote' not in x:continue
 p=Path(x['remote']);s=p.stat()
 assert stat.S_ISREG(s.st_mode) and s.st_size==x['bytes'] and C.sha(p)==x['sha256'],str(p)
 checked.append(dict(path=str(p),bytes=s.st_size,sha256=x['sha256']))
maps={}
for n in ['LOADED_REGULAR_MAPPED_SHA.json','POSTFLIGHT_REGULAR_MAPPED_SHA.json']:
 for x in json.loads((r/n).read_text())['files']:maps[(x['device'],x['inode'])]=x
for x in maps.values():
 p=Path(x['path']);s=p.stat()
 assert stat.S_ISREG(s.st_mode) and s.st_size==x['bytes'] and s.st_ino==x['inode'] and s.st_dev==x['device'] and C.sha(p)==x['sha256'],x['path']
model=C.model_pins(r,full=True)
admission=json.loads((r/'provenance/VALIDATION_REUSE_ADMISSION.json').read_text())
for x in admission['external_evidence']:
 p=Path(x['path']);assert p.stat().st_size==x['bytes'] and C.sha(p)==x['sha256']
for x in admission['identical_current_files']:assert C.sha(r/x['path'])==x['sha256']
identities=[]
for base in [r,Path(REMOTE_TASK)]:
 p=base/'runtime/UNIQUE_PARENT_STARTED.json'
 if p.exists():
  q=json.loads(p.read_text());identities.extend([q['owner'],q['bash']])
 for p in (base/'runtime').glob('*/OWNING_EXIT_AND_RELEASE.json'):
  q=json.loads(p.read_text());assert q['release_observed'] is True;identities.append(q['child'])
for ident in identities:
 try:cur=C.identity(ident['pid'])
 except FileNotFoundError:cur=None
 assert cur is None or cur['starttick']!=ident['starttick'],'Old owning process epoch remains'
current=json.loads((r/'runtime/BASH_WAIT.json').read_text())
previous=json.loads((Path(REMOTE_TASK)/'runtime/BASH_WAIT.json').read_text())
assert type(current['actual_Bash_wait_integer']) is int and current['actual_Bash_wait_integer']==0
assert type(previous['actual_Bash_wait_integer']) is int and previous['actual_Bash_wait_integer']==1
print(json.dumps(dict(status='PASS_FRESH55_PRIMARY_V1_PARENT_SOURCES_ALL_RECOVERY_FILES_MAPS_MODEL_AND_REAL_OWNING',
 checked_files=checked,actual_regular_mapped_files=list(maps.values()),model17=model,
 current_Bashwait=current,previous_failed_Bashwait=previous,checked_owned_epochs=len(identities),
 reader=C.identity(os.getpid()),no_GPU_or_foreign_signals=True)))
'''
    remote_code = ('REMOTE='+repr(REMOTE)+'\nREMOTE_TASK='+repr(REMOTE_TASK)+
                   '\nINPUTS='+repr(inputs)+'\n'+remote_code)
    compile(remote_code, '<final-independent-remote-audit>', 'exec')
    if '--validate-plan-only' in sys.argv:
        print(json.dumps(dict(status='PASS_PLAN_AND_EXACT_REMOTE_SYNTAX',inputs=len(inputs),
                              parts=[len(p) for p in parts],claim_created=False)))
        return
    put(OUT/'PLAN.json', plan)
    put(OUT/'CLAIM.json', dict(local_pid=os.getpid(),client_time=time.time(),plan_sha256=sha(OUT/'PLAN.json')))
    try:
        put(OUT/'INPUTS.json', dict(files=inputs,local_prelaunch_archived_not_GPU_started=True))
        local_review, data = audit_local()
        put(OUT/'LOCAL_REVIEW.json',local_review)
        put(OUT/'COMPARISON_DATA.json',data)
        textfile(OUT/'FINAL_COMPARISON.md',render(data))
        shutil.copyfile(__file__,OUT/'final_extra_review_cpu.py')
        shutil.copyfile(ROOT/'_local/recover_terminal_delivery_cpu.py',OUT/'recover_terminal_delivery_cpu.py')
        for name,p in proof_names:
            dest=OUT/'evidence'/name;dest.parent.mkdir(exist_ok=True)
            with dest.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
        review=run(remote_code,timeout=None)
        put(OUT/'REMOTE_FRESH_REVIEW.json',review)
        receipts=publish(parts,plan)
        put(OUT/'PUBLISH_RECEIPTS.json',receipts)
        verification=verify_remote(receipts)
        put(OUT/'POST_PUBLISH_VERIFICATION.json',verification)
        # Rehash every fixed input and published byte on Mac after all transfers.
        for x in inputs:
            p=Path(x['local']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
        assert sha(ROOT/'MANIFEST.sha256')==plan['original_main_seal_sha256']
        for part in receipts:
            local=OUT/part['name']
            for line in (local/'MANIFEST.sha256').read_text().splitlines():
                digest,rel=line.split('  ',1);assert sha(local/rel)==digest
        result=dict(status='PASS_ALL_REQUIRED_DP32_FINAL_EXTRA_CPU_REVIEW_55_MAC_NOREPLACE',
                    scientific_status='COMPLETE_NO_CLEAR_SPEEDUP',source_sha256=SOURCE,
                    primary_manifest_sha256=sha(ROOT/'MANIFEST.sha256'),
                    parts=receipts,independent_remote_post_read=True,
                    all_fixed_inputs_rehashed_after_delivery=True,extra_review_pending=False,
                    original_failed_claim_preserved=True,new_GPU_or_AIME_or_accuracy_runs=0)
        put(OUT/'RESULT.json',result)
        print(json.dumps(dict(status=result['status'],parts=len(receipts),extra_review_pending=False)),flush=True)
    except BaseException as exc:
        put(OUT/'FIRST_FAILURE.json',dict(error=repr(exc),no_automatic_repeat=True))
        raise

def audit_local():
    full=read(ROOT/'BENCHMARK_RESULTS.json');rows=full['measurements']
    assert len(rows)==18 and len({(x['mode'],x['repeat'],x['decode_period']) for x in rows})==18
    assert full['new_AIME_samples']==0 and full['new_accuracy_evaluations']==0 and full['same_kernel_all_periods']
    runtime=full['runtime']
    assert runtime['driver_uuid']==runtime['CVD']==read(ROOT/'CONFIG.json')['gpu_uuid']
    assert runtime['threads']==[24,24] and runtime['deterministic'] is False and runtime['TF32'] is False
    measurements=[]
    for row in rows:
        p=row['decode_period'];mode=row['mode'];layers=row['quantizer_audit']['storage_by_layer']
        raw=ROOT/'measurements'/f"{mode}_r{row['repeat']}_dp{p}.json"
        assert read(raw)==row
        assert row['input_sha256']=='98f2b40afaefe6cab615dfeefd9aab7589fdd4ce42f10e81187f58522da6c401'
        assert row['fixed_replayed_inputs'] and row['original_sampler_executed'] and row['no_AIME'] and not row['free_generation']
        assert len(layers)==24
        n=160 if mode=='short' else 16384
        totals=collections.Counter()
        for name,b in layers.items():
            assert b['decode_tokens']==n and b['decode_direct_calls']==n
            assert b['decode_dense_materializations']==n//p and b['validation_dense_materializations']==0
            assert not b['dense_state_persistent'] and b['no_dense_state_between_decode_calls']
            assert b['production_backend']=='TRITON_PACKED_DIRECT'
            assert b['key_preparation']=='BATCHED_ORIGINAL_CODEC_AND_NATIVE_BF16_NORM'
            assert b['pending_tokens']==0 and b['finalization_policy']=='preserve'
            assert b['decode_buffer_capacity_nbytes']==17696*p and b['peak_decode_buffer_nbytes']==17696*p
            if mode=='short':
                before=row['before_measurement_storage'][name]
                assert b['decode_dense_materializations']-before['decode_dense_materializations']==128//p
            for k in ['base_nbytes','buffer_nbytes','peak_decode_buffer_nbytes','decode_buffer_capacity_nbytes','vector_scratch_peak_payload_bytes','decode_dense_materializations','packed_kernel_launches']:
                totals[k]+=b[k]
        measured_n=row['measured_decode_forwards']
        assert measured_n==(128 if mode=='short' else 16352)
        assert math.isclose(row['decode_TPS'],measured_n/row['decode_wall_s'],rel_tol=1e-12)
        item={k:row[k] for k in ['mode','repeat','position','decode_period','decode_TPS','decode_wall_s','mean_decode_token_ms','prefill_latency_s','peak_allocated_bytes','peak_reserved_bytes','measured_decode_forwards']}
        item.update(totals=dict(totals),writebacks_measured_all24=24*measured_n//p,
                    max_pending_after_append_before_commit=p,max_pending_between_calls=p-1,
                    terminal_exponent_bytes=sum(t['nbytes'] for b in layers.values() for t in b['inventory'] if 'exponent' in t['name']))
        if mode=='long':
            assert [(z['start_decode_forward'],z['end_decode_forward']) for z in row['intervals']]==[(32,16384),(2048,8192),(8192,16384)]
            for z in row['intervals']:assert math.isclose(z['TPS'],z['forwards']/z['wall_s'],rel_tol=1e-12)
            item.update(intervals=row['intervals'],host_iteration_ms=row['host_iteration_ms'])
        measurements.append(item)
    numerical=[]
    def unexplained(obj):
        if isinstance(obj,dict):return int(obj.get('reason')=='UNEXPLAINED')+sum(unexplained(x) for x in obj.values())
        if isinstance(obj,list):return sum(unexplained(x) for x in obj)
        return 0
    for n in [1,2]:
        z=read(ROOT/f'validation/DP32_FRESH{n}.json')
        assert z['status']=='PASS_DP32_ORIGINAL_NUMERICAL_GATES' and len(z['cases'])==24
        assert z['original_unchanged_limits']['cross_replay_max_normalized_RMSE']==.005
        assert z['original_unchanged_limits']['self_actual_delta_state_readout_max_normalized_RMSE']==.001
        assert all(z['worst_NRMSE'][k]<=.005 for k in ['output','delta','boundary_state'])
        assert z['worst_NRMSE']['actual_delta_state_readout']<=.001
        assert z['local_unexplained_code_mismatches']==0 and z['all_cases_with_legacy_UNEXPLAINED_reviewed']
        needed=set()
        for c in z['cases']:
            key=tuple(c[k] for k in ['period','length','prompt','scenario','heads'])
            assert c['no_tail_flush'] and c['no_extra_forward']
            # validate_dp32's length denotes decode length, in addition to prompt.
            assert c['direct_storage']['decode_tokens']==c['length']
            assert c['direct_storage']['tokens']==c['length']+c['prompt']
            assert c['actual_decode_boundary_materializations']==c['length']//32
            assert c['terminal_pending']==c['length']%32
            if unexplained(c):needed.add(key)
        covered={tuple(c[k] for k in ['period','length','prompt','scenario','heads']) for c in z['same_base_lineage']}
        assert needed<=covered
        numerical.append(dict(fresh=n,cases=len(z['cases']),legacy_cases_requiring_same_base=[list(x) for x in sorted(needed)],
                              actual_same_base_cases=[list(x) for x in sorted(covered)],
                              worst_NRMSE=z['worst_NRMSE'],local_unexplained=0))
    admission=read(ROOT/'provenance/VALIDATION_REUSE_ADMISSION.json')
    for q in admission['original_gate_receipts']:
        assert type(q['owning_Popen_exit']) is int and q['owning_Popen_exit']==0 and q['release_observed']
        assert q['gpu_uuid']==runtime['CVD']
    assert admission['new_GPU_gate_runs']==0 and admission['reused_benchmark_trajectories']==0
    identity=read(ROOT/'SOURCE_IDENTITY.json')
    for q in identity['copied_exact_files']:
        assert sha(ROOT/q['path'])==sha(PARENT/q['path'])==q['parent_sha256']==q['new_sha256']
    original=(PARENT/'src/lq_unroll_buffer.py').read_text()
    new=(ROOT/'src/lq_unroll_buffer.py').read_text()
    restored=new.replace('(8, 16, 32)', '(8, 16)').replace('decode_period must be 8, 16 or 32','decode_period must be 8 or 16')
    # Exact authorized diff is retained by SOURCE_IDENTITY; don't normalize other math.
    if restored!=original:
        import difflib
        changes=[x for x in difflib.unified_diff(original.splitlines(),new.splitlines()) if x.startswith(('+','-')) and not x.startswith(('+++','---'))]
        assert len(changes)==4 and all('8' in x and '16' in x for x in changes),changes
    cpu=read(ROOT/'CPU_CONTROL_VALIDATION.json')
    assert cpu['real_exit_codes']==[0,7,143] and cpu['GPU_credit']==0 and cpu['all_fixtures_are_not_measurements']
    assert all(q['expected']==q['actual'] for q in cpu['report_fixtures'])
    # Audit true receipts rather than converting historical failures to passes.
    assert read(TASK/'runtime/BASH_WAIT.json')['actual_Bash_wait_integer']==1
    assert read(ROOT/'runtime/BASH_WAIT.json')['actual_Bash_wait_integer']==0
    for unit in ['THROUGHPUT_SUITE','CPU_FINAL_REPORT']:
        q=read(ROOT/f'runtime/{unit}/OWNING_EXIT_AND_RELEASE.json')
        assert q['owning_Popen_exit']==0 and q['release_observed'] and q['source_sha256']==SOURCE
    assert read(TASK/'_local/CPU_RECOVERY_V1/RESULT.json')['files']==len(read(TASK/'_local/CPU_RECOVERY_V1/REMOTE_REVIEW.json')['review']['files'])
    kernel=read(ROOT/'KERNEL_RAW.json');micro=[]
    for z in kernel['rows']:
        p=z['period'];boundary=z['boundaries']
        assert len(boundary)==10
        for b in boundary:
            inv=b['actual_precommit_inventory']['inventory']
            assert b['actual_precommit_inventory']['pending']==p
            assert sum(t['bytes'] for t in inv if t['name'].startswith('buffer.'))==17696*p
            for t in inv:
                elements=math.prod(t['shape']);size={'torch.uint8':1,'torch.float16':2,'torch.float32':4,'torch.int32':4}[t['dtype']]
                assert elements*size==t['bytes']
        med=lambda fn:S.median([fn(x) for x in boundary])
        micro.append(dict(period=p,readouts=z['readouts'],
                          whole_boundary_us=med(lambda x:x['whole_boundary_host_synchronized_us']),
                          reconstruction_event_us=med(lambda x:x['components']['state_reconstruction']['event_stream_elapsed_us']),
                          reconstruction_host_us=med(lambda x:x['components']['state_reconstruction']['host_submit_wall_us']),
                          encode_event_us=med(lambda x:x['components']['state_encode_total_including_packing']['event_stream_elapsed_us']),
                          encode_host_us=med(lambda x:x['components']['state_encode_total_including_packing']['host_submit_wall_us']),
                          packing_event_us=med(lambda x:x['packing_event_stream_us']),
                          encode_residual_event_us=med(lambda x:x['encode_remainder_event_us']),
                          whole_boundary_amortized_us=med(lambda x:x['whole_boundary_host_synchronized_us'])/p,
                          measured_single_layer_pending_record_bytes=17696*p))
    compiler=read(ROOT/'COMPILER_TELEMETRY.json')
    assert len(compiler['rows'])>0 and all(x['compiler_flags_unchanged'] for x in compiler['rows'])
    compile_summary=[]
    for hi in [7,15,31]:
        rr=[x for x in compiler['rows'] if int(x['constants']['P'])<=hi]
        compile_summary.append(dict(max_pending=hi,observed_specializations=len(rr),
                                   registers_min=min(x['registers_per_thread'] for x in rr),
                                   registers_max=max(x['registers_per_thread'] for x in rr),
                                   spill_values=sorted({x['spill_count_reported_by_loader'] for x in rr}),
                                   shared_bytes_values=sorted({x['shared_bytes'] for x in rr})))
    data=dict(scientific_status='COMPLETE_NO_CLEAR_SPEEDUP',runtime=runtime,measurements=measurements,
              numeric=numerical,micro=micro,compiler=compile_summary,
              compiler_calls=len(compiler['rows']),compiler_call_wall_s=sum(x['compile_or_disk_cache_load_wall_s'] for x in compiler['rows']),
              first_touch=read(ROOT/'FIRST_TOUCH_AND_COMPILE.json'),
              primary_result=read(ROOT/'RESULT.json'),
              unmeasured=['actual SM occupancy','SM active time','exclusive fused base/delta/pending sub-times','full-model-exclusive writeback time','complete transient tensor peak/lifetime'],
              slowdown_cause='UNRESOLVED; same fast source and no observed launch-blocking/cgroup throttling; no isolated causal profiling performed')
    review=dict(status='PASS_ALL_REQUIRED_LOCAL_NUMERIC_LIFECYCLE_PERFORMANCE_STORAGE_AND_PROVENANCE',
                measurements=18,numeric=numerical,source_identity_equal_files=len(identity['copied_exact_files']),
                original_gate_true_exit0_release_reused=True,thresholds_unchanged=True,
                actual_precommit_tensor_shape_bytes_checked=True,
                original_failure_claims_and_CPU_wrappers_retained=True,new_GPU_or_AIME_runs=0,
                local_preclaim_scope_notes='The read-only reviewer preflight initially treated case.length as total length; original actual counters show it is decode length plus separate prompt. Corrected only this reviewer assertion before any claim or remote operation. Production source/outputs/counters untouched.')
    return review,data

def render(d):
    lines=['# dP32 吞吐专项：独立最终审核与完整比较','',
           '**最终状态：COMPLETE_NO_CLEAR_SPEEDUP。** 全部18组真实测量完成。仅非AIME固定输入；未运行dP32准确率评估。',
           '本补充只读取已有工件；主SOURCE、主MANIFEST、数学和旧任务均未修改。单位TPS=实际decode forward/s，非生产serving吞吐。','',
           '## 同卡完整比较','',
           '短测每轮32 warmup+128计时forward；长测实际到16384，前32个作warmup，计时16352个。三轮顺序8→16→32、16→32→8、32→8→16。','']
    rr=d['measurements'];mm={x['period']:x for x in d['micro']}
    def med(mode,p,k):return S.median(x[k] for x in rr if x['mode']==mode and x['decode_period']==p)
    metrics=[
        ('短测固定工作量TPS',[med('short',p,'decode_TPS') for p in [8,16,32]]),
        ('相对本轮dP8短测%',[100*(med('short',p,'decode_TPS')/med('short',8,'decode_TPS')-1) for p in [8,16,32]]),
        ('相对本轮dP16短测%',[100*(med('short',p,'decode_TPS')/med('short',16,'decode_TPS')-1) for p in [8,16,32]]),
        ('长测32–16384 TPS',[med('long',p,'decode_TPS') for p in [8,16,32]]),
        ('短测平均TPOT中位ms',[med('short',p,'mean_decode_token_ms') for p in [8,16,32]]),
        ('长测平均TPOT中位ms',[med('long',p,'mean_decode_token_ms') for p in [8,16,32]]),
        ('边界前最后一个非边界读出fused GPU图计时µs',[next(x['median_fused_us'] for x in mm[p]['readouts'] if x['pending_records']==p-2) for p in [8,16,32]]),
        ('单层边界整体同步host中位µs',[mm[p]['whole_boundary_us'] for p in [8,16,32]]),
        ('短测128步全部24层写回次数',[24*128//p for p in [8,16,32]]),
        ('长测计时16352步全部24层写回次数',[24*16352//p for p in [8,16,32]]),
        ('全部24层buffer容量bytes',[24*p*17696 for p in [8,16,32]]),
        ('实测瞬时buffer payload峰值bytes',[max(x['totals']['peak_decode_buffer_nbytes'] for x in rr if x['decode_period']==p) for p in [8,16,32]]),
        ('全部24层packed base bytes',[8896512]*3),
        ('末尾optional exponent bytes',[0]*3),
        ('短测CUDA allocated peak bytes',[max(x['peak_allocated_bytes'] for x in rr if x['mode']=='short' and x['decode_period']==p) for p in [8,16,32]]),
        ('长测CUDA allocated peak bytes',[max(x['peak_allocated_bytes'] for x in rr if x['mode']=='long' and x['decode_period']==p) for p in [8,16,32]]),
        ('长测CUDA reserved peak bytes',[max(x['peak_reserved_bytes'] for x in rr if x['mode']=='long' and x['decode_period']==p) for p in [8,16,32]]),
    ]
    lines+=['|指标|dP8|dP16|dP32|','|---|---:|---:|---:|']
    for label,values in metrics:
        lines.append('|'+label+'|'+'|'.join(str(v) if isinstance(v,int) else f'{v:.6f}' for v in values)+'|')
    lines+=['','RUNTIME_VERIFIED：实际measurement和quantizer_audit；写回计时段由实测计数与已对齐的warmup边界相减。长测计数总值49152/24576/12288含warmup，计时段为49056/24528/12264。单层诊断不等于全模型排他耗时。',
            '短测dP32相对16 +6.43%、相对8 +9.71%；长测相对16 +3.89%、相对8 +7.32%。长测16的5.47–6.16与32的5.62–5.83重叠；短测8的4.40–5.29与32的4.97–5.19也重叠。每配置只有3轮，不能称统计显著或普遍稳定加速。','',
            '## 全部原始测量及同上下文区间','',
            '|模式|轮次|配置|TPS|Decode wall s|平均TPOT ms|Host P50 ms|Host P95 ms|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in sorted(rr,key=lambda x:(x['mode'],x['repeat'],x['position'])):
        h=x.get('host_iteration_ms',{})
        lines.append(f"|{x['mode']}|{x['repeat']}|{x['decode_period']}|{x['decode_TPS']:.9f}|{x['decode_wall_s']:.9f}|{x['mean_decode_token_ms']:.6f}|{h.get('P50','UNMEASURED')}|{h.get('P95','UNMEASURED')}|")
    lines+=['','Host P50/P95为实际长测host迭代延迟，包含原sampler、Python、原检查以及原检查自身可能的同步；没有新增逐token CUDA同步，不是CUDA kernel P50/P95。短测原fixed_run没有逐token采样，保留UNMEASURED。','',
            '|区间|轮次|配置|相同token TPS|Wall s|','|---|---:|---:|---:|---:|']
    for x in sorted((x for x in rr if x['mode']=='long'),key=lambda x:(x['repeat'],x['position'])):
        for z in x['intervals'][1:]:
            lines.append(f"|{z['start_decode_forward']}–{z['end_decode_forward']}|{x['repeat']}|{x['decode_period']}|{z['TPS']:.9f}|{z['wall_s']:.9f}|")
    lines+=['','## 单层边界与fused诊断','',
            '|配置|pending|边界token?|两fused GPU图中位µs|普通host提交中位µs|',
            '|---|---:|---|---:|---:|']
    for m in d['micro']:
        for x in m['readouts']:
            lines.append(f"|{m['period']}|{x['pending_records']}|{x['query_token_is_boundary']}|{x['median_fused_us']:.6f}|{x['median_host_dispatch_us']:.6f}|")
    lines+=['','每次Direct Read明确发射2个mixed-head fused kernels；完整K4 decode/norm和其他PyTorch操作的总launch数NOT_PROFILED。访问pending为0..P−1条，读出后增加本步u/log-decay；P−1 pending是边界token的读出阶段，P−2是其前一个非边界token，不能将边界前读出当完整提交。','',
            '|配置|边界整体host µs|重建event µs|encode含packing event µs|packing event µs|encode减packing µs|边界/P µs|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for m in d['micro']:
        lines.append('|'+str(m['period'])+'|'+'|'.join(f"{m[k]:.6f}" for k in ['whole_boundary_us','reconstruction_event_us','encode_event_us','packing_event_us','encode_residual_event_us','whole_boundary_amortized_us'])+'|')
    lines+=['','event为同一stream的elapsed，可能包含host提交空隙；不是SM active时间。encode减packing是INFERRED residual，含scale/zp/exponent/RNE/layout/checks/dispatch，不是独立编码kernel测量。边界/P是单层诊断摊销估计，不是全模型实测节省；各列各自取中位，不能把中位数当严格可相加分解。',
            'dP32写回次数对16减半、对8减75%，但单次重建随pending增多变慢；codec每次encode并没有按周期等比例减少。不能把次数下降说成总写回耗时已实测减半。dP32满窗口读出47.16µs高于16的28.24µs；长缓存确有访问/寄存器成本，收益与成本相抵。','',
            '## State、buffer与scratch内存','',
            'RUNTIME_VERIFIED：单层单token的packed K4 codes [1024] uint8=1024；FP16 scales [1,16,4]×2=128；packed zero [32] uint8=32；u [1,32,128] FP32=16384；log-decay [1,32] FP32=128，总计17696bytes。每层容量P×17696，24层容量24×P×17696。Python _slots里的None不分配tensor payload，已提交的记录清为None。',
            '最大跨调用pending=P−1；本步append后提交前瞬时pending=P，独立boundary tensor inventory确实逐shape×element_size验证。全模型报告的peak是各层tensor payload峰值之和，不声称各层峰值在同一时刻同时驻留；容量不是始终live。末尾全部周期对齐pending=0，optional exponent末尾0；不能从末尾0推断整个长轨迹从未分配指数，轨迹指数峰值UNMEASURED。',
            'packed base每层370688bytes，24层8896512bytes；codes-only为5bit/value，含scale/zp的base为5.65625bit/value（当前e0）。runtime head-map每层128bytes另记。完整State形状[1,32,128,128] FP32=2097152bytes，只在边界重建时临时出现，没有跨decode持久dense State。边界重建是原始u序列计算，不缓存完整高精度State。',
            'storage_report的vector_scratch_peak_payload_bytes是每层向量scratch payload计数峰值再求和，dP8/16/32为6316032/12632064/25264128bytes；源码逐次累加(keys, updates, alphas, key, query)的numel×element_size，FP32 alphas、batched K、FP32 u stack等临时分配确实存在，但此计数不等于全部同时live的真实峰值，更不等于allocator reserved。重建FP32 State和量化临时张量另计，完整生命周期峰值未被逐分配跟踪。',
            'dP32对16多6795264bytes buffer容量；本轮长测allocated峰值只多7274496bytes，整模型约18.5GB allocated、22.3GB reserved。没有OOM或nonfinite记录。不能由这些数据推出SM occupancy不受影响；occupancy/SM-active没有测量。','',
            '## 编译器实际元数据与测量限度','',
            '|可出现pending范围|观测specialization数|registers/thread范围|loader spills|','|---|---:|---|---|']
    for x in d['compiler']:
        lines.append(f"|0..{x['max_pending']}|{x['observed_specializations']}|{x['registers_min']}..{x['registers_max']}|{x['spill_values']}|")
    lines+=['',f"实际编译API共{d['compiler_calls']}次，总wall {d['compiler_call_wall_s']:.6f}s，包含可能的磁盘cache加载，不能称纯编译。loader提供的spills全0，寄存器上限随pending到167；这是资源需求证据，不是实测occupancy。shared/warps原flags不变，详见COMPILER_TELEMETRY.json。加载/first-touch/诊断与主TPS分别计时。",'',
            '## 身份、数值、失败及交付范围','',
            'CODE_VERIFIED/RUNTIME_VERIFIED：SOURCE_IDENTITY的41份原dP16/dP8科学文件逐SHA相同；LQUnrollBuffer仅允许周期8/16/32的构造器准入文本变更。正式快Triton kernel、原fixed_run、原codec/BF16norm批量K、quantizer/mask/sampler/输入保持原字节。无低效replay fallback、无额外数学/period专属优化。',
            '两fresh真实CUDA资格各24cases均exit0/release；V2按30份相同科学文件和原真实STARTED/EXIT证据复用资格，没有重跑GPU门禁或复用任何benchmark状态/轨迹。原0.5%跨轨迹、0.1%同actual-state阈值不变；最坏0.3187185%、0.0474805%。31/32/33、63/64/65与preserve、prompt边界均直接检查计数；所有legacy UNEXPLAINED case都有same-base反事实，局部未解释0。并非bitwise等价证明。',
            '59文件e604快照在GPU前因lineage审查扩展被替换，保留未启动身份；V1 SOURCE0237的120文件、0有效benchmark、真实benchmark/report/parent/Bash失败保留。漏原vendor资产使模型前校验失败；报告glob把.compile.json当数值结果导致CPU报告失败。224文件V1独立CPU recovery逐55/MacSHA复核。V2补原资产、CPU报告glob/回归，原真实CPU exit0/7/143和STOP/COMPLETE/OVERLAP分支不当GPU结果。',
            'V2最终GPU测速及CPU报告/Bash都真实0；原终态helper的remote manifest换行SyntaxError发生于远程程序编译阶段，旧claim/FIRST_FAILURE保留；另发现文本allowlist漏.log。新的独立CPU recovery完成197主/恢复文本SHA，原helper和恢复claim均未重试。本额外审核再次核全部固定输入、主seal、模型17及实际regular mapped并另NOREPLACE sibling封存；不修改历史SOURCE、主报告或主seal。相关wrapper负记录随evidence保留。','',
            '## 为什么低于旧6.65/6.79，以及后续判断','',
            '确实使用原快速配置。旧快基准在GPU1，当前同型号GPU3；不是同一时刻的隔离因果对照，不能将绝对差异归因于换卡/并发。旧全模型短测TPOT约150.30/147.18ms，本轮约215.24/208.80ms，增约65/62ms；独立fused core只从19.10→24.46µs、27.95→28.24µs（旧kernel观测在GPU2），量级解释不了全模型差额。未改算法的prefill也由约2.97s升到约3–4.8s，因此INFERRED指向更广的运行/host/同步/环境开销，但根因仍UNRESOLVED。时钟1695MHz、无launch blocking、cgroup无throttle只是观察，不能排除未采到的因素。',
            'dP32显示小幅潜在吞吐收益，同时buffer容量翻倍且pending读出/寄存器成本增加。仅从吞吐，没有充分证据优先投入一次完整准确率评估；若未来独立研究关注降低写回频率且接受额外缓存，可再单独授权评估。当前不自动运行dP32 AIME或新增任何实验。',
            '未测项明确保留：实际occupancy、SM active、融合核三个内部排他时间、全模型排他writeback占比、全临时张量同时峰值/指数轨迹峰值。','',
            '源码证据（冻结副本相对路径）：scripts/throughput_dp32.py:20 micro和actual_precommit库存，:103 long_run和:115逐token循环、:126区间；src/lq_unroll_buffer.py:43批量原codec K、:66临时State重建、:118 K编码/:120 pending读出、:134 scratch计数、:149追加u记录；原始计时/容量/退出以相邻JSON和runtime收据为准。','']
    return '\n'.join(lines)

def publish(parts,plan):
    receipts=[]
    for i,names in enumerate(parts,1):
        partname=f'part{i:02d}';local=OUT/partname;local.mkdir()
        rows=[]
        for name in names:
            source=OUT/name;dest=local/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(source.read_bytes());f.flush();os.fsync(f.fileno())
            rows.append(dict(path=name,bytes=dest.stat().st_size,sha256=sha(dest)))
        put(local/'PACKAGE.json',dict(source_sha256=SOURCE,files=rows,part=partname,main_manifest=plan['original_main_seal_sha256']))
        textfile(local/'MANIFEST.sha256',''.join(x['sha256']+'  '+x['path']+'\n' for x in rows)+sha(local/'PACKAGE.json')+'  PACKAGE.json\n')
        allrows=rows+[dict(path=name,bytes=(local/name).stat().st_size,sha256=sha(local/name)) for name in ['PACKAGE.json','MANIFEST.sha256']]
        assert len(allrows)<=24 and sum(x['bytes'] for x in allrows)<=32*1024*1024
        payload=[dict(x,data=base64.b64encode((local/x['path']).read_bytes()).decode()) for x in allrows]
        code=r'''
import base64,ctypes,hashlib,json,os,stat
from pathlib import Path
base=Path(TARGET);base.mkdir(parents=True,exist_ok=True)
target=base/PART
assert not target.exists(), 'NOREPLACE final target already exists'
stage=base/('.incoming_'+PART)
stage.mkdir(exist_ok=False)
assert len(PAYLOAD)<=24 and sum(x['bytes'] for x in PAYLOAD)<=32*1024*1024
for x in PAYLOAD:
 rel=Path(x['path']);assert not rel.is_absolute() and '..' not in rel.parts
 b=base64.b64decode(x['data']);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
 p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
for p in sorted([stage]+[p for p in stage.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
 fd=os.open(str(p),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
libc=ctypes.CDLL(None,use_errno=True)
rename=libc.renameat2;rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
if rename(-100,os.fsencode(stage),-100,os.fsencode(target),1)!=0:
 err=ctypes.get_errno();raise OSError(err,os.strerror(err))
fd=os.open(str(base),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
for x in PAYLOAD:
 p=target/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
print(json.dumps(dict(status='PASS_FSYNC_RENAMEAT2_NOREPLACE_AND_POST_SHA',name=PART,root=str(target),files=len(PAYLOAD),bytes=sum(x['bytes'] for x in PAYLOAD),manifest_sha256=hashlib.sha256((target/'MANIFEST.sha256').read_bytes()).hexdigest())))
'''
        code='TARGET='+repr(TARGET)+'\nPART='+repr(partname)+'\nPAYLOAD='+repr(payload)+'\n'+code
        compile(code,'<cpu-sibling-noreplace-publication>','exec')
        receipts.append(run(code,timeout=None))
    return receipts

def verify_remote(receipts):
    code=r'''
import hashlib,json,stat
from pathlib import Path
out=[]
for x in RECEIPTS:
 r=Path(x['root']);assert hashlib.sha256((r/'MANIFEST.sha256').read_bytes()).hexdigest()==x['manifest_sha256']
 for line in (r/'MANIFEST.sha256').read_text().splitlines():
  digest,name=line.split('  ',1);p=r/name
  assert stat.S_ISREG(p.stat().st_mode) and hashlib.sha256(p.read_bytes()).hexdigest()==digest,str(p)
 out.append(dict(root=str(r),manifest_sha256=x['manifest_sha256'],files=x['files']))
print(json.dumps(dict(status='PASS_INDEPENDENT_POST_PUBLICATION_READ_ALL_PARTS',parts=out)))
'''
    code='RECEIPTS='+repr(receipts)+'\n'+code
    compile(code,'<independent-cpu-post-publication-read>','exec')
    return run(code,timeout=None)

if __name__=='__main__':
    main()
