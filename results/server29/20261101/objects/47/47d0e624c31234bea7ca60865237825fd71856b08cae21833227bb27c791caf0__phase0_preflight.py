"""Read-only historical audit and one bounded physical-resource window; no CUDA."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1')
D = ROOT / 'execution_03'
OLD = ROOT / 'execution_02'
W = Path('/data/zypan/worktrees/ling-kda-prospective-engineering-recovery')
sys.dont_write_bytecode = True
sys.path.insert(0, str(W))
os.environ['LING_PROSPECTIVE_RESULTS'] = str(D)
from gpu_telemetry import sample, query, inventory
from gpu_admission import classify_window, classify

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def save(name, value):
    with (D/name).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def audited_manifest(path, expected=None):
    if expected: assert sha(path/'RESULT_AND_DELIVERY_HASHES.json') == expected
    m = read(path/'RESULT_AND_DELIVERY_HASHES.json')
    for name, info in m['files'].items():
        assert sha(path/name) == info['sha256'], (str(path), name)
    lines = (path/'SHA256SUMS.txt').read_text().splitlines()
    for line in lines:
        h, name = line.split(None, 1)
        assert sha(path/name.strip().lstrip('*')) == h, name
    return dict(path=str(path), manifest_sha256=sha(path/'RESULT_AND_DELIVERY_HASHES.json'), files=len(m['files']), SHA_records=len(lines))

def audit():
    records = [audited_manifest(ROOT/'execution_01', '202bf8f64d7a2bfeab93451a607d575f7cc7052ab96692d88de02176f536516b'),
               audited_manifest(OLD, '3d30ed6066e4f4be6c8db5302f53cf8a7c721c9f76bfc9f5dc62c50966209954')]
    success = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')
    records.append(audited_manifest(success))
    assert sha(OLD/'FROZEN_4DOC_SELECTION.json') == 'a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b'
    assert sha(OLD/'preregistration/NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md') == '54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722'
    assert read(OLD/'SELECTION_FREEZE_RECEIPT.json')['sha256'] == sha(OLD/'FROZEN_4DOC_SELECTION.json')
    spec = read(OLD/'FROZEN_SPEC.json')
    for doc in spec['documents']:
        for key in ['token_file', 'raw_text']: assert sha(doc[key]['path']) == doc[key]['sha256']
    partial = read(OLD/'PARTIAL_LONG_HORIZON_QUALIFICATION.json')
    assert partial['qualified_conditions'] == 8 and partial['parent_complete'] == 9
    checks = read(OLD/'CHECKPOINT_MANIFEST.json')['checkpoints']
    assert len(checks) == 9
    fields = ['uninterrupted_suffix_full_logit_sha256','final_full_native_cache_sha256','audit_state_fingerprint','checkpoint_sha256']
    for row in checks:
        assert Path(row['path']).stat().st_size == row['bytes'] and sha(row['path']) == row['sha256']
        assert sha(row['parent_manifest_path']) == row['parent_manifest_sha256']
        p = read(row['parent_manifest_path'])
        assert p['status'] == 'PASS' and p['begin_position'] == 0 and p['endpoint'] == 8256 and p['observer_enabled'] is True
    for row in partial['units']:
        if row['fresh64_status'] == 'PASS':
            assert sha(row['fresh_manifest_path']) == row['fresh_manifest_sha256']
            a,b = read(row['parent_manifest_path']),read(row['fresh_manifest_path'])
            assert b['status'] == 'PASS' and b['begin_position'] == 8192 and b['endpoint'] == 8256 and b['observer_enabled'] is False
            assert all(a[k] == b[k] for k in fields)
    sf = read(OLD/'SOURCE_CODE_FREEZE.json')
    assert subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip() == sf['source_commit'] == 'dfcbb7e8497a585f8ef94c0eec6d4146ade860b0'
    assert subprocess.check_output(['git','-C',str(W),'status','--porcelain'],text=True) == ''
    for name,h in sf['files'].items(): assert sha(W/name) == h, name
    for name,h in read(OLD/'FORMAL_CODE_FREEZE.json')['dependencies'].items(): assert sha(name) == h,name
    assets = read(OLD/'PREREGISTRATION_RECONCILIATION.json')['source_assets']
    for a in assets:
        st=Path(a['path']).stat(); assert (st.st_size,st.st_mtime_ns)==(a['bytes'],a['mtime_ns'])
        if a.get('sha256'): assert sha(a['path']) == a['sha256'],a['path']
    import csv
    scores=list(csv.DictReader((OLD/'SHORT_HORIZON_SELECTION.csv').open()))
    assert len(scores)==64 and len({(r['document'],r['rotation']) for r in scores})==64
    for decision in (OLD/'screening').glob('*.SELECTION_DECISION.json'):
        s=read(decision); assert sha(OLD/'screening'/(s['document']+'.json'))==s['short_result_sha256']
    save('PHASE0_HISTORICAL_ASSET_AUDIT.json', dict(status='PASS',epoch=time.time(),manifests=records,checkpoints=checks,checkpoint_count=9,original_complete_qualification=8,parent_only=1,short_score_count=64,source_assets=len(assets),science_workers_launched=0,source_commit=sf['source_commit'],all_history_read_only=True,pre_registration_sha256=sha(OLD/'preregistration/NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md'),selection_sha256=sha(OLD/'FROZEN_4DOC_SELECTION.json'),selection_receipt_sha256=sha(OLD/'SELECTION_FREEZE_RECEIPT.json')))

class Process(C.Structure):
    _fields_=[('pid',C.c_uint),('usedGpuMemory',C.c_ulonglong),('gpuInstanceId',C.c_uint),('computeInstanceId',C.c_uint)]

def nvml_processes(uuid):
    lib=C.CDLL('libnvidia-ml.so.1'); assert lib.nvmlInit_v2()==0
    try:
        h=C.c_void_p();assert lib.nvmlDeviceGetHandleByUUID(uuid.encode(),C.byref(h))==0
        out={}
        for key,fn in [('compute','nvmlDeviceGetComputeRunningProcesses_v3'),('graphics','nvmlDeviceGetGraphicsRunningProcesses_v3')]:
            count=C.c_uint(0); rc=getattr(lib,fn)(h,C.byref(count),None)
            if rc==0 and count.value==0: out[key]=dict(status='PASS',processes=[],api=fn); continue
            if rc!=7: out[key]=dict(status='UNKNOWN',returncode=rc,api=fn);continue
            values=(Process*(count.value+8))();count.value=len(values)
            rc=getattr(lib,fn)(h,C.byref(count),values)
            out[key]=dict(status='PASS',api=fn,processes=[{n:getattr(p,n) for n,_ in Process._fields_} for p in values[:count.value]]) if rc==0 else dict(status='UNKNOWN',returncode=rc,api=fn)
        return out
    finally:lib.nvmlShutdown()

def process_metadata(pid):
    try:
        d=Path('/proc')/str(pid);st=d.stat();text=(d/'stat').read_text();fields=text[text.rfind(')')+2:].split()
        return dict(status='PASS',pid=pid,uid=st.st_uid,start_ticks=int(fields[19]),comm=(d/'comm').read_text().strip())
    except Exception as e:return dict(status='UNKNOWN',pid=pid,error=repr(e))

def enriched_sample(uuid):
    s=sample(uuid);s['direct_NVML_processes']=nvml_processes(uuid)
    x=query(['nvidia-smi','-i',uuid,'-q','-x']);s['SMI_XML']=x
    try:
        assert x['returncode']==0
        tree=ET.fromstring(x['stdout']);gpu=tree.find('gpu');assert gpu.findtext('uuid')==uuid
        ps=[dict(pid=p.findtext('pid'),type=p.findtext('type'),name=p.findtext('process_name'),used_memory=p.findtext('used_memory')) for p in gpu.findall('processes/process_info')]
        s['SMI_all_processes']=dict(status='PASS',processes=ps)
    except Exception as e:s['SMI_all_processes']=dict(status='UNKNOWN',error=repr(e))
    pids={int(p['pid']) for p in s['SMI_all_processes'].get('processes',[])}
    for group in s['direct_NVML_processes'].values():pids.update(p['pid'] for p in group.get('processes',[]))
    s['process_ownership']=[process_metadata(p) for p in sorted(pids)]
    s['resource_classification']=classify(s)
    if any(v['status']!='PASS' for v in s['direct_NVML_processes'].values()) or s['SMI_all_processes']['status']!='PASS':s['resource_classification']='PROCESS_VISIBILITY_UNKNOWN'
    elif pids:s['resource_classification']='BUSY_VISIBLE_PROCESS'
    return s

def preflight():
    rows,inv=inventory();save('INITIAL_GPU_INVENTORY.json',dict(epoch=time.time(),rows=rows,evidence=inv))
    selected=[r for r in rows if r['index'] in ['4','5','6']]
    save('DEVICE_USE_AUTHORIZATION.json',dict(epoch=time.time(),user_authorization='我下面给你的指令，你可以选用两个空闲的GPU来进行',max_simultaneous_devices=2,user_uid=os.getuid(),device_permissions={r['index']:dict(read=os.access('/dev/nvidia'+r['index'],os.R_OK),write=os.access('/dev/nvidia'+r['index'],os.W_OK),mode=oct(Path('/dev/nvidia'+r['index']).stat().st_mode)) for r in selected},scheduler_environment={k:os.environ.get(k) for k in ['SLURM_JOB_ID','SLURM_JOB_GPUS','PBS_JOBID']},scheduler_status='NO_JOB_ALLOCATION_ENVIRONMENT; USER_DIRECT_AUTHORIZATION_AND_EXCLUSIVE_CONTEXT_REQUIRED',process_namespace_visibility='UNKNOWN_IF_INIT_NAMESPACE_UNREADABLE',actual_context_ownership='REQUIRED_PER_WORKER_EXACT_PID_UID_STARTTIME_UUID; NOT_YET_A_CUDA_QUALIFICATION'))
    records=[];start=time.monotonic()
    with (D/'GPU_PREFLIGHT_WINDOW_01.jsonl').open('x') as f:
        for k in range(13):
            if k:time.sleep(max(0,start+5*k-time.monotonic()))
            for r in selected:
                s=enriched_sample(r['uuid']);records.append(s);f.write(json.dumps(s,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    result=[]
    for r in selected:
        samples=[s for s in records if s['uuid']==r['uuid']]
        status=classify_window(samples)
        if any(s['resource_classification']!='AVAILABLE_VERIFIED' for s in samples[-3:]):status=samples[-1]['resource_classification'] if samples[-1]['resource_classification']!='AVAILABLE_VERIFIED' else 'WINDOW_STABILITY_FAILED'
        result.append(dict(index=r['index'],uuid=r['uuid'],status=status,sample_count=len(samples),states=[s['resource_classification'] for s in samples],window_sha256=sha(D/'GPU_PREFLIGHT_WINDOW_01.jsonl')))
    save('GPU_PREFLIGHT_RESULT.json',dict(epoch=time.time(),window_count=1,maximum_windows=3,results=result,CUDA_context_created=False,numerical_qualification='NOT_RUN',per_worker_original_60s_final3_rule='UNCHANGED'))
    print(json.dumps(dict(event='PREFLIGHT_COMPLETE',results=result)),flush=True)

if __name__=='__main__':
    try:audit();preflight()
    except BaseException as e:
        import traceback
        save('STOP.json',dict(status='STOP',stage='PHASE0_OR_RESOURCE_PREFLIGHT',reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True));raise
