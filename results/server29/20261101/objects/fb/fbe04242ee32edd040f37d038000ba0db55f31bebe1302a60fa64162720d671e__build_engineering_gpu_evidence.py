#!/usr/bin/env python3
"""CPU-only fresh identity audit/binding of all native-exit work; no GPU dispatch."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import qualification_guards as G

def sha(p):return G.sha(p)
def read(p):return json.loads(Path(p).read_text())
def ref(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
def new(p,v):
    p=Path(p)
    with p.open('xb')as f:f.write(json.dumps(v,sort_keys=True,indent=2,allow_nan=False).encode()+bytes([10]));f.flush();os.fsync(f.fileno())
    return ref(p)

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and 'torch'not in sys.modules
assert not(ROOT/'SOURCE_FREEZE.json').exists()and not(ROOT/'CONTROLLER_STARTED.json').exists()
series=read(ROOT/'operations/ENGINEERING_CUPTI_ORDER_REPAIR_TWO_FRESH_RESULT.json')
assert series['status']=='PASS_TWO_FRESH_FULL_9B_LOAD_ONLY_NORMAL_EXIT_NOT_QUALIFICATION'
repaired=[ROOT/('operations/ENGINEERING_CUPTI_ORDER_REPAIR_R'+str(i)+'_COMPLETION.json')for i in (0,1)]
assert all(read(p)['actual_wait_returncode']==0 for p in repaired)
native=G.verify_native_closure(ROOT,fresh=True)
new(ROOT/'operations/FINAL_PRE_SCIENCE_NATIVE_CLOSURE_FRESH_REVIEW.json',native)
review_rows=[]
for index,p in enumerate(repaired):
    completion=read(p);payload=read(ROOT/completion['result']['path']);stages=[]
    for row in payload['observations']:
        q=ROOT/row['path'];assert q.stat().st_size==row['bytes']and sha(q)==row['sha256']
        actual=G.assess_native_maps(ROOT,read(q)['raw_maps'])
        assert actual['status']=='PASS_ALL_ACTUAL_NATIVE_MAPPINGS_EXACTLY_PINNED'and not actual['errors']
        stages.append(dict(**ref(q),mapped_native_count=len(actual['mapped_native']),assessment_status=actual['status']))
    review=dict(status='PASS_ALL_LOAD_ONLY_NATIVE_MAP_STAGES_EXACTLY_PINNED',completion_sha256=sha(p),
        native_manifest_sha256=sha(ROOT/G.NATIVE_CLOSURE),actual_fresh_native_closure_count=native['count'],stages=stages,
        full9B_load_only_exit_is_not_scientific_qualification=True)
    review_rows.append(review)
    new(ROOT/('operations/ENGINEERING_CUPTI_ORDER_REPAIR_R'+str(index)+'_NATIVE_REVIEW.json'),review)
active_review=new(ROOT/'operations/ENGINEERING_NATIVE_MAPS_FINAL_REVIEW.json',review_rows[-1])
combined=new(ROOT/'operations/ENGINEERING_CUPTI_ORDER_REPAIR_ALL_NATIVE_REVIEWS.json',dict(
    status='PASS_BOTH_REPAIRED_LOADER_MAPS_ALL_STAGES_IDENTITY_ONLY',loaders=review_rows,native_manifest_sha256=sha(ROOT/G.NATIVE_CLOSURE)))

gdb_dir=ROOT/'operations/model_native_exit_gdb_v1'
launch=read(gdb_dir/'LAUNCH.json');text=(gdb_dir/'gdb.log').read_text(errors='replace');worker=read(gdb_dir/'WRAPPER_STARTED.json')['identity']
assert 'NATIVE_DIAGNOSTIC_SIGNAL '+json.dumps(dict(signal='SIGSEGV',inferior_pid=worker['pid']))in text
assert not G.vendor(ROOT).same_process(worker)and not G.vendor(ROOT).same_process(launch['wrapper_identity'])
nvml=G.vendor(ROOT).query_gpu();assert not any(x['pid']==worker['pid']for x in nvml['apps'])
now=time.monotonic()
gdb_completion=new(gdb_dir/'COMPLETION.json',dict(status='REAL_NATIVE_SIGSEGV_REPRODUCED_DURING_LIBCUDART_EXIT_NOT_RECOVERY',
    inferior_identity=worker,wrapper_identity=launch['wrapper_identity'],inferior_native_signal='SIGSEGV',
    actual_Popen_wait_returncode='UNRESOLVED_DETACHED_GDB_PARENT',own_exact_process_exited=True,own_GPU_UUID_released=True,
    resource_interval=dict(dispatch_monotonic_lower=launch['dispatch_monotonic_lower'],identity_monotonic_upper=launch['identity_monotonic_upper'],
        exit_first_observed_monotonic=now,release_first_observed_monotonic=time.monotonic(),monotonic_same_host=True,
        exit_observation_scope='FIRST_DURABLE_LATE_READONLY_OBSERVATION_UPPER_NOT_EXACT_EXIT_TIME'),
    gdb_log=ref(gdb_dir/'gdb.log'),model_forward_calls=0,QDQ_calls=0,AIME_generations=0))
resources=[dict(id='ORIGINAL_FAILED_FULL_LOAD',kind='loader',receipt=ref(ROOT/'operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json')),
           dict(id='FULL_LOAD_GDB_SIGSEGV',kind='loader',receipt=gdb_completion)]
for directory,cases in [
 ('cupti_exit_diagnostic_v1',['baseline','profile']),
 ('cupti_optional_runtime_diagnostic_v1',['warm_torch','warm_torchvision','warm_text_only']),
 ('cupti_load_order_diagnostic_v1',['driver_first','cupti_first','torch_first'])]:
    for case in cases:resources.append(dict(id=directory+'_'+case,kind='diagnostic',receipt=ref(ROOT/'operations'/directory/(case+'_COMPLETION.json'))))
for i,p in enumerate(repaired):resources.append(dict(id='REPAIRED_FULL_LOAD_R'+str(i),kind='loader',receipt=ref(p)))
assert len(resources)==len({x['id']for x in resources})==12
binding=dict(status='PASS_NATIVE_LOAD_ORDER_REPAIR_TWO_FRESH_EXITS_HISTORY_PRESERVED',
    native_exit_repair_authorization_sha256=sha(ROOT/'HUMAN_NATIVE_EXIT_REPAIR_AUTHORIZATION.json'),
    original_failed_completion=ref(ROOT/'operations/ENGINEERING_LOAD_ONLY_PROBE_COMPLETION.json'),
    original_failed_probe_preserved_copy_map='provenance/ORIGINAL_FAILED_LOAD_ONLY_PROBE_SOURCE/COPY_MAP.json',
    successful_loader_completions=[ref(p)for p in repaired],active_completion=ref(repaired[-1]),active_native_review=active_review,
    all_repaired_native_reviews=combined,engineering_resource_records=resources,
    dispatch_counts=dict(scientific_qualification_GPU_dispatches=0,engineering_native_loader_GPU_probes=4,engineering_diagnostic_GPU_dispatches=8,total_GPU_work_dispatches=12),
    EXACT_physical_GPUh='UNRESOLVED_NOT_ZERO',mathematical_core_unchanged=True,no_qualification_or_formal_credit=True)
out=new(ROOT/'operations/ENGINEERING_GPU_EVIDENCE_BINDING.json',binding)
print(json.dumps(dict(status=binding['status'],binding=out,native_fresh_count=native['count'],map_stages=sum(len(x['stages'])for x in review_rows))))
