"""Strict54 reuse through a fresh, exact frozen predecessor CPU interpreter."""
import json,os,subprocess,sys
from pathlib import Path
import pace_control as C
OLD='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_lifecycle_repair_20261026'
OLD_SOURCE='8a6ca3a6b0f45c55f190f3310ba0450389cc91567539f4328bf514198953e3d5'
OLD_PERMIT='a85684f1675cfd1e68f82d76020bfcc2f6a3843716cf711f88c9da3683f06937'
OLD_MANIFEST='87ee3c0a785a23295d418df3609f47db51efe681f6a9aede8f7667727d625ff8'
CHILD=r'''
import sys,json,os
from pathlib import Path
r=Path(OLD);sys.path.insert(0,str(r/'src'))
import pace_control as C
import pace_four_gpu_results as K
assert C.sha(r/'source_manifest.json')==SOURCE and C.sha(r/'GPU_PERMIT.json')==PERMIT and C.sha(r/'DELIVERY_MANIFEST.json')==MANIFEST
C.authorize(r);C.verify_science(r);C.model_guard(r);K.verify_reuse(r,full_sources=True)
bad=[];pairs=K.completed_rows(r,bad);assert len(pairs)==54 and len(bad)==2 and {x['unit'] for x in bad}=={'aime26_25_R2_seed1','aime26_26_R2_seed1'}
units=[]
for j,w,t in pairs:
 v=C.read(t/'RESOURCE_INTERVAL.json');assert v['returncode']==0 and v['released'] is True
 units.append(dict(unit=j['id'],payload_sha256=w['payload_sha256'],result_sha256=C.sha(t/'RESULT.json'),interval_sha256=C.sha(t/'RESOURCE_INTERVAL.json'),origin_target=str(t),origin_source_sha256=w['source_manifest_sha256'],origin_gpu=w['gpu'],origin_uuid=w['gpu_uuid']))
controls=['source_manifest.json','GPU_PERMIT.json','SCIENCE_FREEZE.json','SESSION_RESULT.json','RESOURCE_LEDGER.json','final_verdict.json','DELIVERY_MANIFEST.json','ARCHIVE_VERIFICATION.json','DELIVERY_READY.json','OFFLINE_PARENT_STARTED.json','OFFLINE_PARENT_RESULT.json','PARENT_CONTROLLER_EXIT.json','PARENT_FINISHER_EXIT.json','OFFLINE_CHAIN_STARTED.json','OFFLINE_CHAIN_ACTUAL_CHILD.json','OFFLINE_CHAIN_OBSERVED_EXIT.json','FINISHER_EXIT.json']
ctrl=C.read(r/'PARENT_CONTROLLER_EXIT.json');fin=C.read(r/'PARENT_FINISHER_EXIT.json');chain=C.read(r/'OFFLINE_CHAIN_OBSERVED_EXIT.json');assert ctrl['actual_exited'] and ctrl['identity_gone'] and ctrl['returncode']==1 and fin['actual_exited'] and fin['identity_gone'] and fin['returncode']==0 and chain['actual_shell_wait_returncode']==0
ids=[C.read(r/'OFFLINE_PARENT_STARTED.json')['identity'],C.read(r/'OFFLINE_CHAIN_STARTED.json')['bash'],ctrl['identity'],fin['identity']]+[x['identity'] for x in C.read(r/'RESOURCE_LEDGER.json')['intervals'] if x.get('identity')]
assert all(not C.same_process(x) for x in ids)
gates=[]
for g in range(4):
 u='SCALE_REPAIR_GATES_GPU'+str(g);t=C.unit_target(r,'scale_repair_gates',u);v=C.read(t/'RESOURCE_INTERVAL.json');w=C.verify_result(r,'scale_repair_gates',u,v['identity']);assert v['returncode']==0 and v['released'] and w['gpu']==g and w['gpu_uuid']==C.UUIDS[g];d=C.read(t/'DURABLE29_RECEIPT.json');assert d['actual_exit_returncode']==0 and d['released']
 gates.append(dict(unit=u,gpu=g,result_sha256=C.sha(t/'RESULT.json'),interval_sha256=C.sha(t/'RESOURCE_INTERVAL.json'),durable29_manifest_sha256=d['manifest_sha256']))
print(json.dumps(dict(status='PASS_FROZEN_PREDECESSOR_54_CPU_INSPECTORS_AND_REAL_OWNING_EXITS',units=units,reused_ids=[x['unit'] for x in units],origin_terminal_controls={n:dict(sha256=C.sha(r/n),bytes=(r/n).stat().st_size) for n in controls},gates=gates,source_sha256=SOURCE,permit_sha256=PERMIT,primary_manifest_sha256=MANIFEST,old_parent_controller_workers_finisher_gone=True,model_loads=0,GPU_calls=0)))
'''

def predecessor_inventory():
    program='OLD='+repr(OLD)+'\nSOURCE='+repr(OLD_SOURCE)+'\nPERMIT='+repr(OLD_PERMIT)+'\nMANIFEST='+repr(OLD_MANIFEST)+'\n'+CHILD
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    p=subprocess.run(['/data/ydai/miniconda3/envs/bitdecode/bin/python','-B','-'],input=program,text=True,capture_output=True,env=env,timeout=300)
    C.require(p.returncode==0,'Frozen predecessor CPU inspection failed: '+p.stderr[-4000:]);v=json.loads(p.stdout)
    C.require(v['status']=='PASS_FROZEN_PREDECESSOR_54_CPU_INSPECTORS_AND_REAL_OWNING_EXITS' and len(v['units'])==54,'Exact54 predecessor admission missing');return v

def verify_reuse(root,full_sources=False):
    root=Path(root);saved=C.read(root/'REUSE_ADMISSION.json');v=predecessor_inventory()
    C.require(saved['status']=='PASS_CLOSED_PREDECESSOR_FIXED_PAYLOAD_REUSE' and saved['origin_root']==OLD and saved['origin_source_sha256']==OLD_SOURCE,'Exact predecessor scope differs')
    C.require(saved['units']==v['units'] and saved['reused_ids']==v['reused_ids'] and saved['origin_terminal_controls']==v['origin_terminal_controls'],'Fresh original inspector differs from frozen54 admission')
    for x in saved['units']:
        t=root/'provenance/REUSED_FORMAL'/x['unit'];C.require(C.sha(t/'RESULT.json')==x['result_sha256'] and C.sha(t/'RESOURCE_INTERVAL.json')==x['interval_sha256'],'Copied original54 payload/interval differs')
    C.validate_jobs(root)
    return dict(status='PASS_FRESH_FROZEN_PREDECESSOR54_EXIT0_RELEASE_INSPECTORS',count=54,units=v['reused_ids'],source_sha256=OLD_SOURCE,model_loads=0,GPU_calls=0)

def verify_lifecycle(root,target,interval):
    root=Path(root);target=Path(target);p=target/'OWNING_LIFECYCLE_CLOSED.json';v=C.read(p);journal=target/'OWNING_LIFECYCLE.jsonl'
    C.require(interval.get('failure') is None and interval['owning_lifecycle_sha256']==C.sha(p),'Interval/lifecycle failure or binding differs')
    C.require(v['status']=='PASS_OWNING_LIFECYCLE_EXIT0_RELEASED' and v['actual_returncode']==0 and v['actual_exited'] and v['released'] and v['ownership_or_exit_failure'] is None,'Actual owning lifecycle exit0/release proof missing')
    C.require(v['identity']==interval['identity'] and v['unit']==interval['unit'] and v['stage']==interval['stage'] and v['gpu_uuid']==interval['gpu_uuid'],'Lifecycle scope/identity differs')
    C.require(v['source_manifest_sha256']==C.sha(root/C.SOURCE) and v['permit_sha256']==C.sha(root/'GPU_PERMIT.json') and v['journal_sha256']==C.sha(journal) and v['journal_bytes']==journal.stat().st_size,'Lifecycle source/journal binding differs')
    events=[json.loads(b) for b in journal.read_bytes().splitlines()]
    C.require(events[0]['phase']=='REGISTERED' and events[-1]['phase']=='CLOSED' and any(e['phase']=='EXITED' and e['returncode']==0 for e in events),'Actual exit event missing')
    C.require(not any(e['phase'] in ('OWNERSHIP_VIOLATION','EXIT_RELEASE_TIMEOUT') for e in events) and all(e['expected']==interval['identity'] and e['source_manifest_sha256']==v['source_manifest_sha256'] and e['permit_sha256']==v['permit_sha256'] for e in events),'Lifecycle journal contains failure or different identity')
    return v

def completed_rows(root,invalid=None):
    root=Path(root);verify_reuse(root);reuse=set(C.read(root/'REUSE_ADMISSION.json')['reused_ids']);rows=[]
    for job in C.validate_jobs(root):
        u=job['id']
        if u in reuse:
            t=root/'provenance/REUSED_FORMAL'/u;rows.append((job,C.read(t/'RESULT.json'),t));continue
        t=C.unit_target(root,'formal',u)
        if not (t/'RESULT.json').exists() and not (t/'RESOURCE_INTERVAL.json').exists():continue
        try:
            v=C.read(t/'RESOURCE_INTERVAL.json');C.require(v['returncode']==0 and v['released'] is True and v['gpu']==job['gpu'] and v['gpu_uuid']==job['gpu_uuid'],'Current actual exit/release/device differs');verify_lifecycle(root,t,v);w=C.verify_result(root,'formal',u,v['identity']);rows.append((job,w,t))
        except BaseException as e:
            if invalid is None:raise
            invalid.append(dict(unit=u,error=repr(e),completion_credit=0))
    C.require(len(rows)==len({j['id'] for j,w,t in rows}),'Duplicate legal completion');return rows
