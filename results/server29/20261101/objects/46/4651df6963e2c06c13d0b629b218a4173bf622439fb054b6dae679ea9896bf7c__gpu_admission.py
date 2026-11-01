"""Single bounded resource admission and original exact context ownership."""
import json
import os
import pathlib
import time
from gpu_telemetry import sample, observe, inventory
import v3_guard as legacy

UUID='GPU-de53f243-b271-521c-d875-d96611ed8574'

def graphics_allowed(s,baseline=None):
    if baseline is None:
        try:baseline=json.loads((pathlib.Path(os.environ['LING_PROSPECTIVE_RESULTS'])/'EXECUTION_ROUTING.json').read_text())['baseline_graphics'][s['uuid']]
        except (OSError,KeyError,ValueError):return False
    groups=s['direct_NVML_processes'];xml=s['SMI_all_processes']
    if any(g['status']!='PASS' for g in groups.values()) or xml['status']!='PASS':return False
    if groups['graphics']['processes']!=baseline['NVML_graphics']:return False
    graphics=[p for p in xml['processes'] if p['type']=='G']
    if graphics!=baseline['SMI_graphics']:return False
    owners={p['pid']:p for p in s['process_ownership']}
    return all(owners.get(p['pid'])==p for p in baseline['ownership'])

def classify(s):
    try:
        if 'direct_NVML_processes' in s:
            groups=s['direct_NVML_processes']
            if any(g['status']!='PASS' for g in groups.values()) or s['SMI_all_processes']['status']!='PASS':
                return 'PROCESS_VISIBILITY_UNKNOWN'
            if groups['compute']['processes'] or any(p['type']!='G' for p in s['SMI_all_processes']['processes']):
                return 'BUSY_VISIBLE_PROCESS'
            if not graphics_allowed(s):return 'GRAPHICS_IDENTITY_NOT_QUALIFIED'
        if s['physical']['returncode'] != 0 or s['compute']['returncode'] != 0 or s['direct_NVML']['status'] != 'PASS':
            return 'DRIVER_OR_NVML_ERROR'
        t=s['target']; n=s['direct_NVML']
        if len(t)!=1 or t[0]['uuid']!=s['uuid']: return 'TELEMETRY_INCONSISTENT'
        t=t[0]; m=n['memory_v2']; v=n['memory']
        if 'used' not in m or 'used' not in v: return 'DRIVER_OR_NVML_ERROR'
        if m['total']!=m['free']+m['used']+m['reserved'] or v['used']!=m['used']+m['reserved']:
            return 'TELEMETRY_INCONSISTENT'
        if abs(m['used']/2**20-float(t['used_MiB']))>1 or m['total']/2**20!=float(t['total_MiB']):
            return 'TELEMETRY_INCONSISTENT'
        if t['name']!='NVIDIA GeForce RTX 3090' or float(t['total_MiB'])<24576:
            return 'INSUFFICIENT_MEMORY'
        processes=s['compute']['stdout'].strip()
        if processes: return 'BUSY_VISIBLE_PROCESS'
        a,b=float(t['GPU_util']),n['utilization']['gpu']
        if (a==0)!=(b==0): return 'TELEMETRY_INCONSISTENT'
        if a!=0 or b!=0: return 'BUSY_UNATTRIBUTED'
        if float(t['used_MiB'])>64 or m['used']>64*2**20: return 'BUSY_UNATTRIBUTED'
        if s['fuser']['returncode'] not in (0,1): return 'DRIVER_OR_NVML_ERROR'
        return 'AVAILABLE_VERIFIED'
    except (KeyError,TypeError,ValueError):
        return 'DRIVER_OR_NVML_ERROR'

def classify_window(rows):
    if len(rows)<3: return 'TELEMETRY_INCONSISTENT'
    if len({r['uuid'] for r in rows})!=1: return 'TELEMETRY_INCONSISTENT'
    states=[classify(r) for r in rows]
    return 'AVAILABLE_VERIFIED' if states[-3:]==['AVAILABLE_VERIFIED']*3 else states[-1] if states[-1]!='AVAILABLE_VERIFIED' else 'TELEMETRY_INCONSISTENT'

def choose(states, numerical_pass, preferred=UUID):
    ordered=[preferred]+sorted(u for u in states if u!=preferred)
    for u in ordered:
        if states.get(u)=='AVAILABLE_VERIFIED' and numerical_pass.get(u) is True:return u
    return None

def route(cfg):
    uuid=cfg.get('selected_gpu_uuid',UUID)
    rows,_=inventory(); targets=[r for r in rows if r['uuid']==uuid]
    assert len(targets)==1,'UUID_NOT_UNIQUELY_PRESENT'
    gpu=int(targets[0]['index']);cfg['gpu_uuids'][str(gpu)]=uuid
    return gpu

def admit(cfg,path):
    uuid=cfg.get('selected_gpu_uuid',UUID)
    rows=observe([uuid],path)
    state=classify_window(rows)
    report=dict(status=state,uuid=uuid,index=rows[-1]['target'][0]['index'] if rows[-1]['target'] else None,
        completed_epoch=time.time(),sample_count=len(rows),states=[classify(s) for s in rows],
        execution=os.environ['LING_PROSPECTIVE_RESULTS'],driver=cfg['runtime']['driver'],
        window_sha256=__import__('recovery_state').sha(path),window_path=str(path))
    assert all(r['target'][0]['driver']==cfg['runtime']['driver'] for r in rows if r['target']),'DRIVER_CHANGED'
    from recovery_state import atomic_json
    atomic_json(ticket_path(uuid),report,replace=True)
    if state!='AVAILABLE_VERIFIED':raise RuntimeError(('GPU_ADMISSION_STOP',report))
    return report

def ticket_path(uuid):
    return pathlib.Path(os.environ['LING_PROSPECTIVE_RESULTS'])/'admission_tickets'/(uuid+'.json')

def device(gpu,cfg,pid=None,*,binding=None,evidence=None,require_context=False):
    if binding is not None:
        result=legacy.device(gpu,cfg,pid,binding=binding,evidence=None,require_context=require_context)
        s=sample(cfg['gpu_uuids'][str(gpu)])
        groups=s['direct_NVML_processes'];xml=s['SMI_all_processes']
        visible=all(g['status']=='PASS' for g in groups.values()) and xml['status']=='PASS'
        listed={p['pid'] for p in groups['compute'].get('processes',[])}
        listed.update(int(p['pid']) for p in xml.get('processes',[]) if p['type']!='G')
        own=listed=={pid}
        if not visible or not own or not graphics_allowed(s):
            result['exclusive']=False;result['failure_conditions'].append('COMPUTE_GRAPHICS_EXACT_OWNERSHIP_OR_VISIBILITY_FAILED')
        result['compute_graphics_visibility']=s
        if evidence:
            with pathlib.Path(evidence).open('a') as f:f.write(json.dumps(result)+'\n');f.flush();os.fsync(f.fileno())
        return result
    p=ticket_path(cfg['gpu_uuids'][str(gpu)])
    try:
        ticket=json.loads(p.read_text());s=sample(cfg['gpu_uuids'][str(gpu)])
        state=classify(s)
        good=(ticket['status']=='AVAILABLE_VERIFIED' and ticket['uuid']==s['uuid']
            and ticket['execution']==os.environ['LING_PROSPECTIVE_RESULTS'] and 0<=time.time()-ticket['completed_epoch']<=20
            and state=='AVAILABLE_VERIFIED' and int(s['target'][0]['index'])==gpu)
        result=dict(exclusive=good,failure_conditions=[] if good else ['ADMISSION_TICKET_OR_CURRENT_TELEMETRY_FAILED'],
            gpu=gpu,gpu_uuid=s['uuid'],epoch=time.time(),ticket=ticket,telemetry=s)
    except Exception as e:result=dict(exclusive=False,failure_conditions=['ADMISSION_UNKNOWN'],error=repr(e))
    if evidence:
        with pathlib.Path(evidence).open('a') as f:f.write(json.dumps(result)+'\n');f.flush();os.fsync(f.fileno())
    return result

proc_identity=legacy.proc_identity
