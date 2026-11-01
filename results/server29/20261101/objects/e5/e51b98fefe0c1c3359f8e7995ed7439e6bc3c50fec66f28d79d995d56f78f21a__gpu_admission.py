"""Prospectively frozen Clean-Room resource policy, independent of old gates."""
import os,json,time,sys,threading
from pathlib import Path
from gpu_telemetry import sample as raw_sample,inventory
TELEMETRY_LOCK=threading.Lock()
def sample(uuid):
 with TELEMETRY_LOCK:return raw_sample(uuid)
from v3_guard import proc_identity
from cleanroom_common import D,read,publish
GIB=2**30
HEADROOM=2*GIB
# Maximum of recorded CUDA reserved and NVML owned-process peaks, plus 2 GiB.
PEAK=max(20128464896,19536*2**20)

def graphics_identity(s):
 groups=s['direct_NVML_processes'];xml=s['SMI_all_processes']
 assert all(v['status']=='PASS' for v in groups.values()) and xml['status']=='PASS','PROCESS_VISIBILITY_UNKNOWN'
 gs=groups['graphics']['processes'];owners={p['pid']:p for p in s['process_ownership']}
 result=[]
 for g in gs:
  p=owners.get(g['pid']);assert p and p['status']=='PASS' and p['comm']=='Xorg','UNAUTHORIZED_GRAPHICS_PROCESS'
  assert p['uid'] in [0,120] and g['usedGpuMemory']<=16*2**20,'GRAPHICS_BASELINE_NOT_SYSTEM_XORG'
  result.append(p)
 assert {int(p['pid']) for p in xml['processes'] if p['type']=='G'}=={g['pid'] for g in gs},'GRAPHICS_API_IDENTITY_MISMATCH'
 return sorted(result,key=lambda x:x['pid'])
def assess(s,uuid,owner=None,baseline=None):
 assert s['uuid']==uuid and len(s['target'])==1 and s['target'][0]['uuid']==uuid,'DEVICE_IDENTITY_FAILURE'
 t=s['target'][0];n=s['direct_NVML'];assert n['status']=='PASS' and n['memory_v2'].get('free') is not None,'NVML_UNAVAILABLE'
 assert t['name']=='NVIDIA GeForce RTX 3090' and float(t['total_MiB'])>=24576,'UNAUTHORIZED_DEVICE_TYPE'
 m=n['memory_v2'];assert m['total']==m['free']+m['used']+m['reserved'],'MEMORY_ARITHMETIC_FAILURE'
 if owner is None:assert abs(m['used']/2**20-float(t['used_MiB']))<=64,'MATERIAL_MEMORY_TELEMETRY_MISMATCH'
 # During owned model loading, these two non-atomic reads can legitimately
 # straddle large allocations. Exact PID attribution and true free headroom
 # govern safety; preserve the difference as telemetry, not a false conflict.
 graphics=graphics_identity(s)
 if baseline is not None:assert graphics==baseline,'GRAPHICS_IDENTITY_CHANGED'
 union={p['pid'] for p in s['direct_NVML_processes']['compute']['processes']}
 union.update(int(p['pid']) for p in s['SMI_all_processes']['processes'] if p['type']!='G')
 allowed=set() if owner is None else {owner['pid']}
 assert not (union-allowed),'UNAUTHORIZED_EXTERNAL_COMPUTE_PROCESS'
 if owner is not None:
  assert proc_identity(owner['pid'])==owner,'OWN_PROCESS_IDENTITY_CHANGED'
  if m['used']>64*2**20:assert owner['pid'] in union,'MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID'
 assert m['free']>=HEADROOM+(PEAK if owner is None else 0),'INSUFFICIENT_FREE_MEMORY_AND_2GIB_HEADROOM'
 return dict(SMI_NVML_allocated_difference_bytes=m['used']-float(t['used_MiB'])*2**20,graphics=graphics,allocated_bytes=m['used'],free_bytes=m['free'],SMI_util=float(t['GPU_util']),NVML_util=n['utilization']['gpu'],compute_PIDs=sorted(union))
def assess_window(rows,uuid):
 assert len(rows)==3,'THREE_SAMPLES_REQUIRED'
 values=[assess(s,uuid) for s in rows];base=values[0]['graphics']
 assert all(v['graphics']==base for v in values),'GRAPHICS_IDENTITY_NOT_STABLE'
 mem=[v['allocated_bytes'] for v in values]
 assert max(mem)-min(mem)<=64*2**20,'UNATTRIBUTED_MEMORY_GROWTH'
 assert all(v['allocated_bytes']<=64*2**20 for v in values),'UNATTRIBUTED_DEVICE_ALLOCATION'
 util=[max(v['SMI_util'],v['NVML_util']) for v in values]
 # Low <=2% differences are non-blocking; persistent higher use is unsafe.
 assert sum(x>2 for x in util)<2 and util[-1]<=2,'SUSTAINED_OR_CURRENT_MATERIAL_GPU_OCCUPANCY'
 return dict(status='PASS',UUID=uuid,sample_count=3,graphics_baseline=base,min_free_bytes=min(v['free_bytes'] for v in values),measured_peak_budget_bytes=PEAK,required_extra_headroom_bytes=HEADROOM,low_util_difference_warnings=[i for i,v in enumerate(values) if v['SMI_util']!=v['NVML_util']],values=values)
def admit(uuid,tag):
 path=D/'admission'/(tag+'.json');assert not path.exists(),'ADMISSION_ALREADY_DONE_NO_RETRY'
 rows=[]
 for i in range(3):
  rows.append(sample(uuid))
  if i<2:time.sleep(5)
 publish(path,dict(samples=rows,epoch=time.time()))
 result=assess_window(rows,uuid);result.update(epoch=time.time(),sample_path=str(path))
 publish(D/'admission'/(tag+'.PASS.json'),result)
 return result

def route(cfg):
 uuid=cfg['selected_gpu_uuid'];routing=read(D/'EXECUTION_ROUTING.json');rows,_=inventory();r=[r for r in rows if r['uuid']==uuid]
 assert len(r)==1 and int(r[0]['index'])==routing['authorized_devices'][uuid],'FROZEN_UUID_INDEX_MAPPING_CHANGED'
 index=int(r[0]['index']);cfg['gpu_uuids'][str(index)]=uuid
 return index
def device(gpu,cfg,pid=None,*,binding=None,evidence=None,require_context=False):
 uuid=cfg['selected_gpu_uuid'];assert route(cfg)==gpu
 baseline=read(D/'resource_baselines.json')[uuid]
 s=sample(uuid);owner=binding
 if owner is None and pid is not None:owner=proc_identity(pid)
 # Prelaunch may have this worker's CUDA context from helper imports, so only
 # own exact PID is permitted; no foreign process can be accepted.
 if owner is None:
  current=proc_identity(os.getpid());union={p['pid'] for p in s['direct_NVML_processes']['compute'].get('processes',[])}
  if current['pid'] in union:owner=current
 try:
  value=assess(s,uuid,owner,baseline)
  if require_context:assert owner is not None and owner['pid'] in value['compute_PIDs'],'OWN_CUDA_CONTEXT_NOT_VISIBLE'
  result=dict(exclusive=True,failure_conditions=[],gpu=gpu,gpu_uuid=uuid,epoch=time.time(),resource=value)
 except Exception as e:result=dict(exclusive=False,failure_conditions=[repr(e)],gpu=gpu,gpu_uuid=uuid,epoch=time.time())
 if evidence:
  with Path(evidence).open('a') as f:f.write(json.dumps(dict(result,snapshot=s))+'\n');f.flush();os.fsync(f.fileno())
 return result
