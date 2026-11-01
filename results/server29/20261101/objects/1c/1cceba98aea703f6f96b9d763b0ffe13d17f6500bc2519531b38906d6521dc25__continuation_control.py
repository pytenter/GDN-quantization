"""Resource-only continuation controls; no model/codec imports or GPU API on import."""
from pathlib import Path
import gpu_control as C
PARENT_MANIFEST='340f4c2e4285a7d229742dacc52f9dc5e14d5aa4bc094749e75261abc6e2e95b'
PARENT_SOURCE='5979bb5a40d7ea46f76d542b9d664b1f7d936acd3bb35f392c6659958f992d41'
RESOURCE_ONLY_CHANGES={'GPU_PLAN.json','src/gpu_control.py','scripts/gpu_controller.py'}

def admit_parent(root,plan):
 p=Path(root)/plan['parent_delivery_relative']
 C.require(C.sha(p/'SHA256_MANIFEST.json')==PARENT_MANIFEST,'Parent manifest identity differs')
 m=C.read(p/'SHA256_MANIFEST.json')
 for x in m['files']:
  q=p/x['path'];C.require(q.stat().st_size==x['bytes'] and C.sha(q)==x['sha256'],'Parent seal member differs:'+x['path'])
 C.require(C.verify_sources(p)==PARENT_SOURCE,'Parent source identity differs')
 old=C.read(p/'SOURCE_FREEZE.json')['files']
 for x in old:
  if x['path'] not in RESOURCE_ONLY_CHANGES:
   q=Path(root)/x['path'];C.require(q.stat().st_size==x['bytes'] and C.sha(q)==x['sha256'],'Science/worker changed:'+x['path'])
 s=C.read(p/'execution/SESSION_RESULT.json')
 C.require(s['status']=='BUDGET_INSUFFICIENT_FOR_NEXT_LEGAL_UNIT' and s['own_all_UUID_release'] is True,'Parent not a clean budget terminal')
 C.require(not (p/'execution/FIRST_FAILURE.json').exists(),'Parent shared failure present')
 permit_sha=C.sha(p/'GPU_PERMIT.json');results={}
 for name in plan['inherited_units']:
  u=p/'execution'/name;r=C.read(u/'RESULT.json')
  C.require(r['unit']==name and r['permit_sha256']==permit_sha,'Inherited result permit/unit differs')
  C.require(not any(u.glob('FIRST_FAILURE*.json')),'Inherited failure present')
  C.require(C.read(u/'DRIVER_IDENTITY.json')['actual_uuid']==C.UUID,'Inherited driver UUID differs')
  for n in ('PREFLIGHT_LIBRARY.json','POSTFLIGHT_LIBRARY.json'):
   v=C.read(u/n);C.require(v['status']=='PASS' and v['identity']['actual_UUIDs']==[C.UUID],'Inherited runtime/owner differs')
  if name=='CODEC_MICRO':
   C.require(r['status']=='PASS_MICRO_TERMINAL' and all(r['conditions'][b]['status']=='GPU_CODEC_QUALIFIED' for b in ('B0','B1','B2')),'Inherited codec micro gate missing')
  else:
   audit=r['audit'];C.require(r['status']=='PASS' and r['tokens']==32 and audit['qdq_layer_updates']==768 and audit['next_consumption_checks']==744 and audit['actual_values_and_bytes_compared'] and not audit['unquantized_shadow'],'Inherited persistent gate missing')
   C.require(len(list(u.glob('TOKEN_*.json')))==32,'Inherited token roster incomplete')
   if name.endswith('R1'):
    for t in range(32):
     a=C.read(u/('TOKEN_%04d.json'%t));b=C.read(p/'execution'/name.replace('R1','R0')/('TOKEN_%04d.json'%t))
     C.require(a['fresh_actual_bytes_compared'] is True and a['raw']==b['raw'] and a['tensors']==b['tensors'],'Inherited actual fresh trace differs')
  results[name]=r
 C.require(set(results)==set(s['completed']),'Parent completed roster differs')
 return results,dict(status='PASS_READ_ONLY_PARENT_CONTINUATION_ADMISSION',manifest_sha256=PARENT_MANIFEST,source_freeze_sha256=PARENT_SOURCE,verified_manifest_members=len(m['files']),inherited_units=list(results),science_and_worker_unchanged=True,parent_budget_stop_preserved=True)

def ready(unit,complete):
 method=unit['method'];prior=32 if unit.get('tokens')==128 else 128
 if unit['kind']=='benchmark':return all(f'{method}_512_R{i}' in complete for i in (0,1))
 if unit.get('repeat')==1 and unit['reference'] not in complete:return False
 return all(f'{method}_{prior}_R{i}' in complete for i in (0,1))

def progress_signature(target):
 # Only new immutable token commits or real worker phase receipts renew this.
 names=('STARTED.json','DRIVER_IDENTITY.json','PREFLIGHT_LIBRARY.json','POSTFLIGHT_LIBRARY.json','RESULT.json','FIRST_FAILURE.json')
 return (len(list(Path(target).glob('PROGRESS_*.json'))),tuple(n for n in names if (Path(target)/n).is_file()))

def stalled(last_progress,now,seconds=1800):return now-last_progress>=seconds

def reservation(dispatch,last_alive,release):
 C.require(dispatch<=last_alive<=release,'Resource clock order differs')
 return dict(reservation_lower_seconds=last_alive-dispatch,reservation_upper_seconds=release-dispatch)
