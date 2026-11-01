"""Read-only remote fragments for the separate final CPU review."""

CURRENT55 = r'''
import glob,importlib.metadata
r=Path(P['current']);sys.path.insert(0,str(r/'src'));import execution_common as E
req(sha(r/'FORMAL_SOURCE_MANIFEST.json')==P['source'] and sha(r/'GPU_PERMIT.json')==P['permit'],'Current source/permit differs')
sourceproof=verify(r,read(r/'FORMAL_SOURCE_MANIFEST.json')['files'])
for x in P['primary']:
 p=safe(r,x['path']);req(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Primary snapshot changed: '+str(p))
old=Path(P['unused_root']);req(sha(old/'FORMAL_SOURCE_MANIFEST.json')==P['unused_source'] and sha(old/'GPU_PERMIT.json')==P['unused_permit'],'Rejected V1 source identity differs')
unused=verify(old,read(old/'FORMAL_SOURCE_MANIFEST.json')['files'])
req(not (old/'FORMAL_STARTED.json').exists(),'Rejected candidate unexpectedly started')
jobs=read(r/'FORMAL_JOBS.json')['jobs'];identities=[];workers=[];maps={}
events=[json.loads(s) for s in (r/'runtime/CONTROLLER_EVENTS.jsonl').read_text().splitlines()]
dispatch={x['unit']:x for x in events if x['phase']=='UNIT_DISPATCHED'}
closed={x['unit']:x for x in events if x['phase']=='UNIT_CLOSED_AND_ARCHIVED'}
req(len(dispatch)==len(closed)==len(jobs),'Controller lifecycle event denominator differs')
for job in jobs:
 t=r/'formal'/job['id'];d=read(t/'DISPATCH.json');v=read(t/'RESULT.json');c=read(t/'OWNING_CLOSED.json');a=read(t/'ARCHIVE29.json')
 req(d['job']==v['job']==job and c['RESULT_sha256']==sha(t/'RESULT.json'),'Job/result closure differs')
 req(d['source_sha256']==c['source_sha256']==v['source_manifest_sha256']==P['source'],'Source binding differs')
 req(d['permit_sha256']==c['permit_sha256']==v['permit_sha256']==P['permit'],'Permit binding differs')
 for k in ['pid','ppid','starttick','cmdline']:req(d['child'][k]==c['child'][k]==v['identity'][k],'Worker identity differs')
 req(d['child']['ppid']==d['owner']['pid'] and d['CVD']==job['gpu_uuid'],'Owner/CVD differs')
 runtime=read(t/'LOADED_REGULAR_MAPPED_SHA.json')['runtime']
 req(runtime==v['runtime'] and runtime['driver_uuid']==runtime['CVD']==job['gpu_uuid'],'Actual Driver UUID differs')
 req(type(c['owning_Popen_exit']) is int and c['owning_Popen_exit']==0 and c['release_observed'] is True,'Real worker exit/release differs')
 req(a['status']=='PASS_29_DURABLE_NOREPLACE' and closed[job['id']]['RESULT_sha256']==c['RESULT_sha256'],'Archive binding differs')
 identities.extend([d['child'],d['owner']])
 for p in [t/'LOADED_REGULAR_MAPPED_SHA.json',t/'POSTFLIGHT_REGULAR_MAPPED_SHA.json']:
  for x in read(p)['files']:
   key=(x['device'],x['inode'])
   if key in maps:req(maps[key]['sha256']==x['sha256'],'Conflicting mapped identity')
   maps[key]=x
 workers.append(dict(unit=job['id'],child=d['child'],uuid=job['gpu_uuid'],real_exit=0,released=True,RESULT_sha256=c['RESULT_sha256'],archive=a))
for gpu in sorted({j['gpu'] for j in jobs}):
 lane=[j for j in jobs if j['gpu']==gpu]
 for before,after in zip(lane,lane[1:]):
  req(closed[before['id']]['monotonic']<dispatch[after['id']]['monotonic'],'Queue dispatched before previous archive closure')
for x in maps.values():verify(r,[x],True)
model=verify(r,read(r/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files'],True)
for p in [r/'runtime/PARENT_DISPATCH.json']+list((r/'runtime').glob('*/STARTED.json')):
 if not p.exists():continue
 z=read(p)
 for k in ['child','owner','bash']:
  if k in z:identities.append(z[k])
epochs=[]
for x in {json.dumps(x,sort_keys=True):x for x in identities}.values():
 p=Path('/proc',str(x['pid']),'stat')
 if p.exists():req(int(p.read_text().rsplit(')',1)[1].split()[19])!=x['starttick'],'Exact owning epoch still live');state='PID_REUSED_FOREIGN'
 else:state='GONE'
 epochs.append(dict(pid=x['pid'],starttick=x['starttick'],status=state))
bash=read(r/'runtime/BASH_WAIT.json');req(type(bash['bash_wait_integer']) is int and bash['bash_wait_integer']==0 and bash['source_sha256']==P['source'] and bash['permit_sha256']==P['permit'],'Actual Bashwait differs')
for n in ['CPU_SCORE_FRESH1','CPU_SCORE_FRESH2','CPU_REPORT_FINISHER','FORMAL_CONTROLLER']:
 x=read(r/('runtime/'+n+'/OWNING_EXIT.json'));req(type(x['owning_Popen_exit']) is int and x['owning_Popen_exit']==0,'Original parent child actual exit differs')
req(read(r/'runtime/SCORE_PASS1.json')==read(r/'runtime/SCORE_PASS2.json'),'Two original score outputs differ')
cache_root=Path('/dev/shm/pace48_leap_unroll_dp8_dp16_v1/triton')
cache_records=[]
if cache_root.exists():
 for p in sorted(cache_root.rglob('*')):
  if not p.is_file():continue
  x=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
  if p.suffix=='.json':
   try:z=read(p)
   except (ValueError,UnicodeError):z={}
   if isinstance(z,dict):x['metadata']={k:z[k] for k in ['name','num_warps','num_ctas','shared','n_regs','n_spills','maxnreg'] if k in z}
  cache_records.append(x)
config={}
for x in read(r/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
 if Path(x['path']).name=='config.json':
  z=read(x['path']);z=z.get('text_config',z)
  config={k:z[k] for k in ['num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','hidden_size','layer_types','torch_dtype','dtype'] if k in z}
unchanged()
print(json.dumps(dict(status='PASS_CURRENT_ALL60_SOURCE_IDENTITY_PAYLOAD_LIFECYCLE_QUEUE_MAPS_MODEL_AND_UNUSED_V1',
 source=sourceproof,unused_V1=unused,workers=workers,regular_maps=list(maps.values()),model17=model,owning_epochs=epochs,
 actual_Bashwait=bash,all_original_child_exits0=True,compile_cache_SHA_metadata_only=cache_records,model_config=config,
 actual_per_specialization_Driver_load_trace='UNMEASURED; cache is not a load receipt; executed packed backend is bound by runtime counters and source',
 GPU_launches=0,new_scores=0,remote_mutations=0)))
'''

CURRENT29 = r'''
base=Path(P['archive']);components=[]
for m in sorted(base.rglob('ARCHIVE_MANIFEST.json')):
 if any(x.startswith('.incoming_') for x in m.parts):continue
 rows=read(m);components.append(dict(root=str(m.parent),manifest_sha256=sha(m),verification=verify(m.parent,rows)))
req(any(x['root'].endswith('/SOURCE_BOOTSTRAP') for x in components),'Current bootstrap absent')
by={x['root']:x for x in components}
for x in P['unit_archives']:
 req(x['root'] in by and by[x['root']]['manifest_sha256']==x['manifest_sha256'] and by[x['root']]['verification']['files']==x['files'] and by[x['root']]['verification']['bytes']==x['bytes'],'Current unit archive receipt differs')
main=P['main_archive'];req(by[main['root']]['manifest_sha256']==main['manifest_sha256'],'Main29 receipt differs')
unchanged();print(json.dumps(dict(status='PASS_CURRENT29_BOOTSTRAP_ALL60_UNITS_MAIN_AND_OTHER_PUBLISHED_COMPONENTS',components=components,GPU_queries=0,remote_mutations=0)))
'''

PUBLISH = r'''
import base64,ctypes,hashlib,json,os,platform
from pathlib import Path
req(platform.machine()=='x86_64','CPU publication syscall architecture differs')
base=Path(DEST);base.mkdir(parents=True,exist_ok=True)
target=base/PART;req(not target.exists(),'NOREPLACE target already exists')
stage=base/('.incoming_'+PART);stage.mkdir(exist_ok=False)
req(len(FILES)<=24 and sum(x['bytes'] for x in FILES)<=32*1024*1024,'Supplement hard bound exceeded')
for x in FILES:
 p=safe(stage,x['path']);p.parent.mkdir(parents=True,exist_ok=True)
 b=base64.b64decode(x['base64']);req(len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],'Publication payload differs')
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
for p in sorted([stage]+[p for p in stage.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
 fd=os.open(str(p),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
lib=ctypes.CDLL(None,use_errno=True);fn=lib.syscall;fn.restype=ctypes.c_long
ret=fn(ctypes.c_long(316),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(stage)),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(target)),ctypes.c_uint(1))
if ret!=0:
 err=ctypes.get_errno();raise OSError(err,os.strerror(err))
fd=os.open(str(base),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
verify(target,[{k:v for k,v in x.items() if k!='base64'} for x in FILES])
print(json.dumps(dict(status='PASS_FSYNC_ATOMIC_SYSCALL316_NOREPLACE_POST_SHA',root=str(target),part=PART,files=len(FILES),bytes=sum(x['bytes'] for x in FILES),manifest_sha256=sha(target/'MANIFEST.sha256'))))
'''

POST_READ = r'''
rows=[]
for x in RECEIPTS:
 root=Path(x['root']);req(sha(root/'MANIFEST.sha256')==x['manifest_sha256'],'Published manifest differs')
 for line in (root/'MANIFEST.sha256').read_text().splitlines():
  digest,name=line.split('  ',1);req(sha(safe(root,name))==digest,'Published bytes differ: '+name)
 rows.append(x)
unchanged();print(json.dumps(dict(status='PASS_NEW_INDEPENDENT_READER_ALL_PUBLISHED_PARTS_SHA',parts=rows)))
'''
