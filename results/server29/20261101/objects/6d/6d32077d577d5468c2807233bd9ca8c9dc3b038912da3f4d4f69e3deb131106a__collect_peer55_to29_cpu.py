"""After actual node exits, snapshot immutable55 science into29 with SHA/bytes and no-clobber publication."""
import hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
import node_rpc as RPC
C.require(C.NODE['id']=='server55_gpu0','Collector only55CPU');C.require(not(R/'GLOBAL_COLLECTION_STARTED.json').exists(),'No repeated collection/transfer')
C.require(C.read(R/'GLOBAL_COORDINATOR_ACTUAL_EXIT.json')['controller_exact_alive']is False,'Actual global coordinator must exit')
if(R/'GPU_RUN_STARTED.json').exists():C.require(not C.same_process(C.read(R/'GPU_RUN_STARTED.json')['identity']),'Node55 controller alive')
local_intervals=C.read(R/'RESOURCE_LEDGER.json')['intervals']if(R/'RESOURCE_LEDGER.json').exists()else[]
C.require(all(not C.same_process(x['identity'])and x['release']=='PASS'for x in local_intervals),'Own55 worker/release unresolved')
obs=C.query_gpu();C.require(not any(a['pid']in{x['identity']['pid']for x in local_intervals}for a in obs['apps']),'Own55 NVML not released')
remote_state=RPC.query29(RPC.state_code());C.require(not remote_state['controller_alive']and not remote_state['active'],'Node29 not exited')
started=dict(identity=C.process_identity(os.getpid()),monotonic_on55=time.monotonic(),source55_sha256=C.sha(R/'SOURCE_FREEZE.json'),source29_sha256=remote_state['source_sha256'],no_GPU_work=True)
C.atomic_new(R/'GLOBAL_COLLECTION_STARTED.json',started)
rows=[]
for p in sorted(R.rglob('*')):
    C.require(not p.is_symlink(),'Snapshot symlink forbidden')
    if not p.is_file():continue
    rel=p.relative_to(R)
    if rel.parts[0]in('monitoring','operations')or p.name in ('global_finisher.log','global_collector.log','GLOBAL_COLLECTION_STARTED.json'):continue
    rows.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=C.sha(p)))
manifest=dict(status='IMMUTABLE_SERVER55_SCIENCE_AFTER_NODE_AND_COORDINATOR_EXIT',source55_sha256=C.sha(R/'SOURCE_FREEZE.json'),source29_sha256=remote_state['source_sha256'],files=rows,count=len(rows),total_bytes=sum(x['bytes']for x in rows),models_and_environment_outside_snapshot=True)
C.atomic_new(R/'HOST55_SNAPSHOT_MANIFEST.json',manifest)
prepare=RPC.prefix()+"manifest="+repr(manifest)+r'''
stage=R/'peer55.partial';final=R/'peer55';assert not stage.exists()and not final.exists();stage.mkdir()
linked=[];missing=[]
for x in manifest['files']:
 rel=Path(x['path']);assert not rel.is_absolute()and'..'not in rel.parts
 p=R/rel;q=stage/rel
 if p.is_file()and not p.is_symlink()and p.stat().st_size==x['bytes']and C.sha(p)==x['sha256']:
  q.parent.mkdir(parents=True,exist_ok=True);os.link(p,q);linked.append(x['path'])
 else:missing.append(x['path'])
C.atomic_new(R/'GLOBAL_COLLECTION_RECEIVER_CLAIM.json',dict(status='CLAIMED_ONCE_NO_RETRY',manifest=manifest,missing=missing,linked=linked,monotonic=time.monotonic()))
print(json.dumps(dict(status='PASS_RECEIVER_PREPARED',missing=missing,linked_count=len(linked))))
'''
ready=RPC.query29(prepare,timeout=180);missing=ready['missing']
receiver=RPC.prefix()+r'''
import tarfile
stage=R/'peer55.partial';claim=C.read(R/'GLOBAL_COLLECTION_RECEIVER_CLAIM.json');names=claim['missing']
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz')as t:
 for m in t:
  rel=Path(m.name);assert m.isfile()and not rel.is_absolute()and'..'not in rel.parts and str(rel)in names
  p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:
   source=t.extractfile(m)
   while True:
    b=source.read(8*1024**2)
    if not b:break
    f.write(b)
for x in claim['manifest']['files']:
 p=stage/x['path'];assert p.is_file()and not p.is_symlink()and p.stat().st_size==x['bytes']and C.sha(p)==x['sha256'],x['path']
C.atomic_new(stage/'HOST55_SNAPSHOT_MANIFEST.json',claim['manifest'])
import ctypes
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;assert fn(-100,os.fsencode(stage),-100,os.fsencode(R/'peer55'),1)==0
print(json.dumps(dict(status='PASS_RECEIVED_PEER55_SHA_BYTES_NOREPLACE',count=len(claim['manifest']['files']))))
'''
# The receiver Python program is passed as -c; stdin carries only the tar stream.
cmd=RPC.SSH+['env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -c '+__import__('shlex').quote(receiver)]
err=(R/'operations/GLOBAL_COLLECTION_TRANSFER.stderr').open('xb');p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
with tarfile.open(fileobj=p.stdin,mode='w|gz')as t:
    for name in missing:
        source=R/name;C.require(source.is_file()and not source.is_symlink(),'Changed snapshot source');t.add(source,arcname=name,recursive=False)
p.stdin.close();stdout=p.stdout.read();code=p.wait();err.close();C.require(code==0,'Real peer55 transfer failed, preserve receiver claim; no retry')
transfer=json.loads(stdout);C.require(transfer['status']=='PASS_RECEIVED_PEER55_SHA_BYTES_NOREPLACE','Receiver did not verify')
terminal=C.read(R/'GLOBAL_SESSION_RESULT.json')
receipt=dict(status='PASS_BOTH_NODE_TERMINALS_AND_PEER55_SNAPSHOT_SHA_BYTES',snapshot55_manifest_sha256=C.sha(R/'HOST55_SNAPSHOT_MANIFEST.json'),snapshot55_files=manifest['count'],snapshot55_total_bytes=manifest['total_bytes'],node55_actual_controller_and_workers_exited=True,node29_actual_controller_and_workers_exited=True,source55_sha256=manifest['source55_sha256'],source29_sha256=manifest['source29_sha256'],global_terminal=terminal['status'],transfer=transfer,local_control_clock='server55_monotonic',monotonic=time.monotonic())
publish=RPC.prefix()+"terminal="+repr(terminal)+"\nreceipt="+repr(receipt)+"\n"+r'''
C.atomic_new(R/'GLOBAL_SESSION_RESULT.json',terminal)
if terminal['status']=='GPU_COMPLETE_CROSS_HOST_GROUP64_120_FORMAL':C.atomic_new(R/'GLOBAL_FORMAL_GENERATIONS_COMPLETE.json',dict(status='PASS_ALL120_BOTH_HOSTS_ACTUAL_EXIT_AND_RELEASE',count=120,source55_sha256=receipt['source55_sha256'],source29_sha256=receipt['source29_sha256']))
else:C.atomic_new(R/'GLOBAL_FIRST_FAILURE.json',terminal['first_observed_failure'])
C.atomic_new(R/'GLOBAL_COLLECTION_RECEIPT.json',receipt)
print(json.dumps(receipt))
'''
result=RPC.query29(publish);C.atomic_new(R/'operations/GLOBAL_COLLECTION_COMPLETED.json',result);print(json.dumps(result))
