"""Once after controller exit: two fresh CPU scores, immutable archive29, no GPU retries."""
import os,sys,subprocess,json,traceback,statistics,csv,shlex
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
root=C.read(R/'ROOT.json');archive=root['archive_root'];host='zypan@202.38.247.29';sock='/tmp/qwen_replication_archive29.sock'
def remote(code,*args):
 cmd=['ssh','-S',sock,'-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(['/usr/bin/python3','-c',code,*args])]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=2400);C.require(p.returncode==0,'Read-only/archive CPU SSH failure:'+p.stderr);return json.loads(p.stdout)
try:
 C.atomic_new(R/'FINALIZER_STARTED.json',dict(identity=C.process_identity(os.getpid())))
 session=C.read(R/'SESSION_RESULT.json');ledger=C.read(R/'RESOURCE_LEDGER.json');ids=[x['identity'] for x in ledger['intervals']]
 C.require(not C.same_process(session['controller_identity']) and all(not C.same_process(i) for i in ids),'Own exact processes have not exited')
 nvml=C.query_gpu();C.require(not any(x['pid'] in {i['pid'] for i in ids} for x in nvml['apps']) and all(x['release']=='PASS' for x in ledger['intervals']),'Own exact GPU release unresolved')
 C.verify_sources(R);C.model_guard(R)
 import native_map_guards as G
 deps=G.verify_model_and_external(R,fresh_external=True)
 C.atomic_new(R/'analysis/FINAL_DEPENDENCY_MODEL_SOURCE_RELEASE.json',dict(status='PASS',source_count=C.read(R/'SOURCE_FREEZE.json')['count'],external_fresh_SHA=61,native_fresh_SHA=293,model_stat=17,own_exact_processes_exited=True,own_NVML_released=True,observation=nvml))
 passes=[]
 for i in range(2):
  p=R/('SCORE_PASS_%d.json'%i);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');q=subprocess.run([sys.executable,'-B',str(R/'scripts/score_cpu.py'),str(R),str(p)],env=env,capture_output=True,text=True)
  C.require(q.returncode==0,'CPU scoring failed:'+q.stderr);passes.append(C.read(p))
 C.require(passes[0]==passes[1],'Independent CPU scores differ');scores=passes[0]['scores'];n=len(scores);correct=sum(x['correct'] for x in scores)
 summary=dict(status='P1_ORIGINAL15_PLUS_GPU0_REMAINDER15_COMPLETE_30_30_AND_STOP' if session['count']==15 else 'P1_REMAINDER_PARTIAL_FAIL_CLOSED_STOP',new_legal=session['count'],preserved_original=15,combined_legal=n,correct=correct,accuracy=correct/30 if n==30 else None,full_N30=n==30,EOS=sum(x['finish_reason']=='eos' for x in scores),length=sum(x['finish_reason']=='length' for x in scores),total_tokens=sum(x['generated_token_count'] for x in scores),mean_tokens=statistics.mean(x['generated_token_count'] for x in scores),failure=session['failure'],EXACT_physical_GPUh='UNRESOLVED',cross_UUID=True,old_cancellation_preserved=True,old_failed_C1_S3_unchanged=True,limitations=['Exploratory N30 seed1; AIME previously observed','Original15 retained across GPU1/2/3; remaining15 newly run on GPU0 under separate explicit authorization','No cross-GPU bitwise identity claim; original3 interrupted partial trajectories not reused','P1 normalization and persistent QDQ code unchanged; resident C1/S3 uses different normalization batching/runtime','FP32 decoded persistent carrier is not physically packed INT8'])
 C.atomic_new(R/'P1_FINAL_SCORES.json',passes[0]);C.atomic_new(R/'FINAL_TASK_STATUS.json',summary)
 with (R/'P1_PER_QUESTION.csv').open('x',newline='') as f:
  fields=['problem_id','correct','generated_token_count','finish_reason','gpu_uuid','source','payload_sha256'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:x[k] for k in fields} for x in scores)
 with (R/'FINAL_REPORT.md').open('x') as f:f.write('# P1 GPU0 remainder\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 excluded={'finisher.log','SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json','FINALIZER_FAILURE.json'}
 files=[]
 for p in sorted(R.rglob('*')):
  C.require(not p.is_symlink(),'Scientific catalog refuses symlink:'+str(p))
  if p.is_file() and str(p.relative_to(R)) not in excluded:files.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=C.sha(p)))
 C.require(sum(x['bytes'] for x in files)<2*1024**3,'P1 archive output bound')
 C.atomic_new(R/'SHA256_MANIFEST.json',dict(task_id=C.TASK,status=summary['status'],files=files,count=len(files),total_bytes=sum(x['bytes'] for x in files)))
 (R/'SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in files))
 stage=archive+'.staging_'+C.sha(R/'SHA256_MANIFEST.json')[:16]
 remote("import os,sys,json;from pathlib import Path;f=Path(sys.argv[1]);s=Path(sys.argv[2]);assert not os.path.lexists(f) and not os.path.lexists(s);v=os.statvfs('/data');assert v.f_bavail*v.f_frsize>120*1024**3;s.mkdir(parents=True);print(json.dumps({'status':'RESERVED_NEW_STAGE'}))",archive,stage)
 p=subprocess.run(['rsync','-a','--exclude','finisher.log','-e',shlex.join(['ssh','-S',sock,'-o','BatchMode=yes']),str(R)+'/',host+':'+stage+'/'],capture_output=True,text=True,timeout=2400);C.require(p.returncode==0,'Archive CPU transfer failed:'+p.stderr)
 result=remote('''import sys,os,json,hashlib,ctypes
from pathlib import Path
s=Path(sys.argv[1]);f=Path(sys.argv[2]);m=json.loads((s/'SHA256_MANIFEST.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
for x in m['files']:
 p=s/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
v=dict(status='PASS_INDEPENDENT29_ALL_FILE_SHA_BYTES',count=m['count'],manifest_sha256=sha(s/'SHA256_MANIFEST.json'),archive_root=str(f));vb=(json.dumps(v,sort_keys=True)+'\\n').encode();(s/'FINAL_ARCHIVE_VERIFICATION.json').write_bytes(vb)
seal=dict(status='PASS_FINAL_SEAL',manifest_sha256=v['manifest_sha256'],archive_verification_sha256=hashlib.sha256(vb).hexdigest(),scientific_status=m['status']);sb=(json.dumps(seal,sort_keys=True)+'\\n').encode();(s/'FINAL_SEAL_RECEIPT.json').write_bytes(sb)
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
assert fn(-100,os.fsencode(s),-100,os.fsencode(f),1)==0,os.strerror(ctypes.get_errno());fd=os.open(f.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(verification_text=vb.decode(),seal_text=sb.decode())))
''',stage,archive)
 for name,key in [('FINAL_ARCHIVE_VERIFICATION.json','verification_text'),('FINAL_SEAL_RECEIPT.json','seal_text')]:
  with (R/name).open('x') as f:f.write(result[key])
except BaseException as exc:
 if not (R/'FINALIZER_FAILURE.json').exists():C.atomic_new(R/'FINALIZER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0))
 raise
