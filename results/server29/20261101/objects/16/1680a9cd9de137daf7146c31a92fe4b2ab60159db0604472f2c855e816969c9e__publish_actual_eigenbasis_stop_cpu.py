"""CPU-only legal60 scoring or honest STOP, followed by independent no-clobber29 seal."""
import sys,os,json,subprocess,shlex,time,traceback,csv,statistics
from pathlib import Path
R=Path(sys.argv[1]);sys.path.insert(0,str(R/'src'));import execution_control as C
ARCHIVE=C.read(R/'ROOT.json')['archive_root'];SOCKET='/tmp/qwen_replication_archive29.sock';HOST='zypan@202.38.247.29'
def remote(code,*args):
    p=subprocess.run(['ssh','-S',SOCKET,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,shlex.join(['/usr/bin/python3','-u','-B','-c',code,*args])],capture_output=True,timeout=300)
    C.require(p.returncode==0,'Archive CPU operation failed:'+p.stderr.decode());return json.loads(p.stdout)
def dump(name,obj):C.atomic_new(R/name,obj)
def write(name,text):
    with (R/name).open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
def cpu_pass(program,name):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',USE_TORCH='0',USE_TF='0')
    p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts'/program),str(R),str(R/name)],env=env,capture_output=True,timeout=1200)
    (R/(name+'.log')).write_bytes(p.stdout+p.stderr);C.require(p.returncode==0,'Independent CPU scoring failed:'+name);return C.read(R/name)
C.require(not (R/'FINALIZER_STARTED.json').exists(),'STOP finalizer already started; no duplicate publication')
dump('FINALIZER_STARTED.json',dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic(),scope='CPU_ONLY_QUERY_EIGENBASIS_FAIL_CLOSED_PUBLISHER'))
try:
 C.verify_sources(R);C.model_guard(R);C.require(C.read(R/'FIRST_FAILURE.json')['status']=='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP','Not the original actual first failure')
 audit=C.read(R/'calibration/INDEPENDENT_FIRST_EIGENBASIS_FAILURE_AUDIT.json');C.require(not audit['own_alive'] and not audit['own_NVML'] and not audit['source_errors'],'Independent identity/source audit failed')
 ids=[C.read(R/'CALIBRATION_RUN_STARTED.json')['identity'],C.read(R/'calibration/execution/STARTED.json')['identity']];obs=C.query_gpu();C.require(all(not C.same_process(i) for i in ids) and not any(x['pid'] in {i['pid'] for i in ids} for x in obs['apps']),'Own actual process/NVML not released')
 interval=C.read(R/'calibration/execution/RESOURCE_INTERVAL.json');C.require(interval['exit_returncode']==0 and interval['release']=='PASS','Calibration GPU exit/release failed')
 external=C.verify_external(R)
 dump('analysis/FINAL_SOURCE_DEPENDENCY_MODEL_RELEASE_AUDIT.json',dict(status='PASS',source_count=C.read(R/'SOURCE_FREEZE.json')['count'],external_fresh_SHA_files=external,model_stat_files=17,own_calibration_identities_exited=True,own_NVML_released=True,observation=obs))
 dump('RESOURCE_LEDGER.json',dict(resource_policy='NO_GPU_TIME_BUDGET',intervals=[interval],completed_units=['QUERY_CALIBRATION_4096'],qualification_91_and_formal90_dispatches=0,EXACT_physical_GPUh='UNRESOLVED'))
 dump('SESSION_RESULT.json',dict(status='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP',completed=['QUERY_CALIBRATION_4096'],formal_completed=0,new_rotation_GPU_qualification_dispatched=0,own_all_UUID_release=True))
 dump('FORMAL_PROGRESS.json',dict(task_id=C.TASK,completed=0,total=90,dispatched=0,status='NOT_STARTED_DUE_TO_ACTUAL_QUERY_EIGENBASIS_FAILURE',partial=True))
 tensors=[]
 for p in sorted((R/'calibration/rotations').glob('*.npz')):tensors.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=C.sha(p)))
 dump('U_EIGENBASIS_PARTIAL_NOT_QUALIFIED_CATALOG.json',dict(status='PARTIAL_UNQUALIFIED_DIAGNOSTIC_ONLY_NOT_DEPLOYED',completed_layer_files=tensors,shared_U_seal_created=False,not_accepted_for_inference=True,first_failing_case=audit['first_failing_case']))
 dump('UHP_ROTATION_MATH_AUDIT.json',dict(status='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP',first_failure=C.read(R/'FIRST_FAILURE.json'),independent_first_failure=audit['first_failing_case'],tested_layer_heads_before_first_failure=audit['tested_layer_heads'],full_768_head_qualification=False,all_layer_shared_U_seal=False,FP_model_rotation_test_executed=False,thresholds_relaxed=False,definition_adjusted=False))
 cpu_sources=[]
 for n in ['scripts/build_eigenbasis_cpu.py','scripts/rotation_cpu_qualification.py','src/rotation_codec.py','src/rotation_adapter.py']:
  p=R/n;cpu_sources.append(dict(path=n,bytes=p.stat().st_size,sha256=C.sha(p)))
 dump('analysis/CPU_STAGE_ACTUAL_ADDED_SOURCE_IDENTITY.json',dict(status='ACTUAL_EXECUTED_CPU_STAGE_AND_STAGED_ROTATION_SOURCES',files=cpu_sources,original_GPU_phaseA_SOURCE_FREEZE_unchanged=True,explanation='These files were added before CPU controls/eigensolve. No dispatched GPU calibration source was replaced. Production rotation runtime was never dispatched.'))
 failure=audit['first_failing_case'];status='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP';counts={'P1':0,'C1':0,'S3':0}
 report='# Qwen/GDN OSCAR-style Sym32\uff1a\u771f\u5b9e\u8d44\u683c\u505c\u6b62\u4ea4\u4ed8\n\n\u72b6\u6001\uff1a'+status+'\u3002\u65b0\u589eAIME\u5b9e\u96450/90\uff0cP1/C1/S3\u5747\u672a\u6d3e\u53d1\uff0c\u672a\u4ea7\u751f\u672c\u4efb\u52a1accuracy\u3001\u8bc4\u5206\u8868\u6216\u79d1\u5b66CSV\uff1bP0\u6ca1\u6709\u91cd\u8dd1\u3002\n\n\u552f\u4e00GPU\u9636\u6bb5\u4e3a4096-token\u975eAIME Native-FP Query\u91c7\u96c6\uff1a\u56db\u7bc7\u51bb\u7ed3RepoBench\u6587\u6863\u54041024token\uff0c24\u5c42\u3001\u6bcf\u5c4232\u4e2aactual recurrent head\u54044096\u4e2a\u672a\u4e2d\u5fc3\u5316Query\u4e8c\u9636\u77e9\u6837\u672c\uff0cCPU FP64\u6309\u56fa\u5b9a\u6587\u6863/Token\u987a\u5e8f\u7d2f\u8ba1\u3002\u5b9e\u964516\u4e2aKey/Query head\u5728\u9012\u5f52\u524drepeat_interleave2\u523032\u4e2arecurrent/value head\uff1b\u76f8\u90bb\u5171\u4eabQuery\u5b57\u8282\u5b8c\u5168\u4e00\u81f4\uff0c\u4e0d\u8de8\u5c42Head\u805a\u5408\uff0c\u6837\u672c\u65f6\u95f4\u76f8\u5173\u6027\u4e0d\u5f53IID\uff0c\u6e90\u9879\u76ee\u72ec\u7acb\u6027\u672a\u77e5\u3002\n\n\u9996\u6b21\u5931\u8d25\u4e3a\u7b2c'+str(failure['layer'])+'\u5c42\u3001\u7b2c'+str(failure['recurrent_head'])+'\u4e2arecurrent head\uff08\u96f6\u8d77\u70b9\uff09\u3002FP32\u90e8\u7f72A=U@H\u7684Query\u654f\u611f\u6027\u5bf9\u89d2\u5747\u5300\u6027\u6700\u5927\u76f8\u5bf9\u8bef\u5dee'+str(failure['actual_checks']['FP32_query_diagonal_uniform_relative_max'])+'\uff0c\u9884\u5148\u51bb\u7ed3\u4e0a\u96502e-6\uff0c\u8d85\u9650\u7ea61.46%\uff1bA\u6b63\u4ea4\u6027\u8bef\u5dee'+str(failure['actual_checks']['FP32_A_orthogonality_max_abs'])+'\u4ecd\u57282e-6\u4ee5\u5185\u3002FP64\u7279\u5f81\u5206\u89e3\u76f8\u5bf9\u6b8b\u5dee'+str(failure['FP64_eigendecomposition_relative'])+'\u3001FP64 U\u6b63\u4ea4\u8bef\u5dee'+str(failure['FP64_U_orthogonality'])+'\u3001FP64\u7406\u60f3A\u5bf9\u89d2\u5747\u5300\u8bef\u5dee'+str(failure['FP64_A_diagonal_uniform_relative_max'])+'\u3002\n\n\u8be5\u8bc1\u636e\u5b9a\u4f4d\u5230\u9884\u5148\u7ea6\u5b9a\u7684FP32\u90e8\u7f72\u7cbe\u5ea6\u68c0\u67e5\uff0c\u4e0d\u80fd\u636e\u6b64\u5ba3\u5e03OSCAR\u65cb\u8f6c\u6570\u5b66\u9519\u8bef\u6216\u51c6\u786e\u7387\u5931\u8d25\u30022e-6\u662f\u672c\u8f6e\u6267\u884c\u524d\u51bb\u7ed3\u7684\u5de5\u7a0b\u9608\u503c\uff0c\u4e0d\u662fAIME\u5b9e\u6d4b\u6548\u5e94\u754c\uff1b\u6ca1\u6709\u770b\u5230AIME\u540e\u8c03\u9608\u503c\uff0c\u4e5f\u6ca1\u6709\u91cd\u65b0\u6821\u51c6\u3001\u91cd\u65b0\u62df\u5408\u9009\u77e9\u9635\u3001\u6269\u5927\u5bb9\u5dee\u6216patch-and-retry\u3002\u672c\u6b21\u53ea\u80fd\u62a5\u544a\u5176\u672a\u8fbe\u5230\u672c\u8f6e\u65e2\u5b9a\u8d44\u683c\u5408\u540c\u3002\n\n\u72ec\u7acbCPU\u8bca\u65ad\u4ee5\u540c\u4e00\u51bb\u7ed3Cq\u548c\u539f\u59cbbasis\u51fd\u6570\u53ea\u8bfb\u91cd\u73b0\u540c\u4e00\u9996\u6b21\u5931\u8d25\uff0c\u4e0d\u6784\u9020\u65b0\u7684\u5019\u9009\u77e9\u9635\u3001\u4e0d\u6267\u884cGPU\u3002\u539fCq\u3001\u9996\u6b21\u5931\u8d25\u5b8c\u6574Cq/U/A\u8bca\u65adTensor\u3001\u5df2\u5199\u5165\u768410\u4e2a\u672a\u5b8c\u6210U\u5c42\u6587\u4ef6\u5168\u90e8\u4fdd\u7559\u670d\u52a1\u5668\u5e76\u9010\u6587\u4ef6SHA\u5c01\u5b58\uff1b\u8fd9\u4e9b\u90e8\u5206\u77e9\u9635\u6ca1\u6709U\u5408\u683cseal\uff0c\u672a\u90e8\u7f72\u5230\u63a8\u7406\u3002\u539f\u59cb\u5f02\u5e38\u3001\u6e90\u7801\u3001\u56fa\u5b9a\u9608\u503c\u3001\u8f93\u5165\u8eab\u4efd\u548cpartial\u72b6\u6001\u4fdd\u7559\u3002\n\nCPU PBR/\u539fSym32\u72ec\u7acb80\u9879\u5b57\u6bb5\u5b57\u8282\u68c0\u67e5\u3001Identity U/P\u3001Group128\u6392\u5217\u8d1f\u63a7\u548c64-token FP64\u72ec\u7acb\u65cb\u8f6c\u9012\u63a8\u901a\u8fc7\uff1b\u8fd9\u4e9b\u662fCPU\u6570\u5b66\u8bc1\u636e\uff0c\u4e0d\u80fd\u5192\u5145\u672c\u8f6e\u4e09GPU\u77ed\u7a0b\u8d44\u683c\u3001\u771f\u5b9e9B\u65cb\u8f6cFP\u7b49\u4ef7\u6216AIME\u8bc1\u660e\u3002\u540e\u7eedFP\u7b49\u4ef7\u300191\u5355\u4f4dGPU\u8d44\u683c\u300190\u6761\u6b63\u5f0f\u3001FrozenV4\u8bc4\u5206\u548c\u4ea4\u4e92\u6548\u5e94\u5747\u672a\u8fd0\u884c\uff0c\u56e0\u6b64\u4e0d\u751f\u6210\u7a7a\u7ed3\u679c\u6587\u4ef6\u3002\n\n\u4ec5GPU1\u4e0a\u7684\u4e00\u6b21Query\u91c7\u96c6\uff0c\u771f\u5b9e\u8fdb\u7a0b\u5df2\u9000\u51fa\u4e14NVML\u91ca\u653e\uff1bGPU2/3\u672a\u6267\u884c\u672c\u4efb\u52a1GPU\u5de5\u4f5c\uff0c\u65e7GPU0\u4efb\u52a1\u4ece\u672a\u88ab\u505c\u6b62/\u62a2\u5360/\u8fc1\u79fb\u3002\u65f6\u95f4\u533a\u95f4\u89c1RESOURCE_LEDGER\u548cappend-onlyGPU_RESOURCE_LEDGER\uff0c\u6821\u51c6\u5360\u7528\u89c2\u6d4b\u4e0b\u754c\u7ea6501.54\u79d2\u3001\u4efb\u52a1\u4fdd\u7559\u4e0a\u754c\u7ea6642.53\u79d2\uff08\u5305\u62ecCPU\u51c6\u5165/\u52a0\u8f7d\u7b49\uff09\uff0cEXACT\u7269\u7406GPUh\u4ecdUNRESOLVED\u3002\u6a21\u578b\u6743\u91cd\u4e0d\u6539\uff0c17\u6a21\u578b/Tokenizer\u6587\u4ef6freshSHA\u4e0e61\u5b9e\u9645\u5916\u90e8\u4f9d\u8d56\u5df2\u6838\uff1b\u539f\u5386\u53f2\u79d1\u7814\u8d44\u4ea7\u53ea\u8bfb\u672a\u6539\u5199\u3002\n'
 write('FINAL_SCIENTIFIC_REPORT.md',report)
 dump('FINAL_TASK_STATUS.json',dict(status=status,completed=counts,formal_completed=0,formal_total=90,new_GPU_qualification_dispatched=0,own_GPU_processes_released=True,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),original_failure_preserved=True))
 # Freeze all actual files except live finisher output; no raw model is under R.
 rows=[]
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.is_symlink() or str(p.relative_to(R)) in ['finisher.log','SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','FINAL_ARCHIVE_VERIFICATION.json','FINALIZER_FAILURE.json']:continue
  rows.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=C.sha(p)))
 C.require(sum(x['bytes'] for x in rows)<=20*1024**3,'New archive exceeds20GiB bound')
 dump('SHA256_MANIFEST.json',dict(task_id=C.TASK,status=status,files=rows,count=len(rows),total_bytes=sum(x['bytes'] for x in rows)))
 write('SHA256SUMS.txt',''.join(x['sha256']+'  '+x['path']+'\n' for x in rows))
 stage=ARCHIVE+'.staging_'+C.sha(R/'SHA256_MANIFEST.json')[:16]
 remote('''import sys,json,os
from pathlib import Path
final=Path(sys.argv[1]);stage=Path(sys.argv[2]);assert not final.exists() and not stage.exists(),'Archive publication/staging already exists; inspect, no duplicate'
s=os.statvfs('/data/zypan');assert s.f_bavail*s.f_frsize>=120*1024**3,'Archive reservation+headroom insufficient';stage.mkdir(parents=True)
print(json.dumps(dict(status='NEW_STAGE_RESERVED')))
''',ARCHIVE,stage)
 sshcmd=shlex.join(['ssh','-S',SOCKET,'-o','BatchMode=yes']);p=subprocess.run(['rsync','-a','--exclude','finisher.log','-e',sshcmd,str(R)+'/',HOST+':'+stage+'/'],capture_output=True,timeout=2400);C.require(p.returncode==0,'Archive transfer failure:'+p.stderr.decode())
 verification=remote('''import sys,json,hashlib,os,ctypes
from pathlib import Path
stage=Path(sys.argv[1]);final=Path(sys.argv[2]);manifest=json.loads((stage/'SHA256_MANIFEST.json').read_text());count=0
for x in manifest['files']:
 p=stage/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==x['sha256'],'Independent archived SHA mismatch:'+x['path'];count+=1
v=dict(status='PASS_INDEPENDENT29_ALL_FILE_SHA_BYTES_ATOMIC_SEAL',count=count,total_bytes=manifest['total_bytes'],manifest_sha256=hashlib.sha256((stage/'SHA256_MANIFEST.json').read_bytes()).hexdigest(),archive_root=str(final))
(stage/'FINAL_ARCHIVE_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\\n')
seal=dict(status='PASS_FINAL_SEAL',manifest_sha256=v['manifest_sha256'],archive_verification_sha256=hashlib.sha256((stage/'FINAL_ARCHIVE_VERIFICATION.json').read_bytes()).hexdigest(),scientific_status=manifest['status'])
(stage/'FINAL_SEAL_RECEIPT.json').write_text(json.dumps(seal,indent=2)+'\\n')
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;rc=fn(-100,os.fsencode(stage),-100,os.fsencode(final),1);assert rc==0,os.strerror(ctypes.get_errno());fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(verification=v,seal=seal,verification_text=(final/'FINAL_ARCHIVE_VERIFICATION.json').read_text(),seal_text=(final/'FINAL_SEAL_RECEIPT.json').read_text())))
''',stage,ARCHIVE)
 write('FINAL_ARCHIVE_VERIFICATION.json',verification['verification_text']);write('FINAL_SEAL_RECEIPT.json',verification['seal_text'])
except BaseException as exc:
 if not (R/'FINALIZER_FAILURE.json').exists():dump('FINALIZER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0))
 raise
