#!/usr/bin/env python3
"""One-shot CPU-only seal and exact small-file Mac ->55 ->29 delivery.

Default is a local preview. --run is for the parent only after report review.
On any error inspect STATE.jsonl and existing remote paths; never blindly rerun.
No GPU APIs, raw archive publication, model loading, or old-root writes exist.
"""
import argparse,base64,datetime,gzip,hashlib,io,json,os,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
EXCLUDED={'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','LOCAL_DELIVERY_VERIFICATION.json'}
REQUIRED=['EXECUTION_PLAN.md','STRICT_RUNTIME_CONTRACT.md','HISTORICAL_EVIDENCE_REUSE.md','CUBLAS_CONFIG_QUALIFICATION.md','UNINITIALIZED_MEMORY_SAFETY_AUDIT.md','NUMERICAL_PARITY_RESULTS.json','PERFORMANCE_COMPARISON.csv','FAST_RUNTIME_DECISION.json','OPTIMIZATION_DIFF.md','RESOURCE_LEDGER.json','FINAL_SCIENTIFIC_REPORT.md']
AUDIT_REQUIRED=['audits/source/NUMERICAL_FAILURE_CPU_AUDIT.json','audits/source/FROZEN_SOURCE_DEPENDENCIES_POST_STOP.json','audits/resource/RESOURCE_TERMINAL_AUDIT.json','audits/receipts/FINAL_ARCHIVE_STOP_AUDIT.json','audits/final_review/REPORT_INDEPENDENT_REVIEW.json']
MAX_FILE=16*1024**2;MAX_TOTAL=512*1024**2

def require(ok,message):
 if not ok:raise RuntimeError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def encoded(x):return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def stable(p):
 p=Path(p);require(not p.is_symlink()and p.is_file(),'Not regular file:'+str(p));a=p.stat();b=p.read_bytes();z=p.stat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'File changed:'+str(p));return b
def describe(p):
 b=stable(p);return {'path':str(p),'bytes':len(b),'sha256':digest(b)}
def new_bytes(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.incoming');require(not p.exists()and not tmp.exists(),'No overwrite:'+str(p))
 with tmp.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.link(tmp,p);tmp.unlink();return describe(p)
def read(p):return json.loads(stable(p))
def roster(d,allow_external=False):
 rows=[]
 for p in sorted(Path(d).rglob('*')):
  require(not p.is_symlink(),'Delivery symlink prohibited:'+str(p))
  if not p.is_file():continue
  rel=str(p.relative_to(d));q=PurePosixPath(rel);require(not q.is_absolute()and'..'not in q.parts,'Unsafe delivery path')
  if rel in EXCLUDED and not allow_external:continue
  require(p.suffix.lower()not in {'.pt','.pth','.bin','.safetensors','.npy','.npz','.tensor'},'Raw/model/tensor prohibited:'+rel)
  require(p.stat().st_size<MAX_FILE,'Large file prohibited:'+rel)
  require(not(rel.startswith('runtime_presets/fast')and p.suffix in {'.sh','.py'}),'Unapproved Fast launcher:'+rel)
  b=stable(p);rows.append({'path':rel,'bytes':len(b),'sha256':digest(b)})
 require(sum(x['bytes']for x in rows)<MAX_TOTAL,'Small delivery exceeded bounded total');return rows

def check_payload(d,settings,with_receipt=False):
 m=read(d/'SHA256_MANIFEST.json');expected={x['path']:x for x in m['files']}
 require(len(expected)==m['files_count']and sum(x['bytes']for x in expected.values())==m['total_bytes'],'Manifest counts invalid')
 for n,k in [('SHA256_MANIFEST.json','manifest_sha256'),('SHA256SUMS.txt','SHA256SUMS_sha256'),('FINAL_SEAL_RECEIPT.json','seal_sha256')]:require(digest(stable(d/n))==settings[k],'Anchor changed:'+n)
 sums=[]
 for line in stable(d/'SHA256SUMS.txt').decode().splitlines():
  h,n=line.split('  ',1);sums.append((n,h))
 want={n:x['sha256']for n,x in expected.items()};want['SHA256_MANIFEST.json']=settings['manifest_sha256'];require(len(sums)==len(want)and len(dict(sums))==len(sums)and dict(sums)==want,'SHA256SUMS roster mismatch')
 rows=roster(d,True);actual={x['path']:x for x in rows};names=set(expected)|{'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json'}
 if with_receipt:names.add('LOCAL_DELIVERY_VERIFICATION.json')
 require(set(actual)==names,'Actual delivery roster differs')
 for name,row in expected.items():require(actual[name]==row,'Indexed bytes/SHA mismatch:'+name)
 require(len(expected)==settings['indexed_files']and m['total_bytes']==settings['indexed_bytes'],'Expected payload counts changed')
 return {'status':'PASS_EXACT_SMALL_DELIVERY_SHA_BYTES','destination':str(d),'indexed_files':len(expected),'indexed_bytes':m['total_bytes'],'actual_files':len(names),'actual_bytes':sum(x['bytes']for x in rows),'manifest_sha256':settings['manifest_sha256'],'SHA256SUMS_sha256':settings['SHA256SUMS_sha256'],'seal_sha256':settings['seal_sha256'],'payload_roster_sha256':digest(canonical(rows)),'large_model_or_tensor_in_delivery':False}

REMOTE_UPLOAD=r'''import hashlib,json,os,sys
from pathlib import Path
s=json.loads(sys.argv[1]);p=Path(s['path']);tmp=Path(str(p)+'.incoming');assert not p.exists()and not tmp.exists(),'No overwrite remote upload';p.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
with tmp.open('xb')as f:
 for b in iter(lambda:sys.stdin.buffer.read(1048576),b''):count+=len(b);assert count<=s['bytes'];h.update(b);f.write(b)
 f.flush();os.fsync(f.fileno())
assert count==s['bytes']and h.hexdigest()==s['sha256'],'Upload byte/SHA mismatch';os.link(tmp,p);tmp.unlink();z=p.stat();q=hashlib.sha256()
with p.open('rb')as f:
 for b in iter(lambda:f.read(1048576),b''):q.update(b)
a=p.stat();assert(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns)==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns);assert z.st_size==s['bytes']and q.hexdigest()==s['sha256'];print(json.dumps(dict(status='PASS_REMOTE_NEW_FILE_SHA_BYTES',path=str(p),bytes=z.st_size,sha256=q.hexdigest())))
'''
REMOTE_VERIFY=r'''import ctypes,hashlib,json,os,sys,tarfile
from pathlib import Path,PurePosixPath
s=json.load(sys.stdin);bundle=Path(s['bundle']);dest=Path(s['destination']);tmp=Path(str(dest)+'.incoming');assert not dest.exists()and not tmp.exists(),'Existing delivery: inspect, never rerun blindly'
def raw(p):
 assert p.is_file()and not p.is_symlink();a=p.stat();b=p.read_bytes();z=p.stat();assert(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns);return b
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
h=hashlib.sha256()
with bundle.open('rb')as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert bundle.stat().st_size==s['bundle_bytes']and h.hexdigest()==s['bundle_sha256'];tmp.mkdir(parents=True)
seen=set();total=0
with tarfile.open(bundle,'r:gz')as tf:
 for member in tf:
  pp=PurePosixPath(member.name);assert member.isfile()and not pp.is_absolute()and'..'not in pp.parts and member.name not in seen and member.size<16*1024**2;seen.add(member.name);total+=member.size;assert total<512*1024**2
  p=tmp/member.name;assert p.resolve().is_relative_to(tmp.resolve());p.parent.mkdir(parents=True,exist_ok=True)
  with tf.extractfile(member)as src,p.open('xb')as f:
   for b in iter(lambda:src.read(1048576),b''):f.write(b)
   f.flush();os.fsync(f.fileno())
  p.chmod(member.mode&0o777)
for n,k in [('SHA256_MANIFEST.json','manifest_sha256'),('SHA256SUMS.txt','SHA256SUMS_sha256'),('FINAL_SEAL_RECEIPT.json','seal_sha256')]:assert sha(raw(tmp/n))==s[k]
m=json.loads(raw(tmp/'SHA256_MANIFEST.json'));expected={x['path']:x for x in m['files']};assert len(expected)==m['files_count']==s['indexed_files'];assert sum(x['bytes']for x in expected.values())==m['total_bytes']==s['indexed_bytes']
rows=[]
for p in sorted(tmp.rglob('*')):
 assert not p.is_symlink()
 if p.is_file():b=raw(p);rows.append(dict(path=str(p.relative_to(tmp)),bytes=len(b),sha256=sha(b)))
actual={x['path']:x for x in rows};names=set(expected)|{'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json'};assert set(actual)==seen==names
for name,row in expected.items():assert actual[name]==row,name
pairs=[line.split('  ',1)for line in raw(tmp/'SHA256SUMS.txt').decode().splitlines()];want={n:x['sha256']for n,x in expected.items()};want['SHA256_MANIFEST.json']=s['manifest_sha256'];assert len(pairs)==len(want)and{n:h for h,n in pairs}==want
assert sha(canon(rows))==s['payload_roster_sha256'];libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None)
if rename is not None:
 rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int;rc=rename(-100,os.fsencode(tmp),-100,os.fsencode(dest),1)
else:
 assert os.uname().machine=='x86_64','No known atomic no-clobber directory publish syscall';libc.syscall.restype=ctypes.c_long;rc=libc.syscall(ctypes.c_long(316),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(tmp)),ctypes.c_int(-100),ctypes.c_char_p(os.fsencode(dest)),ctypes.c_uint(1))
if rc!=0:raise OSError(ctypes.get_errno(),'Atomic no-clobber renameat2 failed; incoming retained')
d=os.open(dest.parent,os.O_RDONLY)
try:os.fsync(d)
finally:os.close(d)
print(json.dumps(dict(status='PASS_EXACT_SMALL_DELIVERY_SHA_BYTES',destination=str(dest),indexed_files=len(expected),indexed_bytes=m['total_bytes'],actual_files=len(names),actual_bytes=sum(x['bytes']for x in rows),manifest_sha256=s['manifest_sha256'],SHA256SUMS_sha256=s['SHA256SUMS_sha256'],seal_sha256=s['seal_sha256'],payload_roster_sha256=s['payload_roster_sha256'],transfer_bundle=dict(path=str(bundle),bytes=s['bundle_bytes'],sha256=s['bundle_sha256']),atomic_no_clobber_publication=True,large_model_or_tensor_in_delivery=False)))
'''
REMOTE_RECEIPT_CHECK=r'''import hashlib,json,sys
from pathlib import Path
s=json.load(sys.stdin);d=Path(s['destination']);p=d/'LOCAL_DELIVERY_VERIFICATION.json';a=p.stat();b=p.read_bytes();z=p.stat();assert(a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns);assert len(b)==s['bytes']and hashlib.sha256(b).hexdigest()==s['sha256'];m=json.loads((d/'SHA256_MANIFEST.json').read_bytes());want={x['path']for x in m['files']}|{'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','LOCAL_DELIVERY_VERIFICATION.json'};assert{str(q.relative_to(d))for q in d.rglob('*')if q.is_file()}==want
print(json.dumps(dict(status='PASS_EXTERNAL_RECEIPT_SHA_BYTES_AND_FINAL_ROSTER',path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),actual_files=len(want))))
'''

class Transport:
 def __init__(self,root,record):self.root=root;self.record=record;self.ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',root['ssh_control_path'],root['remote_user']+'@'+root['remote_host']];self.py=root['remote_python']
 def call55(self,code,args=(),data=None):
  p=subprocess.run(self.ssh+[shlex.join([self.py,'-B','-c',code]+list(args))],input=data,capture_output=True,timeout=600)
  require(p.returncode==0,'SSH55 action failed; inspect remote state before any retry: '+p.stderr.decode(errors='replace')[:3000]);return json.loads(p.stdout)
 def call29(self,code,settings,data=b''):
  wrapper="import base64,json,shlex,subprocess,sys;v=json.load(sys.stdin);p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',v['control'],v['host'],shlex.join(['python3','-B','-c',v['code']]+v['args'])],input=base64.b64decode(v['body']),capture_output=True,timeout=600);sys.stdout.buffer.write(p.stdout);sys.stderr.buffer.write(p.stderr);sys.exit(p.returncode)"
  # Verifier/check functions use stdin JSON; uploader uses argv settings+raw stdin.
  body=data if code==REMOTE_UPLOAD else encoded(settings);args=[json.dumps(settings)]if code==REMOTE_UPLOAD else[]
  env={'control':self.root['archive_control_from55'],'host':self.root['remote_user']+'@'+self.root['archive_host'],'code':code,'args':args,'body':base64.b64encode(body).decode()};return self.call55(wrapper,data=encoded(env))
 def forward_bundle29(self,path55,path29,desc):
  settings={'path':path29,'bytes':desc['bytes'],'sha256':desc['sha256']}
  code="import json,pathlib,shlex,subprocess,sys;v=json.load(sys.stdin);p=pathlib.Path(v['source']);assert p.stat().st_size==v['settings']['bytes'];p29=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',v['control'],v['host'],shlex.join(['python3','-B','-c',v['code'],json.dumps(v['settings'])])],input=p.read_bytes(),capture_output=True,timeout=600);sys.stdout.buffer.write(p29.stdout);sys.stderr.buffer.write(p29.stderr);sys.exit(p29.returncode)"
  return self.call55(code,data=encoded({'source':path55,'settings':settings,'control':self.root['archive_control_from55'],'host':self.root['remote_user']+'@'+self.root['archive_host'],'code':REMOTE_UPLOAD}))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--audit-bindings',type=Path);args=p.parse_args();r=args.root.resolve();d=r/'delivery';root=read(r/'ROOT.json');rows=roster(d);require(all((d/n).is_file()for n in REQUIRED+AUDIT_REQUIRED),'Required reports/audits missing')
 review=read(d/'audits/final_review/REPORT_INDEPENDENT_REVIEW.json');require(str(review.get('status','')).startswith('PASS'),'Independent report review not PASS')
 archive=read(d/'audits/receipts/FINAL_ARCHIVE_STOP_AUDIT.json');require(archive['status']=='PASS_STORAGE_BYTES_AND_SMALL_DELIVERY_ONLY','Final raw archive audit missing');decision=read(d/'FAST_RUNTIME_DECISION.json');require(decision.get('execution_status')=='FAIL_CLOSED_STOP'and decision.get('status')in {'FAIL_CLOSED_STOP','FAST_CANDIDATE_NUMERICAL_FAIL'},'Expected true STOP delivery with unqualified or numerical-failed candidate')
 require(not any((d/n).exists()for n in EXCLUDED),'Existing seal/receipt: inspect state; do not reseal/overwrite')
 preview={'status':'PREPARED_NOT_EXECUTED','root':str(r),'indexed_files':len(rows),'indexed_bytes':sum(x['bytes']for x in rows),'actual_files_after_seal':len(rows)+3,'all_required_reports_and_audits_present':True,'run_requested':args.run,'no_GPU_calls':True}
 if not args.run:print(json.dumps(preview));return
 receipt_dir=r/'independent_delivery/receipts';receipt_dir.mkdir(parents=True,exist_ok=True);state=receipt_dir/'DELIVERY_TRANSFER_STATE.jsonl';require(not state.exists(),'One-shot transfer state exists: inspect; no blind retry')
 def record(event,**data):
  with state.open('a')as f:f.write(json.dumps(dict(event=event,unix=time.time(),**data),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 def save(name,value):return new_bytes(receipt_dir/name,encoded(value))
 record('START',**preview)
 try:
  manifest={'task_id':root['task_id'],'execution_run_id':root['execution_run_id'],'status':'SEALED_FAIL_CLOSED_STOP_REPORT_AND_EVIDENCE','science_status':'INCOMPLETE_OR_ENGINEERING_STOP','files_count':len(rows),'total_bytes':sum(x['bytes']for x in rows),'large_model_or_tensor_downloaded':False,'excluded_external_anchors':sorted(EXCLUDED),'files':rows};new_bytes(d/'SHA256_MANIFEST.json',encoded(manifest));mh=digest(stable(d/'SHA256_MANIFEST.json'));sums=''.join(x['sha256']+'  '+x['path']+'\n'for x in rows)+mh+'  SHA256_MANIFEST.json\n';new_bytes(d/'SHA256SUMS.txt',sums.encode())
  anchors=[x for x in rows if x['path'].startswith('audits/')and x['path'].endswith('.json')];freezes=[]
  for q in [r/'SOURCE_FREEZE.json',r/'staging/payload_v1/SOURCE_FREEZE.json']:
   if q.is_file():freezes.append(describe(q))
  require(freezes,'Frozen execution SOURCE_FREEZE external anchor missing')
  frozen_control_payload={}
  for name,key in [('CONTROL_SOURCE_MANIFEST.json','control_manifest_sha256'),('PAYLOAD_BUILD_MANIFEST.json','payload_build_manifest_sha256')]:
   candidates=[q for q in [r/name,r/'staging/payload_v1'/name]if q.is_file()];require(candidates,'Frozen execution anchor missing:'+name);descriptions=[describe(q)for q in candidates];require(all(q['sha256']==root[key]for q in descriptions),'Frozen execution anchor differs from ROOT:'+name);frozen_control_payload[name]=descriptions
  additional=read(args.audit_bindings)if args.audit_bindings else None
  seal={'task_id':root['task_id'],'execution_run_id':root['execution_run_id'],'status':'SEALED_FAIL_CLOSED_STOP','science_status':'INCOMPLETE_OR_ENGINEERING_STOP','sealed_utc_local_host':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_verified':len(rows),'bytes_verified':manifest['total_bytes'],'manifest_sha256':mh,'SHA256SUMS_sha256':digest(stable(d/'SHA256SUMS.txt')),'FINAL_SCIENTIFIC_REPORT_sha256':digest(stable(d/'FINAL_SCIENTIFIC_REPORT.md')),'archive_audit_sha256':digest(stable(d/'audits/receipts/FINAL_ARCHIVE_STOP_AUDIT.json')),'review_sha256':digest(stable(d/'audits/final_review/REPORT_INDEPENDENT_REVIEW.json')),'audit_anchors':anchors,'frozen_execution_anchors':{'control_manifest_sha256':root['control_manifest_sha256'],'payload_build_manifest_sha256':root['payload_build_manifest_sha256'],'SOURCE_FREEZE':freezes,'actual_control_and_payload_files':frozen_control_payload},'additional_external_audit_bindings':additional,'GPU_time_limit':None,'failed_GPU_unit_retried':False,'Fast_launcher_created':False,'local_external_delivery_verification_pending':True,'external_anchor_excluded_from_manifest':True};new_bytes(d/'FINAL_SEAL_RECEIPT.json',encoded(seal))
  settings={'indexed_files':len(rows),'indexed_bytes':manifest['total_bytes'],'manifest_sha256':mh,'SHA256SUMS_sha256':seal['SHA256SUMS_sha256'],'seal_sha256':digest(stable(d/'FINAL_SEAL_RECEIPT.json'))};local=check_payload(d,settings);settings['payload_roster_sha256']=local['payload_roster_sha256'];save('DELIVERY_LOCAL_BASE_VERIFICATION.json',local);record('LOCAL_SEAL_VERIFIED',**settings)
  bundle=r/'independent_delivery/final_small_delivery.tar.gz';require(not bundle.exists(),'Bundle already exists')
  with bundle.open('xb')as output,gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0)as gz,tarfile.open(fileobj=gz,mode='w')as tf:
   for row in roster(d,True):
    q=d/row['path'];raw=stable(q);require(digest(raw)==row['sha256'],'File changed while bundling');info=tarfile.TarInfo(row['path']);info.size=len(raw);info.mode=q.stat().st_mode&0o777;info.mtime=0;info.uid=info.gid=0;tf.addfile(info,io.BytesIO(raw))
  bundle_desc=describe(bundle);record('BUNDLE_READY',bundle=bundle_desc);transport=Transport(root,record);remote55=root['remote_root']+'/independent_delivery/final_delivery';bundle55=root['remote_root']+'/independent_delivery/final_small_delivery.tar.gz';remote29=root['archive_root']+'/STOP_DELIVERY';bundle29=root['archive_root']+'/STOP_DELIVERY_SOURCE_AND_REPORTS.tar.gz';base=dict(settings,bundle_bytes=bundle_desc['bytes'],bundle_sha256=bundle_desc['sha256'])
  record('UPLOAD55_STARTED',path=bundle55);up55=transport.call55(REMOTE_UPLOAD,[json.dumps({'path':bundle55,'bytes':bundle_desc['bytes'],'sha256':bundle_desc['sha256']})],stable(bundle));save('DELIVERY_SERVER55_UPLOAD.json',up55);record('UPLOAD55_PASS')
  verify55=transport.call55(REMOTE_VERIFY,data=encoded(dict(base,bundle=bundle55,destination=remote55)));require(verify55['payload_roster_sha256']==local['payload_roster_sha256'],'55 payload differs');save('DELIVERY_SERVER55_VERIFICATION.json',verify55);record('SERVER55_VERIFIED')
  record('FORWARD29_STARTED',path=bundle29);up29=transport.forward_bundle29(bundle55,bundle29,bundle_desc);save('DELIVERY_ARCHIVE29_UPLOAD.json',up29);record('FORWARD29_PASS')
  verify29=transport.call29(REMOTE_VERIFY,dict(base,bundle=bundle29,destination=remote29));require(verify29['payload_roster_sha256']==local['payload_roster_sha256'],'29 payload differs');save('DELIVERY_ARCHIVE29_VERIFICATION.json',verify29);record('ARCHIVE29_VERIFIED')
  local_again=check_payload(d,settings);require(local_again['payload_roster_sha256']==local['payload_roster_sha256'],'Mac payload changed during transfer')
  external={'task_id':root['task_id'],'execution_run_id':root['execution_run_id'],'status':'PASS_VERIFIED_FAIL_CLOSED_STOP_DELIVERY','science_status':'INCOMPLETE_OR_ENGINEERING_STOP','verified_utc_local_host':datetime.datetime.now(datetime.timezone.utc).isoformat(),'indexed_files':len(rows),'indexed_bytes':manifest['total_bytes'],'actual_payload_files_before_external_receipt':len(rows)+3,'actual_files_including_external_receipt':len(rows)+4,'manifest_sha256':mh,'SHA256SUMS_sha256':settings['SHA256SUMS_sha256'],'FINAL_SEAL_RECEIPT_sha256':settings['seal_sha256'],'FINAL_SCIENTIFIC_REPORT_sha256':seal['FINAL_SCIENTIFIC_REPORT_sha256'],'archive_audit_sha256':seal['archive_audit_sha256'],'review_sha256':seal['review_sha256'],'all_three_hosts_payload_exact_same_SHA_bytes_and_roster':True,'payload_roster_sha256':local['payload_roster_sha256'],'local_verification':local_again,'server55_verification':verify55,'archive29_verification':verify29,'transfer_bundle':bundle_desc,'external_receipt_excluded_from_manifest_by_design':True,'seal_pending_external_verification_resolved_by_this_receipt':True,'external_receipt_replication_proof_separate':'independent_delivery/receipts/EXTERNAL_RECEIPT_REPLICATION.json','no_GPU_calls':True,'no_archive_catalog_republication':True,'no_large_raw_or_model_transfer':True};ext=new_bytes(d/'LOCAL_DELIVERY_VERIFICATION.json',encoded(external));record('EXTERNAL_RECEIPT_CREATED',receipt=ext)
  body=stable(d/'LOCAL_DELIVERY_VERIFICATION.json');transport.call55(REMOTE_UPLOAD,[json.dumps({'path':remote55+'/LOCAL_DELIVERY_VERIFICATION.json','bytes':ext['bytes'],'sha256':ext['sha256']})],body);transport.call29(REMOTE_UPLOAD,{'path':remote29+'/LOCAL_DELIVERY_VERIFICATION.json','bytes':ext['bytes'],'sha256':ext['sha256']},body)
  rec55=transport.call55(REMOTE_RECEIPT_CHECK,data=encoded({'destination':remote55,'bytes':ext['bytes'],'sha256':ext['sha256']}));rec29=transport.call29(REMOTE_RECEIPT_CHECK,{'destination':remote29,'bytes':ext['bytes'],'sha256':ext['sha256']});final_local=check_payload(d,settings,True);require(digest(stable(d/'LOCAL_DELIVERY_VERIFICATION.json'))==ext['sha256'],'Mac external receipt changed');replication={'status':'PASS_EXTERNAL_RECEIPT_REPLICATED_EXACT_ALL_THREE_HOSTS','local':ext,'server55':rec55,'archive29':rec29,'actual_files_all_hosts':len(rows)+4,'no_hash_cycle':True};proof=save('EXTERNAL_RECEIPT_REPLICATION.json',replication);record('COMPLETE',external_receipt=ext,replication_proof=proof);print(json.dumps({'status':'PASS_VERIFIED_FAIL_CLOSED_STOP_DELIVERY','manifest_sha256':mh,'indexed_files':len(rows),'indexed_bytes':manifest['total_bytes'],'LOCAL_DELIVERY_VERIFICATION':ext,'external_receipt_replication':proof,'no_GPU_calls':True}))
 except BaseException as exc:
  record('FAILED_INSPECT_STATE_NO_BLIND_RETRY',error=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
