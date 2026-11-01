#!/usr/bin/env python3
"""Local synthetic CPU tests only; fake transport never opens SSH or a GPU."""
import copy,hashlib,importlib.util,json,os,sys,tarfile,tempfile
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import seal_cpu_delivery as C
checks=[]
def ok(name,value=True):assert value,name;checks.append(dict(name=name,status='PASS'))
def reject(name,fn):
 try:fn()
 except RuntimeError:ok(name);return
 raise AssertionError(name+' unexpectedly accepted')
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(C.encoded(x))
def fixture(base):
 root=base/'root';root.mkdir();state=dict(task_id=C.TASK,status=C.STATUS,ALLOW_GPU_RESTORE='NO',GPU_queries=0,GPU_dispatches=0,restoration_mode=C.MODE,remote_root='/synthetic55/task',archive_root='/synthetic29/task',remote_host='192.0.2.55',archive_host='192.0.2.29',remote_user='synthetic',ssh_control_path='/synthetic/control55',archive_control_from55='/synthetic/control29',remote_python='/synthetic/python')
 for name in C.REQUIRED:
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic CPU-only evidence\n')
 save(root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json',dict(task_id=C.TASK,ALLOW_GPU_RESTORE='NO',GPU_dispatches=0));save(root/'GPU_RESOURCE_LEDGER.json',dict(task_id=C.TASK,ALLOW_GPU_RESTORE='NO',GPU_queries=0,GPU_dispatches=0,actual_GPU_worker_seconds=0,actual_GPU_hours_for_new_dispatches=0,first_dispatch_events=[],first_exit_events=[],first_release_events=[]));save(root/'RESTORATION_MODE.json',dict(mode=C.MODE));save(root/'RUNTIME_RESTORATION_DECISION.json',dict(status=C.STATUS,restoration_mode=C.MODE));save(root/'analysis/final_review/FINAL_REPORT_REVIEW.json',dict(status='PASS_SYNTHETIC_SCOPE',checks_passed=25,checks_total=25,findings=[],receipts=[dict(path='FINAL_REPORT.md',bytes=(root/'FINAL_REPORT.md').stat().st_size,sha256=C.digest((root/'FINAL_REPORT.md').read_bytes()))]))
 for name in ['qualification/REMOTE_ISOLATED_CPU_CHECK.json','qualification/ACTUAL_CPU_ENTRY_CHECKS.json','analysis/HISTORICAL_UNCHANGED_VERIFICATION.json']:save(root/name,dict(status='PASS_SYNTHETIC_CPU_PROVENANCE'))
 for name in ['analysis/final_review/CONTROLLER_WORKER_INTEGRATION_REVIEW.json','analysis/environment/CONTROLLER_IMPLEMENTATION_REVIEW.json','analysis/protocol/WORKER_PREPARATION_REVIEW.json']:save(root/name,dict(status='PASS_SYNTHETIC_CPU_NOT_GPU'))
 raw=(root/'USER_REQUEST.txt').read_bytes();save(root/'SOURCE_FREEZE.json',dict(task_id=C.TASK,ALLOW_GPU_RESTORE='NO',files=[dict(path='USER_REQUEST.txt',bytes=len(raw),sha256=C.digest(raw))]));state['source_freeze_sha256']=C.digest((root/'SOURCE_FREEZE.json').read_bytes());save(root/'ROOT.json',state);return root,state
class FakeTransport:
 base=None
 def __init__(self,root,record):self.root=root;self.record=record;self.settings={}
 def path(self,host,path):return self.base/host/str(path).lstrip('/')
 def action(self,host,code,args=(),data=None,settings=None):
  if code==C.P.REMOTE_UPLOAD:
   s=settings if settings is not None else json.loads(args[0]);p=self.path(host,s['path']);C.require(C.digest(data)==s['sha256']and len(data)==s['bytes'],'Fake incoming SHA mismatch');desc=C.new_bytes(p,data);return dict(status='PASS_REMOTE_NEW_FILE_SHA_BYTES',path=s['path'],bytes=desc['bytes'],sha256=desc['sha256'])
  s=settings if settings is not None else json.loads(data)
  if code==C.P.REMOTE_VERIFY:
   bundle=self.path(host,s['bundle']);dest=self.path(host,s['destination']);C.require(not dest.exists(),'Fake no-clobber');C.require(C.digest(bundle.read_bytes())==s['bundle_sha256']and bundle.stat().st_size==s['bundle_bytes'],'Fake bundle identity');dest.mkdir(parents=True)
   with tarfile.open(bundle,'r:gz')as tf:
    for item in tf:
     C.require(item.isfile()and not Path(item.name).is_absolute()and'..'not in Path(item.name).parts,'Fake safe tar');C.new_bytes(dest/item.name,tf.extractfile(item).read())
   self.settings[(host,s['destination'])]=s;return C.check_payload(dest,s)
  if code==C.P.REMOTE_RECEIPT_CHECK:
   dest=self.path(host,s['destination']);p=dest/'LOCAL_DELIVERY_VERIFICATION.json';C.require(p.stat().st_size==s['bytes']and C.digest(p.read_bytes())==s['sha256'],'Fake receipt identity');result=C.check_payload(dest,self.settings[(host,s['destination'])],True);return dict(status='PASS_EXTERNAL_RECEIPT_SHA_BYTES_AND_FINAL_ROSTER',path=s['destination']+'/LOCAL_DELIVERY_VERIFICATION.json',bytes=s['bytes'],sha256=s['sha256'],actual_files=result['actual_files'])
  raise AssertionError('Unknown simulated transport action')
 def call55(self,code,args=(),data=None):return self.action('55',code,args,data)
 def call29(self,code,settings,data=b''):return self.action('29',code,data=data,settings=settings)
 def forward_bundle29(self,path55,path29,desc):return self.action('29',C.P.REMOTE_UPLOAD,data=self.path('55',path55).read_bytes(),settings=dict(path=path29,bytes=desc['bytes'],sha256=desc['sha256']))
with tempfile.TemporaryDirectory(prefix='CPU_DELIVERY_SYNTHETIC_')as temp:
 base=Path(temp);root,state=fixture(base);state,freeze=C.validate_scope(root);ok('CPU_HPORT_NO_scope_fixture_accepted');rows=C.source_roster(root);ok('default_roster_before_prepare_is_read_only',not(root/'delivery').exists()and not(root/'delivery_tools').exists());(root/'operations').mkdir();(root/'operations/stage.tar.gz').write_bytes(b'omitted');(root/'__pycache__').mkdir();(root/'__pycache__/cache.pyc').write_bytes(b'omitted');(root/'.DS_Store').write_bytes(b'omitted');ok('explicit_archives_caches_omitted',C.source_roster(root)==rows)
 raw=root/'forbidden.safetensors';raw.write_bytes(b'x');reject('model_file_rejected',lambda:C.source_roster(root));raw.unlink();raw=root/'bad.json';raw.write_bytes(b'\x7fELFbad');reject('ELF_disguised_small_file_rejected',lambda:C.source_roster(root));raw.unlink();(root/'alias').symlink_to(root/'USER_REQUEST.txt');reject('symlink_rejected',lambda:C.source_roster(root));(root/'alias').unlink()
 original=C.read(root/'ROOT.json');bad=dict(original,ALLOW_GPU_RESTORE='YES');save(root/'ROOT.json',bad);reject('ROOT_YES_scope_rejected',lambda:C.validate_scope(root));save(root/'ROOT.json',original);auth=C.read(root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json');save(root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json',dict(auth,ALLOW_GPU_RESTORE='YES'));reject('human_YES_scope_rejected',lambda:C.validate_scope(root));save(root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json',auth);ledger=C.read(root/'GPU_RESOURCE_LEDGER.json');save(root/'GPU_RESOURCE_LEDGER.json',dict(ledger,GPU_queries=1));reject('nonzero_GPU_ledger_rejected',lambda:C.validate_scope(root));save(root/'GPU_RESOURCE_LEDGER.json',ledger)
 request=(root/'USER_REQUEST.txt').read_bytes();(root/'USER_REQUEST.txt').write_text('mutated');reject('changed_frozen_source_rejected',lambda:C.validate_scope(root));(root/'USER_REQUEST.txt').write_bytes(request);state,freeze=C.validate_scope(root);rows=C.source_roster(root);C.prepare(root,rows);ok('prepare_exact_small_file_SHA_bytes',C.roster(root/'delivery')==rows);ok('prepare_does_not_seal_or_transfer',not(root/'delivery/SHA256_MANIFEST.json').exists()and not(root/'delivery_tools/receipts/DELIVERY_TRANSFER_STATE.jsonl').exists());reject('prepare_cannot_overwrite',lambda:C.prepare(root,rows));(root/'later.md').write_text('changed after preparation');reject('changed_prepared_payload_rejected',lambda:C.run(root,state,freeze,C.source_roster(root),FakeTransport));(root/'later.md').unlink();FakeTransport.base=base/'virtual_remote'
 with mock.patch.object(C.P.subprocess,'run',side_effect=AssertionError('No SSH in synthetic test')):
  result=C.run(root,state,freeze,rows,FakeTransport)
 ok('synthetic_three_host_delivery_completed',result['status']=='PASS_VERIFIED_CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED_DELIVERY');local=root/'delivery';receipt=C.read(local/'LOCAL_DELIVERY_VERIFICATION.json');ok('external_receipt_no_hash_cycle',receipt['external_receipt_excluded_from_manifest_by_design']and receipt['all_three_hosts_payload_exact_same_SHA_bytes_and_roster']);proof=C.read(root/'delivery_tools/receipts/EXTERNAL_RECEIPT_REPLICATION.json');ok('all_three_exact_actual_count_not_hardcoded',proof['actual_files_all_hosts']==len(rows)+4 and proof['server55']['actual_files']==proof['archive29']['actual_files']==len(rows)+4);ok('all_three_receipts_exact_same_SHA_bytes',C.stable(local/'LOCAL_DELIVERY_VERIFICATION.json')==C.stable(FakeTransport.base/'55/synthetic55/task/final_delivery/LOCAL_DELIVERY_VERIFICATION.json')==C.stable(FakeTransport.base/'29/synthetic29/task/CPU_DELIVERY/LOCAL_DELIVERY_VERIFICATION.json'));ok('delivery_manifest_CPU_only_no_STOP_classification',C.read(local/'SHA256_MANIFEST.json')['status']==C.STATUS and C.read(local/'FINAL_SEAL_RECEIPT.json')['status']==C.STATUS);reject('one_shot_completed_run_cannot_reseal',lambda:C.run(root,state,freeze,rows,FakeTransport));ok('no_scientific_GPU_or_engine_import',not any(x=='torch'or x.startswith('torch.')or x=='transformers'or x.startswith('transformers.')for x in sys.modules))
print(json.dumps(dict(status='PASS_CPU_SYNTHETIC_DELIVERY_PREVIEW_AND_TRANSPORT_TESTS_NOT_REAL_SEAL',checks_passed=len(checks),checks=checks,GPU_queries=0,GPU_dispatches=0,real_network_transfers=0,real_root_sealed=False,synthetic_temporary_fixture_removed=True,sealer_sha256=hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),transfer_primitives_sha256=hashlib.sha256(Path(C.P.__file__).read_bytes()).hexdigest()),sort_keys=True,indent=2))
