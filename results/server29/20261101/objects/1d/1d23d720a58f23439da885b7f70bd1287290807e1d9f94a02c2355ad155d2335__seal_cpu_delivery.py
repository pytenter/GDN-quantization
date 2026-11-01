#!/usr/bin/env python3
"""CPU-only exact small-file delivery. Default: read-only preview.

--prepare copies the current bounded root payload once to delivery/; no seal or
network action. --run seals/transfers once only after final CPU report review.
The imported historical transfer primitives are byte-exact evidence; their
old task-specific main is never called. No GPU API or runtime import exists.
"""
import argparse,datetime,gzip,hashlib,io,json,os,sys,tarfile,time,traceback
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
import transfer_primitives as P
TASK='QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1'
STATUS='CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED'
MODE='HISTORICAL_PORT'
PRIMITIVES_SHA='0214931830433d9e9d7fab27f418d21b838e3464d1a4307b7375c5a535d67948'
EXCLUDED=P.EXCLUDED
REQUIRED=['USER_REQUEST.txt','EXECUTION_PLAN.md','ROOT.json','HUMAN_GPU_RESTORE_AUTHORIZATION.json','SOURCE_FREEZE.json','FINAL_REPORT.md','HISTORICAL_RUNTIME_IDENTITY.md','OLD_VS_NEW_PROTOCOL_DIFF.md','RUNTIME_NUMERICAL_QUALIFICATION.md','INT8_COMPATIBILITY_AUDIT.md','RESTORATION_MODE.json','RESTORED_FP_BENCHMARK_STATUS.json','RUNTIME_RESTORATION_DECISION.json','GPU_RESOURCE_LEDGER.json','HISTORICAL_BENCHMARK_REFERENCE.csv','historical_runtime_candidate.sh','analysis/final_review/FINAL_REPORT_REVIEW.json','analysis/final_review/CONTROLLER_WORKER_INTEGRATION_REVIEW.json','analysis/environment/CONTROLLER_IMPLEMENTATION_REVIEW.json','analysis/protocol/WORKER_PREPARATION_REVIEW.json','qualification/REMOTE_ISOLATED_CPU_CHECK.json','qualification/ACTUAL_CPU_ENTRY_CHECKS.json','analysis/HISTORICAL_UNCHANGED_VERIFICATION.json']
require=P.require;stable=P.stable;read=P.read;digest=P.digest;encoded=P.encoded;canonical=P.canonical;roster=P.roster;check_payload=P.check_payload;Transport=P.Transport

def new_bytes(path,raw):
 desc=P.new_bytes(path,raw);fd=os.open(Path(path).parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return desc

def omitted(rel):
 parts=PurePosixPath(rel).parts
 if parts and parts[0]=='delivery':return True
 if '__pycache__'in parts or '.DS_Store'in parts or rel.endswith('.tar.gz'):return True
 if rel.startswith('delivery_tools/receipts/'):return True
 if rel.startswith('operations/')and any(x in PurePosixPath(rel).name.upper()for x in ['DELIVERY_TRANSFER_STATE','LOCAL_DELIVERY_VERIFICATION','EXTERNAL_RECEIPT_REPLICATION','FINAL_DELIVERY_']):return True
 if '/'not in rel and rel in EXCLUDED:return True
 return False

def source_roster(root):
 rows=[]
 for p in sorted(root.rglob('*')):
  rel=str(p.relative_to(root))
  if omitted(rel):continue
  require(not p.is_symlink(),'Payload symlink prohibited:'+rel)
  if not p.is_file():continue
  require(not any(x in rel for x in ['\n','\r','\x00']),'Unsupported payload filename')
  require(p.suffix.lower()not in {'.pt','.pth','.bin','.safetensors','.npy','.npz','.tensor','.so','.dylib','.dll'}and'.so.'not in p.name,'Model/tensor/native library prohibited:'+rel)
  require(p.stat().st_size<P.MAX_FILE,'Large root file prohibited:'+rel);raw=stable(p);require(not raw.startswith(b'\x7fELF'),'ELF native object prohibited:'+rel);rows.append(dict(path=rel,bytes=len(raw),sha256=digest(raw)))
 require(sum(x['bytes']for x in rows)<P.MAX_TOTAL,'Small root payload total bound exceeded');return rows

def verify_freeze(root,state):
 fp=root/'SOURCE_FREEZE.json';require(digest(stable(fp))==state['source_freeze_sha256'],'ROOT source freeze SHA differs');f=read(fp);require(f.get('task_id')==TASK and f.get('ALLOW_GPU_RESTORE')=='NO','Wrong source freeze CPU-only task');rows=f.get('files');require(isinstance(rows,list)and rows,'Empty source freeze');seen=set()
 for row in rows:
  name=row['path'];pp=PurePosixPath(name);require(not pp.is_absolute()and'..'not in pp.parts and name not in seen,'Unsafe/duplicate frozen file');seen.add(name);raw=stable(root/name);require(len(raw)==row['bytes']and digest(raw)==row['sha256'],'Frozen scientific source changed:'+name)
 return dict(path='SOURCE_FREEZE.json',bytes=fp.stat().st_size,sha256=digest(stable(fp)),files_verified=len(rows),status='PASS_CPU_FROZEN_SOURCE_SHA_BYTES')

def validate_scope(root):
 require(digest(stable(Path(__file__).with_name('transfer_primitives.py')))==PRIMITIVES_SHA,'Historical transfer primitive bytes differ');require(all((root/n).is_file()for n in REQUIRED),'Required CPU deliverable/audit missing');state=read(root/'ROOT.json');auth=read(root/'HUMAN_GPU_RESTORE_AUTHORIZATION.json');ledger=read(root/'GPU_RESOURCE_LEDGER.json');decision=read(root/'RUNTIME_RESTORATION_DECISION.json');mode=read(root/'RESTORATION_MODE.json')
 require(state.get('task_id')==TASK and state.get('status')==STATUS and state.get('ALLOW_GPU_RESTORE')=='NO','ROOT must remain CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED and NO');require(state.get('GPU_queries')==state.get('GPU_dispatches')==0,'ROOT GPU use must be zero');require(state.get('restoration_mode')==MODE,'Wrong restoration mode');require(auth.get('task_id')==TASK and auth.get('ALLOW_GPU_RESTORE')=='NO','Human authorization must remain NO');require(auth.get('GPU_dispatches')==0,'Human CPU preparation dispatch count differs')
 require(ledger.get('task_id')==TASK and ledger.get('ALLOW_GPU_RESTORE')=='NO'and ledger.get('GPU_queries')==ledger.get('GPU_dispatches')==ledger.get('actual_GPU_worker_seconds')==ledger.get('actual_GPU_hours_for_new_dispatches')==0,'Ledger nonzero or authorization changed');require(all(ledger.get(x)==[]for x in ['first_dispatch_events','first_exit_events','first_release_events']),'Unexpected GPU resource events');require(decision.get('status')==STATUS and decision.get('restoration_mode')==MODE and mode.get('mode')==MODE,'CPU-only HPORT report classification differs')
 review=read(root/'analysis/final_review/FINAL_REPORT_REVIEW.json');require(str(review.get('status','')).startswith('PASS_')and review.get('checks_passed')==review.get('checks_total')and review.get('checks_passed')>=25 and not review.get('findings'),'Final independent fact/scope review not complete PASS')
 review_rows=review.get('receipts');require(isinstance(review_rows,list)and any(x.get('path')=='FINAL_REPORT.md'for x in review_rows),'Final review report SHA binding missing')
 for row in review_rows:
  if row['path']=='ROOT.json':continue # Review explicitly deferred final ROOT status/transport finalization.
  pp=PurePosixPath(row['path']);require(not pp.is_absolute()and'..'not in pp.parts,'Unsafe report review binding');raw=stable(root/row['path']);require(len(raw)==row['bytes']and digest(raw)==row['sha256'],'Reviewed report/source bytes changed:'+row['path'])
 for name in ['qualification/REMOTE_ISOLATED_CPU_CHECK.json','qualification/ACTUAL_CPU_ENTRY_CHECKS.json','analysis/HISTORICAL_UNCHANGED_VERIFICATION.json']:require(str(read(root/name).get('status','')).startswith('PASS'),'Final CPU provenance/entry check not PASS:'+name)
 for name in ['analysis/final_review/CONTROLLER_WORKER_INTEGRATION_REVIEW.json','analysis/environment/CONTROLLER_IMPLEMENTATION_REVIEW.json','analysis/protocol/WORKER_PREPARATION_REVIEW.json']:
  require(str(read(root/name).get('status','')).startswith('PASS_'),'Worker/controller review not PASS:'+name)
 freeze=verify_freeze(root,state);return state,freeze

def prepare(root,rows):
 dest=root/'delivery';receipt_dir=root/'delivery_tools/receipts';require(not dest.exists(),'Delivery exists: do not overwrite/reprepare');receipt_dir.mkdir(parents=True,exist_ok=True);state=receipt_dir/'PREPARATION_STARTED.json';require(not state.exists(),'Preparation already attempted: inspect, no blind retry');new_bytes(state,encoded(dict(status='CPU_DELIVERY_PREPARATION_STARTED',task_id=TASK,source_roster_sha256=digest(canonical(rows)),indexed_files=len(rows))));dest.mkdir()
 for row in rows:
  raw=stable(root/row['path']);require(len(raw)==row['bytes']and digest(raw)==row['sha256'],'Root changed during prepare:'+row['path']);target=dest/row['path'];new_bytes(target,raw);os.chmod(target,(root/row['path']).stat().st_mode&0o777)
 actual=roster(dest);require(actual==rows,'Prepared copy roster/SHA/bytes differs');require(source_roster(root)==rows,'Root changed while copying');new_bytes(receipt_dir/'PREPARATION_COMPLETE.json',encoded(dict(status='PASS_PREPARED_CPU_SMALL_DELIVERY_NOT_SEALED_NOT_TRANSFERRED',task_id=TASK,files_verified=len(rows),bytes_verified=sum(x['bytes']for x in rows),source_roster_sha256=digest(canonical(rows)),GPU_queries=0,GPU_dispatches=0)));return actual

def bundle_delivery(dest,bundle):
 require(not bundle.exists(),'Bundle exists: inspect transfer state')
 with bundle.open('xb')as output,gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0)as gz,tarfile.open(fileobj=gz,mode='w')as tf:
  for row in roster(dest,True):
   p=dest/row['path'];raw=stable(p);require(digest(raw)==row['sha256'],'File changed while bundling');info=tarfile.TarInfo(row['path']);info.size=len(raw);info.mode=p.stat().st_mode&0o777;info.mtime=0;info.uid=info.gid=0;tf.addfile(info,io.BytesIO(raw))
 return P.describe(bundle)

def run(root,state,freeze,rows,transport_factory=Transport):
 dest=root/'delivery';require(dest.is_dir(),'Run requires --prepare first');require(not any((dest/n).exists()for n in EXCLUDED),'Existing seal/receipt: no overwrite or reseal');require(roster(dest)==rows,'Prepared root/copy roster differs; inspect before sealing');require(read(root/'delivery_tools/receipts/PREPARATION_COMPLETE.json')['source_roster_sha256']==digest(canonical(rows)),'Preparation receipt differs')
 for key in ['remote_root','archive_root','remote_host','archive_host','remote_user','ssh_control_path','archive_control_from55','remote_python']:require(isinstance(state.get(key),str)and state[key],'Transport ROOT missing:'+key)
 for key in ['remote_root','archive_root']:require(PurePosixPath(state[key]).is_absolute()and'..'not in PurePosixPath(state[key]).parts,'Unsafe remote path')
 receipts=root/'delivery_tools/receipts';operation=receipts/'DELIVERY_TRANSFER_STATE.jsonl';require(not operation.exists(),'One-shot transfer state exists: inspect all host paths; no blind retry')
 def record(event,**data):
  with operation.open('a')as f:f.write(json.dumps(dict(event=event,unix=time.time(),**data),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 def save(name,value):return new_bytes(receipts/name,encoded(value))
 with operation.open('x')as initial:initial.flush();os.fsync(initial.fileno())
 record('START_CPU_ONLY_SEAL_AND_TRANSFER',status=STATUS,mode=MODE,indexed_files=len(rows),indexed_bytes=sum(x['bytes']for x in rows))
 try:
  manifest=dict(task_id=TASK,status=STATUS,restoration_mode=MODE,ALLOW_GPU_RESTORE='NO',GPU_queries=0,GPU_dispatches=0,files_count=len(rows),total_bytes=sum(x['bytes']for x in rows),large_model_or_tensor_downloaded=False,excluded_external_anchors=sorted(EXCLUDED),files=rows);new_bytes(dest/'SHA256_MANIFEST.json',encoded(manifest));mh=digest(stable(dest/'SHA256_MANIFEST.json'));new_bytes(dest/'SHA256SUMS.txt',(''.join(x['sha256']+'  '+x['path']+'\n'for x in rows)+mh+'  SHA256_MANIFEST.json\n').encode())
  audit_anchors=[x for x in rows if x['path'].endswith('.json')and x['path'].startswith(('analysis/','qualification/','preregistration/'))];seal=dict(task_id=TASK,status=STATUS,restoration_mode=MODE,ALLOW_GPU_RESTORE='NO',GPU_queries=0,GPU_dispatches=0,sealed_utc_local_host=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_verified=len(rows),bytes_verified=manifest['total_bytes'],manifest_sha256=mh,SHA256SUMS_sha256=digest(stable(dest/'SHA256SUMS.txt')),FINAL_REPORT_sha256=digest(stable(dest/'FINAL_REPORT.md')),report_review_sha256=digest(stable(dest/'analysis/final_review/FINAL_REPORT_REVIEW.json')),source_freeze=freeze,audit_anchors=audit_anchors,tool_source_bindings=[P.describe(Path(__file__)),P.describe(Path(__file__).with_name('transfer_primitives.py'))],no_new_GPU_qualification_claim=True,local_external_delivery_verification_pending=True,external_anchor_excluded_from_manifest=True);new_bytes(dest/'FINAL_SEAL_RECEIPT.json',encoded(seal));settings=dict(indexed_files=len(rows),indexed_bytes=manifest['total_bytes'],manifest_sha256=mh,SHA256SUMS_sha256=seal['SHA256SUMS_sha256'],seal_sha256=digest(stable(dest/'FINAL_SEAL_RECEIPT.json')));local=check_payload(dest,settings);settings['payload_roster_sha256']=local['payload_roster_sha256'];save('DELIVERY_LOCAL_BASE_VERIFICATION.json',local);record('LOCAL_SEAL_VERIFIED',**settings)
  bundle=root/'delivery_tools/CPU_SMALL_DELIVERY.tar.gz';bundle_desc=bundle_delivery(dest,bundle);record('BUNDLE_READY',bundle=bundle_desc);transport=transport_factory(state,record);remote55=state['remote_root']+'/final_delivery';bundle55=state['remote_root']+'/CPU_SMALL_DELIVERY.tar.gz';remote29=state['archive_root']+'/CPU_DELIVERY';bundle29=state['archive_root']+'/CPU_DELIVERY_SOURCE_AND_REPORTS.tar.gz';base=dict(settings,bundle_bytes=bundle_desc['bytes'],bundle_sha256=bundle_desc['sha256'])
  record('UPLOAD55_STARTED',path=bundle55);up55=transport.call55(P.REMOTE_UPLOAD,[json.dumps(dict(path=bundle55,bytes=bundle_desc['bytes'],sha256=bundle_desc['sha256']))],stable(bundle));save('DELIVERY_SERVER55_UPLOAD.json',up55);record('UPLOAD55_PASS');verify55=transport.call55(P.REMOTE_VERIFY,data=encoded(dict(base,bundle=bundle55,destination=remote55)));require(verify55['payload_roster_sha256']==local['payload_roster_sha256'],'55 payload differs');save('DELIVERY_SERVER55_VERIFICATION.json',verify55);record('SERVER55_VERIFIED')
  record('FORWARD29_STARTED',path=bundle29);up29=transport.forward_bundle29(bundle55,bundle29,bundle_desc);save('DELIVERY_ARCHIVE29_UPLOAD.json',up29);record('FORWARD29_PASS');verify29=transport.call29(P.REMOTE_VERIFY,dict(base,bundle=bundle29,destination=remote29));require(verify29['payload_roster_sha256']==local['payload_roster_sha256'],'29 payload differs');save('DELIVERY_ARCHIVE29_VERIFICATION.json',verify29);record('ARCHIVE29_VERIFIED');local_again=check_payload(dest,settings);require(local_again['payload_roster_sha256']==local['payload_roster_sha256'],'Mac payload changed during transfer')
  external=dict(task_id=TASK,status='PASS_VERIFIED_CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED_DELIVERY',restoration_mode=MODE,ALLOW_GPU_RESTORE='NO',GPU_queries=0,GPU_dispatches=0,verified_utc_local_host=datetime.datetime.now(datetime.timezone.utc).isoformat(),indexed_files=len(rows),indexed_bytes=manifest['total_bytes'],actual_payload_files_before_external_receipt=len(rows)+3,actual_files_including_external_receipt=len(rows)+4,manifest_sha256=mh,SHA256SUMS_sha256=settings['SHA256SUMS_sha256'],FINAL_SEAL_RECEIPT_sha256=settings['seal_sha256'],FINAL_REPORT_sha256=seal['FINAL_REPORT_sha256'],source_freeze=freeze,report_review_sha256=seal['report_review_sha256'],all_three_hosts_payload_exact_same_SHA_bytes_and_roster=True,payload_roster_sha256=local['payload_roster_sha256'],local_verification=local_again,server55_verification=verify55,archive29_verification=verify29,transfer_bundle=bundle_desc,external_receipt_excluded_from_manifest_by_design=True,seal_pending_external_verification_resolved_by_this_receipt=True,external_receipt_replication_proof_separate='delivery_tools/receipts/EXTERNAL_RECEIPT_REPLICATION.json',no_large_raw_or_model_transfer=True,no_new_GPU_qualification_claim=True);ext=new_bytes(dest/'LOCAL_DELIVERY_VERIFICATION.json',encoded(external));record('EXTERNAL_RECEIPT_CREATED',receipt=ext);body=stable(dest/'LOCAL_DELIVERY_VERIFICATION.json');transport.call55(P.REMOTE_UPLOAD,[json.dumps(dict(path=remote55+'/LOCAL_DELIVERY_VERIFICATION.json',bytes=ext['bytes'],sha256=ext['sha256']))],body);transport.call29(P.REMOTE_UPLOAD,dict(path=remote29+'/LOCAL_DELIVERY_VERIFICATION.json',bytes=ext['bytes'],sha256=ext['sha256']),body);rec55=transport.call55(P.REMOTE_RECEIPT_CHECK,data=encoded(dict(destination=remote55,bytes=ext['bytes'],sha256=ext['sha256'])));rec29=transport.call29(P.REMOTE_RECEIPT_CHECK,dict(destination=remote29,bytes=ext['bytes'],sha256=ext['sha256']));check_payload(dest,settings,True);require(digest(stable(dest/'LOCAL_DELIVERY_VERIFICATION.json'))==ext['sha256'],'Mac external receipt changed');replication=dict(status='PASS_CPU_EXTERNAL_RECEIPT_REPLICATED_EXACT_ALL_THREE_HOSTS',local=ext,server55=rec55,archive29=rec29,actual_files_all_hosts=len(rows)+4,no_hash_cycle=True);proof=save('EXTERNAL_RECEIPT_REPLICATION.json',replication);record('COMPLETE_CPU_ONLY_DELIVERY',external_receipt=ext,replication_proof=proof);return dict(status='PASS_VERIFIED_CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED_DELIVERY',manifest_sha256=mh,indexed_files=len(rows),indexed_bytes=manifest['total_bytes'],LOCAL_DELIVERY_VERIFICATION=ext,external_receipt_replication=proof,GPU_queries=0,GPU_dispatches=0)
 except BaseException as exc:record('FAILED_INSPECT_STATE_NO_BLIND_RETRY',error=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc());raise

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True,type=Path);actions=ap.add_mutually_exclusive_group();actions.add_argument('--prepare',action='store_true');actions.add_argument('--run',action='store_true');a=ap.parse_args();root=a.root.resolve();state,freeze=validate_scope(root);rows=source_roster(root)
 if a.prepare:prepare(root,rows);result=dict(status='PASS_PREPARED_CPU_SMALL_DELIVERY_NOT_SEALED_NOT_TRANSFERRED',indexed_files=len(rows),indexed_bytes=sum(x['bytes']for x in rows),GPU_queries=0,GPU_dispatches=0)
 elif a.run:result=run(root,state,freeze,rows)
 else:result=dict(status='PREVIEW_CPU_ONLY_NO_WRITES_NO_TRANSFER_NOT_SEALED',task_id=TASK,restoration_status=STATUS,restoration_mode=MODE,indexed_files=len(rows),indexed_bytes=sum(x['bytes']for x in rows),delivery_exists=(root/'delivery').exists(),SOURCE_FREEZE=freeze,all_required_CPU_reports_reviews_present=True,GPU_queries=0,GPU_dispatches=0)
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
