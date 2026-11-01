#!/usr/bin/env python3
"""One-shot post-release local small delivery seal. Default is read-only preview.

Runs only file/stdlib work on the compute host after independent scientific,
source, archive and release reviews. It never transfers files, queries GPUs,
imports a runtime, deletes payload assets, or writes an external delivery PASS.
"""
import argparse,datetime,gzip,hashlib,io,json,os,stat,sys,tarfile,time,traceback
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
TASK='QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1'
MAX_FILE=64*1024**2;MAX_TOTAL=256*1024**2;MAX_BUNDLE=128*1024**2;MAX_FILES=20000
ANCHORS={'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','LOCAL_DELIVERY_VERIFICATION.json'}
SUFFIXES={'.json','.jsonl','.md','.txt','.py','.sh','.log','.csv','.patch','.h','.cpp','.cu','.cuh'}
TOPDIRS={'payload','analysis','qualification','provenance','scripts','tests','operations','archive','benchmarks'}
SKIPDIRS={'final_delivery','delivery','temp','tmp','drafts','monitoring','__pycache__','.git'}
FORBIDDEN={'.bin','.pt','.pth','.safetensors','.npy','.npz','.tensor','.so','.dylib','.dll','.a','.o'}
REPORT='FINAL_SCIENTIFIC_REPORT.md';DECISION='RUNTIME_RESTORATION_DECISION.json';REVIEW='analysis/FINAL_INDEPENDENT_RESULT_REVIEW.json'
SOURCE='analysis/FINAL_SOURCE_DEPENDENCY_VERIFICATION.json';RELEASE='analysis/POST_GPU_RELEASE_ADMISSION.json';ARCHIVE='archive/ARCHIVE_VERIFICATION.json';CATALOG='archive/ASSET_CATALOG.json'
REQUIRED=['ROOT.json',REPORT,DECISION,'RESOURCE_LEDGER.json',REVIEW,SOURCE,RELEASE,ARCHIVE,CATALOG,'analysis/SCIENTIFIC_FINISHER_RESULT.json','payload/SOURCE_FREEZE.json']
SCIENCE_STATUSES={'HISTORICAL_PORT_FP_QUALIFIED_SPEED_RECOVERED','HISTORICAL_PORT_FP_QUALIFIED_PARTIAL_SPEED','FP_NUMERICAL_QUALIFICATION_FAIL','GPU_RESOURCE_OR_IDENTITY_FAIL','FAIL_CLOSED_STOP','PARTIAL_GPU_RESULTS_PRESERVED_AND_STOP'}

def require(value,message):
 if not value:raise RuntimeError(message)
def encoded(value):return(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def signature(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def safe_relative(name):
 require(isinstance(name,str),'Path must be a string');p=PurePosixPath(name);require(not p.is_absolute()and p.parts and'..'not in p.parts and'.'not in p.parts and not any(x in name for x in ['\\','\n','\r','\0']),'Unsafe relative path');return p
def stable(path,limit=MAX_FILE):
 p=Path(path);a=p.lstat();require(stat.S_ISREG(a.st_mode)and a.st_size<=limit,'Not a bounded regular file:'+str(p))
 with p.open('rb')as f:b=f.read(limit+1);z=os.fstat(f.fileno())
 require(len(b)<=limit and signature(a)==signature(z)==signature(p.lstat()),'File changed or exceeds limit:'+str(p));return b
def descriptor(root,name,limit=MAX_FILE):
 safe_relative(name);b=stable(Path(root)/name,limit);return dict(path=name,bytes=len(b),sha256=sha(b))
def read(root,name):return json.loads(stable(Path(root)/name))
def new_bytes(path,raw):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(p),bytes=len(raw),sha256=sha(raw))
def save(path,value):return new_bytes(path,encoded(value))
def excluded(name):
 parts=safe_relative(name).parts
 if any(x in SKIPDIRS for x in parts):return 'excluded_directory'
 if parts[0]not in TOPDIRS and len(parts)>1:return 'outside_selected_metadata_trees'
 if any(x=='.DS_Store'or x.endswith(('.tmp','.incoming','.tar','.tar.gz','.tgz'))for x in parts):return 'temporary_or_bundle'
 if parts[0]=='operations'and PurePosixPath(name).name.startswith(('SMALL_DELIVERY_','SMALL_ASSET_')):return 'external_seal_operation_receipt'
 if len(parts)==1 and name in ANCHORS:return 'reserved_final_anchor'
 p=PurePosixPath(name)
 if p.suffix.lower()in FORBIDDEN or'.so.'in p.name:return 'raw_model_or_native_binary'
 if p.suffix.lower()not in SUFFIXES:return 'suffix_not_in_small_evidence_allowlist'
 return None
def source_roster(root,hashes=True):
 root=Path(root);rows=[];omitted=[]
 for p in sorted(root.rglob('*')):
  name=str(p.relative_to(root));reason=excluded(name)
  if reason:continue
  a=p.lstat();require(not stat.S_ISLNK(a.st_mode),'Selected source symlink prohibited:'+name)
  if stat.S_ISDIR(a.st_mode):continue
  require(stat.S_ISREG(a.st_mode)and a.st_size<=MAX_FILE,'Selected source not bounded regular:'+name)
  row=dict(path=name,bytes=a.st_size)
  if hashes:
   b=stable(p);require(not b.startswith((b'\x7fELF',b'MZ',b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf')),'Native executable disguised as small file:'+name)
   if name.startswith('benchmarks/')and p.suffix=='.csv':require(len([x for x in b.splitlines()if x.strip()])>=2,'Empty/placeholder scientific benchmark CSV prohibited:'+name)
   row['sha256']=sha(b)
  rows.append(row)
 require(len(rows)<MAX_FILES and sum(x['bytes']for x in rows)<MAX_TOTAL,'Small evidence bounds exceeded');return rows
def verify_descriptor(root,row):
 actual=descriptor(root,row['path']);require(actual=={k:row[k]for k in ['path','bytes','sha256']},'Reviewed evidence changed:'+row['path']);return actual
def validate(root):
 require(all((root/n).is_file()for n in REQUIRED),'Post-release report/source/archive/review prerequisites missing');state=read(root,'ROOT.json');decision=read(root,DECISION);require(state.get('task_id')==TASK and decision.get('task_id')==TASK and decision.get('status')in SCIENCE_STATUSES,'Wrong task or scientific classification');freeze=read(root,'payload/SOURCE_FREEZE.json');freeze_desc=descriptor(root,'payload/SOURCE_FREEZE.json');rows=freeze.get('files');require(isinstance(rows,list)and len(rows)==407,'Expected exact407 effective frozen source entries');seen=set()
 for row in rows:
  safe_relative(row['path']);require(row['path']not in seen,'Duplicate frozen source');seen.add(row['path']);require(excluded('payload/'+row['path'])is None,'Frozen source would be excluded');verify_descriptor(root,dict(row,path='payload/'+row['path']))
 require(state.get('source_freeze_sha256')==freeze_desc['sha256'],'ROOT effective freeze differs');source=read(root,SOURCE);require(source.get('status')=='PASS_FRESH_POST_GPU407_AND249_EXTERNAL_STREAM_SHA_BYTES'and source.get('files_verified')==407 and source.get('external_files_verified')==249 and source.get('source_freeze_sha256')==freeze_desc['sha256'],'Fresh post-GPU source/external verification not complete')
 proof=read(root,RELEASE);require(proof.get('schema')=='QWEN_RECOVERY_RELEASE_ADMISSION_V1'and proof.get('task_id')==TASK and proof.get('status')=='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED','Independent release not proved');require(proof.get('payload_root')==str((root/'payload').resolve())and proof.get('source_freeze_sha256')==freeze_desc['sha256']and proof.get('final_owned_NVML_records')==[],'Independent release binds another source/payload or live ownGPU');require(all(proof.get(k)is True for k in ['controller_exit_verified','all_owned_processes_exited','all_owned_GPU_resources_released']),'Release aggregate incomplete')
 own=proof.get('own_processes');require(isinstance(own,list)and own and all(x.get('process_exit_verified')is True and x.get('all_UUID_NVML_release_verified')is True for x in own),'Per-process release unresolved')
 archive=read(root,ARCHIVE);catalog=read(root,CATALOG);require(archive.get('status')=='PASS_POST_RELEASE_FULL_PAYLOAD_ARCHIVE_BYTES_ONLY'and archive.get('source_two_stable_SHA_passes')is True and archive.get('remote_two_independent_disk_SHA_passes')is True and archive.get('source_deleted')is False and archive.get('payload_unchanged')is True,'Full retained payload archive not verified');require(catalog.get('source_freeze_sha256')==freeze_desc['sha256']and catalog.get('release_proof',{}).get('sha256')==descriptor(root,RELEASE)['sha256'],'Archive binds another source/release');require(archive.get('source_catalog',{}).get('sha256')==descriptor(root,CATALOG)['sha256'],'Archive catalog SHA differs');remote=archive.get('remote',{});require(remote.get('ASSET_CATALOG',{}).get('sha256')==descriptor(root,CATALOG)['sha256']and remote.get('verification',{}).get('catalog_sha256')==descriptor(root,CATALOG)['sha256'],'Remote independently verified catalog differs')
 raw=[];catalog_rows={r['path']:r for r in catalog.get('files',[])};require(len(catalog_rows)==catalog.get('files_count'),'Catalog duplicate/count error')
 for p in sorted((root/'payload').rglob('*.bin')):
  name=str(p.relative_to(root/'payload'));a=p.lstat();require(stat.S_ISREG(a.st_mode),'Raw not regular');row=catalog_rows.get(name);require(row and row['bytes']==a.st_size,'Omitted raw lacks permanent archive SHA/bytes');raw.append(dict(row,local_payload_path='payload/'+name,permanent_archive_path=catalog['archive_destination']+'/'+name))
 review=read(root,REVIEW);require(str(review.get('status','')).startswith('PASS')and review.get('findings')==[] and type(review.get('checks_total'))is int and review['checks_total']>0 and review['checks_total']==review.get('checks_passed'),'Independent final result review incomplete');receipts=review.get('receipts');require(isinstance(receipts,list),'Review exact receipt roster missing');bindings={r['path']:r for r in receipts};require(len(bindings)==len(receipts),'Duplicate final review receipt');required_review={REPORT,DECISION,'RESOURCE_LEDGER.json',SOURCE,RELEASE,ARCHIVE};require(required_review<=set(bindings),'Review must bind report, decision, resources, source, release and archive')
 for row in receipts:verify_descriptor(root,row)
 require(read(root,'analysis/SCIENTIFIC_FINISHER_RESULT.json').get('decision')==decision['status'],'Finisher and decision disagree')
 return dict(state=state,decision=decision,freeze=dict(freeze_desc,files_verified=407),raw=raw,anchors=[descriptor(root,n)for n in [REPORT,DECISION,'RESOURCE_LEDGER.json',REVIEW,SOURCE,RELEASE,ARCHIVE,CATALOG]])
def roster(dest):
 rows=[]
 for p in sorted(dest.rglob('*')):
  require(not p.is_symlink(),'Delivery symlink')
  if p.is_file():rows.append(descriptor(dest,str(p.relative_to(dest))))
 return rows
def check_payload(dest,settings):
 manifest=read(dest,'SHA256_MANIFEST.json');rows=manifest['files'];expected={x['path']:x for x in rows};require(len(rows)==len(expected)==manifest['files_count']==settings['indexed_files']and sum(x['bytes']for x in rows)==manifest['total_bytes']==settings['indexed_bytes'],'Manifest count/byte mismatch')
 for n,k in [('SHA256_MANIFEST.json','manifest_sha256'),('SHA256SUMS.txt','SHA256SUMS_sha256'),('FINAL_SEAL_RECEIPT.json','seal_sha256')]:require(descriptor(dest,n)['sha256']==settings[k],'Anchor SHA mismatch')
 want={x['path']:x['sha256']for x in rows};want['SHA256_MANIFEST.json']=settings['manifest_sha256'];pairs=[line.split('  ',1)for line in stable(dest/'SHA256SUMS.txt').decode().splitlines()];require(len(pairs)==len(want)and{n:h for h,n in pairs}==want,'SUMS exact roster differs');actual=roster(dest);actual_map={x['path']:x for x in actual};require(set(actual_map)==set(expected)|{'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json'},'Actual N+3 roster differs');require(all(actual_map[n]==r for n,r in expected.items()),'Indexed payload SHA/bytes differs');return dict(status='PASS_EXACT_SMALL_DELIVERY_SHA_BYTES',destination=str(dest),indexed_files=len(rows),indexed_bytes=manifest['total_bytes'],actual_files=len(actual),actual_bytes=sum(x['bytes']for x in actual),manifest_sha256=settings['manifest_sha256'],SHA256SUMS_sha256=settings['SHA256SUMS_sha256'],seal_sha256=settings['seal_sha256'],payload_roster_sha256=sha(canonical(actual)),large_model_or_tensor_in_delivery=False)
def seal(root):
 root=Path(root).resolve();validated=validate(root);rows=source_roster(root);dest=root/'final_delivery';operations=root/'operations';operations.mkdir(exist_ok=True);statepath=operations/'SMALL_DELIVERY_STATE.jsonl';require(not statepath.exists()and not dest.exists()and not(root/'SMALL_DELIVERY.tar.gz').exists(),'Seal already attempted/destination exists: inspect state; no blind retry')
 with statepath.open('x')as f:f.flush();os.fsync(f.fileno())
 def event(kind,**kw):
  with statepath.open('a')as f:f.write(json.dumps(dict(event=kind,unix=time.time(),**kw),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 event('POST_RELEASE_SMALL_SEAL_STARTED',scientific_status=validated['decision']['status'],source_freeze_sha256=validated['freeze']['sha256'])
 try:
  dest.mkdir()
  for row in rows:
   raw=stable(root/row['path']);require(len(raw)==row['bytes']and sha(raw)==row['sha256'],'Input changed while copying');new_bytes(dest/row['path'],raw);(dest/row['path']).chmod((root/row['path']).stat().st_mode&0o777)
  require(source_roster(root)==rows,'Source small metadata changed during copy');selection=dict(task_id=TASK,status='SMALL_EVIDENCE_SELECTION_RAW_RETAINED_REMOTE',source_files=rows,omitted_raw_files=validated['raw'],omitted_raw_bytes=sum(x['bytes']for x in validated['raw']),raw_catalog=descriptor(root,CATALOG),archive_verification=descriptor(root,ARCHIVE),selection_suffixes=sorted(SUFFIXES),excluded_directories=sorted(SKIPDIRS),no_large_raw_model_or_native_download=True);save(dest/'SMALL_ASSET_SELECTION.json',selection);indexed=roster(dest);require(len(indexed)<=MAX_FILES and sum(x['bytes']for x in indexed)<=MAX_TOTAL,'Generated small delivery bound exceeded');require(all('payload/'+x['path']in {r['path']for r in indexed}for x in read(root,'payload/SOURCE_FREEZE.json')['files']),'Not all407 frozen sources indexed')
  status=validated['decision']['status'];manifest=dict(task_id=TASK,run_id=validated['state'].get('run_id'),status='SEALED_POST_RELEASE_SMALL_EVIDENCE',scientific_status=status,files_count=len(indexed),total_bytes=sum(x['bytes']for x in indexed),source_freeze_sha256=validated['freeze']['sha256'],large_model_or_tensor_downloaded=False,excluded_external_anchors=sorted(ANCHORS),files=indexed);save(dest/'SHA256_MANIFEST.json',manifest);mh=descriptor(dest,'SHA256_MANIFEST.json')['sha256'];new_bytes(dest/'SHA256SUMS.txt',(''.join(x['sha256']+'  '+x['path']+'\n'for x in indexed)+mh+'  SHA256_MANIFEST.json\n').encode());seal_receipt=dict(task_id=TASK,status='SEALED_POST_RELEASE_SMALL_EVIDENCE_EXTERNAL_VERIFICATION_PENDING',scientific_status=status,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_verified=len(indexed),bytes_verified=manifest['total_bytes'],manifest_sha256=mh,SHA256SUMS_sha256=descriptor(dest,'SHA256SUMS.txt')['sha256'],FINAL_SCIENTIFIC_REPORT_sha256=descriptor(dest,REPORT)['sha256'],source_freeze=validated['freeze'],source_freeze_sha256=validated['freeze']['sha256'],report_review_sha256=descriptor(dest,REVIEW)['sha256'],release_proof_sha256=descriptor(dest,RELEASE)['sha256'],archive_verification_sha256=descriptor(dest,ARCHIVE)['sha256'],audit_anchors=validated['anchors'],local_external_delivery_verification_pending=True,external_anchor_excluded_from_manifest=True,all407_source_entries_indexed=True,raw_payload_retained55_and29=True,large_tensor_or_model_in_small_delivery=False,tool_source_sha256=sha(stable(Path(__file__))));save(dest/'FINAL_SEAL_RECEIPT.json',seal_receipt);settings=dict(indexed_files=len(indexed),indexed_bytes=manifest['total_bytes'],manifest_sha256=mh,SHA256SUMS_sha256=seal_receipt['SHA256SUMS_sha256'],seal_sha256=descriptor(dest,'FINAL_SEAL_RECEIPT.json')['sha256']);checked=check_payload(dest,settings);settings['payload_roster_sha256']=checked['payload_roster_sha256'];save(operations/'SMALL_DELIVERY_LOCAL_VERIFICATION.json',dict(checked,task_id=TASK,settings=settings,scientific_status=status,source_freeze_sha256=validated['freeze']['sha256'],report_sha256=seal_receipt['FINAL_SCIENTIFIC_REPORT_sha256'],release_proof_sha256=seal_receipt['release_proof_sha256'],report_review_sha256=seal_receipt['report_review_sha256'],archive_verification_sha256=seal_receipt['archive_verification_sha256'],external_delivery_verification_pending=True))
  bundle=root/'SMALL_DELIVERY.tar.gz'
  with bundle.open('xb')as out,gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0)as gz,tarfile.open(fileobj=gz,mode='w')as tf:
   for row in roster(dest):
    raw=stable(dest/row['path']);require(sha(raw)==row['sha256'],'Sealed asset changed while bundling');info=tarfile.TarInfo(row['path']);info.size=len(raw);info.mode=(dest/row['path']).stat().st_mode&0o777;info.mtime=0;info.uid=info.gid=0;tf.addfile(info,io.BytesIO(raw))
  bundle_desc=descriptor(root,'SMALL_DELIVERY.tar.gz',MAX_BUNDLE);require(bundle_desc['bytes']<=MAX_BUNDLE,'Compressed small bundle exceeds128MiB: preserve and inspect');bundle_desc['path']=str(bundle);require(check_payload(dest,settings)['payload_roster_sha256']==settings['payload_roster_sha256'],'Final local seal roster changed');receipt=dict(task_id=TASK,status='PASS_POST_RELEASE_SMALL_DELIVERY_SEALED_BUNDLE',bundle=bundle_desc,settings=settings,destination=str(dest),scientific_status=status,external_delivery_verification_pending=True,no_network_or_GPU_calls=True);save(operations/'SMALL_DELIVERY_BUNDLE.json',receipt);event('PASS_LOCAL_SEAL_AND_BUNDLE_EXTERNAL_VERIFICATION_PENDING',bundle=bundle_desc,settings=settings);return receipt
 except BaseException as exc:event('FAILED_INSPECT_STATE_NO_BLIND_RETRY',error=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc());raise
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--run',action='store_true');a=p.parse_args();root=a.root.resolve()
 if not a.run:
  rows=source_roster(root,hashes=False);print(json.dumps(dict(status='PREVIEW_ONLY_NO_WRITES_NO_SEAL_NO_NETWORK_NO_GPU',selected_files=len(rows),selected_bytes=sum(x['bytes']for x in rows),missing_prerequisites=[n for n in REQUIRED if not(root/n).is_file()],per_file_limit=MAX_FILE,total_limit=MAX_TOTAL,compressed_bundle_limit=MAX_BUNDLE)));return
 print(json.dumps(seal(root),sort_keys=True))
if __name__=='__main__':main()
