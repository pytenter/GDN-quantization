#!/usr/bin/env python3
"""Synthetic temporary CPU-only tests, never seals the real run or networks."""
import copy,hashlib,importlib.util,json,os,sys,tarfile,tempfile
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
RUN=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('small_seal_cpu',RUN/'scripts/seal_small_delivery.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
checks=[]
def ok(name,value=True):assert value,name;checks.append(dict(name=name,status='PASS'))
def reject(name,fn):
 try:fn()
 except (RuntimeError,OSError,AssertionError,KeyError):ok(name);return
 raise AssertionError(name+' unexpectedly accepted')
def save(root,name,value):p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(S.encoded(value))
def fixture(root,status='FAIL_CLOSED_STOP'):
 root.mkdir();payload=root/'payload';payload.mkdir();rows=[]
 for i in range(407):
  name='source/f%03d.json'%i;save(payload,name,dict(SYNTHETIC_CPU_SOURCE=i));rows.append(S.descriptor(payload,name))
 save(payload,'SOURCE_FREEZE.json',dict(task_id=S.TASK,files=rows));freeze=S.descriptor(root,'payload/SOURCE_FREEZE.json');save(root,'ROOT.json',dict(task_id=S.TASK,source_freeze_sha256=freeze['sha256'],run_id='SYNTHETIC_CPU_ONLY'))
 save(root,S.DECISION,dict(task_id=S.TASK,status=status));(root/S.REPORT).write_text('SYNTHETIC actual STOP report; no scientific measurements.\n');save(root,'RESOURCE_LEDGER.json',dict(status='SYNTHETIC_CPU_NO_GPU'))
 who=dict(pid=10,starttick=20,cmdline=['synthetic']);proof=dict(schema='QWEN_RECOVERY_RELEASE_ADMISSION_V1',task_id=S.TASK,status='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED',payload_root=str(payload.resolve()),source_freeze_sha256=freeze['sha256'],controller_exit_verified=True,all_owned_processes_exited=True,all_owned_GPU_resources_released=True,final_owned_NVML_records=[],own_processes=[dict(who,process_exit_verified=True,all_UUID_NVML_release_verified=True)])
 save(root,S.RELEASE,proof);save(root,S.SOURCE,dict(status='PASS_FRESH_POST_GPU407_AND249_EXTERNAL_STREAM_SHA_BYTES',files_verified=407,external_files_verified=249,source_freeze_sha256=freeze['sha256']));save(root,'analysis/SCIENTIFIC_FINISHER_RESULT.json',dict(decision=status))
 raw=payload/'execution/raw/token0.bin';raw.parent.mkdir(parents=True);raw.write_bytes(b'SYNTHETIC CPU RAW no actual model');catalog_rows=[S.descriptor(payload,str(p.relative_to(payload)))for p in sorted(payload.rglob('*'))if p.is_file()];catalog=dict(task_id=S.TASK,source_freeze_sha256=freeze['sha256'],release_proof=S.descriptor(root,S.RELEASE),files=catalog_rows,files_count=len(catalog_rows),archive_destination='/SYNTHETIC29/PAYLOAD_ASSETS');save(root,S.CATALOG,catalog);cs=S.descriptor(root,S.CATALOG)['sha256'];save(root,S.ARCHIVE,dict(status='PASS_POST_RELEASE_FULL_PAYLOAD_ARCHIVE_BYTES_ONLY',source_two_stable_SHA_passes=True,remote_two_independent_disk_SHA_passes=True,source_deleted=False,payload_unchanged=True,source_catalog=S.descriptor(root,S.CATALOG),remote=dict(ASSET_CATALOG=dict(sha256=cs),verification=dict(catalog_sha256=cs))))
 names=[S.REPORT,S.DECISION,'RESOURCE_LEDGER.json',S.SOURCE,S.RELEASE,S.ARCHIVE];save(root,S.REVIEW,dict(status='PASS_SYNTHETIC_CPU_REVIEW',findings=[],checks_total=6,checks_passed=6,receipts=[S.descriptor(root,n)for n in names]));return root
with tempfile.TemporaryDirectory(prefix='SMALL_DELIVERY_CPU_ONLY_')as temporary:
 t=Path(temporary);root=fixture(t/'one');v=S.validate(root);ok('trueSTOP_classification_preserved',v['decision']['status']=='FAIL_CLOSED_STOP');ok('full407_current_source_SHA_verified',v['freeze']['files_verified']==407);ok('raw_omission_binds_permanent29_catalog_SHA',len(v['raw'])==1 and v['raw'][0]['permanent_archive_path'].startswith('/SYNTHETIC29/'))
 for suffix in ['.bin','.pt','.pth','.safetensors','.npy','.npz','.tensor','.so','.dylib','.dll']:
  ok('forbidden_'+suffix+'_excluded',S.excluded('payload/forbidden'+suffix)=='raw_model_or_native_binary')
 ok('versioned_DSO_excluded',S.excluded('payload/libx.so.12')=='raw_model_or_native_binary');ok('draft_monitor_delivery_temp_excluded',all(S.excluded(x)for x in ['drafts/a.py','monitoring/a.json','final_delivery/a.json','operations/SMALL_DELIVERY_STATE.jsonl','payload/a.tmp']))
 fake=root/'analysis/disguised.json';fake.write_bytes(b'\x7fELFpretend');reject('ELF_disguised_small_file_rejected',lambda:S.source_roster(root));fake.unlink();fake.symlink_to(root/S.REPORT);reject('selected_symlink_rejected',lambda:S.source_roster(root));fake.unlink();(root/'benchmarks').mkdir();(root/'benchmarks/empty.csv').write_text('only,header\n');reject('placeholder_scienceCSV_rejected',lambda:S.source_roster(root));(root/'benchmarks/empty.csv').unlink()
 for label,mutator,name in [('review_notPASS',lambda x:x.update(status='PENDING'),S.REVIEW),('review_findings',lambda x:x.update(findings=['issue']),S.REVIEW),('review_count_notcomplete',lambda x:x.update(checks_passed=0),S.REVIEW),('release_own_NVML',lambda x:x.update(final_owned_NVML_records=[dict(pid=10)]),S.RELEASE),('release_unknown_process',lambda x:x['own_processes'][0].update(process_exit_verified=False),S.RELEASE),('archive_notPASS',lambda x:x.update(source_two_stable_SHA_passes=False),S.ARCHIVE),('external_not249',lambda x:x.update(external_files_verified=248),S.SOURCE)]:
  previous=(root/name).read_bytes();value=json.loads(previous);mutator(value);save(root,name,value);reject(label+'_blocksseal',lambda:S.validate(root));(root/name).write_bytes(previous)
 with mock.patch.object(S,'MAX_FILE',5):reject('64MiB_perfile_bound_checked',lambda:S.source_roster(root))
 with mock.patch.object(S,'MAX_TOTAL',5):reject('256MiB_total_bound_checked',lambda:S.source_roster(root))
 original=(root/'payload/source/f000.json').read_bytes();(root/'payload/source/f000.json').write_bytes(b'changed');reject('frozen_source_SHA_change_rejected',lambda:S.validate(root));(root/'payload/source/f000.json').write_bytes(original)
 result=S.seal(root);ok('one_shot_local_seal_and_bundle_PASS',result['status']=='PASS_POST_RELEASE_SMALL_DELIVERY_SEALED_BUNDLE');ok('STOP_not_promoted_or_CPUONLY',result['scientific_status']=='FAIL_CLOSED_STOP');dest=root/'final_delivery';settings=result['settings'];verified=S.check_payload(dest,settings);ok('manifest_SUMS_exactNplus3_roster',verified['actual_files']==settings['indexed_files']+3);manifest=S.read(dest,'SHA256_MANIFEST.json');ok('all407_sources_indexed',sum(x['path'].startswith('payload/source/')for x in manifest['files'])==407);ok('raw_tensor_not_in_small_delivery',not any(p.suffix=='.bin'for p in dest.rglob('*')));ok('externalreceipt_not_created',not(dest/'LOCAL_DELIVERY_VERIFICATION.json').exists());ok('source_raw_retained',(root/'payload/execution/raw/token0.bin').is_file());ok('no_false_threehost_physicalclaim','all_three_hosts_payload_exact_same_SHA_bytes_and_roster'not in S.read(dest,'FINAL_SEAL_RECEIPT.json'))
 with tarfile.open(result['bundle']['path'],'r:gz')as archive:
  members=archive.getmembers();ok('tar_flat_relative_Nplus3_no_links',len(members)==settings['indexed_files']+3 and all(m.isfile()and not m.name.startswith('/')and'..'not in Path(m.name).parts for m in members));ok('bundle_source_SHA_exact',hashlib.sha256(Path(result['bundle']['path']).read_bytes()).hexdigest()==result['bundle']['sha256'])
 reject('one_shot_seal_repeated_blocked',lambda:S.seal(root));changed=dest/S.REPORT;prior=changed.read_bytes();changed.write_bytes(b'tamper');reject('sealedfile_tamper_rejected',lambda:S.check_payload(dest,settings));changed.write_bytes(prior)
 ok('stdlib_only_no_GPU_Runtime_or_network','torch'not in sys.modules and not hasattr(S,'subprocess'))
print(json.dumps(dict(status='PASS_SYNTHETIC_CPU_SMALL_DELIVERY_TESTS_NOT_REAL_SEAL',checks_passed=len(checks),checks=checks,source_sha256=hashlib.sha256((RUN/'scripts/seal_small_delivery.py').read_bytes()).hexdigest(),GPU_calls=0,Driver_calls=0,network_calls=0,actual_run_sealed=False),sort_keys=True,indent=2))
