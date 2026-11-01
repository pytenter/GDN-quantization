"""One authorized independent CPU schema recovery after the preserved extra55 failure.

This unfrozen operator is separate from all scientific sources/claims/seals.
It is not the official delivery helper, finisher, collector or a GPU launcher.
"""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[1]
FAILED = ROOT / 'operations/FINAL_PACE_SCALE_REPAIR55_EXTRA_INDEPENDENT_REVIEW.json'
FAILED_SHA = '01cfbba39d35f5e487fa643f0b83f2f1bbce53fb65d456d06e2744ba1f8628df'
OUTPUT = ROOT / 'operations/FINAL_PACE_SCALE_REPAIR55_EXTRA_CPU_SCHEMA_RECOVERY_INDEPENDENT_REVIEW.json'
TRANSPORT_SHA = 'db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40'
SCHEMA_DIFF = {
    'before': "if external and 'inode' in row: require((actual_inode,actual_mtime)==(row['inode'],row['mtime_ns']))",
    'after': "if external and 'inode' in row: require(actual_inode==row['inode']); if external and 'mtime_ns' in row: require(actual_mtime==row['mtime_ns'])",
    'unchanged': 'Actual SHA/bytes and before/after inode/device/size/mtime stability remain mandatory; archived mapped-native inode is separately checked against current actual inode and historical full_maps.',
}

REMOTE_BODY = r'''
from pathlib import Path
import os,json,hashlib,datetime,subprocess
r=Path(P['root']);OLD=Path('/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1');UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422';cache={}
def req(v,s):
 if not v:raise AssertionError(s)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def check(root,rows,external=False):
 seen=set();verified=[]
 for row in rows:
  name=row['path'];req(name not in seen,'Duplicate file row');seen.add(name)
  p=Path(name) if external else root/name
  if not external:
   rel=Path(name);req(not rel.is_absolute() and '..' not in rel.parts and str(rel)==name,'Unsafe science path')
   cur=p
   while cur!=root:req(not cur.is_symlink(),'Science symlink forbidden');cur=cur.parent
  a=p.stat();req(p.is_file() and a.st_size==row['bytes'],'Byte size differs:'+str(p))
  key=(str(p.resolve()),a.st_ino,a.st_dev,a.st_size,a.st_mtime_ns)
  if key not in cache:cache[key]=sha(p)
  req(cache[key]==row['sha256'],'Fresh SHA differs:'+str(p))
  b=p.stat();req((a.st_ino,a.st_dev,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_dev,b.st_size,b.st_mtime_ns),'File changed during read:'+str(p))
  # Schema-only correction: loaded-native receipts legitimately omit mtime_ns.
  # Inode stays independently mandatory whenever the original row records it.
  if external and 'inode' in row:req(a.st_ino==row['inode'],'External inode differs:'+str(p))
  if external and 'mtime_ns' in row:req(a.st_mtime_ns==row['mtime_ns'],'External mtime differs:'+str(p))
  if external and 'resolved_path' in row:req(str(p.resolve())==row['resolved_path'],'External symlink target differs')
  verified.append({'path':name,'bytes':a.st_size,'sha256':cache[key],'inode':a.st_ino,'device':a.st_dev,'mtime_ns':a.st_mtime_ns})
 return {'status':'PASS_FRESH_ALL_SHA_BYTES','count':len(verified),'bytes':sum(x['bytes'] for x in verified),'fresh_tree_sha256':hashlib.sha256(canonical(verified)).hexdigest()}
def identity(pid):
 p=Path('/proc')/str(pid);parts=(p/'stat').read_text().rsplit(')',1)[1].split()
 return {'pid':pid,'starttick':int(parts[19]),'cmdline':[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]}
def gone(expected):
 try:x=identity(expected['pid'])
 except (FileNotFoundError,ProcessLookupError):return {'status':'GONE','expected':expected,'owned_alive':False}
 if x['starttick']!=expected['starttick']:return {'status':'PID_REUSED_FOREIGN_NOT_SIGNALLED','expected':expected,'actual':x,'owned_alive':False}
 return {'status':'OWNED_ALIVE_OR_SAME_TICK_UNRESOLVED','expected':expected,'actual':x,'owned_alive':True}
def current_identities():
 xs=[read(r/'OFFLINE_PARENT_STARTED.json')['identity'],read(r/'SESSION_STARTED.json')['identity'],read(r/'FINISHER_STARTED.json')['identity']]
 xs += [x['identity'] for x in read(r/'RESOURCE_LEDGER.json')['intervals']]
 rows={canonical(x):x for x in xs};req(set(rows)=={canonical(x) for x in P['expected_own_identities']},'Official exact own identity roster changed')
 return [gone(x) for x in rows.values()]
req(sha(r/'source_manifest.json')==P['source']=='f51840870f017786bf761a53804a81bfa5c4a941a2207f136349761e0272256f','Source543 changed')
req(sha(r/'GPU_PERMIT.json')==P['permit']=='32ca98b22157ec6b5c03ea648c39cc8f6e88b3d773e3ee245380fd53c5dbf2d7','Permit changed')
req(sha(r/'operations/HUMAN_PACE_SCALE_REPAIR_AUTHORIZATION.json')=='f9cfbada775455554bf3d33a4d22522e48ad0a75fab03ed2caa3611b72232ef3','New human grant changed')
before=current_identities();req(not any(x['owned_alive'] for x in before),'An original own process remains alive')
s=read(r/'source_manifest.json');req(s['count']==len(s['files'])==543,'Frozen543 count differs');source=check(r,s['files'])
req(sha(r/'DELIVERY_MANIFEST.json')==P['manifest'],'Official current primary manifest differs')
m=read(r/'DELIVERY_MANIFEST.json');req(m['count']==len(m['files']) and m['total_bytes']==sum(x['bytes'] for x in m['files']) and m['source_manifest_sha256']==P['source'] and m['permit_sha256']==P['permit'],'Current all-science manifest schema/pins differ');science=check(r,m['files']);late=check(r,P['late'])
externals={};per_manifest={}
for n in ['EXTERNAL_NATIVE_BINDINGS.json','EXTERNAL_PYTHON_BINDINGS.json','MODEL_TOKENIZER_FRESH_SHA.json']:
 p=r/'analysis'/n;data=read(p)
 if n=='MODEL_TOKENIZER_FRESH_SHA.json':req(len(data['files'])==17,'Full model/tokenizer17 required')
 per_manifest[n]={'manifest_sha256':sha(p),'verification':check(None,data['files'],True)}
 for row in data['files']:
  key=str(Path(row['path']).resolve());old=externals.setdefault(key,row)
  req((old['bytes'],old['sha256'])==(row['bytes'],row['sha256']),'Conflicting external catalog')
external=check(None,list(externals.values()),True)
t=P['triton'];trows=t['files']+t['distribution_metadata']+t['host_c_compiler'];req(len({x['path'] for x in trows})==len(trows)==322,'Fixed Triton+metadata+compiler322 roster differs');triton_proof=check(None,trows,True)
package=Path(t['package_root']);actual={str(p) for p in package.rglob('*') if p.is_file() and not any(part in ('__pycache__','.cache','cache') for part in p.relative_to(package).parts) and p.suffix!='.pyc'}
req(actual=={x['path'] for x in t['files']},'Full Triton311 package inventory differs')
ledger=read(r/'RESOURCE_LEDGER.json');intervals=ledger['intervals'];req(ledger['active_unit'] is None and len(intervals)==2 and ledger['all_recorded_workers_released'] is True,'Current interval roster not terminal two')
req(all(x['released'] is True and x['accounting_final'] is True and x['lower_bound_only'] is False for x in intervals),'Current actual release not PASS')
req([(x['stage'],x['unit'],x['returncode']) for x in intervals]==[('scale_repair_gates','SCALE_REPAIR_GATES',0),('formal','aime26_02_R3_seed1',1)],'New actual gate/failed formal statuses changed')
req(abs(ledger['cumulative_GPU_seconds']-sum(x['GPU_seconds'] for x in intervals))<1e-5 and ledger['physical_GPU_active_seconds']=='UNRESOLVED_NOT_MEASURED','Current cost accounting differs')
units=[('scale_repair_gates','SCALE_REPAIR_GATES',r/'execution/SCALE_REPAIR_GATES'),('formal','aime26_02_R3_seed1',r/'R3_PACE48_DP1/trajectories/aime26_02_R3_seed1')]
native=[];driver=[]
for stage,unit,target in units:
 i=read(target/'RESOURCE_INTERVAL.json');d=read(target/'DISPATCH.json');started=read(target/'STARTED.json');child=read(target/'ACTUAL_CHILD_IDENTITY.json');req(child==i['identity']==started['identity'] and child['pid']==i['pid'] and started['dispatch_sha256']==sha(target/'DISPATCH.json'),'Exact new unit actual identity binding differs')
 req(d['unit']==unit and d['stage']==stage and d['source_manifest_sha256']==P['source'] and d['permit_sha256']==P['permit'] and d['gpu_uuid']==UUID and d['gpu']==0,'New dispatch source/permit/UUID differs')
 dr=read(target/'DRIVER_IDENTITY.json');req(dr['status']=='PASS' and dr['actual_uuid']==UUID and dr['visible_device_count']==1 and dr['CUDA_VISIBLE_DEVICES']==UUID,'Recorded actual driver UUID differs');driver.append({'unit':unit,'driver_sha256':sha(target/'DRIVER_IDENTITY.json'),'actual_uuid':dr['actual_uuid']})
 for path in sorted(target.glob('LOADED_LIBRARY_*.json'))+([target/'POSTFLIGHT_LIBRARY.json'] if (target/'POSTFLIGHT_LIBRARY.json').is_file() else []):
  a=read(path);req(a['status']=='PASS_ACTUAL_LOADED_NATIVE_SHA_INODE' and a['selected'],'Actual native record differs')
  mapped={}
  for line in a['full_maps'].splitlines():
   f=line.split(maxsplit=5)
   if len(f)==6 and f[5].startswith('/'):
    mapped.setdefault(str(Path(f[5]).resolve()),set()).add(int(f[4]))
  for row in a['selected']:
   key=str(Path(row['path']).resolve());req(key in externals and row['sha256']==externals[key]['sha256'] and row['bytes']==externals[key]['bytes'],'Actual mapped native not in full frozen external catalog')
   check(None,[row],True);req(Path(key).stat().st_ino==row['inode'] and row['inode'] in mapped.get(key,set()),'Actual historical mapped inode/fresh native identity differs')
  native.append({'unit':unit,'artifact':path.relative_to(r).as_posix(),'artifact_sha256':sha(path),'actual_mapped_count':len(a['selected']),'fresh_native_rows_tree_sha256':hashlib.sha256(canonical(a['selected'])).hexdigest()})
gate=read(units[0][2]/'RESULT.json');gcopy=dict(gate);gp=gcopy.pop('payload_sha256');req(hashlib.sha256(canonical(gcopy)).hexdigest()==gp and gate['status']=='PASS' and gate['identity']==intervals[0]['identity'],'New gate original payload differs')
failed=units[1][2];req(not (failed/'RESULT.json').exists(),'Failed current formal gained a completion receipt');failure=read(failed/'FIRST_FAILURE.json');req(failure['formal_completion_credit']==0 and failure['no_automatic_retry'] is True,'Failed formal credit/retry changed')
session=read(r/'SESSION_RESULT.json');verdict=read(r/'final_verdict.json');req(session['status']==verdict['status']=='FAIL_CLOSED_STOP' and session['current_formal_count']==0 and session['reused_formal_count']==3 and session['formal_total_unique']==verdict['formal_completed']==3 and verdict['R2_completed']==2 and verdict['R3_completed']==1,'Current0/prior3 STOP counts differ')
req(verdict['full_accuracy_reported'] is False and verdict['partial_scientific_CSV_created'] is False and verdict['original_uninterrupted_60'] is False,'Partial/mixed scientific scope changed')
req(not list(r.glob('FROZEN_V4_FRESH_PASS_*.json')) and not (r/'comparison/paired_results.csv').exists() and not (r/'R2_PACE48_DP8/results.jsonl').exists() and not (r/'R3_PACE48_DP1/results.jsonl').exists(),'Partial generated scores/full scientific table')
for name,expected_rc in [('CONTROLLER',1),('FINISHER',0)]:
 x=read(r/('PARENT_'+name+'_EXIT.json'));req(x['actual_exited'] is True and x['identity_gone'] is True and x['returncode']==expected_rc and x['source_manifest_sha256']==P['source'] and x['permit_sha256']==P['permit'],'Actual parent-owned child exit receipt differs')
 req(x['identity']==read(r/('PARENT_'+name+'_ACTUAL_CHILD_IDENTITY.json')),'Parent actual child identity differs')
parent=read(r/'OFFLINE_PARENT_RESULT.json');req(parent['status']=='PARENT_TERMINAL' and parent['failure'] is None and parent['source_manifest_sha256']==P['source'] and parent['permit_sha256']==P['permit'],'Actual offline parent receipt differs')
req(sha(OLD/'source_manifest.json')=='700b707d450a14fb5dbd52aef3df6c0fa6ac59649359eb0712cac87a951113ca' and sha(OLD/'GPU_PERMIT.json')=='3cb0900e85fea2124fbcbfd3b1c764ce833e3de3e037cadd06f990c230914c31' and sha(OLD/'SCIENCE_FREEZE.json')=='9943928865d7259e200bc4b2df0e368ee6d230821ae23e230a5b278ed26b4a2e','Original133 origin pins changed')
osrc=read(OLD/'source_manifest.json');om=read(OLD/'DELIVERY_MANIFEST.json');req(osrc['count']==133 and om['count']==317 and sha(OLD/'DELIVERY_MANIFEST.json')=='a803c18c7aa8bf90e69b9d01e9ece3d1599a6bf51fa371f4ff1250d08c332c18','Original count/seal differs');old_source=check(OLD,osrc['files']);old_science=check(OLD,om['files']);old_late=check(OLD,P['old_late'])
req(read(OLD/'final_verdict.json')['formal_completed']==3 and read(OLD/'final_verdict.json')['status']=='FAIL_CLOSED_STOP','Original STOP reclassified')
original=read(r/'SCALE_REPAIR_UNIQUE_LAUNCH_CLAIM.json');receipt=read(r/'UNIQUE_OFFLINE_LAUNCH_RECEIPT.json');compat=read(r/'CPU_TMUX_COMPAT_LAUNCH_CLAIM.json');actual=read(r/'CPU_TMUX_COMPAT_ACTUAL_LAUNCH_RECEIPT.json')
req(receipt['returncode']==1 and receipt['claim_sha256']==sha(r/'SCALE_REPAIR_UNIQUE_LAUNCH_CLAIM.json') and compat['original_failed_launch_receipt_sha256']==actual['original_failed_launch_receipt_sha256']==sha(r/'UNIQUE_OFFLINE_LAUNCH_RECEIPT.json'),'Original CPU launch failure/claim changed')
req(compat['original_claim_preserved'] is True and compat['zero_prior_scientific_dispatch'] is True and compat['no_frozen_source_modified'] is True and compat['human_authorization_sha256']=='f9cfbada775455554bf3d33a4d22522e48ad0a75fab03ed2caa3611b72232ef3','CPU compatibility authorization/scope changed')
opsha='ecb481ad605a04f6bbdc0f48d6d69342223ea54ad862fd27f1a7975dabf87a87';opdir=r/'operations/operator_supplements'/opsha
req(sha(opdir/'CPU_ONLY_TMUX_COMPAT_LAUNCH_RECOVERY.py')==compat['operator_supplement_sha256']==actual['operator_supplement_sha256']==opsha and sha(opdir/'TMUX_COMPAT_NO_SCIENTIFIC_DISPATCH_INDEPENDENT_REVIEW.json')==compat['independent_zero_dispatch_review_sha256'],'CPU operator/review content differs')
req(actual['returncode']==0 and actual['claim_sha256']==sha(r/'CPU_TMUX_COMPAT_LAUNCH_CLAIM.json') and actual['source_manifest_sha256']==compat['source_manifest_sha256']==P['source'] and actual['permit_sha256']==compat['permit_sha256']==P['permit'],'Single actual launch changed')
initial=actual['initial_pane_identities'][0];req(initial['pid']==parent['identity']['pid'] and initial['starttick']==parent['identity']['starttick'],'Initial exec-transition parent identity differs')
p=subprocess.run(['nvidia-smi','--id='+UUID,'--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);req(p.returncode==0,'GPU0 NVML inventory unavailable');device=[x.strip() for x in p.stdout.strip().split(',')];req(len(device)==5 and device[0]=='0' and device[1]==UUID,'PhysicalGPU0 UUID differs')
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);req(p.returncode==0,'NVML ownership inventory unavailable');apps=[]
for line in p.stdout.splitlines():
 if not line.strip():continue
 f=[x.strip() for x in line.split(',')];req(len(f)==3,'Malformed NVML app');apps.append({'pid':int(f[0]),'uuid':f[1],'memory':f[2]})
after=current_identities();req(not any(x['owned_alive'] for x in after),'An original own identity revived')
for row in after:
 for app in apps:
  if app['pid']==row['expected']['pid']:req(row['status']=='PID_REUSED_FOREIGN_NOT_SIGNALLED','Original ownPID still in NVML')
p=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True,timeout=10)
if p.returncode==0:req('qwen-pace48-scale-repair-gpu0-20261025' not in p.stdout.splitlines(),'Exact original tmux session remains')
else:req(p.returncode==1 and any(x in p.stderr.lower() for x in ('no server running','failed to connect','no sessions')),'Tmux absence unresolved')
req(sha(r/'source_manifest.json')==P['source'] and sha(r/'GPU_PERMIT.json')==P['permit'] and sha(r/'DELIVERY_MANIFEST.json')==P['manifest'],'Frozen/seal pin changed during extra review')
print(json.dumps({'status':'PASS_FINAL_PACE_SCALE_REPAIR55_EXTRA_INDEPENDENT_CPU_SCHEMA_RECOVERY','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source543':source,'all_current_science':science,'late_current_exit_receipts':late,'external_unique_all':external,'external_manifests':per_manifest,'extra_triton322':{'verification':triton_proof,'inventory_origin_sha256':P['triton_manifest_sha256'],'inventory_scope':'Additional fixed same-bitdecode environment Triton311+metadata6+hostcompiler5 inherited frozen CPU inventory; no WScale process/output operation'},'actual_native_records':native,'actual_driver_UUIDs':driver,'identities_before':before,'identities_after':after,'current_intervals':[{'stage':x['stage'],'unit':x['unit'],'returncode':x['returncode'],'released':x['released'],'identity':x['identity'],'GPU_seconds':x['GPU_seconds']} for x in intervals],'counts':{'new_formal_completed':0,'reused_formal_completed':3,'unique_formal_completed':3,'new_repair_gate_passed':1},'current_failure_error':failure['error'],'current_failure_sha256':sha(failed/'FIRST_FAILURE.json'),'old_source133':old_source,'old_science317':old_science,'old_late55_mirrors':old_late,'old_nine_file_supplement_on29':'Separate29extra_review_scope_not_queried_here','CPU_tmux_operator_and_claims':'PASS_ORIGINAL_FAILED_COMMAND_RETAINED_FIRST_ACTUAL_LAUNCH_SINGLE_BOUND_SCOPE','original_parent_initial_exec_transition':'SAME_PID_AND_STARTTICK_ORIGINAL_ARGV_PRESERVED','exact_tmux_session_absent':True,'NVML_readonly_queries':2,'GPU0_device':device,'foreign_NVML_app_count':len(apps),'GPU_calls':0,'GPU_model_imports':0,'model_loads':0,'scores_recomputed':0,'signals':0,'remote_mutations':0,'transfers':0,'helper_or_finisher_calls':0,'science_seals_modified':False,'physical_GPU_active_seconds':'UNRESOLVED_NOT_MEASURED'},sort_keys=True))
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload():
    d = ROOT / 'independent_delivery'
    result = json.loads((d / 'RESULT.json').read_text())
    audit = json.loads((d / 'AUDIT55.json').read_text())
    assert result['status'] == 'PASS_INDEPENDENT_TERMINAL_DELIVERY_AND_STOP'
    assert result['formal_completed'] == 3 and result['final_status'] == 'FAIL_CLOSED_STOP'
    old = ROOT.with_name('QWEN_GDN_PACE48_AIME26_SEED1_60_V1')
    old55 = json.loads((old / 'independent_delivery/AUDIT55.json').read_text())
    triton_path = ROOT.with_name('QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_CUPTI_REPAIR_20261025') / 'analysis/TRITON_RUNTIME_FRESH_SHA.json'
    assert digest(triton_path) == 'bc695cc73e549272d9231cfe1f6b833dbf3519c9964e3820746acfee92dcafe5'
    return dict(root=json.loads((ROOT / 'ROOT.json').read_text())['remote_root'],
                source=result['source_manifest_sha256'], permit=result['permit_sha256'],
                manifest=result['core_manifest_sha256'], late=audit['late_files'],
                old_late=old55['late_files'], triton=json.loads(triton_path.read_text()),
                triton_manifest_sha256=digest(triton_path),
                expected_own_identities=[x['expected'] for x in audit['terminal_snapshot']['identities']])


def main():
    assert digest(FAILED) == FAILED_SHA, 'Preserved failed extra55 review changed'
    assert not OUTPUT.exists(), 'This distinct CPU recovery already produced a receipt; never retry'
    transport = ROOT / 'operations/remote.py'
    assert digest(transport) == TRANSPORT_SHA
    p = payload()
    code = 'P=' + repr(p) + '\n' + REMOTE_BODY
    compile(code, '<extra55_schema_recovery>', 'exec')
    local_inputs = {n: digest(ROOT / 'independent_delivery' / n) for n in
                    ('RESULT.json', 'AUDIT55.json', 'AUDIT29.json', 'SUPPLEMENT_ARCHIVE_VERIFICATION.json')}
    spec = importlib.util.spec_from_file_location('extra55_schema_recovery_transport', transport)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    try:
        observed = module.run(code, role='55', python='/data/ydai/miniconda3/envs/bitdecode/bin/python', timeout=300)
        report = dict(status=observed['status'], remote=observed,
                      scope='DISTINCT_ONE_OFF_CPU_SCHEMA_RECOVERY_FULL_FRESH_PASS_NOT_OFFICIAL_HELPER_REPEAT',
                      preserved_failed_review_sha256=FAILED_SHA, schema_diff=SCHEMA_DIFF,
                      official_local_input_sha256=local_inputs, operator_sha256=digest(__file__),
                      remote_verifier_sha256=hashlib.sha256(code.encode()).hexdigest())
    except BaseException as exc:
        report = dict(status='FAIL_EXTRA_CPU_SCHEMA_RECOVERY_NO_AUTOMATIC_RETRY', error=repr(exc),
                      preserved_failed_review_sha256=FAILED_SHA, schema_diff=SCHEMA_DIFF,
                      official_local_input_sha256=local_inputs, operator_sha256=digest(__file__),
                      remote_verifier_sha256=hashlib.sha256(code.encode()).hexdigest(),
                      GPU_calls=0, remote_mutations=0, no_automatic_retry=True)
    assert digest(FAILED) == FAILED_SHA, 'Preserved failed review changed during recovery'
    raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    with OUTPUT.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(status=report['status'], path=str(OUTPUT), sha256=hashlib.sha256(raw).hexdigest())))


if __name__ == '__main__':
    main()
