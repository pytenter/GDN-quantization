#!/usr/bin/env python3
"""CPU-only immutable four-card run-payload builder. Never queries/uses GPUs/SSH.

Default is preview; --build creates a NEW local staging tree. Remote transfer,
real device/storage admission, and GPU dispatch belong to the parent launcher.
"""
import argparse,copy,datetime,hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
TASK='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'
RUN='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_GPU_20261019_V1'
REMOTE='/data/zypan/analyses/qwen_gdn_fast_runtime_qualification_v1/gpu_runs/CUBLAS_4GPU_20261019_V1'
ARCHIVE='/data/zypan/qwen_fast_runtime_qualification_archive/CUBLAS_4GPU_20261019_V1'
DEVICES=[{'physical_index':0,'uuid':'GPU-a78176ed-f97f-c54f-6201-5a01b570f422'},{'physical_index':1,'uuid':'GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3'},{'physical_index':2,'uuid':'GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31'},{'physical_index':3,'uuid':'GPU-30b07aba-a728-7649-4899-e09d33f8bc75'}]
REQUIRED_SCRIPTS=['cublas_worker.py','cublas_harness.py','cublas_compare.py','archive_io.py','run_cublas_controller.py']
def require(ok,message):
 if not ok:raise RuntimeError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def encode(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(p):return json.loads(Path(p).read_text())
def stable(p):
 p=Path(p);a=p.stat();raw=p.read_bytes();b=p.stat();require((a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns),'Source changed during read:'+str(p));return raw
def safe_relative(name):
 p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe relative asset path:'+name);return p

def verify_cpu(cpu):
 receipt=read(cpu/'FINAL_SEAL_RECEIPT.json');raw=stable(cpu/'SHA256_MANIFEST.json');manifest=json.loads(raw)
 require(receipt['status']=='SEALED_CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED' and receipt['manifest_sha256']==digest(raw),'Original CPU seal invalid')
 require(receipt['files_verified']==manifest['files_count']==151,'Unexpected original151-file CPU seal')
 for row in manifest['files']:
  data=stable(cpu/safe_relative(row['path']));require(len(data)==row['bytes'] and digest(data)==row['sha256'],'CPU sealed asset changed:'+row['path'])
 srcraw=stable(cpu/'SOURCE_MANIFEST.json');require(digest(srcraw)==receipt['source_manifest_sha256'],'Original CPU source manifest mismatch');src=json.loads(srcraw);require(len(src['execution_sources'])==95,'Expected95 original execution assets')
 require(digest(stable(cpu/'SHA256SUMS.txt'))==receipt['SHA256SUMS_sha256'],'Original CPU SHA256SUMS changed')
 return receipt,manifest,src

def validate_human(raw):
 h=json.loads(raw);require(h['task_id']==TASK and h['execution_run_id']==RUN and h['ALLOW_GPU_DIAG']=='YES','Wrong human authorization')
 require(h['maximum_active_GPUs']==4 and h['GPU_time_limit'] is None,'Human resource amendment missing')
 require(all(x in h['overrides'] for x in ['previoussingleGPUresourcelimit','previous1800s0.5GPUhcap','previous1650sprotectivetimecap','previousperunitwalltimeallowancekill']),'Prior resource overrides incomplete');return h

def freeze_inputs(cpu,new):
 receipt,manifest,src=verify_cpu(cpu);human=stable(new/'HUMAN_GPU_AUTHORIZATION.json');validate_human(human)
 amendment=stable(new/'analysis/GPU_EXECUTION_AMENDMENT.json');a=json.loads(amendment);require(a['human_authorization_sha256']==digest(human),'Amendment human binding changed');require(a['shared_control_root']==REMOTE and a['device_strata']==DEVICES,'Unexpected amendment root/devices')
 original={row['path']:stable(cpu/safe_relative(row['path'])) for row in src['execution_sources']}
 overlays={str(p.relative_to(new)):stable(p) for p in sorted((new/'scripts').glob('*.py'))}
 for name in REQUIRED_SCRIPTS:require('scripts/'+name in overlays,'Amended runtime script not prepared:'+name)
 require('estimated_seconds' in overlays['scripts/cublas_harness.py'].decode(),'Harness estimates not yet amended')
 module_name='_frozen_gpu_builder_harness';spec=importlib.util.spec_from_file_location(module_name,new/'scripts/cublas_harness.py');mod=importlib.util.module_from_spec(spec);sys.modules[module_name]=mod;spec.loader.exec_module(mod);units=mod.build_stage_schedule()
 require('torch' not in sys.modules and len(units)==16 and len({x['unit_id'] for x in units})==16,'Unsafe harness import or wrong finite16 roster')
 require([x['unit_id'] for x in units[:4]]==['B0_FP_Q0','B0_FP_Q1','B1_FP_Q0','B1_FP_Q1'],'Numeric-first schedule changed')
 require([x['unit_id'] for x in units[4:10]]==['B0_FP_P0','B1_FP_P0','B1_FP_P1','B0_FP_P1','B0_FP_P2','B1_FP_P2'],'Performance roster changed')
 require(all('allowance_seconds' not in u for u in units),'No old perunit allowance cap allowed')
 anchors={name:stable(cpu/name) for name in ['FINAL_SEAL_RECEIPT.json','SHA256_MANIFEST.json','SOURCE_MANIFEST.json','SOURCE_FREEZE.json','SHA256SUMS.txt']}
 evidence_paths=set((new/'qualification').glob('*.json'))|set((new/'analysis').glob('*CPU*.json'))
 for name in ['INDEPENDENT_MATH_AST_REVIEW.json','REMOTE_PREFLIGHT_FRESH_DEPENDENCIES.json','FINAL_INDEPENDENT_SOURCE_REVIEW.json']:
  if(new/'analysis'/name).is_file():evidence_paths.add(new/'analysis'/name)
 new_evidence={str(p.relative_to(new)):stable(p) for p in sorted(evidence_paths)}
 return dict(new_evidence=new_evidence,receipt=receipt,manifest=manifest,src=src,human=human,amendment=amendment,amendment_md=stable(new/'analysis/GPU_EXECUTION_AMENDMENT.md'),original=original,overlays=overlays,units=units,anchors=anchors)

def card_contract(base,device):
 out=copy.deepcopy(base);out['historical_CPU_contract_sha256']=digest(encode(base)) # serialized identity additionally exact below
 out.update(status='GPU_AUTHORIZED_PENDING_REAL_ADMISSION_NOT_NUMERIC_PASS',ALLOW_GPU_DIAG='YES',execution_run_id=RUN,execution_device_binding=copy.deepcopy(device),no_GPU_time_limit=True)
 # Historical recorded environment remains separate; only prospective mapping changes.
 out['environment']['CUDA_VISIBLE_DEVICES']=str(device['physical_index']);out['environment']['preferred_GPU_UUID']=device['uuid'];out['environment']['physical_gpu']=device['physical_index']
 out['qualification_status']={k:('FILL_OPT_SAFETY_UNPROVEN_NO_FILL_GPU_AUTHORIZATION' if k=='fill_off' else 'NOT_CREATED' if k=='fast_script' else 'NOT_RUN_GPU_AUTHORIZED_PENDING_REAL_GATES') for k in out['qualification_status']}
 out['candidate_limits']['GPU_stage_admission']='HumanYES amended4strata+frozennewsource+freshactualdevice/storage+numericalfirst; no time caps;1800s no-progress watchdog only.'
 out['historical_CPU_contract_note']='Original CPU151-file seal untouched; prospective physical mapping separately authorized, logical cuda:0/GDNmath unchanged.'
 return out

def card_plan(base,device,root,card_roots,units,human_sha,amendment_sha,archive_sha,contract_sha):
 out=copy.deepcopy(base);out.update(status='GPU_AUTHORIZED_FROZEN_PENDING_REAL_ADMISSION',ALLOW_GPU_DIAG='YES',execution_run_id=RUN,future_execution_root=root,shared_control_root=REMOTE,card_roots=card_roots,device=copy.deepcopy(device),no_GPU_time_limit=True,progress_watchdog_seconds=1800,human_authorization_sha256=human_sha,amendment_sha256=amendment_sha,archive_config_sha256=archive_sha,contract_sha256=contract_sha,max_active_GPUs=4,per_card_max_active_GPUs=1)
 for key in ['GPU_hard_seconds','protective_seconds','cleanup_seconds','mandatory_conservative_seconds']:out[key]=None
 out['conditional_allowances_are_not_extra_budget']=False
 out['prior_time_limits']='Overridden only by bound humanresourceamendment; no elapsedtime/perunitallowance kill. Numeric/source/storage/physicalfailclosed retained.'
 out['admission_rule']='Exact human/source/parentdispatch/device/archive checks before eachunit. All4FPnumericalPASS barrier before globally serialized no-contention performance. All4performance terminal barrier before conditionally eligible profile/H; no time-cap gate.'
 out['minimum_free_new_root_bytes']=4*1024**3
 out['raw_snapshots']='Directstream each B0_Q0 token to independently SHA/bytesverified atomic29archive;55andMac metadataonly. Otherfresh runs read exactremote reference. Keepfirstdifference remote. Nohistoricaldeletion.'
 out['units']=copy.deepcopy(units)
 for group in out['conditional_groups']:
  group.pop('seconds_allowance',None)
  group['requires']='Thiscard strictFPnumericPASS and frozen>=5percent+nonoverlap PASS_BENEFIT; globalmandatorybarriers; no new configuration/retry. NoGPUtime limit.'
 out['numeric_first_global_barrier']=True;out['performance_global_flock_serialization']=True;out['performance_all_cards_barrier']=True;out['no_cross_card_byte_reference']=True;out['no_pooled_TPS']=True
 out['D3_Dual_M']='Conditional isolated subsequent Codec roots/source/preregfreeze only after allcurrentworkerprocessesexit and NVMLrelease; pereligibledevice B0two+B1two fresh256 numericalgroup,4total. Current16-unit authorization doesnotcontainDual-M; noactivehotpatch/noFPH-TPSinheritance.'
 return out

def describe_inputs(info):
 return dict(status='CPU_PREPARED_NOT_DISPATCHED',task_id=TASK,execution_run_id=RUN,original_CPU_seal_files=info['manifest']['files_count'],original_execution_files=len(info['original']),amended_script_files=len(info['overlays']),new_CPU_evidence_files=len(info['new_evidence']),units_per_device=len(info['units']),devices=DEVICES,shared_control_root=REMOTE,archive_root=ARCHIVE,maximum_active_GPUs=4,GPU_time_limit=None,progress_watchdog_seconds=1800,source_overlay_SHA={n:digest(v) for n,v in info['overlays'].items()},human_sha256=digest(info['human']),amendment_sha256=digest(info['amendment']))

def build(cpu,new,destination,info):
 cpu=cpu.resolve();new=new.resolve();destination=destination.resolve();require(destination!=cpu and cpu not in destination.parents,'Never stage within sealed CPU root');require((new/'staging').resolve() in destination.parents,'Staging output must be NEW task staging descendant');require(not destination.exists(),'Existing payload cannot overwrite/restart');destination.mkdir(parents=True,exist_ok=False)
 def put(directory,name,raw):
  p=directory/safe_relative(name);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
 def obj(directory,name,value):put(directory,name,encode(value))
 card_roots={f'gpu{x["physical_index"]}':REMOTE+'/gpu'+str(x['physical_index']) for x in DEVICES};cards=[]
 put(destination,'HUMAN_GPU_AUTHORIZATION.json',info['human']);put(destination,'GPU_EXECUTION_AMENDMENT.json',info['amendment']);put(destination,'GPU_EXECUTION_AMENDMENT.md',info['amendment_md']);obj(destination,'CARD_ROSTER.json',dict(task_id=TASK,execution_run_id=RUN,shared_control_root=REMOTE,card_roots=card_roots,devices=DEVICES,no_GPU_time_limit=True,no_pooled_TPS=True,source_CPU_seal_manifest_sha256=info['receipt']['manifest_sha256']))
 put(destination,'scripts/run_four_card_session.py',info['overlays']['scripts/run_four_card_session.py'])
 for n,raw in info['new_evidence'].items():put(destination,n,raw)
 basecontract=json.loads(info['original']['analysis/STRICT_BASELINE_CONTRACT.json']);baseplan=json.loads(info['original']['qualification/CUBLAS_PREREGISTRATION.json'])
 for device in DEVICES:
  key='gpu'+str(device['physical_index']);directory=destination/key;directory.mkdir();remote=card_roots[key]
  payload=dict(info['original']);payload.update(info['overlays']);payload['HUMAN_GPU_AUTHORIZATION.json']=info['human'];payload['GPU_EXECUTION_AMENDMENT.json']=info['amendment'];payload['GPU_EXECUTION_AMENDMENT.md']=info['amendment_md']
  for n,v in info['anchors'].items():payload['evidence/ORIGINAL_CPU_'+n]=v
  archive=dict(task_id=TASK,execution_run_id=RUN,host='zypan@202.38.247.29',control_path='/tmp/qwen_replication_archive29.sock',archive_root=ARCHIVE,card_key=key,required_quota_bytes=200*1024**3,global_reservation_bytes=800*1024**3,global_headroom_bytes=100*1024**3,remote_python='python3',io_timeout_seconds=300,metadata_only_local=True)
  payload['ARCHIVE_CONFIG.json']=encode(archive);contract=card_contract(basecontract,device);contract['historical_CPU_contract_sha256']=digest(info['original']['analysis/STRICT_BASELINE_CONTRACT.json']);payload['analysis/STRICT_BASELINE_CONTRACT.json']=encode(contract)
  plan=card_plan(baseplan,device,remote,card_roots,info['units'],digest(info['human']),digest(info['amendment']),digest(payload['ARCHIVE_CONFIG.json']),digest(payload['analysis/STRICT_BASELINE_CONTRACT.json']));payload['qualification/CUBLAS_PREREGISTRATION.json']=encode(plan)
  origin=dict(task_id=TASK,execution_run_id=RUN,status='NEW_AUTHORIZED_RUN_COPY_OLD_CPU_SEAL_UNCHANGED',CPU_source_root=str(cpu),CPU_manifest_sha256=info['receipt']['manifest_sha256'],CPU_source_manifest_sha256=info['receipt']['source_manifest_sha256'],CPU_seal_sha256=digest(info['anchors']['FINAL_SEAL_RECEIPT.json']),CPU_files_verified=151,original_execution_files_copied=95,authorized_amendment_sha256=digest(info['amendment']),human_authorization_sha256=digest(info['human']),execution_root=remote,device=device,overlay_original_source_changes=[dict(path=n,CPU_sha256=digest(info['original'][n]) if n in info['original'] else None,new_sha256=digest(raw)) for n,raw in sorted(info['overlays'].items())])
  for name,raw in payload.items():put(directory,name,raw)
  source=dict(task_id=TASK,execution_run_id=RUN,status='FROZEN_BEFORE_ANY_GPU_DISPATCH_NOT_NUMERICAL_PASS',ALLOW_GPU_DIAG='YES',execution_sources=[dict(path=n,bytes=len(raw),sha256=digest(raw)) for n,raw in sorted(payload.items())],external_dependencies=copy.deepcopy(info['src']['external_dependencies']),external_weight_hash_scope=info['src'].get('external_weight_hash_scope'),device=device,no_GPU_time_limit=True,immutable_CPU_seal_manifest_sha256=info['receipt']['manifest_sha256'])
  sourceraw=encode(source);put(directory,'SOURCE_MANIFEST.json',sourceraw)
  origin['source_manifest_sha256']=digest(sourceraw);originraw=encode(origin);put(directory,'RUN_COPY_ORIGIN.json',originraw)
  auth=dict(task_id=TASK,execution_run_id=RUN,status='PASS',authorization_meaning='Explicit human GPU authorization plus frozenpayload; actualdevice/storage/numericaladmissionNOT_RUN',ALLOW_GPU_DIAG='YES',root=remote,device=device,shared_control_root=REMOTE,card_roots=card_roots,card_key=key,max_active_GPUs=4,per_card_max_active_GPUs=1,no_GPU_time_limit=True,GPU_time_limit=None,progress_watchdog_seconds=1800,source_manifest_sha256=digest(sourceraw),run_copy_origin_sha256=digest(originraw),contract_sha256=digest(payload['analysis/STRICT_BASELINE_CONTRACT.json']),preregistration_sha256=digest(payload['qualification/CUBLAS_PREREGISTRATION.json']),human_authorization_sha256=digest(info['human']),amendment_sha256=digest(info['amendment']),archive_config_sha256=digest(payload['ARCHIVE_CONFIG.json']),CPU_seal_manifest_sha256=info['receipt']['manifest_sha256'],units=copy.deepcopy(info['units']),allowed_modes=['qualify','benchmark','profile'],allow_new_experiments=False,no_fill_GPU_dispatch=True,no_automatic_fast_script=True)
  obj(directory,'AUTHORIZATION.json',auth);obj(directory,'SOURCE_FREEZE.json',dict(task_id=TASK,execution_run_id=RUN,status='FROZEN_BEFORE_ANY_GPU_DISPATCH',source_manifest_sha256=digest(sourceraw),authorization_sha256=digest(encode(auth)),contract_sha256=auth['contract_sha256'],preregistration_sha256=auth['preregistration_sha256'],human_authorization_sha256=auth['human_authorization_sha256'],device=device,GPU_dispatches=0,no_GPU_time_limit=True))
  obj(directory,'ROOT.json',dict(task_id=TASK,execution_run_id=RUN,remote_root=remote,device=device,archive_root=ARCHIVE+'/'+key,shared_control_root=REMOTE,source_manifest_sha256=digest(sourceraw),authorization_sha256=digest(encode(auth)),GPU_dispatches=0,status='STAGED_NOT_DISPATCHED'))
  cards.append(dict(card_key=key,remote_root=remote,source_manifest_sha256=digest(sourceraw),authorization_sha256=digest(encode(auth)),contract_sha256=auth['contract_sha256'],preregistration_sha256=auth['preregistration_sha256'],archive_config_sha256=auth['archive_config_sha256']))
 # Refuse a concurrent edit to any overlay/human/amendment used for the snapshot.
 for n,raw in info['overlays'].items():require(stable(new/n)==raw,'Overlay changed during payload build:'+n)
 for n,raw in info['new_evidence'].items():require(stable(new/n)==raw,'CPU evidence changed during payload build:'+n)
 require(stable(new/'HUMAN_GPU_AUTHORIZATION.json')==info['human'],'Human authorization changed during build');require(stable(new/'analysis/GPU_EXECUTION_AMENDMENT.json')==info['amendment'],'Amendment changed during build');require(stable(new/'analysis/GPU_EXECUTION_AMENDMENT.md')==info['amendment_md'],'Amendment prose changed during build')
 catalog=[]
 for p in sorted(destination.rglob('*')):
  if p.is_file():raw=stable(p);catalog.append(dict(path=str(p.relative_to(destination)),bytes=len(raw),sha256=digest(raw)))
 payload_manifest=encode(dict(describe_inputs(info),status='EXACT_LOCAL_PAYLOAD_BUILT_NOT_DISPATCHED',files=catalog,cards=cards,build_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_remote_or_GPU_calls=True));put(destination,'PAYLOAD_BUILD_MANIFEST.json',payload_manifest)
 common_names=list(info['new_evidence'])+['HUMAN_GPU_AUTHORIZATION.json','GPU_EXECUTION_AMENDMENT.json','GPU_EXECUTION_AMENDMENT.md','CARD_ROSTER.json','PAYLOAD_BUILD_MANIFEST.json','scripts/run_four_card_session.py']
 for card in cards:
  common_names += [card['card_key']+'/'+n for n in ['SOURCE_MANIFEST.json','AUTHORIZATION.json','RUN_COPY_ORIGIN.json','SOURCE_FREEZE.json','ROOT.json']]
 common_files=[]
 for n in sorted(common_names):
  raw=stable(destination/n);common_files.append(dict(path=n,bytes=len(raw),sha256=digest(raw)))
 common=encode(dict(task_id=TASK,execution_run_id=RUN,status='FROZEN_SHARED_CONTROL_BEFORE_ANY_GPU_DISPATCH',files=common_files,cards=cards,shared_control_root=REMOTE,original_CPU_seal_manifest_sha256=info['receipt']['manifest_sha256'],no_GPU_time_limit=True))
 put(destination,'CONTROL_SOURCE_MANIFEST.json',common)
 obj(destination,'SOURCE_FREEZE.json',dict(task_id=TASK,execution_run_id=RUN,status='FROZEN_SHARED_CONTROL_BEFORE_ANY_GPU_DISPATCH',control_manifest_sha256=digest(common),payload_build_manifest_sha256=digest(payload_manifest),human_authorization_sha256=digest(info['human']),amendment_sha256=digest(info['amendment']),GPU_dispatches=0,no_GPU_time_limit=True,external_anchor_excluded_from_manifests=True))
 final_files=[p for p in destination.rglob('*') if p.is_file()]
 return dict(status='BUILT_NOT_DISPATCHED',destination=str(destination),cards=cards,files=len(final_files),bytes=sum(p.stat().st_size for p in final_files),control_manifest_sha256=digest(common),payload_build_manifest_sha256=digest(payload_manifest))

def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu-root',type=Path,required=True);p.add_argument('--gpu-root',type=Path,required=True);p.add_argument('--destination',type=Path);p.add_argument('--build',action='store_true');a=p.parse_args();cpu=a.cpu_root.resolve();new=a.gpu_root.resolve();info=freeze_inputs(cpu,new)
 if a.build:require(a.destination is not None,'--build needs explicit new --destination');out=build(cpu,new,a.destination,info)
 else:out=describe_inputs(info)
 print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
