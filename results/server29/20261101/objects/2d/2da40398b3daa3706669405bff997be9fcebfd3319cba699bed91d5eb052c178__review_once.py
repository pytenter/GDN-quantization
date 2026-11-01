"""Extra independent CPU terminal review. No source/seal/GPU/process mutation."""
import base64,hashlib,importlib.util,json,os,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1_4GPU_CONTINUE_CONTROL_V2_20261026'
s=importlib.util.spec_from_file_location('frozen_terminal_primitives',ROOT/'operations/pace_four_gpu_monitor.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M);M.configure();O=M.O
old={};exec(M.OLD_COMMON,old)

COMMON=r'''
import hashlib,json,os,stat,sys,time
from pathlib import Path
cache={}
def req(ok,message):
 if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 p=Path(p);a=p.stat();key=str(p.resolve());finger=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)
 if key in cache:
  req(cache[key][0]==finger,'File changed during independent audit: '+key);return cache[key][1]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 z=p.stat();req(finger==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'File changed while hashing: '+key);cache[key]=(finger,h.hexdigest());return h.hexdigest()
def safe(root,name):
 p=Path(name);req(not p.is_absolute() and '..' not in p.parts and p.as_posix()==name and name!='.','Unsafe manifest path');q=Path(root)/p
 req(not q.is_symlink(),'Manifest symlink forbidden');return q
def verify(root,rows,external=False):
 seen=set();total=0
 for x in rows:
  req(x['path'] not in seen,'Duplicate inventory path');seen.add(x['path']);p=Path(x['path']) if external else safe(root,x['path']);s=p.stat();req(stat.S_ISREG(s.st_mode) and s.st_size==x['bytes'] and sha(p)==x['sha256'],'Fresh SHA/bytes differ: '+str(p))
  if external:
   for k,attr in [('inode','st_ino'),('device','st_dev'),('mtime_ns','st_mtime_ns')]:
    if k in x:req(getattr(s,attr)==x[k],'Frozen external stat differs: '+str(p))
  total+=s.st_size
 return dict(status='PASS_ALL_FRESH_SHA_BYTES',files=len(rows),bytes=total)
def unchanged():
 for name,(finger,digest) in cache.items():
  s=Path(name).stat();req(finger==(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns),'File changed before audit completion: '+name)
def regular_manifest(directory,expected):
 r=Path(directory);candidates=[p for p in r.iterdir() if p.is_file() and 'manifest' in p.name.lower() and p.suffix.lower()=='.json'];found=[p for p in candidates if sha(p)==expected];req(len(found)==1,'Pinned supplement manifest not found: '+directory);m=read(found[0]);return dict(root=directory,manifest=found[0].name,manifest_sha256=expected,verification=verify(r,m['files']))
'''
AUDIT55=r'''
identities=[];roots=[]
for spec in P['roots']:
 r=Path(spec['remote']);req(sha(r/'source_manifest.json')==spec['source'] and sha(r/'GPU_PERMIT.json')==spec['permit'] and sha(r/'DELIVERY_MANIFEST.json')==spec['manifest'],'Root source/permit/science anchor differs: '+str(r))
 source=read(r/'source_manifest.json');m=read(r/'DELIVERY_MANIFEST.json');proof=dict(root=str(r),source=verify(r,source['files']),science=verify(r,m['files']))
 ledger=read(r/'RESOURCE_LEDGER.json');req(not ledger.get('active_units') and ledger.get('active_unit') is None,'Closed origin ledger active')
 for x in ledger['intervals']:
  req(x['released'] and x['returncode'] is not None,'Unreleased origin worker');
  if x.get('identity'):identities.append(x['identity'])
 for n,key in [('OFFLINE_CHAIN_STARTED.json','bash'),('OFFLINE_CHAIN_ACTUAL_CHILD.json','parent'),('OFFLINE_PARENT_STARTED.json','identity'),('SESSION_STARTED.json','identity'),('FINISHER_STARTED.json','identity'),('PARENT_CONTROLLER_EXIT.json','identity'),('PARENT_FINISHER_EXIT.json','identity')]:
  if (r/n).is_file():
   v=read(r/n)
   if v.get(key):identities.append(v[key])
   if v.get('parent'):identities.append(v['parent'])
 for q in r.glob('FROZEN_V4_FRESH_PASS_*.PROCESS_STARTED.json'):identities.append(read(q)['identity'])
 roots.append(proof)
r=Path(P['current']);sys.path.insert(0,str(r/'src'));import pace_control as C
from pace_four_gpu_results import completed_rows
C.authorize(r);C.verify_science(r);bindings=C.bindings(r);models=read(r/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files'];req(len(models)==17,'Expected all17 model/tokenizer files');external=verify(r,list(bindings.values()),True);model=verify(r,models,True);bindings.update({str(Path(x['path']).resolve()):x for x in models})
triton=[]
for p in [r/'analysis/TRITON_RUNTIME_BINDINGS.json',r/'analysis/EXTERNAL_TRITON_BINDINGS.json']:
 if p.exists():triton.append(dict(inventory=str(p),verification=verify(r,read(p)['files'],True)))
mapped={};devices=[];locale=[];artifacts=0
for spec in P['roots']:
 base=Path(spec['remote']);ledger=read(base/'RESOURCE_LEDGER.json')
 for interval in ledger['intervals']:
  if spec['remote']==P['current']:t=C.unit_target(base,interval['stage'],interval['unit'])
  else:
   unit=interval['unit'];condition='R2' if '_R2_' in unit else 'R3' if '_R3_' in unit else None
   t=base/(condition+'_PACE48_DP'+('8' if condition=='R2' else '1'))/'trajectories'/unit if condition else base/'execution'/unit
  for q in list(t.glob('LOADED_LIBRARY_*.json'))+list(t.glob('POSTFLIGHT_LIBRARY.json')):
   v=read(q);artifacts+=1;verify(base,v['selected'],True)
   for line in v.get('full_maps','').splitlines():
    x=line.split(maxsplit=5)
    if len(x)!=6 or not x[5].startswith('/'):continue
    if x[5].startswith('/dev/'):devices.append(x[5]);continue
    p=Path(x[5]).resolve();s=p.stat();req(stat.S_ISREG(s.st_mode) and s.st_ino==int(x[4]),'Mapped inode/regular file differs: '+str(p));expected=bindings.get(str(p))
    if expected is None:
     req(str(p)=='/usr/lib/locale/locale-archive','Unbound mapped code/native file: '+str(p));locale.append(dict(path=str(p),sha256=sha(p),bytes=s.st_size,inode=s.st_ino,no_frozen_baseline_pin_claimed=True))
    else:verify(base,[dict(expected,path=str(p))],True)
    mapped[str(p)]=dict(path=str(p),bytes=s.st_size,inode=s.st_ino,sha256=sha(p))
ledger=read(r/'RESOURCE_LEDGER.json');failure=read(r/'FIRST_FAILURE.json');req(failure['unit']=='aime26_13_R3_seed1' and 'PID/starttick/argv changed' in failure['error'],'Unexpected first cause')
t=C.unit_target(r,'formal',failure['unit']);interval=read(t/'RESOURCE_INTERVAL.json');req(interval['returncode']==0 and interval['released'],'False-stop unit owning OS exit differs');wrapper=C.verify_result(r,'formal',failure['unit'],interval['identity']);req(wrapper['status']=='PASS','Exited worker did not produce legal result')
invalid=[];pairs=completed_rows(r,invalid);verdict=read(r/'final_verdict.json');req(len(pairs)==verdict['formal_completed']==19,'Legal union differs');req(verdict['R2_completed']==8 and verdict['R3_completed']==11,'Condition counts differ');req(len(invalid)==3 and {x['unit'] for x in invalid}=={'aime26_09_R2_seed1','aime26_10_R2_seed1','aime26_10_R3_seed1'},'Interrupted set differs')
interrupted=[]
for bad in invalid:
 t=C.unit_target(r,'formal',bad['unit']);v=read(t/'RESOURCE_INTERVAL.json');req(v['returncode']==-15 and v['released'],'Expected controller-owned interrupt exit differs');progress=read(t/'PROGRESS.json');interrupted.append(dict(unit=bad['unit'],returncode=v['returncode'],released=True,last_progress=progress,completion_credit=0))
for n in ['PARENT_CONTROLLER_EXIT.json','PARENT_FINISHER_EXIT.json']:
 v=read(r/n);req(v['actual_exited'] and v['identity_gone'] and v['source_manifest_sha256']==P['source'] and v['permit_sha256']==P['permit'],'OwningPopen terminal binding differs');req(v['returncode']==(1 if 'CONTROLLER' in n else 0),'OwningPopen returncode differs')
chain=read(r/'OFFLINE_CHAIN_OBSERVED_EXIT.json');req(chain['actual_shell_wait_returncode']==0 and chain['source_manifest_sha256']==P['source'] and chain['permit_sha256']==P['permit'],'Genuine Bash wait differs')
states=[]
for x in {json.dumps(x,sort_keys=True):x for x in identities}.values():
 p=Path('/proc')/str(x['pid'])/'stat'
 if p.exists():tick=int(p.read_text().rsplit(')',1)[1].split()[19]);req(tick!=x['starttick'],'Exact old owning identity remains alive');status='PID_REUSED_FOREIGN'
 else:status='GONE'
 states.append(dict(identity=x,status=status))
obs=C.query_gpu();req(not any(a['pid']==x['identity']['pid'] and x['status']!='PID_REUSED_FOREIGN' for a in obs['apps'] for x in states),'Own NVML process remains')
req(not list(r.glob('FROZEN_V4_FRESH_PASS_*.json')) and not (r/'comparison/formal_results.csv').exists(),'Partial run unexpected full score output')
unchanged();print(json.dumps(dict(status='PASS_EXTRA_INDEPENDENT55_ALL_CURRENT_AND_ORIGINS',roots=roots,external=external,model17=model,Triton=triton,actual_mapped_artifacts=artifacts,actual_unique_regular_mapped_files=len(mapped),mapped=list(mapped.values()),locale_data=locale,device_mmaps=sorted(set(devices)),identities=states,NVML=obs,first_failure=failure,normal_exit_unit=dict(unit=failure['unit'],actual_returncode=0,released=True,RESULT_sha256=sha(C.unit_target(r,'formal',failure['unit'])/'RESULT.json')),interrupted=interrupted,legal_formal_union=19,reused=15,current_completed=4,ledger_formal_total=ledger['formal_total_unique'],full_accuracy_reported=False,source_manifest_sha256=P['source'],permit_sha256=P['permit'],GPU_calls=0,model_loads=0,signals=0,remote_mutations=0)))
'''
AUDIT29=r'''
roots=[]
for spec in P['roots']:
 r=Path(spec['archive']);req(sha(r/'source_manifest.json')==spec['source'] and sha(r/'GPU_PERMIT.json')==spec['permit'] and sha(r/'DELIVERY_MANIFEST.json')==spec['manifest'],'Archive source/permit/primary anchor differs');m=read(r/'DELIVERY_MANIFEST.json');proof=verify(r,m['files']);req({p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file()}=={x['path'] for x in m['files']}|{'DELIVERY_MANIFEST.json','ARCHIVE_VERIFICATION.json'},'Unexpected primary archive file');roots.append(dict(root=str(r),verification=proof))
supplements=[regular_manifest(x['root'],x['manifest']) for x in P['supplements']]
live=[]
for label in P['live_revisions']:
 base=Path('/data/zypan/qwen_gdn_pace48_archive')/('LIVE_'+label);p=base/'SOURCE_BOOTSTRAP';m=read(p/'SOURCE_BOOTSTRAP_MANIFEST.json');live.append(dict(root=str(p),verification=verify(p,m['files'])))
 units=base/'units'
 if units.exists():
  for p in sorted(units.iterdir()):
   req(p.is_dir() and '.staging' not in p.name,'Unclosed LIVE unit');m=read(p/'DURABLE29_MANIFEST.json');v=read(p/'DURABLE29_RECEIPT.json');req(v['manifest_sha256']==sha(p/'DURABLE29_MANIFEST.json') and v['released'] and m['actual_exit_returncode']==v['actual_exit_returncode'],'LIVE receipt differs');live.append(dict(root=str(p),verification=verify(p,m['files'])))
unchanged();print(json.dumps(dict(status='PASS_EXTRA_INDEPENDENT29_ALL_PRIMARY_SUPPLEMENTS_SOURCE_BOOTSTRAPS_AND_LIVE',roots=roots,supplements=supplements,LIVE=live,GPU_queries=0,remote_mutations=0)))
'''

def specs():
    current=json.loads((ROOT/'independent_delivery/RESULT.json').read_bytes())
    out=[dict(remote=old['OLD_REMOTE'],archive=old['OLD_ARCHIVE'],source=old['OLD_SOURCE'],permit=old['OLD_PERMIT'],manifest=old['OLD_MANIFEST']),dict(remote=old['PRIOR_REPAIR_REMOTE'],archive=old['PRIOR_REPAIR_ARCHIVE'],source=old['PRIOR_REPAIR_SOURCE'],permit=old['PRIOR_REPAIR_PERMIT'],manifest=old['PRIOR_REPAIR_MANIFEST']),dict(remote=M.OLD_REMOTE,archive=M.OLD_ARCHIVE,source=M.OLD_PINS['source_sha256'],permit=M.OLD_PINS['permit_sha256'],manifest=M.OLD_PINS['manifest_sha256']),dict(remote='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_continue_20261026',archive='/data/zypan/qwen_gdn_pace48_archive/QWEN_GDN_PACE48_AIME26_SEED1_60_V1_4GPU_CONTINUE_20261026',source='c76372b0e15dbf0473e228f9c591166cc510d7e51fe2892a59d9e43e95c750f6',permit='b4ca1861df7abb253c4ebfc1744687c362d1c428eb9a1f49d9cf7e31b4571ebb',manifest='808ca9e0846caf97c17ecc377ecea56f3fd6c7b98623f2df7576b2b26426194c'),dict(remote=M.REMOTE,archive=M.ARCHIVE,source=current['source_manifest_sha256'],permit=current['permit_sha256'],manifest=current['core_manifest_sha256'])]
    return out

def inventories():
    roots=[BASE/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1',BASE/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1_SCALE_REPAIR_20261025',BASE/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1_SCALE_EXPONENT_REPAIR_20261025',BASE/'PACE4_CONTROL_CPU_RECOVERY_20261026',ROOT]
    payload=[];supp={}
    allowed=[x['archive'] for x in specs()]
    def visit(v):
        if isinstance(v,dict):
            path=v.get('archive_root');digest=v.get('manifest_sha256')
            if isinstance(path,str) and isinstance(digest,str) and len(digest)==64 and any(path.startswith(a+'.') for a in allowed):supp[path]=digest
            for x in v.values():visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
    for root in roots:
        delivery=root/'independent_delivery'
        options=[delivery/'MAC_VERIFICATION.json',delivery/'MAC_SMALL_FILES_VERIFICATION.json']
        file=next((p for p in options if p.is_file()),None);O.require(file is not None,'Expected original Mac verification absent: '+str(root));v=json.loads(file.read_bytes());core=delivery/'primary_archive_small_files';O.verify_rows(core,v['files']);late=v.get('late',v.get('late_receipts',[]));O.verify_rows(delivery/'late_exit_receipts',late)
        payload.append(dict(root=str(root),verification_sha256=O.digest(file),small_count=len(v['files']),late_count=len(late)))
        for p in root.rglob('*.json'):
            if 'primary_archive_small_files' in p.parts or p.stat().st_size>16*1024**2:continue
            try:visit(json.loads(p.read_bytes()))
            except (OSError,json.JSONDecodeError):pass
    return payload,[dict(root=k,manifest=v) for k,v in sorted(supp.items())]

def call(t,code,payload,role):
    return t.run('import json\nP=json.loads('+repr(json.dumps(payload))+')\n'+COMMON+code,role=role,python='/data/ydai/miniconda3/envs/bitdecode/bin/python' if role=='55' else '/usr/bin/python3',timeout=7200)

def main():
    O.require(json.loads((ROOT/'independent_delivery/RESULT.json').read_bytes())['status']=='PASS_INDEPENDENT_TERMINAL_DELIVERY_AND_STOP','Primary independent delivery not complete')
    O.write_new(HERE/'CLAIM.json',O.canonical(dict(status='EXTRA_FINAL_INDEPENDENT_REVIEW_ONCE',source_manifest_sha256=O.digest(ROOT/'source_manifest.json'),permit_sha256=O.digest(ROOT/'GPU_PERMIT.json'),unix=time.time()))+b'\n')
    try:
        mac,supp=inventories();O.write_new(HERE/'MAC_AND_ORIGIN_VERIFICATION.json',O.canonical(dict(status='PASS_ALL_CURRENT_AND_ORIGINAL_MAC_SMALL_AND_LATE_FILES',roots=mac,supplements=supp))+b'\n')
        roots=specs();payload=dict(roots=roots,current=M.REMOTE,source=roots[-1]['source'],permit=roots[-1]['permit'],supplements=supp,live_revisions=['QWEN_GDN_PACE48_AIME26_SEED1_60_V1_4GPU_CONTINUE_20261026',ROOT.name]);t=O.load_transport(ROOT/'operations/remote.py','db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40')
        a55=call(t,AUDIT55,payload,'55');O.write_new(HERE/'AUDIT55.json',O.canonical(a55)+b'\n');print(json.dumps(dict(status=a55['status'],formal_union=a55['legal_formal_union'])),flush=True)
        a29=call(t,AUDIT29,payload,'29');O.write_new(HERE/'AUDIT29.json',O.canonical(a29)+b'\n');print(json.dumps(dict(status=a29['status'],supplements=len(a29['supplements']))),flush=True)
        # These local CPU review receipts are additional evidence, outside the
        # frozen scientific source; preserve their exact hashes in this review.
        reviews=[]
        for name in ['FORMAL_START_FOUR_GPU_FULL_INDEPENDENT_REVIEW.json','FORMAL_START_FOUR_GPU_ARCHIVE29_INDEPENDENT_REVIEW.json','FORMAL_START_FOUR_GPU_COMPLETION_REVIEW.json']:
            p=ROOT/'operations'/name;reviews.append(dict(path=str(p),bytes=p.stat().st_size,sha256=O.digest(p)))
        result=dict(status='PASS_EXTRA_FULL_TRUE_STOP_INDEPENDENT_REVIEW',source_manifest_sha256=payload['source'],permit_sha256=payload['permit'],formal_completed=19,reused=15,current=4,remaining=41,controller_first_cause='Normal-owning-exit/identity-check race; real worker exit0 misclassified',GPU_calls=0,signals=0,model_loads=0,primary_seals_modified=False,local_formal_start_reviews=reviews,review55_sha256=O.digest(HERE/'AUDIT55.json'),review29_sha256=O.digest(HERE/'AUDIT29.json'),Mac_review_sha256=O.digest(HERE/'MAC_AND_ORIGIN_VERIFICATION.json'))
        O.write_new(HERE/'RESULT.json',O.canonical(result)+b'\n');print(json.dumps(result),flush=True)
    except BaseException as exc:
        O.write_new(HERE/'FIRST_FAILURE.json',O.canonical(dict(status='EXTRA_CPU_REVIEW_FAILURE_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc(),claim_preserved=True,GPU_calls=0))+b'\n');raise

if __name__=='__main__':main()
