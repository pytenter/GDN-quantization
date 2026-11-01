"""One CPU/read-only fresh server55 identity audit; writes local receipts only."""
import hashlib,importlib.util,json,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'QWEN_GDN_GROUP32_FOUR_CODEC_AIME26_SEED1_V1'
transport=ROOT.parent/'QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/operations/remote.py';spec=importlib.util.spec_from_file_location('readonly_transport',transport);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
code=r'''
import concurrent.futures,hashlib,json,os,time
from pathlib import Path
R=Path('/data/zypan/analyses/qwen_gdn_group32_four_codec_aime26_seed1_v1')
def read(p):return json.loads(Path(p).read_text())
def H(p):
 p=Path(p);a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 z=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'Changed duringhash:'+str(p)
 return dict(path=str(p),bytes=z.st_size,sha256=h.hexdigest(),inode=z.st_ino,mtime_ns=z.st_mtime_ns,device=z.st_dev)
def pin(p):x=H(p);return dict(path=x['path'],bytes=x['bytes'],sha256=x['sha256'])
source=read(R/'SOURCE_FREEZE.json');source_rows=[]
for x in source['files']:
 p=R/x['path'];a=H(p);assert a['bytes']==x['bytes'] and a['sha256']==x['sha256'],'Historicalsourcechanged:'+x['path'];source_rows.append(dict(path=x['path'],bytes=a['bytes'],sha256=a['sha256']))
model=read(R/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files'];bindings=[]
for name in ('PROSPECTIVE_MODEL_AND_NATIVE_BINDINGS.json','CPU_FIXTURE_CRITICAL_NATIVE_BINDINGS.json'):
 d=read(R/'inherited/group32/inherited/historical_port_payload/analysis/environment'/name)
 for key in ('model_weight_shards','model_small_files','native_library_inventory','native_library_rows','torch_python_files','external_source_files'):bindings+=d.get(key,[])
 if isinstance(d.get('python_executable'),dict):bindings.append(d['python_executable'])
expected={str(Path(x['path']).resolve()):x for x in bindings+model if all(k in x for k in ('path','bytes','sha256'))};fresh={}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for x in pool.map(H,expected):
  prior=expected[x['path']];assert x['bytes']==prior['bytes'] and x['sha256']==prior['sha256'],'Model/externalidentitychanged:'+x['path'];fresh[x['path']]=x
scores=read(R/'AIME26_FROZEN_V4_SCORES.json');p0=pin(R/'SCORE_PASS_0.json');p1=pin(R/'SCORE_PASS_1.json');canonical=pin(R/'AIME26_FROZEN_V4_SCORES.json');assert p0['sha256']==p1['sha256']==canonical['sha256'],'HistoricalCPUscorersdiffer'
selected={q:[x for x in scores['scores'] if x['condition']==q] for q in ('Q2','Q3')}
assert len(scores['scores'])==120
vectors={};evidence=[]
for q,label,total in [('Q2','H+G32+Sym8',21),('Q3','H+G32+Asym8',25)]:
 rows=selected[q];assert len(rows)==30 and len({x['problem_id'] for x in rows})==30 and sum(x['correct'] for x in rows)==total
 vectors[label]={x['problem_id']:x['correct'] for x in rows}
 for row in rows:
  name=row['sample_id'];t=R/'formal'/name;raw=read(t/'RESULT.json');i=read(t/'RESOURCE_INTERVAL.json');s=read(t/'STARTED.json');dispatch=read(t/'DISPATCH.json')
  cp=dict(raw);payload=cp.pop('payload_sha256');body=json.dumps(cp,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode();assert hashlib.sha256(body).hexdigest()==payload==row['payload_sha256']
  assert raw['status']=='PASS' and i['exit_returncode']==0 and i['release']=='PASS' and i['identity']==s['identity']==raw['identity']
  assert raw['gpu_uuid']==row['gpu_uuid'] and raw['permit_sha256']==dispatch['permit_sha256'] and raw['source_freeze_sha256']==pin(R/'SOURCE_FREEZE.json')['sha256']
  evidence.append(dict(method=label,sample_id=name,problem_id=row['problem_id'],correct=row['correct'],payload_sha256=payload,identity=raw['identity'],gpu_uuid=row['gpu_uuid'],actualexit0=True,release='PASS',result=pin(t/'RESULT.json'),interval=pin(t/'RESOURCE_INTERVAL.json')))
terminal=read(R/'FINAL_TASK_STATUS.json');seal=read(R/'FINAL_SEAL_RECEIPT.json');qual=read(R/'GPU_CODEC_QUALIFICATION.json');assert terminal['status']=='FORMAL_COMPLETE_120_120_AND_STOP' and terminal['own_GPU_processes_released'] and seal['scientific_status']==terminal['status']
model_rows=[fresh[str(Path(x['path']).resolve())] for x in model]
print(json.dumps(dict(status='PASS_FRESH_SERVER55_HISTORICAL_SOURCE_MODEL_EXTERNAL_AND_BASELINE_RAW',unix=time.time(),monotonic=time.monotonic(),historical_root=str(R),source_manifest=pin(R/'SOURCE_FREEZE.json'),source_count=len(source_rows),source_rows=source_rows,model_files=model_rows,external_files=list(fresh.values()),external_count=len(fresh),scores=[p0,p1,canonical],historical_per_problem=vectors,per_sample=evidence,qualification=pin(R/'GPU_CODEC_QUALIFICATION.json'),historical_qualification_status=qual['status'],terminal=pin(R/'FINAL_TASK_STATUS.json'),seal=pin(R/'FINAL_SEAL_RECEIPT.json'),protocol_pins=[pin(R/name) for name in ['AIME26_DATASET_FREEZE.json','AIME26_SEED1_PROTOCOL.json','protocol/sampler.py','protocol/common.py','protocol/aime26_scorer_v4.py','protocol/aime26_frozen.jsonl','src/runtime_engine.py','src/protocol_binding.py','src/four_codec_adapter.py','inherited/group32/src/canonical_h128.py','inherited/group32/src/group_codecs.py','inherited/group32/src/persistent_adapter.py']],GPU_calls=0,remote_writes=0)))
'''
x=m.run(code,role='55',python='/data/ydai/miniconda3/envs/bitdecode/bin/python',timeout=600)
out=ROOT/'analysis/HISTORICAL_RUNTIME_FRESH_READONLY55.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(dict(status=x['status'],source_count=x['source_count'],model_count=len(x['model_files']),external_count=x['external_count'],score_pins=x['scores'],GPU_calls=0,remote_writes=0)))
