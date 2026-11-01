"""Human-requested CPU-only descriptive partial score; production stays frozen."""
import collections,datetime,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_exit_fix_20261026')
HERE=Path(__file__).resolve().parent
SOURCE='64d59ba3b37d092b28c8a509019dd1ca9b5863767d93f406705dd613b3aa8132'
PERMIT='2d13b5fe9ebfd10717d7eb1b6a515956512aa95e781eb132dd49550620b10ad3'
PY='/data/ydai/miniconda3/envs/bitdecode/bin/python'
MODEL='/data/zypan/modelscope_models/Qwen3.5-9B'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def write(p,v):
 with Path(p).open('xb') as f:f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
def control():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert sha(ROOT/'source_manifest.json')==SOURCE and sha(ROOT/'GPU_PERMIT.json')==PERMIT
 sys.path.insert(0,str(ROOT/'src'));import pace_control as C
 C.authorize(ROOT);C.verify_science(ROOT);C.model_guard(ROOT)
 return C
def score(cohort_sha,index):
 C=control();assert sha(HERE/'COHORT.json')==cohort_sha;cohort=read(HERE/'COHORT.json')
 spec=importlib.util.spec_from_file_location('manual_frozen_v4_exact',ROOT/'protocol/aime26_scorer_v4.py')
 S=importlib.util.module_from_spec(spec);sys.modules[spec.name]=S;spec.loader.exec_module(S)
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
 gold={v['problem_id']:v for v in (json.loads(line) for line in (ROOT/'protocol/aime26_frozen.jsonl').read_text().splitlines())}
 rows=[]
 for x in cohort['units']:
  p=Path(x['result_path']);interval=Path(x['interval_path'])
  assert sha(p)==x['result_sha256'] and sha(interval)==x['interval_sha256']
  w=read(p);r=w['result'];v=read(interval);assert v['returncode']==0 and v['released'] is True
  assert w['payload_sha256']==C.objsha({k:z for k,z in w.items() if k!='payload_sha256'})
  assert C.objsha(r['generated_token_ids'])==r['generated_token_ids_sha256']
  decoded=tokenizer.decode(r['generated_token_ids'],skip_special_tokens=True);assert decoded==r['response']
  extracted=S.extract_v4(decoded,r['finish_reason'],gold[x['problem_id']]['problem'])
  correct=bool(S.compare_with_gold(extracted,gold[x['problem_id']]['answer']))
  rows.append(dict(unit=x['unit'],problem_id=x['problem_id'],condition=x['condition'],correct=correct,
                   extraction_status=extracted.status,claimed_value=extracted.normalized_value,
                   finish_reason=r['finish_reason'],generated_tokens=len(r['generated_token_ids']),
                   token_sha256=r['generated_token_ids_sha256'],result_sha256=sha(p),
                   evidence_source_manifest_sha256=w['source_manifest_sha256']))
 arms={}
 for condition in ('R2','R3'):
  a=[x for x in rows if x['condition']==condition];n=len(a);correct=sum(x['correct'] for x in a)
  arms[condition]=dict(N=n,correct=correct,accuracy=correct/n if n else None,
                       missing_final_claim=sum(x['extraction_status']!='EXTRACTED' for x in a))
 by={}
 for x in rows:by.setdefault(x['problem_id'],{})[x['condition']]=x
 paired={k:v for k,v in by.items() if set(v)=={'R2','R3'}}
 paired_summary=dict(N=len(paired),R2_correct=sum(x['R2']['correct'] for x in paired.values()),R3_correct=sum(x['R3']['correct'] for x in paired.values()))
 payload=dict(rows=rows,arms=arms,paired_completed_subset=paired_summary,cohort_sha256=cohort_sha,
              source_manifest_sha256=SOURCE,permit_sha256=PERMIT,scorer_sha256=sha(ROOT/'protocol/aime26_scorer_v4.py'),
              dataset_sha256=sha(ROOT/'protocol/aime26_frozen.jsonl'),model_tokenizer_inventory_sha256=sha(ROOT/'analysis/MODEL_TOKENIZER_FRESH_SHA.json'),
              partial_descriptive_only=True,full60_accuracy=False,model_loaded=False,GPU_calls=0,science_mutations=0)
 C.model_guard(ROOT);C.verify_sources(ROOT)
 write(HERE/('PASS_%d.json'%index),dict(status='PASS_FRESH_CPU_FROZEN_V4_PARTIAL_SAME_SAVED_TOKENS',payload=payload,payload_sha256=hashlib.sha256(canonical(payload)).hexdigest(),identity=C.process_identity(os.getpid())))
def main():
 C=control();assert str(HERE)=='/data/zypan/analyses/pace_manual_partial_score_20261026_current20_v1'
 assert not (HERE/'COHORT.json').exists(),'This CPU report is one-shot'
 write(HERE/'AUTHORIZATION.json',dict(human_request='给我看看现在的正确率是什么',scope='Independent CPU partial scoring of already closed exit0/released saved payloads; no controller or finisher rerun',overrides_blind_monitor_no_extra_scoring_for_this_manual_report_only=True,production_and_queue_unchanged=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 from pace_four_gpu_results import completed_rows
 invalid=[];pairs=completed_rows(ROOT,invalid);reuse=set(C.read(ROOT/'REUSE_ADMISSION.json')['reused_ids']);units=[]
 ledger=C.read(ROOT/'RESOURCE_LEDGER.json');closed={x['unit']:x for x in ledger['completed_units'] if x['stage']=='formal'}
 for job,w,t in pairs:
  interval=C.read(t/'RESOURCE_INTERVAL.json');assert interval['returncode']==0 and interval['released']
  if job['id'] not in reuse:
   assert job['id'] in closed;receipt=C.read(t/'DURABLE29_RECEIPT.json')
   assert receipt['actual_exit_returncode']==0 and receipt['released'] and receipt['manifest_sha256']==closed[job['id']]['durable_manifest_sha256']
  units.append(dict(unit=job['id'],problem_id=job['problem_id'],condition=job['condition'],
                    result_path=str(t/'RESULT.json'),result_sha256=sha(t/'RESULT.json'),
                    interval_path=str(t/'RESOURCE_INTERVAL.json'),interval_sha256=sha(t/'RESOURCE_INTERVAL.json'),origin_source=w['source_manifest_sha256']))
 assert len(units)==len({x['unit'] for x in units}) and 0<len(units)<=60
 cohort=dict(status='FROZEN_CPU_PARTIAL_REPORT_COHORT',units=units,N=len(units),reused=19,current=len(units)-19,
             incomplete_not_scored=60-len(units),invalid_or_pending_receipts=invalid,
             source_manifest_sha256=SOURCE,permit_sha256=PERMIT,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(HERE/'COHORT.json',cohort);cohort_sha=sha(HERE/'COHORT.json');children=[]
 for i in (0,1):
  argv=[PY,'-B',str(HERE/'score_once.py'),'--score',cohort_sha,str(i)];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',USE_TORCH='0')
  for k in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE'):env.pop(k,None)
  with (HERE/('PASS_%d.log'%i)).open('xb',buffering=0) as log:
   p=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);identity=None
   try:identity=C.process_identity(p.pid)
   except FileNotFoundError:pass
   rc=p.wait()
  value=dict(argv=argv,identity=identity,pid=p.pid,actual_returncode=rc,actual_reaped=True,
             identity_gone=not C.same_process(identity) if identity else not Path('/proc',str(p.pid)).exists(),log_sha256=sha(HERE/('PASS_%d.log'%i)))
  write(HERE/('PASS_%d.ACTUAL_EXIT.json'%i),value);children.append(value);assert rc==0 and value['identity_gone'],value
 a,b=read(HERE/'PASS_0.json'),read(HERE/'PASS_1.json');assert a['payload']==b['payload'] and a['payload_sha256']==b['payload_sha256']
 result=dict(status='PASS_TWO_FRESH_CPU_MANUAL_PARTIAL_SCORE',N=cohort['N'],reused=19,current=cohort['current'],
             arms=a['payload']['arms'],paired_completed_subset=a['payload']['paired_completed_subset'],
             cohort_sha256=cohort_sha,score_payload_sha256=a['payload_sha256'],children=children,
             full60_accuracy=False,partial_descriptive_only=True,no_generation_or_GPU_dispatch=True,production_source_and_queue_unchanged=True,
             source_manifest_sha256=SOURCE,permit_sha256=PERMIT,scorer_sha256=a['payload']['scorer_sha256'])
 write(HERE/'RESULT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--score':score(sys.argv[2],int(sys.argv[3]))
 else:
  try:main()
  except BaseException as e:
   if not (HERE/'FAILURE.json').exists():write(HERE/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),GPU_dispatches=0))
   raise
