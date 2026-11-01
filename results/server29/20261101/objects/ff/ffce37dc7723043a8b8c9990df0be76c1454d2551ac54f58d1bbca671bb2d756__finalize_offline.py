"""After actual GPU terminal: CPU science/STOP report and atomic archive only."""
import sys,os,json,subprocess,shlex,time,traceback,csv,math,random
from pathlib import Path
R=Path(sys.argv[1]);sys.path.insert(0,str(R/'src'));import execution_control as C
ARCHIVE=C.read(R/'ROOT.json')['archive_root'];SOCKET='/tmp/qwen_replication_archive29.sock';HOST='zypan@202.38.247.29'
def remote(code,*args):
 p=subprocess.run(['ssh','-S',SOCKET,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,shlex.join(['/usr/bin/python3','-u','-B','-c',code,*args])],capture_output=True,timeout=300);C.require(p.returncode==0,'Read-only/archive operation failed:'+p.stderr.decode());return json.loads(p.stdout)
def dump(name,obj):C.atomic_new(R/name,obj)
def write(name,text):
 p=R/name
 with p.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
C.require(not (R/'FINALIZER_STARTED.json').exists(),'Finisher already started; inspect, no duplicate')
dump('FINALIZER_STARTED.json',dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic()));terminal=C.read(R/'SESSION_RESULT.json');obs=C.query_gpu();children=[e['identity'] for e in C.read(R/'RESOURCE_LEDGER.json')['intervals'] if e['identity'] is not None]
C.require(not C.same_process(C.read(R/'GPU_RUN_STARTED.json')['identity']) and all(not C.same_process(i) for i in children) and all(x['release']=='PASS' for x in C.read(R/'RESOURCE_LEDGER.json')['intervals']),'Actual own process/NVML not released')
try:
 # Reused proof remains separate from all fresh STARTED/RESULT/interval files.
 from qualification_reuse import admit_reused48, remaining_units
 reuse=admit_reused48(R,C.read(R/'GPU_EXECUTION_PLAN.json'))
 C.require(reuse['count']==48 and reuse['status']=='PASS_REUSE48_EXACT_BYTEVIEW_ORIGIN_EXIT0_RELEASE','Terminal exact reused48 admission differs')
 plan=C.read(R/'GPU_EXECUTION_PLAN.json');fresh_qual={u['id'] for u in remaining_units(plan)};formal_ids={j['id'] for j in C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']}
 ledger=C.read(R/'RESOURCE_LEDGER.json')['intervals']
 C.require(len(fresh_qual)==4 and len({x['unit'] for x in ledger})==len(ledger) and all(x['unit'] in fresh_qual|formal_ids for x in ledger),'Terminal current94 roster/interval scope differs')
 fresh_legal=[x for x in ledger if x['unit'] in fresh_qual and x['exit_returncode']==0 and x['release']=='PASS']
 formal_legal=[x for x in ledger if x['unit'] in formal_ids and x['exit_returncode']==0 and x['release']=='PASS']
 durable=[]
 for interval in ledger:
  folder='formal' if interval['unit'] in formal_ids else 'execution';target=R/folder/interval['unit'];receipt=target/'DURABLE29_RECEIPT.json'
  if receipt.exists():
   d=C.read(receipt);C.require(d['status']=='PASS_DURABLE29_CLOSED_UNIT' and d['unit']==interval['unit'] and d['actual_exit_returncode']==interval['exit_returncode'] and d['source_freeze_sha256']==C.sha(R/'SOURCE_FREEZE.json') and d['permit_sha256']==C.sha(R/'GPU_PERMIT.json'),'Terminal durable unit receipt identity differs')
   durable.append(interval['unit'])
 accounting=dict(reused_qualification_count=48,fresh_qualification_legal_exit0_release=len(fresh_legal),fresh_qualification_planned=4,combined_qualification_legal=48+len(fresh_legal),qualification_planned=52,fresh_formal_legal_exit0_release=len(formal_legal),fresh_formal_planned=90,current_intervals=len(ledger),current_durable_units=len(durable),original_reused_result_or_interval_files_created=0,old_failed_or_nonzero_result_credit=0,reuse_admission_sha256=C.sha(R/'QUALIFICATION_REUSE_ADMISSION.json'))
 dump('QUALIFICATION_REUSE48_FRESH4_TERMINAL_ACCOUNTING.json',accounting)
 if terminal['status']=='GPU_COMPLETE_REUSE48_PLUS4_QUALIFICATION_90_FORMAL':
  C.require(len(fresh_legal)==4 and len(formal_legal)==90 and len(ledger)==94 and len(durable)==94,'Full terminal requires all94 fresh actual exit/release and durable29 receipts')
 else:C.require(terminal['status']=='FAIL_CLOSED_STOP' and (R/'FIRST_FAILURE.json').exists(),'Unknown terminal status cannot be scored')
 if terminal['status']=='GPU_COMPLETE_REUSE48_PLUS4_QUALIFICATION_90_FORMAL':
  for rep in (0,1):
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
   command=[sys.executable,'-u','-B',str(R/'scripts/score_pass_cpu.py'),str(R),str(R/('SCORE_PASS_'+str(rep)+'.json'))]
   scorer=None;scorer_identity=None;output=b''
   try:
    scorer=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    scorer_identity=C.process_identity(scorer.pid)
    dump('SCORE_PASS_'+str(rep)+'_STARTED.json',dict(identity=scorer_identity,argv=command,finisher_identity=C.process_identity(os.getpid()),monotonic=time.monotonic(),CPU_only=True))
    output,_=scorer.communicate(timeout=1200)
   finally:
    if scorer is not None:
     if scorer.poll() is None:
      C.require(scorer_identity is not None and C.same_process(scorer_identity),'Scorer identity unavailable or changed; never signal replacementPID')
      scorer.terminate()
      try:output,_=scorer.communicate(timeout=15)
      except subprocess.TimeoutExpired:
       C.require(C.same_process(scorer_identity),'Scorer identity changed before owned kill; never signal replacementPID');scorer.kill();output,_=scorer.communicate(timeout=15)
     (R/('SCORE_PASS_'+str(rep)+'.log')).write_bytes(output or b'')
     dump('SCORE_PASS_'+str(rep)+'_ACTUAL_EXIT.json',dict(identity=scorer_identity,actual_returncode=scorer.returncode,same_identity_alive=C.same_process(scorer_identity) if scorer_identity is not None else 'UNRESOLVED',monotonic=time.monotonic(),CPU_only=True))
   C.require(scorer.returncode==0 and not C.same_process(scorer_identity),'Independent original FrozenV4 CPU pass failed or actual identity still alive')
  a=C.read(R/'SCORE_PASS_0.json');b=C.read(R/'SCORE_PASS_1.json');C.require(a==b,'Independent exact payload scoring disagreement');dump('FROZEN_V4_SCORES.json',a);scores=a['scores'];by={}
  for row in scores:by.setdefault(row['problem_id'],{})[row['condition']]=row
  C.require(len(by)==30 and all(set(v)=={'W','R','RW'} for v in by.values()),'Incomplete30 three-way paired roster')
  labels={'W':'H+G32+Sym8+WScale','R':'H+G32+Sym8+RowBal','RW':'H+G32+Sym8+RowBal+WScale'}
  totals={q:sum(int(v[q]['correct']) for v in by.values()) for q in C.METHODS}
  historical=C.read(R/'HISTORICAL_G32_BASELINE_REUSE.json');h=historical['historical_per_problem']
  for key,count in [('H+G32+Sym8',21),('H+G32+Asym8',25)]:
   C.require(set(h[key])==set(by) and all(type(v)==bool for v in h[key].values()) and sum(h[key].values())==count,'Historical exact30 vector/identity differs')
  vectors={q:{pid:v[q]['correct'] for pid,v in by.items()} for q in C.METHODS};vectors.update(h);pairs={}
  for new,base in [('W','R'),('RW','W'),('RW','R')]+[(q,hist) for q in C.METHODS for hist in h]:
   rescue=sum(vectors[new][p] and not vectors[base][p] for p in by);lost=sum(vectors[base][p] and not vectors[new][p] for p in by);n=rescue+lost;pvalue=1. if not n else min(1.,2*sum(math.comb(n,k) for k in range(min(rescue,lost)+1))/2**n)
   pairs[new+' minus '+base]=dict(rescued=rescue,lost=lost,net_correct=rescue-lost,delta_accuracy=(rescue-lost)/30,exact_McNemar_two_sided_p=pvalue,exploratory_unadjusted_multiple_comparisons=True)
  interaction=(totals['RW']-totals['R']-totals['W']+21)/30
  dump('WSCALE_ROWBAL_PAIRWISE_INTERACTION.json',dict(N=30,seeds=[1],labels=labels,totals=totals,historical_totals=historical['historical_scores'],correct_vectors=vectors,pairwise=pairs,I=interaction,interaction_descriptive_historical_base_not_matched_new_dispatch=True,historical_baseline_receipt_sha256=C.sha(R/'HISTORICAL_G32_BASELINE_REUSE.json'),previously_observed_AIME=True))
  with (R/'AIME26_PER_PROBLEM_RESULTS.csv').open('x',newline='') as f:
   fields=['problem_id']+[q+'_'+s for q in C.METHODS for s in ['method','correct','answer','tokens','termination','gpu_uuid','finite','wall_seconds','payload_sha256']]+['historical_H_G32_Sym8_correct','historical_H_G32_Asym8_correct'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for pid,v in by.items():
    row={'problem_id':pid,'historical_H_G32_Sym8_correct':h['H+G32+Sym8'][pid],'historical_H_G32_Asym8_correct':h['H+G32+Asym8'][pid]}
    for q,x in v.items():
     for k,val in [('method',labels[q]),('correct',x['correct']),('answer',x['extracted_answer']),('tokens',x['generated_token_count']),('termination',x['finish_reason']),('gpu_uuid',x['gpu_uuid']),('finite',x['finite']),('wall_seconds',x['wall_seconds']),('payload_sha256',x['payload_sha256'])]:row[q+'_'+k]=val
    w.writerow(row)
  with (R/'FORMAL_GENERATIONS_90.jsonl').open('x') as f:
   for j in C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']:f.write(json.dumps(C.read(R/'formal'/j['id']/'RESULT.json'),ensure_ascii=False,sort_keys=True)+'\n')
  lengths={q:dict(total_tokens=sum(x['generated_token_count'] for x in scores if x['condition']==q),EOS=sum(x['finish_reason']=='eos' for x in scores if x['condition']==q),length=sum(x['finish_reason']=='length' for x in scores if x['condition']==q),seconds=sum(x['wall_seconds'] for x in scores if x['condition']==q)) for q in C.METHODS}
  for v in lengths.values():v['aggregate_generation_TPS']=v['total_tokens']/v['seconds']
  write('GENERATION_LENGTH_AND_RESOURCES.md','# Actual generation length and resources\n\n'+json.dumps(lengths,indent=2)+'\n\nGeneration TPS includes prompt/sampling/QDQ; model loading is instead in resource intervals. Matched non-AIME timings and kernel/memory metadata are in GPU_THROUGHPUT_AND_MEMORY.json. Physical exact GPUh remains UNRESOLVED; bounds and actual exit/release are in RESOURCE_LEDGER.json and append-only GPU_RESOURCE_LEDGER.jsonl.\n')
  status='FORMAL_COMPLETE_90_90_AND_STOP';table='\n'.join('| '+labels[q]+' | '+str(totals[q])+'/30 | '+f'{totals[q]/30:.2%}'+' |' for q in C.METHODS)
  report='# Qwen/GDN H+G32+Sym8 WScale, RowBal and combined AIME26 seed1\n\nStatus: '+status+'. All90 fresh generations completed with actualexit0/releasePASS. Qualification is48 strictly reused legal exit0/release units from the preserved BYTEVIEW execution plus4 fresh matched-performance units in this new execution; none of the reused48 is relabelled fresh. Two independent CPU passes over originalFrozenV4 and the same saved payload token re-decode agree.\n\n| Method | Correct/30 | Accuracy |\n|---|---:|---:|\n| H+G32+Sym8 (historical read-only) |21/30|70.00%|\n'+table+'\n| H+G32+Asym8 (historical read-only) |25/30|83.33%|\n\nPairwise:\n\n'+json.dumps(pairs,indent=2)+f'\n\nDescriptive historical-base interaction I={interaction:.6f}. This is single seed1,30 questions, already-observed AIME26, with multiple exploratory pairwise comparisons. Historical baseline UUID/question assignment and timing may differ; the interaction is descriptive rather than a matched causal effect.\n\nAll methods share one non-AIME calibrated omega and R/RW share one chosen(a,b). Local MSE, weightedMSE/readout diagnostics never eliminate any formal method or drive answer-dependent changes. Native GDN update/readout, historical32head BF16norm/H128, original sampler and81920 cap are unchanged. Prompt token0 and decode use true24layer QDQ/writeback; next consumption uses actual reconstructed state. Numerical81K equivalence is not claimed from shortqualification or synthetic context pressure.\n\nThe recurrent carrier is FP32 postQDQ, not packedINT8. Codes/scales/negative-zero flags, staticomega and dynamicRowBal factors have distinct storage/computation costs. Actual matched GPU timings, CUDAkernel counts and peakmemory accompany the result. No untampered highprecision continuation shadow, futurequery oracle, calibration on AIME, shortened cap, GPU retries or answer-selected repeat were used. Historical source/seals and stopped attempts remain unchanged.\n'
 else:
  status='PARTIAL_RESULTS_PRESERVED_AND_STOP';counts={q:0 for q in C.METHODS}
  for j in C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']:
   p=R/'formal'/j['id']/'RESULT.json';interval=R/'formal'/j['id']/'RESOURCE_INTERVAL.json'
   if p.exists() and interval.exists():
    x=C.read(interval)
    if x['exit_returncode']==0 and x['release']=='PASS':counts[j['condition']]+=1
  report='# True STOP preserved\n\nStatus: '+status+'. Actual fresh formal complete legal counts: '+json.dumps(counts)+'. Qualification accounting: '+json.dumps(accounting)+'. Reused48 remain original BYTEVIEW results; new qualification maximum4 and formal maximum90 are separate. No full30 accuracy, no empty scientific CSV, no automatic rerun or prefix concatenation.\n\n'+json.dumps(C.read(R/'FIRST_FAILURE.json'),ensure_ascii=False,indent=2)+'\n'
 write('FINAL_SCIENTIFIC_REPORT.md',report);dump('FINAL_TASK_STATUS.json',dict(status=status,own_GPU_processes_released=True,raw_model_downloads=0,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),qualification_accounting=accounting,output_policy='55tmpfs_scratch_29_per_closed_unit_SHA_bytes_NOREPLACE_durable'))
 # Freeze all actual files except live finisher output; no raw model is under R.
 rows=[]
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.name in ['finisher.log','SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','FINAL_ARCHIVE_VERIFICATION.json','FINALIZER_FAILURE.json']:continue
  rows.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=C.sha(p)))
 dump('SHA256_MANIFEST.json',dict(task_id=C.TASK,status=status,files=rows,count=len(rows),total_bytes=sum(x['bytes'] for x in rows)))
 write('SHA256SUMS.txt',''.join(x['sha256']+'  '+x['path']+'\n' for x in rows))
 stage=ARCHIVE+'.staging_'+C.sha(R/'SHA256_MANIFEST.json')[:16]
 remote('''import sys,json,os
from pathlib import Path
final=Path(sys.argv[1]);stage=Path(sys.argv[2]);assert not final.exists() and not stage.exists(),'Archive publication/staging already exists; inspect, no duplicate'
s=os.statvfs('/data/zypan');assert s.f_bavail*s.f_frsize>=120*1024**3,'Archive reservation+headroom insufficient';stage.mkdir(parents=True)
print(json.dumps(dict(status='NEW_STAGE_RESERVED')))
''',ARCHIVE,stage)
 sshcmd=shlex.join(['ssh','-S',SOCKET,'-o','BatchMode=yes']);p=subprocess.run(['rsync','-a','--exclude','finisher.log','-e',sshcmd,str(R)+'/',HOST+':'+stage+'/'],capture_output=True,timeout=2400);C.require(p.returncode==0,'Archive transfer failure:'+p.stderr.decode())
 verification=remote('''import sys,json,hashlib,os,ctypes
from pathlib import Path
stage=Path(sys.argv[1]);final=Path(sys.argv[2]);manifest=json.loads((stage/'SHA256_MANIFEST.json').read_text());count=0
for x in manifest['files']:
 p=stage/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==x['sha256'],'Independent archived SHA mismatch:'+x['path'];count+=1
v=dict(status='PASS_INDEPENDENT29_ALL_FILE_SHA_BYTES_ATOMIC_SEAL',count=count,total_bytes=manifest['total_bytes'],manifest_sha256=hashlib.sha256((stage/'SHA256_MANIFEST.json').read_bytes()).hexdigest(),archive_root=str(final))
(stage/'FINAL_ARCHIVE_VERIFICATION.json').write_text(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\\n')
seal=dict(status='PASS_FINAL_SEAL',manifest_sha256=v['manifest_sha256'],archive_verification_sha256=hashlib.sha256((stage/'FINAL_ARCHIVE_VERIFICATION.json').read_bytes()).hexdigest(),scientific_status=manifest['status'])
(stage/'FINAL_SEAL_RECEIPT.json').write_text(json.dumps(seal,sort_keys=True,separators=(',',':'),allow_nan=False)+'\\n')
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;rc=fn(-100,os.fsencode(stage),-100,os.fsencode(final),1);assert rc==0,os.strerror(ctypes.get_errno());fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(verification=v,seal=seal)))
''',stage,ARCHIVE)
 dump('FINAL_ARCHIVE_VERIFICATION.json',verification['verification']);dump('FINAL_SEAL_RECEIPT.json',verification['seal'])
except BaseException as exc:
 if not (R/'FINALIZER_FAILURE.json').exists():dump('FINALIZER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0))
 raise
