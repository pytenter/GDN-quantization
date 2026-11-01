"""CPU-only legal60 scoring or honest STOP, followed by independent no-clobber29 seal."""
import sys,os,json,subprocess,shlex,time,traceback,csv,statistics
from pathlib import Path
R=Path(sys.argv[1]);sys.path.insert(0,str(R/'src'));import execution_control as C
ARCHIVE=C.read(R/'ROOT.json')['archive_root'];SOCKET='/tmp/qwen_replication_archive29.sock';HOST='zypan@202.38.247.29'
def remote(code,*args):
    p=subprocess.run(['ssh','-S',SOCKET,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,shlex.join(['/usr/bin/python3','-u','-B','-c',code,*args])],capture_output=True,timeout=300)
    C.require(p.returncode==0,'Archive CPU operation failed:'+p.stderr.decode());return json.loads(p.stdout)
def dump(name,obj):C.atomic_new(R/name,obj)
def write(name,text):
    with (R/name).open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
def cpu_pass(program,name):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',USE_TORCH='0',USE_TF='0')
    p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts'/program),str(R),str(R/name)],env=env,capture_output=True,timeout=1200)
    (R/(name+'.log')).write_bytes(p.stdout+p.stderr);C.require(p.returncode==0,'Independent CPU scoring failed:'+name);return C.read(R/name)
C.require(not (R/'FINALIZER_STARTED.json').exists(),'Finalizer already started; never duplicate')
dump('FINALIZER_STARTED.json',dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic()))
try:
 terminal=C.read(R/'SESSION_RESULT.json');counts=terminal['counts'];ledger=C.read(R/'RESOURCE_LEDGER.json');obs=C.query_gpu();children=[x['identity'] for x in ledger['intervals'] if x['identity']]
 C.require(all(not C.same_process(i) for i in children) and not C.same_process(C.read(R/'GPU_RUN_STARTED.json')['identity']) and not any(x['pid'] in {i['pid'] for i in children} for x in obs['apps']),'Own coordinator/worker/NVML still active')
 C.require(all(x['release']=='PASS' for x in ledger['intervals']),'Own worker GPU release unresolved; no seal')
 C.verify_sources(R);C.model_guard(R);external=C.verify_external(R)
 dump('analysis/FINAL_SOURCE_DEPENDENCY_MODEL_RELEASE_AUDIT.json',dict(status='PASS',source_count=C.read(R/'SOURCE_FREEZE.json')['count'],external_fresh_SHA_files=external,model_stat_files=17,own_exact_coordinator_and_workers_exited=True,own_NVML_released=True,observation=obs))
 jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];conditions=['P1','C1','S3'];scores=[];by={};totals={};lengths={};pairs={};history=None
 if sum(counts.values()):
  a=cpu_pass('score_pass_cpu.py','SCORE_PASS_0.json');b=cpu_pass('score_pass_cpu.py','SCORE_PASS_1.json');C.require(a==b,'Two independent originalFrozenV4 CPU/token-decode passes differ');dump('FROZEN_V4_SCORES.json',a);scores=a['scores']
  for row in scores:by.setdefault(row['problem_id'],{})[row['condition']]=row
  C.require(len(scores)==sum(counts.values()),'Actual scored payload count differs')
  with (R/('FORMAL_GENERATIONS_90.jsonl' if sum(counts.values())==90 else 'FORMAL_GENERATIONS_PARTIAL.jsonl')).open('x') as f:
   for j in jobs:
    p=R/'formal'/j['id']/'RESULT.json'
    if p.exists() and (p.parent/'RESOURCE_INTERVAL.json').exists() and C.read(p.parent/'RESOURCE_INTERVAL.json')['exit_returncode']==0:f.write(json.dumps(C.read(p),ensure_ascii=False,sort_keys=True)+'\n')
  with (R/'AIME26_PER_PROBLEM_RESULTS.csv').open('x',newline='') as f:
   keys=['problem_id','condition','correct','extracted_answer','generated_token_count','finish_reason','gpu_uuid','wall_seconds','TPS'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
   for row in scores:w.writerow({k:row[k] for k in keys})
  for c in conditions:
   rr=[x for x in scores if x['condition']==c];N=len(rr);correct=sum(x['correct'] for x in rr);totals[c]=dict(completed=N,correct=correct,denominator=30 if N==30 else N,accuracy=correct/30 if N==30 else None,status='FULL30' if N==30 else 'PARTIAL' if N else 'NOT_RUN')
   if N:lengths[c]=dict(N=N,total_tokens=sum(x['generated_token_count'] for x in rr),mean_tokens=statistics.mean(x['generated_token_count'] for x in rr),median_tokens=statistics.median(x['generated_token_count'] for x in rr),EOS=sum(x['finish_reason']=='eos' for x in rr),length=sum(x['finish_reason']=='length' for x in rr),generation_wall_seconds=sum(x['wall_seconds'] for x in rr),aggregate_TPS=sum(x['generated_token_count'] for x in rr)/sum(x['wall_seconds'] for x in rr),description='Per-sample true prompt/decode QDQ+sampler time; model loading, driver/library checks, exit/release separate in GPU intervals')
  P=Path('/data/zypan/analyses/qwen_gdn_group32_four_codec_aime26_seed1_v1');hist=[]
  for j in C.read(P/'FORMAL_JOB_MANIFEST.json')['jobs']:
   p=P/'formal'/j['id']/'RESULT.json';interval=p.parent/'RESOURCE_INTERVAL.json'
   if j['condition']=='Q2' and p.exists() and interval.exists():
    ir=C.read(interval);C.require(ir['exit_returncode']==0 and ir['release']=='PASS','Original P0 not cleanly released');hist.append(dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=C.sha(p)))
  dump('analysis/HISTORICAL_P0_INPUT_MANIFEST.json',dict(root=str(P),files=hist,read_only=True,observed_monotonic=time.monotonic()))
  h0=cpu_pass('history_p0_score_cpu.py','P0_HISTORY_SCORE_PASS_0.json');h1=cpu_pass('history_p0_score_cpu.py','P0_HISTORY_SCORE_PASS_1.json');C.require(h0==h1,'Two originalP0 CPU score passes differ');history=h0;hb={x['problem_id']:x for x in h0['scores']}
  for q,row in hb.items():by.setdefault(q,{})['P0']=row
  for c,d in [('P1','P0'),('C1','P0'),('S3','C1'),('S3','P1'),('C1','P1')]:
   common=sorted(q for q in by if c in by[q] and d in by[q]);N=len(common)
   if N:
    nc=sum(by[q][c]['correct'] for q in common);nd=sum(by[q][d]['correct'] for q in common);pairs[c+'-'+d]=dict(N=N,problem_ids=common,both_correct=sum(by[q][c]['correct'] and by[q][d]['correct'] for q in common),both_wrong=sum(not by[q][c]['correct'] and not by[q][d]['correct'] for q in common),rescue_problem_ids=[q for q in common if by[q][c]['correct'] and not by[q][d]['correct']],lost_problem_ids=[q for q in common if by[q][d]['correct'] and not by[q][c]['correct']],net=nc-nd,delta_accuracy=(nc-nd)/N,comparison='HISTORICAL_EXPLORATORY' if d=='P0' else 'MATCHED_NEW_SAME_PROBLEM_SAME_UUID_SEED1',partial=N!=30)
  four=sorted(q for q,row in by.items() if all(c in row for c in ['P0','P1','C1','S3']));I=None
  if four:
   n=len(four);t={c:sum(by[q][c]['correct'] for q in four) for c in ['P0','P1','C1','S3']};I=dict(N=n,problem_ids=four,correct_counts=t,value=((t['S3']-t['C1'])-(t['P1']-t['P0']))/n,status='HISTORICAL_EXPLORATORY_NOT_STRICT_CAUSAL_2X2')
  independent_only={c:[q for q in by if c in by[q] and by[q][c]['correct'] and all(d in by[q] and not by[q][d]['correct'] for d in conditions if d!=c)] for c in conditions}
  analysis=dict(new_results=totals,new_counts=counts,pairs=pairs,interaction=I,new_three_way_only_correct_problem_ids=independent_only,historical_P0_N=len(hb),historical_P0_correct=sum(x['correct'] for x in hb.values()),historical_protocol_differences=h0['protocol_differences'],cross_UUID_historical_questions=[q for q in hb if q in by and 'P1' in by[q] and hb[q]['gpu_uuid']!=by[q]['P1']['gpu_uuid']],single_seed_small_N=True,AIME_previously_observed=True,no_cross_device_bitwise_equivalence=True,no_mechanistic_causality_claim=True)
  dump('ROTATION_PERMUTATION_INTERACTION.json',analysis)
 else:
  analysis=dict(status='NO_LEGAL_AIME_PAYLOADS_NO_ACCURACY_PUBLISHED',new_counts=counts,failures=terminal.get('failures',[]))
 u_cpu=C.read(R/'branches/U_CPU_ELIGIBILITY.json') if (R/'branches/U_CPU_ELIGIBILITY.json').exists() else dict(eligible=False)
 fp=C.read(R/'REAL_9B_FP_ROTATION_EQUIVALENCE.json') if (R/'REAL_9B_FP_ROTATION_EQUIVALENCE.json').exists() else None
 if all(n==30 for n in counts.values()):status='FORMAL_COMPLETE_90_90_AND_STOP'
 elif counts['P1']==30 and (not u_cpu['eligible'] or (fp is not None and not fp['all_devices_pass'])):status='P1_COMPLETE_U_BRANCH_FAILED_AND_STOP'
 elif fp is not None and not fp['all_devices_pass']:status='ROTATION_FP_FUNCTIONAL_QUALIFICATION_FAIL_CLOSED_STOP'
 elif not u_cpu['eligible']:status='V2_NUMERICAL_CONTRACT_FAIL_CLOSED_STOP'
 else:status='PARTIAL_RESULTS_PRESERVED_AND_STOP'
 dump('GENERATION_LENGTH_EOS_RUNTIME_AND_COST.json',dict(lengths=lengths,resource_ledger=ledger,failures=terminal.get('failures',[]),EXACT_GPUh='UNRESOLVED'))
 table='\n'.join('|'+c+'|'+str(counts[c])+'/30|'+(str(totals[c]['correct'])+'/30 | '+f"{totals[c]['accuracy']:.2%}" if c in totals and counts[c]==30 else 'NOT_RUN/PARTIAL | N/A')+'|' for c in conditions)
 report='# Qwen/GDN OSCAR Sym32 Unified V2\n\n\u72b6\u6001\uff1a'+status+'\u3002\u5b9e\u9645\u5408\u6cd5\u751f\u6210 '+str(sum(counts.values()))+'/90\uff1b\u6bcf\u5206\u652f\u771f\u5b9e\u6570\u91cf\u89c1\u8868\uff0c\u4e0d\u628aP1\u6210\u529f\u79f0\u4e3a\u6574\u4f5390/90\u3002P0\u6ca1\u6709\u65b0\u589e\u751f\u6210\u3002\n\n|\u914d\u7f6e|\u5408\u6cd5\u5b8c\u6210|\u6b63\u786e/30|\u51c6\u786e\u7387|\n|---|---:|---:|---:|\n'+table+'\n\n'+json.dumps(analysis,ensure_ascii=False,indent=2)+'\n\n'+json.dumps(lengths,ensure_ascii=False,indent=2)+'\n\nV1\u539f\u59cbFAIL\u7ec8\u6001\u4e0e585\u6587\u4ef6\u5c01\u5b58\u53ea\u8bfb\u4fdd\u7559\u3002V2\u590d\u7528\u540c\u4e00\u5b8c\u6574Cq\uff0c24\u5c42\xd716\u552f\u4e00Query\u5934\u3001\u6bcf\u59344096\u65f6\u95f4\u76f8\u5173\u975eAIME\u6837\u672c\uff0crepeat_interleave2\u6620\u5c0432State\u5934\u4e0d\u589e\u52a0\u72ec\u7acb\u6837\u672c\uff1b\u4e0d\u91cd\u91c7Query\u3001\u4e0d\u6362\u8bed\u6599\u3002\u539f\u786e\u5b9a\u6027FP64 eigenbasis\u89c4\u8303\u4e0d\u53d8\uff1b\u552f\u4e00canonical\u8def\u7ebfFP64\u5b8c\u6574A=UH\u540e\u4e00\u6b21FP32\u8f6c\u6362\uff0c\u4e0d\u6309\u65e7\u9608\u503c\u8de8\u8d8a\u60c5\u51b5\u6216AIME\u6210\u7ee9\u9009\u8def\u7ebf\u3002M0\u2013M3\u5168\u5934\u9762\u677f\u3001C\u8f6c\u6362\u3001FP32\u68c0\u67e5\u5668\u3001gamma128/\u8868\u793a\u6270\u52a8\u9884\u7b97\u4e0e\u72ec\u7acb384\u5934\u9a8c\u8bc1\u5b8c\u6574\u4fdd\u7559\uff1bV1\u5b9e\u9645\u68c0\u67e5\u662fFP32\u6784\u9020A\u540e\u5728FP64 C\u4e0a\u8bc4\u4f30\uff0cM3\u53e6\u5217\u4e3a\u5b8c\u6574FP32\u8bca\u65ad\u3002\u65b0\u6570\u503c\u5408\u540c\u5148\u4e8eV2\u9762\u677f\u548cGPU\u51bb\u7ed3\uff0c\u4e0d\u80fd\u5012\u6539V1\u4e3aPASS\u3002\n\nP1\u4e0d\u4f9d\u8d56U\uff0c\u4fdd\u6301\u539fH\u5750\u6807\u548c\u5b8c\u5168\u539fSym32\u3002C1/S3\u5171\u7528\u540c\u4e00U/A\u9759\u6001\u6587\u4ef6\u4e0e\u540c\u4e00\u5b9e\u9645FP\u9012\u5f52\u3001Q/K\u540c\u6b65\u53d8\u6362\uff0c\u957f\u671fState\u4e3aA\u5750\u6807\uff1bPBR\u53ea\u5728QDQ\u5185\u6309Key\u884c\u6392\u5217\u548c\u9006\u6392\u5217\u3002\u5f52\u4e00\u5316/\u56e0\u679c\u5377\u79ef/Gate/Beta/Head\u5171\u4eab\u539f\u751f\u4f4d\u7f6e\u4e0d\u6539\uff0c\u4e0d\u628a\u65cb\u8f6c\u79fb\u5230\u975e\u7ebf\u6027\u4e4b\u524d\u3002\u6240\u6709GPU\u53ef\u6267\u884c\u6e90\u7801\u9996\u6b21GPU\u4e4b\u524d\u4e00\u6b21\u51bb\u7ed3\uff0c\u72ec\u7acbP1\u8eab\u4efd\u4e0d\u542bU\u8d44\u4ea7\u3002\u5404\u5206\u652f\u8d44\u683c\u72ec\u7acb\uff0c\u5931\u8d25\u4e0d\u53d6\u6d88\u5176\u4ed6\u5408\u6cd5\u5206\u652f\uff1b\u6ca1\u6709patch-and-retry\u3001\u5df2STARTED\u91cd\u6d3e\u3001\u8f68\u8ff9\u62fc\u63a5\u6216GPU\u8fc1\u79fb\u3002\n\n\u539fSignedINT8 KeyGroup32\u3001FP32scale\u3001CUDA\u9664\u6cd5/\u820d\u5165/\u88c1\u526a/\u8d1f\u96f6\u548cdecode\u9010\u5b57\u8282\u539f\u6838\u5fc3\u4e0d\u53d8\u3002\u6bcf\u4e2aPrompt token\u4ecetoken0\u548cdecode\u90fd\u771f\u6b63QDQ\u7acb\u5373writeback\uff1bp+n-1\u5b9e\u9645forward\u53ca24(p+n-1)\u5199\u56de\uff0c\u6700\u540e\u8f93\u51fatoken\u4e0d\u518d\u9001\u56de\u6a21\u578b\u3002persistent\u8f7d\u4f53\u5b9e\u9645\u4e3a\u5df2QDQ\u91cd\u5efaFP32\u7ea648MiB\uff0c\u975e\u7269\u7406packedINT8\u3002512\u4e24fresh\u548c\u975eAIME free64\u7684\u5de5\u7a0b\u6bd4\u8f83\u53c2\u8003\u53ea\u5728\u4e13\u7528\u6709\u754cRAM\u4e2d\uff0cpairedPASS\u5e76\u6301\u4e45\u56de\u6267\u540e\u5220\u9664\uff1b\u4e0d\u662fcontinuation state\u3002\u6b63\u5f0f\u4e0e4096\u4f7f\u7528\u51bb\u7ed3\u7a00\u758f\u4f4d\u7f6e\u5b8c\u6574\u5b9e\u9645\u503c/bytes next-kernel\u6d88\u8d39\u6838\u9a8c\uff0c\u4e0d\u80fd\u79f081K\u6570\u503c\u754c\u300282293\u662f\u6d3b\u6a21\u578b\u5408\u6210KV/nativeSDPA\u8d44\u6e90\u538b\u529b\uff0c\u4e0d\u662f81920\u8f68\u8ff9\u8bef\u5dee\u8bc1\u660e\u3002\n\n\u5b9e\u964532/128\u3001512fresh\u3001\u539fSampler/EOS/length\u30014096\u5b9e\u6d4b\u901f\u7387\u4e0e\u957fContext\u8d44\u683c\u89c1\u6bcf\u5206\u652f\u5b9e\u9645\u56de\u6267\uff0c\u672a\u8fd0\u884c\u9636\u6bb5\u4e0d\u4f2a\u9020PASS\u3002\u771f\u5b9e9B Native/H/UH/PBR-off FP\u8fde\u7eed128token\u7684\u6bcf\u5c42State\u3001Q/K\u3001Gate/Beta\u3001\u72ec\u7acbFP64\u66f4\u65b0/Readout\u300132\u5c42\u8f93\u51fa\u53calogits\u6838\u9a8c\u72ec\u7acb\u4e8eCPU\u77e9\u9635\u4ee3\u6570\uff1bC1/S3\u65e0QDQ\u4e3a\u76f8\u540c\u6570\u503c\u51fd\u6570\u3002P/H\u7f13\u5b58\u53ea\u53bb\u9664\u91cd\u590d\u6784\u9020\uff0c\u77e9\u9635\u539f\u5408\u540c\u4fdd\u6301\uff0c\u6b63\u5f0f\u6bcftoken\u4e0d\u8f6c\u6362\u5b8c\u6574State\u56deNative\u3002\n\n\u4e09\u5361TP1\u540c\u9898bundle\u540cUUID\u3001hash10/10/10\u3001\u5faa\u73af\u65b9\u6cd5\u987a\u5e8f\u9884\u51bb\u7ed3\uff0c\u6bcf\u6761\u72ec\u7acbfreshprocess/RNG\u8bf7\u6c42/recurrent/KV/generated-only penalty\u5386\u53f2\uff1b\u539fQwen thinking/Sampler/top-k20\u539fties\u884c\u4e3a\u4e0d\u53d8\uff0c81920\u65b0token cap\u3002EOS\u6216\u5408\u6cd5length\u5747\u4e3a\u5b8c\u6574\u6837\u672c\uff0c\u6ca1\u6709\u6309\u6b63\u786e\u7387\u505c\u8d1f\u7ed3\u679c\u6216\u6311\u9009\u91cd\u8bd5\u3002\u975eAIME\u5b9e\u9645State/Readout\u914d\u5bf9\u6270\u52a8\u3001scale/clip/outlier\u5206\u7ec4\u8bca\u65ad\u89c1CALIBRATION_NUMERICAL_DIAGNOSTICS\uff08\u82e5\u5b9e\u9645\u5b8c\u6210\uff09\uff0c\u4e0d\u636e\u5c40\u90e8MSE\u63a8\u65adAIME\u6539\u5584\uff1b\u65e0FutureKL/\u8bad\u7ec3/INT4/Asym\u65b0\u5206\u652f\u3002\n\n\u5386\u53f2P0\u8f93\u5165\u4e0e\u534f\u8bae\u9010\u9879\u590d\u6838\u3001\u5171\u540c\u9898N\u6309\u5b9e\u9645\u5b8c\u6210\u6570\uff1b\u5386\u53f2\u8bbe\u5907/\u8fd0\u884c\u65f6\u70b9\u53ca\u65e2\u6709AIME\u89c2\u5bdf\u4f7f2\xd72\u4ec5\u63a2\u7d22\u6027\uff0c29\u9898\u7edd\u4e0d\u5f5330\u9898\u3002\u5c0fN\u5355seed\u4e0e\u8de8GPU\u975ebitwise\u9650\u5b9a\u4fdd\u7559\u3002\u5b9e\u9645\u6210\u672c\u6309\u540c\u4e3b\u673amonotonic\u4e0a\u4e0b\u754c\u3001\u52a0\u8f7d/\u8d44\u683c/\u6b63\u5f0f/\u9000\u51fa\u91ca\u653e\u8d26\u672c\uff0cEXACT\u7269\u7406GPUh\u4ecdUNRESOLVED\uff1b\u65e7INT8 TPS\u4e0d\u5f53\u65b0UH\u5b9e\u6d4b\u901f\u7387\uff0c\u77ed/late4096\u901f\u7387\u4e0d\u627f\u8bfa81K\u901f\u5ea6\u6216\u5b8c\u6210\u65e5\u671f\u3002\n'
 write('FINAL_SCIENTIFIC_REPORT.md',report);dump('FINAL_TASK_STATUS.json',dict(status=status,completed=counts,formal_completed=sum(counts.values()),formal_total=90,own_GPU_processes_released=True,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),branch_failures_preserved=True))
 # Freeze all actual files except live finisher output; no raw model is under R.
 rows=[]
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.is_symlink() or str(p.relative_to(R)) in ['finisher.log','SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','FINAL_ARCHIVE_VERIFICATION.json','FINALIZER_FAILURE.json']:continue
  rows.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=C.sha(p)))
 C.require(sum(x['bytes'] for x in rows)<=20*1024**3,'New archive exceeds20GiB bound')
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
(stage/'FINAL_ARCHIVE_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\\n')
seal=dict(status='PASS_FINAL_SEAL',manifest_sha256=v['manifest_sha256'],archive_verification_sha256=hashlib.sha256((stage/'FINAL_ARCHIVE_VERIFICATION.json').read_bytes()).hexdigest(),scientific_status=manifest['status'])
(stage/'FINAL_SEAL_RECEIPT.json').write_text(json.dumps(seal,indent=2)+'\\n')
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;rc=fn(-100,os.fsencode(stage),-100,os.fsencode(final),1);assert rc==0,os.strerror(ctypes.get_errno());fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(verification=v,seal=seal,verification_text=(final/'FINAL_ARCHIVE_VERIFICATION.json').read_text(),seal_text=(final/'FINAL_SEAL_RECEIPT.json').read_text())))
''',stage,ARCHIVE)
 write('FINAL_ARCHIVE_VERIFICATION.json',verification['verification_text']);write('FINAL_SEAL_RECEIPT.json',verification['seal_text'])
except BaseException as exc:
 if not (R/'FINALIZER_FAILURE.json').exists():dump('FINALIZER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0))
 raise
