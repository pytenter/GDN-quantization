"""Unique CPU finisher after controller exit; score only complete120, archive true terminal."""
import sys,os,json,subprocess,time,traceback,csv,math,shutil,ctypes
from pathlib import Path
R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R/'src'));import execution_control as C
ARCHIVE=Path(C.read(R/'ROOT.json')['archive_root'])
def dump(n,a):C.atomic_new(R/n,a)
def write(n,s):
 with(R/n).open('x')as f:f.write(s);f.flush();os.fsync(f.fileno())
def pair(new,base):
 ids=sorted(new);C.require(ids==sorted(base) and len(ids)==30,'Pairing roster differs')
 rescued=[p for p in ids if new[p]and not base[p]];lost=[p for p in ids if base[p]and not new[p]];n=len(rescued)+len(lost);small=min(len(rescued),len(lost));p=1. if n==0 else min(1.,2*sum(math.comb(n,k)for k in range(small+1))/2**n)
 return dict(rescued=len(rescued),lost=len(lost),rescued_ids=rescued,lost_ids=lost,net_correct=len(rescued)-len(lost),delta_accuracy=(len(rescued)-len(lost))/30,exact_McNemar_two_sided_p=p,both_correct=sum(new[p]and base[p]for p in ids),both_wrong=sum(not new[p]and not base[p]for p in ids),test_label='EXPLORATORY_UNADJUSTED_MULTIPLE_COMPARISONS')
C.require(C.NODE['id']=='server29_gpu6','Global CPU finisher must run29 profile')
C.require(C.read(R/'GLOBAL_COLLECTION_RECEIPT.json')['status']=='PASS_BOTH_NODE_TERMINALS_AND_PEER55_SNAPSHOT_SHA_BYTES','Global collection missing')
from cross_host_contract import validate_global_layout
layout=validate_global_layout(R)
def source_for(job):return R if job['node_id']=='server29_gpu6'else R/'peer55'
C.require(not(R/'FINALIZER_STARTED.json').exists(),'Finisher already started; no repeated scoring/publication')
started=C.read(R/'GPU_RUN_STARTED.json')if(R/'GPU_RUN_STARTED.json').exists()else None
C.require(started is None or not C.same_process(started['identity']),'Controller still alive')
dump('FINALIZER_STARTED.json',dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic()));terminal=C.read(R/'GLOBAL_SESSION_RESULT.json');ledger=C.read(R/'RESOURCE_LEDGER.json')if(R/'RESOURCE_LEDGER.json').exists()else dict(status='NODE_NOT_STARTED_NO_GPU_DISPATCH',intervals=[],reservation_GPUh_lower=0,reservation_GPUh_upper=0,EXACT_physical_GPUh='UNRESOLVED');obs=C.query_gpu();children=[x['identity']for x in ledger['intervals']if x['identity']]
C.require(all(not C.same_process(i)for i in children)and not any(x['pid']in{i['pid']for i in children}for x in obs['apps']),'Own worker process/NVML not released')
try:
 C.verify_sources(R);C.verify_sources(R/'peer55');C.model_guard(R)
 host_ledgers={n:(C.read(base/'RESOURCE_LEDGER.json')if(base/'RESOURCE_LEDGER.json').exists()else dict(status='NODE_NOT_STARTED_NO_GPU_DISPATCH',intervals=[],reservation_GPUh_lower=0,reservation_GPUh_upper=0,EXACT_physical_GPUh='UNRESOLVED'))for n,base in [('server29_gpu6',R),('server55_gpu0',R/'peer55')]}
 dump('CROSS_HOST_RESOURCE_LEDGER.json',dict(status='HOST_SEPARATED_SAME_HOST_MONOTONIC_INTERVALS',nodes=host_ledgers,EXACT_physical_GPUh='UNRESOLVED',cross_host_absolute_tick_comparison=False,reservation_GPUh_lower=sum(x['reservation_GPUh_lower']for x in host_ledgers.values()),reservation_GPUh_upper=sum(x['reservation_GPUh_upper']for x in host_ledgers.values()),historical_costs='PRIOR_EXECUTION_HISTORY_AND_COST.json, never reclassified as new credit'))
 conditions=['G64-'+str(i)for i in range(4)];full=terminal['status']=='GPU_COMPLETE_CROSS_HOST_GROUP64_120_FORMAL'
 if full:
  for rep in [0,1]:
   env=C.worker_environment(C.NODE['physical_gpu']);env['CUDA_VISIBLE_DEVICES']=''
   p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts/score_global_pass_cpu.py'),str(R),str(R/f'SCORE_PASS_{rep}.json')],env=env,capture_output=True,timeout=1800)
   (R/f'SCORE_PASS_{rep}.log').write_bytes(p.stdout+p.stderr);C.require(p.returncode==0,'Fresh CPU scoring failed; original payload preserved')
  a=C.read(R/'SCORE_PASS_0.json');b=C.read(R/'SCORE_PASS_1.json');C.require(a==b,'Fresh samepayload score disagreement');dump('FROZEN_V4_SCORES.json',a);scores=a['scores']
  by={}
  for x in scores:by.setdefault(x['problem_id'],{})[x['condition']]=x
  C.require(len(by)==30 and all(set(v)==set(conditions)for v in by.values()),'Incomplete30x4 pairing')
  oldproof=C.read(R/'GROUP32_HISTORY_REUSE_AND_IDENTITY.json');oldpath=R/'history/group32/AIME26_FROZEN_V4_SCORES.json';C.require(C.sha(oldpath)==oldproof['sealed_scores_sha256'],'Old frozen score bytes changed')
  old=C.read(oldpath)['scores'];oldvec={q:{x['problem_id']:bool(x['correct'])for x in old if x['condition']==q}for q in ['Q0','Q1','Q2','Q3']};vec={q:{p:bool(v[q]['correct'])for p,v in by.items()}for q in conditions}
  totals={q:sum(v.values())for q,v in vec.items()};oldtotals={q:sum(v.values())for q,v in oldvec.items()};C.require(list(oldtotals.values())==[20,25,21,25],'Historical frozen totals not original')
  pairs={}
  for new,base in [('G64-1','G64-0'),('G64-3','G64-2'),('G64-2','G64-0'),('G64-3','G64-1')]:pairs[new+'-'+base]=pair(vec[new],vec[base])
  for i in range(4):pairs[f'G64-{i}-Q{i}']=pair(vec[f'G64-{i}'],oldvec[f'Q{i}'])
  dump('GROUP32_GROUP64_PAIRWISE.json',dict(N=30,seed=1,current_totals=totals,historical_totals=oldtotals,pairwise=pairs,correctness_vectors=dict(vec,**oldvec),baseline_readonly=True,baseline_rescored=False,hardware_pairing='WithinG64 samequestion sameUUID; G64 fixed15/15 server55GPU0/server29GPU6; G32 originalserver55UUIDs; crossGroup comparison has host/UUID/driver differences'))
  t=totals;o=oldtotals;I64=(t['G64-3']-t['G64-2']-t['G64-1']+t['G64-0'])/30;I32=(o['Q3']-o['Q2']-o['Q1']+o['Q0'])/30
  interactions=dict(G64_Hadamard_x_Asym=I64,G32_Hadamard_x_Asym=I32,group_x_Asym_Native=(t['G64-1']-t['G64-0']-o['Q1']+o['Q0'])/30,group_x_Asym_H128=(t['G64-3']-t['G64-2']-o['Q3']+o['Q2'])/30,group_x_H_Sym=(t['G64-2']-t['G64-0']-o['Q2']+o['Q0'])/30,group_x_H_Asym=(t['G64-3']-t['G64-1']-o['Q3']+o['Q1'])/30,three_way=I64-I32)
  cells={(g,h,a):((oldvec if g==0 else vec)[('Q'if g==0 else'G64-')+str(2*h+a)])for g in [0,1]for h in [0,1]for a in [0,1]}
  main={}
  for axis,name in [(0,'Group64_minus32'),(1,'H128_minusNative'),(2,'Asym_minusSym')]:
   high=[sum(v.values())/30 for k,v in cells.items()if k[axis]==1];low=[sum(v.values())/30 for k,v in cells.items()if k[axis]==0];main[name]=sum(high)/4-sum(low)/4
  dump('GROUP64_FACTORIAL_ANALYSIS.json',dict(N=30,seeds=[1],main_effects_marginal_accuracy=main,interactions_accuracy=interactions,interpretation='Exploratory observed paired accuracy contrasts, not mechanism/independent confirmation; same previously observed30questions and one seed; host/UUID/driver differences confound group comparison',test_label='EXPLORATORY_UNADJUSTED_MULTIPLE_COMPARISONS'))
  with(R/'AIME26_PER_PROBLEM_RESULTS.csv').open('x',newline='')as f:
   fields=['problem_id']+[q+'_'+s for q in conditions+['Q0','Q1','Q2','Q3']for s in ['correct','answer','tokens','termination','gpu_uuid']];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   oldby={(x['problem_id'],x['condition']):x for x in old}
   for p,v in sorted(by.items()):
    row={'problem_id':p}
    for q in conditions+['Q0','Q1','Q2','Q3']:
     x=v[q]if q in conditions else oldby[p,q]
     for key,val in [('correct',x['correct']),('answer',x['extracted_answer']),('tokens',x['generated_token_count']),('termination',x['finish_reason']),('gpu_uuid',x['gpu_uuid'])]:row[q+'_'+key]=val
    w.writerow(row)
  with(R/'FORMAL_GENERATIONS_120.jsonl').open('x')as f:
   for j in C.read(R/'GLOBAL_FORMAL_JOB_MANIFEST.json')['jobs']:f.write(json.dumps(C.read(source_for(j)/'formal'/j['id']/'RESULT.json'),ensure_ascii=False,sort_keys=True)+'\n')
  lengths={}
  for q in conditions:
   xs=[x for x in scores if x['condition']==q];n=sum(x['generated_token_count']for x in xs);wall=sum(x['wall_seconds']for x in xs);decode=sum(x['decode_and_sampling_seconds']for x in xs);prefill=sum(x['prompt_seconds']for x in xs)
   lengths[q]=dict(tokens=n,EOS=sum(x['finish_reason']=='eos'for x in xs),length=sum(x['finish_reason']=='length'for x in xs),generation_seconds=wall,generation_TPS=n/wall,decode_sample_TPS=n/decode,prefill_TPS=sum(x['input_tokens']for x in xs)/prefill,model_load_seconds=sum(x['model_load_seconds']for x in xs),max_allocated=max(x['peak_allocated_bytes']for x in xs),max_reserved=max(x['peak_reserved_bytes']for x in xs))
  write('GENERATION_LENGTH_AND_GPU_RESOURCES.md','# 实际生成与资源\n\n'+json.dumps(lengths,indent=2)+'\n\n生成TPS含Prompt/采样/QDQ；decode口径含首采样，末token不送回。加载另列并计入进程资源区间。EXACT_GPUh=UNRESOLVED，不能当0；详见samehostmonotonic上下界。理论packed与实际FP32decoded48MiB严格区分。\n')
  state='FORMAL_COMPLETE_120_120_AND_STOP';table='\n'.join('| '+q+' | '+str(totals[q])+'/30 | '+f'{totals[q]/30:.2%}'+' |'for q in conditions)
  report=f'# Group64 四配置 AIME26 Seed1\n\n状态：{state}。120条合法完成且实际exit0/releasePASS；两freshCPU原FrozenV4同payloadtoken重解码评分一致。\n\n| 配置 | 正确数 | 准确率 |\n|---|---:|---:|\n{table}\n\nG64-0 Native Sym64；G64-1 Native Asym64；G64-2 KeyH128 Sym64；G64-3 KeyH128 Asym64。历史Group32原封存20/25/21/25只读，未重新评分或重生成。\n\n'+json.dumps(pairs,ensure_ascii=False,indent=2)+'\n\n交互：\n\n'+json.dumps(interactions,indent=2)+'\n\n所有比较为EXPLORATORY_UNADJUSTED_MULTIPLE_COMPARISONS；30题/seed1/已多次观察AIME，不能视作独立确认或广泛泛化。差中差不是机制因果证明。Group32与Group64使用相同原Prompt/inputIDs/原Sampler/评分/QDQ时序；Group64按固定15/15题分配到server55 GPU0和server29 GPU6；同题四配置同一UUID。原Group32在server55原UUID。server55驱动535与server29驱动580不同，跨GroupSize和跨host比较须披露硬件/驱动因素。H仅缓存原矩阵同字节；原32head BF16归一化位置不变。旧C128/DualM不同Prefill/Runtime只背景，不作纯GroupSize比较。\n\n每token全24层真QDQ、即时写回、真实下一消费；短程完整实际数值/bytes比较，fresh进程完整冻结checkpoint比较。正式稀疏数值审计，非每token保存全State。无高精度continuation shadow。82293slot活模型KV/SDPA为合成显存压力，不是81K数值误差证明。实际FP32decodedState48MiB不等于packedINT8，理论13MiB不可冒充实际压缩。不同题目长度影响聚合TPS，GPU分布只描述；EXACT物理GPUh UNRESOLVED并报告区间。无训练/额外seed/FutureKL/INT4/PBR/OSCAR/答案挑选重试。\n'
 else:
  state='PARTIAL_RESULTS_PRESERVED_AND_STOP';counts={q:0 for q in conditions}
  for j in C.read(R/'GLOBAL_FORMAL_JOB_MANIFEST.json')['jobs']:
   p=source_for(j)/'formal'/j['id']/'RESOURCE_INTERVAL.json'
   if p.exists()and C.read(p)['exit_returncode']==0 and C.read(p)['release']=='PASS'and(source_for(j)/'formal'/j['id']/'RESULT.json').exists():counts[j['condition']]+=1
  failure=C.read(R/'GLOBAL_FIRST_FAILURE.json');dump('PARTIAL_COUNTS.json',dict(status=state,completed_legal_counts=counts,full_accuracy_reported=False,first_failure=failure))
  report='# Group64真实STOP交付\n\n状态：'+state+'。实际合法完成数：'+json.dumps(counts)+'。未完成120，不输出完整准确率或伪造空科学CSV。保留首因、原日志、原失败和真实退出释放，无GPU重试/轨迹拼接/热改。\n\n'+json.dumps(failure,ensure_ascii=False,indent=2)+'\n'
 write('FINAL_SCIENTIFIC_REPORT.md',report);C.status(R/'FORMAL_PROGRESS.json',dict(status=state,formal_completed=120 if full else sum(counts.values()),formal_planned=120))
 dump('FINAL_TASK_STATUS.json',dict(status=state,own_GPU_processes_released=True,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),raw_model_downloads=0))
 # All science files, including history and raw new fixtures, excluding only live log/seal outputs.
 excluded={'finisher.log','SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json','FINALIZER_FAILURE.json'};rows=[]
 for p in sorted(R.rglob('*')):
  C.require(not p.is_symlink(),'Scientific archive cannot contain symlinks:'+str(p))
  if p.is_file()and p.name not in excluded:rows.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=C.sha(p)))
 dump('SHA256_MANIFEST.json',dict(task_id=C.TASK,status=state,files=rows,count=len(rows),total_bytes=sum(x['bytes']for x in rows),external_env_model_not_inside_archive=True))
 write('SHA256SUMS.txt',''.join(x['sha256']+'  '+x['path']+'\n'for x in rows));msha=C.sha(R/'SHA256_MANIFEST.json');stage=Path(str(ARCHIVE)+'.staging_'+msha[:16]);C.require(not ARCHIVE.exists()and not stage.exists(),'Existing archive/stage; no duplicate publication');ARCHIVE.parent.mkdir(parents=True,exist_ok=True);stage.mkdir()
 for x in rows+[dict(path='SHA256_MANIFEST.json'),dict(path='SHA256SUMS.txt')]:
  source=R/x['path'];dest=stage/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 for x in rows:
  p=stage/x['path'];C.require(p.is_file()and not p.is_symlink()and p.stat().st_size==x['bytes']and C.sha(p)==x['sha256'],'Independent archive SHA mismatch:'+x['path'])
 v=dict(status='PASS_ALL_ARCHIVED_FILE_SHA_BYTES',count=len(rows),total_bytes=sum(x['bytes']for x in rows),manifest_sha256=msha,archive_root=str(ARCHIVE),same_host_independent_directory=True)
 C.atomic_new(stage/'FINAL_ARCHIVE_VERIFICATION.json',v);seal=dict(status='PASS_FINAL_SEAL',manifest_sha256=msha,archive_verification_sha256=C.sha(stage/'FINAL_ARCHIVE_VERIFICATION.json'),scientific_status=state,publication='renameat2 RENAME_NOREPLACE')
 C.atomic_new(stage/'FINAL_SEAL_RECEIPT.json',seal);lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;C.require(fn(-100,os.fsencode(stage),-100,os.fsencode(ARCHIVE),1)==0,'Archive atomic NOREPLACE failed:'+os.strerror(ctypes.get_errno()));fd=os.open(ARCHIVE.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 dump('FINAL_ARCHIVE_VERIFICATION.json',v);dump('FINAL_SEAL_RECEIPT.json',seal)
except BaseException as exc:
 if not(R/'FINALIZER_FAILURE.json').exists():dump('FINALIZER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0))
 raise
