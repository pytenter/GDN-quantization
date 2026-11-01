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
C.require(all(not C.same_process(i) for i in children) and not any(x['pid'] in {i['pid'] for i in children} for x in obs['apps']),'Actual own process/NVML not released')
try:
 if terminal['status']=='GPU_COMPLETE_58_QUALIFICATION_120_FORMAL':
  for rep in [0,1]:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');p=subprocess.run([sys.executable,'-u','-B',str(R/'scripts/score_pass_cpu.py'),str(R),str(R/('SCORE_PASS_'+str(rep)+'.json'))],env=env,capture_output=True,timeout=1200)
   (R/('SCORE_PASS_'+str(rep)+'.log')).write_bytes(p.stdout+p.stderr);C.require(p.returncode==0,'Independent FrozenV4 CPU pass failed')
  a=C.read(R/'SCORE_PASS_0.json');b=C.read(R/'SCORE_PASS_1.json');C.require(a==b,'Independent exact payload scoring disagreement');dump('AIME26_FROZEN_V4_SCORES.json',a);scores=a['scores'];by={}
  for row in scores:by.setdefault(row['problem_id'],{})[row['condition']]=row
  C.require(len(by)==30 and all(set(v)=={'Q0','Q1','Q2','Q3'} for v in by.values()),'Incomplete30four-way question pairing')
  totals={q:sum(int(v[q]['correct']) for v in by.values()) for q in ['Q0','Q1','Q2','Q3']};pairs={}
  for new,base in [('Q1','Q0'),('Q3','Q2'),('Q2','Q0'),('Q3','Q1')]:
   rescue=sum(v[new]['correct'] and not v[base]['correct'] for v in by.values());lost=sum(v[base]['correct'] and not v[new]['correct'] for v in by.values());n=rescue+lost;p=1. if not n else min(1.,2*sum(math.comb(n,k) for k in range(min(rescue,lost)+1))/2**n);pairs[new+'-'+base]=dict(rescued=rescue,lost=lost,net_correct=rescue-lost,delta_accuracy=(rescue-lost)/30,exact_McNemar_two_sided_p=p,exploratory_unadjusted_multiple_comparisons=True)
  interaction=(totals['Q3']-totals['Q2']-totals['Q1']+totals['Q0'])/30
  dump('AIME26_PAIRWISE_INTERACTION.json',dict(N=30,seeds=[1],totals=totals,pairwise=pairs,I=interaction,interpretation='Observed accuracy difference-in-differences, not mechanism causal proof',historically_observed_test=True))
  with (R/'AIME26_PER_PROBLEM_RESULTS.csv').open('x',newline='') as f:
   fields=['problem_id']+[q+'_'+s for q in ['Q0','Q1','Q2','Q3'] for s in ['correct','answer','tokens','termination','gpu_uuid','finite','wall_seconds']];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for pid,v in by.items():
    row={'problem_id':pid}
    for q,x in v.items():
     for k,val in [('correct',x['correct']),('answer',x['extracted_answer']),('tokens',x['generated_token_count']),('termination',x['finish_reason']),('gpu_uuid',x['gpu_uuid']),('finite',x['finite']),('wall_seconds',x['wall_seconds'])]:row[q+'_'+k]=val
    w.writerow(row)
  with (R/'FORMAL_RESULTS_120.jsonl').open('x') as f:
   for j in C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']:f.write(json.dumps(C.read(R/'formal'/j['id']/'RESULT.json'),ensure_ascii=False,sort_keys=True)+'\n')
  lengths={q:dict(total_tokens=sum(x['generated_token_count'] for x in scores if x['condition']==q),EOS=sum(x['finish_reason']=='eos' for x in scores if x['condition']==q),length=sum(x['finish_reason']=='length' for x in scores if x['condition']==q),seconds=sum(x['wall_seconds'] for x in scores if x['condition']==q)) for q in totals}
  for v in lengths.values():v['aggregate_generation_TPS']=v['total_tokens']/v['seconds']
  write('GENERATION_LENGTH_AND_THROUGHPUT.md','# 实际生成长度与吞吐\n\n'+json.dumps(lengths,indent=2)+'\n\n含各样本Prompt/采样/QDQ；按问题长度和GPU分层，不把不同题目构成解释为设备速度差。模型加载计入资源账本，未计入此生成循环TPS。\n')
  per_gpu={str(g):dict(problem_ids=[pid for pid,v in by.items() if v['Q0']['gpu']==g],correct_by_condition={q:sum(v[q]['correct'] for v in by.values() if v[q]['gpu']==g) for q in ['Q0','Q1','Q2','Q3']},interpretation='Different question mix, descriptive only; no crossGPU accuracy or speed causal comparison') for g in range(4)};dump('AIME26_PER_GPU_DESCRIPTIVE.json',per_gpu)
  status='FORMAL_COMPLETE_120_120_AND_STOP';table='\n'.join('| '+q+' | '+str(n)+'/30 | '+f'{n/30:.2%}'+' |' for q,n in totals.items())
  report=f'# Qwen/GDN Group32 四配置 AIME26 Seed1\n\n状态：{status}。全部120条合法生成、原FrozenV4及独立CPU复评分一致，所有自身GPU资源已释放。\n\n| 方法 | Correct/30 | Accuracy |\n|---|---:|---:|\n{table}\n\nQ0=Native Sym32，Q1=Native Asym32，Q2=Key-H128 Sym32，Q3=Key-H128 Asym32。四方法共享原Sampler、seed1、81920新token上限、逐Prompt token QDQ和同题同UUID。\n\n配对效果：\n\n'+json.dumps(pairs,ensure_ascii=False,indent=2)+f'\n\n差中差 I={interaction:.6f}（{interaction*100:.3f}个百分点）。正负仅解释本次30题中观察到的增量差，不能宣称已建立机制因果解释。各方法仅30题一个seed，少数题差异不证明广泛泛化。AIME26为历史多次观察的测试集。\n\nNative未量化数学控制、新Q0/Q1与四GPU有限资格均真实通过。长数值资格4096 token，82293 slot为活模型合成KV/原生SDPA压力；两者都不能冒充81K数值上界。Q2/Q3旧GPU0 512资格只读复用。Runtime为HISTORICAL_PORT；历史完整OLD二进制EXACT身份仍UNKNOWN。\n\n24层GDN从token0逐token QDQ并写回，nativeFP32已QDQ carrier48MiB；INT8/UINT8整数编码及FP32尺度是临时量化数据，并非物理packedINT8持久缓存。短程next-consumption比较完整真实值/bytes，正式长程按冻结稀疏位置比较，未逐token保存全state。每样本独立进程、cache、Generator与generated-only presence mask；没有未量化高精度continuation shadow。\n\n逐题配对与各样本GPU/长度/合法EOS或length见CSV，配置对比与交互见JSON。每卡固定8/8/7/7不同问题bundle，描述各卡结果时不能将题目难度差异误归因为硬件精度/速度。新逐token Prompt QDQ与旧AIME Prefill不同，历史FP/Dual-M等正确率仅作背景。实际吞吐/加载/资格/失败/退出释放见资源账本和长度报告。物理精确GPUh无法由轮询获得，保留上下界及UNRESOLVED，不将其当0。\n\n本任务未训练、未改权重/recurrence/Codec公式/原Sampler/FrozenV4、未新增Future-KL/FP正式AIME/额外seed，没有依据部分答案修改方案或迁移问题。来源身份、各层计数、Sampler/输入/输出/GPU绑定与最终退出均独立核验，完整源与结果由SHA清单封存。\n'
 else:
  status='PARTIAL_RESULTS_PRESERVED_AND_STOP';counts={q:0 for q in ['Q0','Q1','Q2','Q3']}
  for j in C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']:
   if (R/'formal'/j['id']/'RESULT.json').exists():counts[j['condition']]+=1
  report='# Qwen/GDN Group32 AIME26 真实STOP交付\n\n状态：'+status+'。\n\n实际完整生成数量：'+json.dumps(counts)+'；全部旧失败及当前FIRST_FAILURE/真实进程退出和释放证据保留。未完成120条，不报告完整30题准确率、不补空科学CSV，不擅自重试GPU、重跑前缀或拼接生成。\n\n首失败：\n\n'+json.dumps(C.read(R/'FIRST_FAILURE.json'),ensure_ascii=False,indent=2)+'\n'
 if terminal['status']=='GPU_COMPLETE_58_QUALIFICATION_120_FORMAL':report+='\n各卡问题与描述性正确数：\n\n'+json.dumps(per_gpu,ensure_ascii=False,indent=2)+'\n'
 write('FINAL_SCIENTIFIC_REPORT.md',report);dump('FINAL_TASK_STATUS.json',dict(status=status,own_GPU_processes_released=True,raw_model_downloads=0,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
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
(stage/'FINAL_ARCHIVE_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\\n')
seal=dict(status='PASS_FINAL_SEAL',manifest_sha256=v['manifest_sha256'],archive_verification_sha256=hashlib.sha256((stage/'FINAL_ARCHIVE_VERIFICATION.json').read_bytes()).hexdigest(),scientific_status=manifest['status'])
(stage/'FINAL_SEAL_RECEIPT.json').write_text(json.dumps(seal,indent=2)+'\\n')
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;rc=fn(-100,os.fsencode(stage),-100,os.fsencode(final),1);assert rc==0,os.strerror(ctypes.get_errno());fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(verification=v,seal=seal)))
''',stage,ARCHIVE)
 dump('FINAL_ARCHIVE_VERIFICATION.json',verification['verification']);dump('FINAL_SEAL_RECEIPT.json',verification['seal'])
except BaseException as exc:
 if not (R/'FINALIZER_FAILURE.json').exists():dump('FINALIZER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0))
 raise
