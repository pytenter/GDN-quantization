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
C.require(not (R/'FINALIZER_STARTED.json').exists(),'Finalizer already dispatched; inspect, no duplicate')
dump('FINALIZER_STARTED.json',dict(identity=C.process_identity(os.getpid()),monotonic=time.monotonic()))
try:
 terminal=C.read(R/'SESSION_RESULT.json');ledger=C.read(R/'RESOURCE_LEDGER.json');obs=C.query_gpu();children=[x['identity'] for x in ledger['intervals'] if x['identity'] is not None]
 C.require(all(not C.same_process(i) for i in children) and not any(x['pid'] in {i['pid'] for i in children} for x in obs['apps']),'Own worker process/NVML still active')
 C.verify_sources(R);C.model_guard(R);external=C.verify_external(R)
 dump('analysis/FINAL_SOURCE_DEPENDENCY_MODEL_RELEASE_AUDIT.json',dict(status='PASS',source_count=len(C.read(R/'SOURCE_FREEZE.json')['files']),external_fresh_SHA_files=external,model_stat_files=len(C.read(R/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']),all_own_exit_and_release=True,observation=obs))
 jobs=C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'];counts={c:sum((R/'formal'/j['id']/'RESULT.json').exists() for j in jobs if j['condition']==c) for c in ['I4-N','I4-H']}
 if terminal['status']=='GPU_COMPLETE_22_QUALIFICATION_60_FORMAL':
     a=cpu_pass('score_pass_cpu.py','SCORE_PASS_0.json');b=cpu_pass('score_pass_cpu.py','SCORE_PASS_1.json');C.require(a==b,'Independent exact FrozenV4 scores differ')
     dump('FROZEN_V4_SCORES.json',a);scores=a['scores'];by={}
     for row in scores:by.setdefault(row['problem_id'],{})[row['condition']]=row
     C.require(len(by)==30 and all(set(v)=={'I4-N','I4-H'} for v in by.values()),'Incomplete paired30 problem set')
     totals={c:sum(int(v[c]['correct']) for v in by.values()) for c in ['I4-N','I4-H']}
     pair=dict(both_correct=sum(v['I4-N']['correct'] and v['I4-H']['correct'] for v in by.values()),both_wrong=sum(not v['I4-N']['correct'] and not v['I4-H']['correct'] for v in by.values()),Native_only=sum(v['I4-N']['correct'] and not v['I4-H']['correct'] for v in by.values()),Hadamard_only=sum(v['I4-H']['correct'] and not v['I4-N']['correct'] for v in by.values()),net_correct_H_minus_N=totals['I4-H']-totals['I4-N'],delta_accuracy=(totals['I4-H']-totals['I4-N'])/30)
     dump('INT4_PAIRED_RESULTS.json',dict(N=30,totals=totals,pair=pair))
     with (R/'PER_PROBLEM_PAIRWISE_RESULTS.csv').open('x',newline='') as f:
         fields=['problem_id']+[c+'_'+s for c in ['I4-N','I4-H'] for s in ['correct','extracted_answer','generated_token_count','finish_reason','gpu_uuid','wall_seconds']];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
         for pid,v in sorted(by.items()):
             row={'problem_id':pid}
             for c,x in v.items():
                 for k in ['correct','extracted_answer','generated_token_count','finish_reason','gpu_uuid','wall_seconds']:row[c+'_'+k]=x[k]
             w.writerow(row)
     with (R/'FORMAL_GENERATIONS_60.jsonl').open('x') as f:
         for j in jobs:f.write(json.dumps(C.read(R/'formal'/j['id']/'RESULT.json'),ensure_ascii=False,sort_keys=True)+'\n')
     lengths={}
     for c in totals:
         rr=[x for x in scores if x['condition']==c];lengths[c]=dict(N=30,total_tokens=sum(x['generated_token_count'] for x in rr),mean_tokens=statistics.mean(x['generated_token_count'] for x in rr),median_tokens=statistics.median(x['generated_token_count'] for x in rr),EOS=sum(x['finish_reason']=='eos' for x in rr),length=sum(x['finish_reason']=='length' for x in rr),nonfinite_count=0,runtime_error_count=0,generation_wall_seconds=sum(x['wall_seconds'] for x in rr))
         lengths[c]['aggregate_generation_TPS']=lengths[c]['total_tokens']/lengths[c]['generation_wall_seconds']
     write('GENERATION_LENGTH_AND_RUNTIME_REPORT.md','# 实测长度、EOS与运行时间\n\n'+json.dumps(lengths,ensure_ascii=False,indent=2)+'\n\n含真实逐Prompt token与Decode QDQ/采样；加载、资格、退出释放成本另见GPU_RESOURCE_LEDGER.jsonl与RESOURCE_LEDGER.json。单卡串行；EXACT_GPUh=UNRESOLVED，不把保守区间当精确值。\n')
     # Freeze the original INT8 available-record set once, without changing its progress or seal.
     P=Path('/data/zypan/analyses/qwen_gdn_group32_four_codec_aime26_seed1_v1');oldjobs=C.read(P/'FORMAL_JOB_MANIFEST.json')['jobs'];hist=[]
     for j in oldjobs:
         p=P/'formal'/j['id']/'RESULT.json';interval=p.parent/'RESOURCE_INTERVAL.json'
         if j['condition'] in ('Q1','Q3') and p.exists() and interval.exists():
             z=C.read(interval);C.require(z['release']=='PASS' and z['exit_returncode']==0,'Historical own process not cleanly exited')
             hist.append(dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=C.sha(p)))
     dump('analysis/INT8_COMPARISON_INPUT_MANIFEST.json',dict(records=hist,root=str(P),created_monotonic=time.monotonic(),read_only=True))
     h0=cpu_pass('history_score_cpu.py','INT8_HISTORY_SCORE_PASS_0.json');h1=cpu_pass('history_score_cpu.py','INT8_HISTORY_SCORE_PASS_1.json');C.require(h0==h1,'Independent history CPU score differs')
     hb={}
     for row in h0['scores']:hb.setdefault(row['problem_id'],{})[row['condition']]=row
     common=sorted(q for q in by if set(hb.get(q,{}))=={'Q1','Q3'});N=len(common)
     full=N==30;label='FULL30_MATCHED_INPUT_PROTOCOL' if full else 'PARTIAL_EXPLORATORY'
     matched={c:sum(int(by[q][c]['correct']) for q in common) for c in ['I4-N','I4-H']}
     matched.update({c:sum(int(hb[q][c]['correct']) for q in common) for c in ['Q1','Q3']})
     deltas=None
     if N:
         deltas=dict(native_4_minus_8=(matched['I4-N']-matched['Q1'])/N,H_4_minus_8=(matched['I4-H']-matched['Q3'])/N,H_minus_N_INT4=(matched['I4-H']-matched['I4-N'])/N,interaction=((matched['I4-H']-matched['I4-N'])-(matched['Q3']-matched['Q1']))/N)
     cross_uuid=[q for q in common if hb[q]['Q1']['gpu_uuid']!=by[q]['I4-N']['gpu_uuid']]
     comparison=dict(status=label,N=N,problem_ids=common,correct_counts=matched,deltas=deltas,protocols_equal=h0['protocols_equal'],cross_UUID_questions=cross_uuid,cross_device_bitwise_claim=False,historical_test_observed=True)
     matched_lengths={}
     for c in ['I4-N','I4-H','Q1','Q3']:
         rr=[by[q][c] if c.startswith('I4') else hb[q][c] for q in common]
         matched_lengths[c]=dict(N=N,mean_tokens=statistics.mean(x['generated_token_count'] for x in rr) if N else None,median_tokens=statistics.median(x['generated_token_count'] for x in rr) if N else None,EOS=sum(x['finish_reason']=='eos' for x in rr),length=sum(x['finish_reason']=='length' for x in rr),nonfinite_count=0,runtime_error_count=0)
     comparison['matched_generation_lengths']=matched_lengths
     comparison['paired_bitwidth_changes']={}
     for newer,older in [('I4-N','Q1'),('I4-H','Q3')]:
         comparison['paired_bitwidth_changes'][newer+'-'+older]=dict(rescue=sum(by[q][newer]['correct'] and not hb[q][older]['correct'] for q in common),lost=sum(hb[q][older]['correct'] and not by[q][newer]['correct'] for q in common),net=matched[newer]-matched[older])
     if N:
         with (R/'INT4_INT8_MATCHED_PER_PROBLEM.csv').open('x',newline='') as f:
             cols=['problem_id']+[c+'_'+k for c in ['I4-N','I4-H','Q1','Q3'] for k in ['correct','generated_token_count','finish_reason','gpu_uuid']];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
             for q in common:
                 rr={'problem_id':q}
                 for c in ['I4-N','I4-H','Q1','Q3']:
                     x=by[q][c] if c.startswith('I4') else hb[q][c]
                     for k in ['correct','generated_token_count','finish_reason','gpu_uuid']:rr[c+'_'+k]=x[k]
                 w.writerow(rr)
     dump('INT4_VS_INT8_COMPARISON.json',comparison)
     write('INT4_VS_INT8_COMPARISON.md','# INT4与历史INT8 Asym32\n\n'+json.dumps(comparison,ensure_ascii=False,indent=2)+'\n\nQ1=INT8 Native Asym32，Q3=INT8 H128 Asym32。使用共同已完成题集与原FrozenV4两独立CPU复评分；未完成历史题不当错误、不以23题/30题异分母相减。即使输入/采样/prefillQDQ协议一致，历史INT8分布于多UUID，本INT4固定GPU3，没有额外验证跨GPU严格bitwise等价。差中差只是观察性准确率差，不能证明机制。\n')
     status='FORMAL_COMPLETE_60_60_AND_STOP'
     table='\n'.join('| '+c+' | '+str(totals[c])+'/30 | '+f'{totals[c]/30:.2%}'+' |' for c in totals)
     report=f'# Qwen/GDN Asymmetric INT4 Group32 AIME26 Seed1\n\n状态：{status}。60/60条合法生成；原FrozenV4与两个独立CPU重评分完全一致；自身进程已退出且NVML已释放。\n\n| Codec | Correct/30 | Accuracy |\n|---|---:|---:|\n{table}\n\n配对结果：\n\n'+json.dumps(pair,ensure_ascii=False,indent=2)+'\n\n'+json.dumps(lengths,ensure_ascii=False,indent=2)+'\n\n'+json.dumps(comparison,ensure_ascii=False,indent=2)+'\n\nI4-N与I4-H只在Native/H128坐标路径存在预期差别；同一个独立Canonical AsymINT4Key32。H矩阵/符号/方向与原Q/K和递归数学保持冻结历史实现。FP32scale、integerzero-point、RNE(x/s)+z、范围0..15、包括0极值、Key轴4x32分组、int16扩展后减z、即时FP32重建写回。rangefloor1e-12先于除15，有效scale下限EPS/15，详见数值合同，与INT8EPS/255的位宽改变显式报告。\n\n逐Prompt token从token0执行QDQ，q_length1，24层每次写回；p+n-1实际forward与24(p+n-1)写回，最后输出token不再消费。32/128/512两fresh逐token完整Tensor值/bytes与下一kernel真实读取检查；正式/4096为冻结稀疏位置全值/字节核验。82293为活模型合成KV及nativeSDPA资源压力，不是81920token数值误差上界。\n\n本轮为逻辑4-bit整数QDQ，codes实际UINT8承载0..15，persistent State实际nativeFP32重建48MiB；理论packedcode6MiB另有scale/zmetadata，未实现物理INT4缓存，不声称实际4倍/8倍显存压缩。显存allocated/reserved峰值见每条原记录和压力回执。每条独立fresh进程/模型请求、Generator、Cache和generated-onlymask；单UUID串行，奇偶题交替顺序冻结。模型BF16权重不改，Sampler/topk20含原生ties行为不改。\n\n这是看到INT8阶段性结果后设计的新探索实验；AIME26历史已多次观察，30题单seed限制必须保留。不能据少数题差异推断泛化或确定机制。跨UUID输入/协议匹配不等于验证跨硬件bitwise等价；旧HISTORICAL_PORT完整历史二进制EXACT身份仍UNKNOWN。INT4质量差本身不触发数学失败；没有按答案挑选重跑、缩短cap或额外科学分支。资源时间只有同主机monotonic区间估计，EXACT_GPUh=UNRESOLVED。\n'
 else:
     failure=C.read(R/'FIRST_FAILURE.json');name=failure.get('unit','')
     resource_error=any(k.lower() in (name+' '+failure.get('error','')).lower() for k in ['CONTEXT','outofmemory','out of memory','VRAM','disk headroom','insufficient exclusive'])
     status='LONG_CONTEXT_RESOURCE_QUALIFICATION_FAIL_CLOSED_STOP' if resource_error else 'INT4_NUMERICAL_QUALIFICATION_FAIL_CLOSED_STOP' if not any(counts.values()) and name!='CONTROLLER' else 'PARTIAL_RESULTS_PRESERVED_AND_STOP'
     report='# INT4真实STOP交付\n\n状态：'+status+'。\n\n实际完成数量：'+json.dumps(counts)+'。未完成60条，未发布完整accuracy或空科学CSV；没有GPU重试、前缀重跑或轨迹拼接。原始错误/每token日志/首差异/临时参考保留，资源释放按真实回执核验。\n\n'+json.dumps(failure,ensure_ascii=False,indent=2)+'\n'
 write('FINAL_SCIENTIFIC_REPORT.md',report)
 dump('FINAL_TASK_STATUS.json',dict(status=status,completed=counts,own_GPU_processes_released=True,source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json')))
 # Archive/publish code appended below; no scientific output is fabricated before its real stage.
 # Freeze all actual files except live finisher output; no raw model is under R.
 rows=[]
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.name in ['finisher.log','SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','FINAL_ARCHIVE_VERIFICATION.json','FINALIZER_FAILURE.json']:continue
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
print(json.dumps(dict(verification=v,seal=seal)))
''',stage,ARCHIVE)
 dump('FINAL_ARCHIVE_VERIFICATION.json',verification['verification']);dump('FINAL_SEAL_RECEIPT.json',verification['seal'])
except BaseException as exc:
 if not (R/'FINALIZER_FAILURE.json').exists():dump('FINALIZER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0))
 raise
