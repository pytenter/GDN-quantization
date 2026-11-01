"""Final presentation appendix from existing metadata, never model/tensor loading."""
import base64,fcntl,hashlib,importlib.util,json,math,os,statistics as S,sys,time
from pathlib import Path
TASK=Path(__file__).resolve().parents[1];ROOT=TASK/'releases/FORMAL_V2'
EXTRA=TASK/'final_review/FINAL_EXTRA_CPU_REVIEW_20261029'
OUT=TASK/'final_review/FINAL_ACCOUNTING_APPENDIX_20261029'
sys.path.insert(0,str(ROOT/'operations'))
from transport import run
import final_extra_templates as T
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def put(p,v):write(p,(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def req(ok,message):
 if not ok:raise RuntimeError(message)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(ROOT/'operations/DELIVERY_READER.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'CLAIM.json').exists(),'Already claimed; no automatic repeat'
 assert read(EXTRA/'RESULT.json')['extra_review_pending'] is False
 inputs=[ROOT/'performance/FAST_MATCHED_THROUGHPUT_V2.json',ROOT/'runtime_accounting.json',
         ROOT/'analysis/MODEL_TOKENIZER_FRESH_SHA.json',EXTRA/'CURRENT55_REVIEW.json',
         EXTRA/'FINAL_MEMORY_RUNTIME_DATA.json',EXTRA/'RESULT.json',
         ROOT/'validation/FINAL_NUMERICAL_AND_RUNTIME_ADMISSION.json',Path(__file__),
         TASK/'operations/final_extra_templates.py']
 for name in ['CPU_SCORE_FRESH1','CPU_SCORE_FRESH2','CPU_REPORT_FINISHER']:
  inputs.extend([ROOT/f'runtime/{name}/STARTED.json',ROOT/f'runtime/{name}/OWNING_EXIT.json'])
 roster=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in inputs]
 names=['PLAN.json','CLAIM.json','WEIGHT_HEADER_METADATA.json','COMPARISON_DATA.json',
        'ACCOUNTING_AND_CONCLUSION.md','final_accounting_appendix_cpu.py']
 plan=dict(scope='Separate CPU-only final accounting presentation; original audit already PASS',
           inputs=roster,outputs=names,payload_files=6,max_files_including_metadata=8,max_bytes=32*1024*1024,
           target55='/data/zypan/analyses/pace48_leap_unroll_dp8_dp16_v1/releases/FORMAL_V2_FINAL_ACCOUNTING_APPENDIX_20261029',
           target29='/data/zypan/qwen_gdn_pace48_archive/PACE48_LEAP_UNROLL_DP8_DP16_V1.FINAL_ACCOUNTING_APPENDIX_20261029',
           no_GPU_or_model_load_or_scoring=True)
 header_code=r'''
import json,math,os,struct,collections
from pathlib import Path
rows=[];totals=collections.Counter();dtypes=collections.Counter();prefixes=collections.Counter()
sizes={'BF16':2,'F16':2,'F32':4,'F64':8,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
for x in PINS:
 p=Path(x['path'])
 if p.suffix!='.safetensors':continue
 before=p.stat()
 with p.open('rb') as f:
  length=struct.unpack('<Q',f.read(8))[0];assert 0<length<16*1024*1024
  h=json.loads(f.read(length))
 after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 assert before.st_size==x['bytes']
 count=payload=0
 for name,t in h.items():
  if name=='__metadata__':continue
  n=math.prod(t['shape']);lo,hi=t['data_offsets'];assert hi-lo==n*sizes[t['dtype']]
  assert 0<=lo<=hi<=before.st_size-8-length
  count+=1;payload+=hi-lo;dtypes[t['dtype']]+=n
  prefixes['.'.join(name.split('.')[:2])]+=hi-lo
 rows.append(dict(path=str(p),file_bytes=before.st_size,header_bytes=length,tensors=count,tensor_payload_bytes=payload))
 totals.update(tensors=count,file_bytes=before.st_size,tensor_payload_bytes=payload)
print(json.dumps(dict(status='READ_ONLY_SAFETENSORS_HEADERS_NO_TENSOR_VALUES_OR_MODEL_LOAD',shards=rows,totals=dict(totals),dtype_elements=dict(dtypes),top_level_payload_bytes=dict(prefixes),
 scope='Exact on-disk checkpoint tensor metadata, not GPU live model parameter payload; original full file SHA verified by final independent audit')))
'''
 header_code='PINS='+repr(read(ROOT/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files'])+'\n'+header_code
 compile(header_code,'<read-only-weight-header-metadata>','exec')
 if '--validate-plan-only' in sys.argv:
  print(json.dumps(dict(status='PASS_FIXED_APPENDIX_PLAN_AND_REMOTE_SYNTAX',claim_created=False)));return
 put(OUT/'PLAN.json',plan);put(OUT/'CLAIM.json',dict(pid=os.getpid(),time=time.time(),plan_sha256=sha(OUT/'PLAN.json')))
 try:
  header=run(header_code,timeout=None);put(OUT/'WEIGHT_HEADER_METADATA.json',header)
  fixed=read(ROOT/'performance/FAST_MATCHED_THROUGHPUT_V2.json')['measurements']
  groups={}
  for name in ['replay8','direct8','direct16']:
   rr=[x for x in fixed if x['path']+str(x['decode_period'])==name];assert len(rr)==3
   all_stores=[x['quantizer_audit']['storage_by_layer'] for x in rr]
   counts=[]
   for x,stores in zip(rr,all_stores):
    counts.append(sum(v['writeback_count']-x['before_measurement_storage'][k]['writeback_count'] for k,v in stores.items()))
   assert len(set(counts))==1
   groups[name]=dict(fixed_TPS=S.median(x['decode_TPS'] for x in rr),
    base_bytes=[sum(b['base_nbytes'] for b in s.values()) for s in all_stores],
    capacity_bytes=[sum(b['decode_buffer_capacity_nbytes'] for b in s.values()) for s in all_stores],
    live_buffer_peak_per_layer_sum_bytes=[sum(b['peak_decode_buffer_nbytes'] for b in s.values()) for s in all_stores],
    CUDA_allocated_peak=max(x['peak_allocated_bytes'] for x in rr),CUDA_reserved_peak=max(x['peak_reserved_bytes'] for x in rr),
    measured_128_forward_writebacks_all24=counts[0])
  original=read(EXTRA/'FINAL_MEMORY_RUNTIME_DATA.json');timing={}
  for name in ['CPU_SCORE_FRESH1','CPU_SCORE_FRESH2','CPU_REPORT_FINISHER']:
   a=read(ROOT/f'runtime/{name}/STARTED.json');b=read(ROOT/f'runtime/{name}/OWNING_EXIT.json')
   assert a['child']==b['child'] and b['owning_Popen_exit']==0
   timing[name]=dict(actual_owner_interval_s=b['observed_monotonic']-a['start_monotonic'],scope='Same55 monotonic real dispatch-to-owning-exit interval; CPU wall, not GPU active')
  current=read(EXTRA/'CURRENT55_REVIEW.json');conf=current['model_config'];full=conf['layer_types'].count('full_attention')
  kv_per_token=full*2*conf['num_key_value_heads']*conf['head_dim']*2
  kv_estimates=[dict(unit=x['unit'],context_tokens=x['input_tokens']+x['generated_tokens']-1,
                    inferred_BF16_KV_payload_bytes=(x['input_tokens']+x['generated_tokens']-1)*kv_per_token) for x in original['rows']]
  data=dict(groups=groups,CPU_wall=timing,formal_arms=original['arms'],inferred_KV_bytes_per_context_token=kv_per_token,
            KV_estimates=kv_estimates,KV_scope='INFERRED full BF16 K/V append-only formula, not actual allocator/KV tensor observation',
            actual_GPU_parameter_bytes='UNMEASURED; all parameter dtype/device were checked BF16/CUDA0 by the frozen loader',
            checkpoint_on_disk_header=header['totals'],unmeasured=['pure IO seconds','exclusive audit seconds per GPU unit','GPU active/SM seconds','actual KV peak/lifetime','actual GPU parameter storage aliases'])
  put(OUT/'COMPARISON_DATA.json',data)
  lines=['# 最终内存、性能口径与工程结论补充','',
   '状态：COMPLETE_BOTH_CONFIGS。全部实验/主交付与完整额外55/29/Mac来源审核已PASS；本件仅补齐比较表、权重与KV/CPU时间作用域，无新模型加载、GPU运行或评分。','',
   '|指标|原Replay dP8|快Direct dP8|快Direct dP16|','|---|---:|---:|---:|']
  for label,key in [('同GPU1固定输入完整模型decode TPS','fixed_TPS'),('Packed base bytes','base_bytes'),
   ('Update buffer容量bytes','capacity_bytes'),('实测各层峰值payload之和bytes','live_buffer_peak_per_layer_sum_bytes'),
   ('CUDA allocated peak bytes','CUDA_allocated_peak'),('CUDA reserved peak bytes','CUDA_reserved_peak'),
   ('128计时forward全部24层State写回','measured_128_forward_writebacks_all24')]:
   vals=[]
   for k in ['replay8','direct8','direct16']:
    v=groups[k][key];vals.append(str(v[0]) if isinstance(v,list) and len(set(v))==1 else str(v))
   lines.append('|'+label+'|'+'|'.join(vals)+'|')
  km=read(ROOT/'runtime_accounting.json')['kernel_measurements']['rows']
  lines += ['|fused readout GPU图中位µs|UNMEASURED|'+str(km[0]['CUDA_event_graph_readout_median_us'])+'|'+str(km[1]['CUDA_event_graph_readout_median_us'])+'|',
   '|本次AIME实际pooled decode forwards/s|UNMEASURED|'+str(original['arms']['8']['pooled_decode_TPS'])+'|'+str(original['arms']['16']['pooled_decode_TPS'])+'|','',
   '固定输入TPS是完整模型decode阶段（排除prefill/加载/编译），不是GDN微核TPS；实测写回以after-before计数扣除warmup和prompt。上表峰值各层求和不是全层同时live承诺；allocator峰值不等于payload。原Replay没有本轮新AIME轨迹，不用旧21/30轨迹的不同上下文TPS冒充严格对照。','',
   '## 权重与端到端footprint','',
   f"METADATA_VERIFIED：只读4个safetensors头、未读取tensor数值，checkpoint文件共{header['totals']['file_bytes']}bytes，头内声明tensor payload共{header['totals']['tensor_payload_bytes']}bytes、{header['totals']['tensors']}个tensor。dtype元素计数：{header['dtype_elements']}。",
   '这描述磁盘checkpoint，包含的命名空间/权重别名可能与AutoModelForCausalLM实际参数集合不同，不能声称是GPU live weight bytes。CODE_VERIFIED/RUNTIME_VERIFIED：冻结loader在src/pace_runtime.py:296实际检查所有已加载parameter为BF16/CUDA0；没有保存完整GPU parameter存储/别名inventory，故该项实际payload仍UNMEASURED。上表整模型allocated/reserved是真实端到端allocator footprint，另给State+marker/head-map的有效bpv，不把两者混成量化bit数。','',
   f"INFERRED：模型config有{full}个full-attention层、{conf['num_key_value_heads']}个KV heads、head_dim={conf['head_dim']}；若完整BF16 K/V追加，每context token为{kv_per_token}bytes。各题context=input_tokens+generated−1的推算见COMPARISON_DATA。没有实际KV hook，真实KV峰值/容量/allocator生命周期UNMEASURED。",'',
   '## CPU、IO与校准时间','',
   '|原CPU阶段|真实owner起止区间s|','|---|---:|']
  for name,x in timing.items():lines.append(f"|{name}|{x['actual_owner_interval_s']:.9f}|")
  lines += ['','均用同55主机monotonic相减，来自原STARTED/OWNING_EXIT实际整数0；没有跨主机比较时钟。两fresh CPU评分只读取既有输出。每题模型加载/生成/prefill/decode/dispatch总值见原FINAL_MEMORY_RUNTIME_DATA；加载含部分IO/核验，纯IO和单元排他审计时间没有单独计时，不能从wall差额伪造。当前两配置未重新校准，继承冻结校准/mask；没有新校准耗时。GPU active/SM seconds未测，不拿dispatch-wall或CPU区间冒充。','',
   '## 对原任务八个问题的结论','',
   '1. 是实际packed-base LeapQuant-style Direct Read：Triton融合base/更新读出、每步临时批量原codec/BF16norm K；没有跨token持久完整FP32 State，只有周期末临时重建。',
   '2. 两个正式配置逐字段核对，只差decode_period=8/16。模型/量化器/mask/校准/seed/协议/评分规则不变。',
   '3. 目标CUDA跨轨迹最坏输出NRMSE0.09813%、同actual-state0.05608%，均通过预注册0.5%/0.1%工程界；300 FP64 oracle支持算子计算。CPU prototype原2.42% FAIL保留。full-model teacher-forcing logits最坏6.409%、GDN输出16.254%的差异也保留，因此仅是原工程准入可接受，不是bitwise/80K轨迹等价或普遍精度保证。',
   '4. 同卡同输入快Direct8/16比Replay8分别+30.93%/+33.71%；Direct16比Direct8 +2.12%。完整正式自由生成TPS另见报告，不能当同token因果速度比较。',
   '5. 本次AIME26 seed1 dP8=20/30、dP16=24/30；配对差为+4题，McNemar p0.289，不能称统计显著。dP8两条到长度上限，dP16全EOS；没有低分停止或答案选参。',
   f"6. dP16相对dP8多3397632bytes更新buffer容量；短固定输入整模型allocated峰值增加{groups['direct16']['CUDA_allocated_peak']-groups['direct8']['CUDA_allocated_peak']}bytes。正式峰值还受不同生成长度/KV影响，不把它归因于周期。更长pending向量会增加读取、准备和临时scratch成本；State写回次数减半不等于全模型写回总耗时已实测减半。",
   '7. 在本次seed1和实现下，dP16以约3.4MB额外buffer容量换来较好正确数与小幅同输入吞吐优势，准确率—速度—State内存权衡更有利；结论限本次数据，不推广到其他模型/seed。',
   '8. 可把本次冻结dP16作为下一版PACE的候选工程基线；它是独立数值轨迹，不自动替换原baseline或宣称官方LeapQuant实现等价。本任务已经封存并停止，不自动追加实验。','']
  write(OUT/'ACCOUNTING_AND_CONCLUSION.md','\n'.join(lines).encode())
  write(OUT/'final_accounting_appendix_cpu.py',Path(__file__).read_bytes())
  records=[dict(path=n,bytes=(OUT/n).stat().st_size,sha256=sha(OUT/n)) for n in names]
  put(OUT/'PACKAGE.json',dict(files=records,source=read(EXTRA/'RESULT.json')['source'],permit=read(EXTRA/'RESULT.json')['permit']))
  write(OUT/'MANIFEST.sha256',(''.join(x['sha256']+'  '+x['path']+'\n' for x in records)+sha(OUT/'PACKAGE.json')+'  PACKAGE.json\n').encode())
  records += [dict(path=n,bytes=(OUT/n).stat().st_size,sha256=sha(OUT/n)) for n in ['PACKAGE.json','MANIFEST.sha256']]
  payload=[dict(x,base64=base64.b64encode((OUT/x['path']).read_bytes()).decode()) for x in records]
  # Use the already validated streaming/safe-path primitives, no previous helper claim.
  spec=importlib.util.spec_from_file_location('readonly_common',(TASK/'operations/final_extra_review_once.py'))
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  receipts=[]
  for role in ['55','29']:
   code='DEST='+repr(plan['target'+role])+'\nPART="part01"\nFILES='+repr(payload)+'\n'+T.PUBLISH
   v=mod.remote(code,{},role);put(OUT/(role+'_PUBLICATION.json'),v);receipts.append(v)
   v2=mod.remote('RECEIPTS='+repr([v])+'\n'+T.POST_READ,{},role);put(OUT/(role+'_INDEPENDENT_POST_READ.json'),v2)
  assert receipts[0]['manifest_sha256']==receipts[1]['manifest_sha256']==sha(OUT/'MANIFEST.sha256')
  for x in roster:
   p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
  put(OUT/'RESULT.json',dict(status='PASS_FINAL_ACCOUNTING_APPENDIX_55_29_MAC_NOREPLACE',extra_review_pending=False,scientific_status='COMPLETE_BOTH_CONFIGS',receipts=receipts,main_extra_result_sha256=sha(EXTRA/'RESULT.json'),new_GPU_model_load_or_score_passes=0))
  print(json.dumps(dict(status='PASS_FINAL_ACCOUNTING_APPENDIX_55_29_MAC_NOREPLACE',files=len(records))),flush=True)
 except BaseException as e:
  put(OUT/'FIRST_FAILURE.json',dict(error=repr(e),no_automatic_repeat=True));raise
if __name__=='__main__':main()
