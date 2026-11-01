"""CPU-only nine-question report using actual, qualified five-doc evidence.
No GPU dispatch; original diagnostic/arithmetic audit functions are reused.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time

TASK='QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
CONDS=('FP_NATIVE','H_C128','H_DUAL_M')
def need(ok,s):
 if not ok:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def optional(r,p):return read(r/p) if (r/p).exists() else None
def fmt(v):return '未获得' if v is None else format(v,'.12g') if isinstance(v,(float,int)) else str(v)

def run(root):
 r=Path(root);gate=read(r/'GATE_STATUS_FINAL.json');ledger=read(r/'gpu_resource_ledger.json')
 need(gate['task_id']==ledger['task_id']==TASK,'Task identity mismatch')
 need(gate['no_further_GPU_work'] and gate['GPU_release_verified'] and ledger['final_GPU_release_verified'],'Actual final GPU release required')
 spec=importlib.util.spec_from_file_location('qualified_diagnostics',r/'analysis_scripts/qualified_diagnostics.py')
 q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
 manifest=read(r/'source_manifest.json');identity_errors=[]
 for e in manifest['execution_sources']:
  p=r/e['path']
  if not p.is_file() or q.sha(p)!=e['sha256']:identity_errors.append(e['path'])
 runs={};availability={};receipt_checks=[]
 for tag in ('N_OFF','A_REFERENCE','A_OFF','A_ON','D_OFF','D_ON','N_FRESH','A_FRESH','D_FRESH'):
  if (r/('analysis/'+tag+'.json')).exists():runs[tag]=read(r/('analysis/'+tag+'.json'))
 if not runs and (r/'qualification_gpu/gpu0/analysis').is_dir():
  for tag in ('N_OFF','A_REFERENCE','A_OFF','A_ON','D_OFF','D_ON','N_FRESH','A_FRESH','D_FRESH'):
   p=r/('qualification_gpu/gpu0/analysis/'+tag+'.json')
   if p.exists():runs[tag]=read(p)
 for gpu in range(1,4):
  for tag in ('N_OFF','A_REFERENCE','A_OFF','A_ON','D_OFF','D_ON','N_FRESH','A_FRESH','D_FRESH'):
   p=r/f'qualification_gpu/gpu{gpu}/analysis/{tag}.json'
   if p.exists():runs[f'gpu{gpu}_{tag}']=read(p)
 for doc in range(1,6):
  for c in CONDS:
   p=f'science_doc{doc}_{c}';availability[p]=gate['gates'].get(p)=='PASS'
   if availability[p]:
    run=read(r/('analysis/'+p+'.json'));rec=read(r/('analysis/RECEIPT_'+p+'.json'));phase=[a for a in ledger['phases'] if a['phase']==p]
    need(len(phase)==1 and phase[0]['released'] and phase[0]['returncode']==0,'Reservation completion mismatch')
    need(q.sha(r/('analysis/RECEIPT_'+p+'.json'))==phase[0]['receipt_sha256'],'Worker receipt identity mismatch')
    need(q.complete_run(run,32768) and rec['result']['scores']==384,'Incomplete32768 trajectory')
    runs[p]=run;receipt_checks.append(p)
 post=optional(r,'analysis/POSTPROCESS_STATUS.json');result=optional(r,'science_results.json') if not identity_errors else None;docs=result['documents'] if result else []
 for d in docs:
  a=read(r/f'analysis/score_rows_doc{d["document_index"]}_A.json');b=read(r/f'analysis/score_rows_doc{d["document_index"]}_D.json')
  q.audit_pair_rows(a,b,d)
 final_identity=optional(r,'qualification_cpu/PARENT_IDENTITY_POSTRUN.json')
 n=len(docs);complete=not identity_errors and n==5 and gate['status']=='COMPLETE_AND_STOP' and bool(final_identity and final_identity['status']=='PASS') and bool(post and post['status']=='COMPLETE')
 scientific=result['DUAL_M_FULLMODEL_RESULT'] if result else 'INCOMPLETE_NOT_CONFIRMATORY'
 if not complete:scientific='INCOMPLETE_NOT_CONFIRMATORY'
 failure=optional(r,'FIRST_FAILURE.json');execution='COMPLETE_AND_STOP' if complete else 'FAIL_AND_STOP'
 completed_scale=q.safety_summary(runs);existing_scale=optional(r,'scale_safety.json')
 lane_scale_failures=[dict(relative_path=str(p.relative_to(r)),actual_receipt=read(p)) for p in r.glob('qualification_gpu/gpu*/scale_safety.json')]
 scale=dict(status=existing_scale.get('status'),actual_first_failure_scale_receipt=existing_scale,completed_trajectory_summary=completed_scale) if existing_scale else completed_scale
 if lane_scale_failures:scale=dict(status='FAIL',lane_first_failure_scale_receipts=lane_scale_failures,other_actual_scale_summary=scale)
 timings=[q.timing_summary(k,v) for k,v in runs.items()]
 ratios=[]
 for i in range(1,6):
  a=next((x for x in timings if x['tag']==f'science_doc{i}_H_C128'),None);b=next((x for x in timings if x['tag']==f'science_doc{i}_H_DUAL_M'),None)
  if a and b:ratios.append(dict(document_index=i,D_over_A_postwarm_median=b['postwarm_full_model_plus_QDQ_seconds']['median']/a['postwarm_full_model_plus_QDQ_seconds']['median'],D_over_A_wall=b['wall_seconds']/a['wall_seconds']))
 runtime=dict(status='ACTUAL_COMPLETED_TRAJECTORIES_ONLY',trajectories=timings,matched_runtime_ratios=ratios,GPU_HOURS_USED=ledger['GPU_HOURS_USED'],
  cost_warning='C128 events sample384 positions; DualM timers cover every token. Observation overhead differs; no deployment latency acceptance threshold or universal acceleration claim.',
  local_QDQ_MSE_scope='Each student own history state; not matched common FP state, not causal mediator evidence')
 write(r/'runtime_cost.json',runtime)
 if scale and not existing_scale:write(r/'scale_safety.json',scale)
 parity=dict(original_C128_reference=q.exact_pair(runs.get('A_REFERENCE'),runs.get('A_ON'),True),
  A_observer=q.exact_pair(runs.get('A_OFF'),runs.get('A_ON')),D_observer=q.exact_pair(runs.get('D_OFF'),runs.get('D_ON')),
  official_DualM=q.official_dual_parity(runs.get('D_ON')),fresh={c:q.exact_pair(runs.get(a),runs.get(b),c!='FP_NATIVE') for c,a,b in [('FP_NATIVE','N_OFF','N_FRESH'),('H_C128','A_ON','A_FRESH'),('H_DUAL_M','D_ON','D_FRESH')]},
  QDQ={k:q.qdq_counts(v) for k,v in runs.items() if v.get('condition')!='FP_NATIVE'},fresh_scope='128 tokens only; no32K fullmodel forward uncertainty bound')
 parity['cross_gpu_canary_exact']=gate['gates'].get('cross_gpu_canary_exact','NOT_RUN')
 parity['per_gpu_fresh']={str(gpu):{c:q.exact_pair(runs.get(f'gpu{gpu}_'+a),runs.get(f'gpu{gpu}_'+b),c!='FP_NATIVE') for c,a,b in [('FP_NATIVE','N_OFF','N_FRESH'),('H_C128','A_ON','A_FRESH'),('H_DUAL_M','D_ON','D_FRESH')]} for gpu in range(1,4)}
 parity['per_gpu_qualification']={}
 for gpu in range(4):
  prefix='' if gpu==0 else f'gpu{gpu}_'
  parity['per_gpu_qualification'][str(gpu)]=dict(original_C128=q.exact_pair(runs.get(prefix+'A_REFERENCE'),runs.get(prefix+'A_ON'),True),
   A_observer=q.exact_pair(runs.get(prefix+'A_OFF'),runs.get(prefix+'A_ON')),D_observer=q.exact_pair(runs.get(prefix+'D_OFF'),runs.get(prefix+'D_ON')),
   official_DualM=q.official_dual_parity(runs.get(prefix+'D_ON')),
   **{f'fresh_{c}':q.exact_pair(runs.get(prefix+a),runs.get(prefix+b),c!='FP_NATIVE') for c,a,b in [('FP_NATIVE','N_OFF','N_FRESH'),('H_C128','A_ON','A_FRESH'),('H_DUAL_M','D_ON','D_FRESH')]})
 write(r/'codec_parity.json',parity)
 docs_manifest=read(r/'frozen_documents.json');prereg=read(r/'preregistration.json');means=(result or {}).get('document_means')
 winners={w:sum(d['Winner']==w for d in docs) for w in ('H_DUAL_M','H_C128','NUMERICALLY_UNRESOLVED')}
 qualified=not identity_errors and all(all(x=='PASS' for x in v.values()) for v in parity['per_gpu_qualification'].values()) and gate['gates'].get('cross_gpu_canary_exact')=='PASS'
 if complete:need(qualified and post['status']=='COMPLETE','Completed science lacks exact current qualification')
 status=dict(TASK=TASK,GPU_AUTHORIZATION='YES',FINAL_EXECUTION_STATUS=execution,COORDINATOR_STATUS=gate['status'],
  PROSPECTIVE_SCIENTIFIC_RESULT=scientific,VALID_PAIRED_DOCS=f'{n}/5',DOCS_FROZEN=5,
  **{c+'_32K':f'{sum(availability[f"science_doc{i}_{c}"] for i in range(1,6))}/5' for c in CONDS},
  D_WIN=winners['H_DUAL_M'],A_WIN=winners['H_C128'],NUMERICALLY_UNRESOLVED=winners['NUMERICALLY_UNRESOLVED'],
  GPU_HOURS_USED=ledger['GPU_HOURS_USED'],MAX_NEW_GPU_HOURS=24.0,GPU_RELEASE_VERIFIED=True,GPU_COUNT=4,GPU_UUIDS=ledger.get('GPU_UUIDS'),doc_GPU_assignment=prereg.get('doc_GPU_assignment'),
  PERSISTENT_INT8_QDQ='QUALIFIED' if qualified else 'NOT_QUALIFIED_OR_INCOMPLETE',PHYSICAL_PACKED_INT8='NOT_QUALIFIED',FROZEN_IDENTITY_ERRORS=identity_errors,
  PROJECT_INDEPENDENCE='UNKNOWN',OLD_FP_NATIVE_H_FAILURE='PRESERVED',OLD_BROKENPIPE_FAILURE='PRESERVED',
  AIME='NOT_RUN',DUAL_W='NOT_RUN',C6R_INTERACTION='NOT_RUN',C8_TRAINING='NOT_AUTHORIZED',ALL_TASK_WORK_STOPS_AFTER_SEAL=True)
 write(r/'FINAL_STATUS.json',status)
 lines=[f'# {TASK}',f'最终执行状态：**{execution}**；科学分类：**{scientific}**。',
  f'有效完整配对 {n}/5，新增占用 {fmt(ledger["GPU_HOURS_USED"])} GPUh /24；GPU释放已核验。',
  '## VERIFIED_ENGINEERING',
  '原core/codec/worker/reducer逐字节复用，控制层仅管理新root、N5、资源与离线托管。实际身份、资格及释放回执见源码manifest、门禁和ledger。旧FP_NATIVE/FP_H token1 relativeL2=0.006718461729707917>0.002仍失败；原BrokenPipe失败与后续完成历史均保留。',
  '当前canary parity：'+json.dumps(parity,ensure_ascii=False),
  '## PROSPECTIVE_SCIENTIFIC_RESULT',
  '|Doc|K_A|K_D|Delta|RelativeGain|算术分辨率|Winner|','|---|---:|---:|---:|---:|---:|---|']
 for d in docs:lines.append('|'+ '|'.join(fmt(x) for x in [d['document_index'],d['K_A'],d['K_D'],d['Delta'],d['RelativeGain'],d['Delta_resolution'],d['Winner']])+'|')
 if not docs:lines.append('未获得完整、合格配对；不创建空壳科学CSV，不把缺失计作平局。')
 lines+=['等权汇总：'+json.dumps(means,ensure_ascii=False),
  'Primary固定384点：t=8192+64k，消费ids[t]后预测ids[t+1]；全248320词表FP32 KL，math.fsum/384。两student各自从token0逐token量化并写回。分辨率沿用原独立标量FP64核对及误差合成，仅约束固定已存logits的条件算术误差，不完整约束GPU fullmodel forward不确定性。',
  '## 用户九项问题',
  f'Q1：新5篇中 {winners["H_DUAL_M"]} 篇可分辨D_WIN，{winners["H_C128"]} 篇A_WIN，{winners["NUMERICALLY_UNRESOLVED"]} 篇未分辨；缺失 {5-n} 篇。',
  'Q2：mean Delta、mean per-doc RelativeGain、mean K_A/K_D与两臂均值KL相对变化见上述等权汇总。样本不完整时不冒充N5主分析。',
  'Q3：历史Pilot独立列于下表；不得合并成N7推断。'+('新N5 meanDelta='+fmt(means['mean_absolute_Delta'])+'，历史为0.014760139682959934；新等权每docRelativeGain='+fmt(means['mean_per_doc_RelativeGain'])+'，历史为0.32944781565765247。新平均方向'+('与历史一致' if means['mean_absolute_Delta']>0 else '反向或无净改善')+'；效应量差异为本次选择样本上的描述，不作显著差异判断。' if means else '新N5主分析不完整，不能用部分样本冒充完整方向/量级复制。'),
  '|历史Pilot|K_A|K_D|Delta|RelativeGain|','|---|---:|---:|---:|---:|',
  '|Doc1|0.031818872970088265|0.02431929538210113|0.007499577587987135|0.23569589014159012|',
  '|Doc2|0.05203382619483617|0.03001312441690344|0.022020701777932732|0.42319974117371484|',
  'Q4：以下是全部已完成文档的预冻结窗口。最大收益按原始Delta判断；不替换整篇Primary。',
  '|Doc|窗口|K_A|K_D|Delta|RelativeGain|Winner|','|---|---|---:|---:|---:|---:|---|']
 for d in docs:
  for name,w in d['windows'].items():lines.append('|'+ '|'.join(fmt(x) for x in [d['document_index'],name,w['K_A'],w['K_D'],w['Delta'],w['RelativeGain'],w['Winner']])+'|')
 late=[dict(doc=d['document_index'],maximum_raw_Delta_window=max(d['windows'],key=lambda x:d['windows'][x]['Delta'])) for d in docs]
 window_support={w:{k:sum(d['windows'][w]['Winner']==k for d in docs) for k in ('H_DUAL_M','H_C128','NUMERICALLY_UNRESOLVED')} for w in ('Early','Middle','Late')}
 lines+=['最大收益窗口：'+json.dumps(late,ensure_ascii=False)+f'；Late为最大原始Delta的篇数={sum(x["maximum_raw_Delta_window"]=="Late" for x in late)}/{n}。各窗口方向计数：'+json.dumps(window_support,ensure_ascii=False),
  '## POSTHOC_DIAGNOSTIC',
  'Q5：实际逐token、per-layer/head安全回执：'+json.dumps(scale,ensure_ascii=False)+ '。非零floor、越界、rounded clamp与饱和码必须分别解释，不将它们混同。严重门禁为同一layer/head能量比例>=1e-4且相对MSE>1e-2。',
  'Q6：实际D/A耗时比：'+json.dumps(ratios,ensure_ascii=False)+'；完整逐token、吞吐、峰值allocated/reserved和codec计时见RUNTIME_COST.json。不同观察覆盖不能转为普遍部署加速宣称。',
  '|轨迹|tokens|wall秒|token/s|peak allocated|peak reserved|','|---|---:|---:|---:|---:|---:|']
 for t in timings:lines.append('|'+ '|'.join(fmt(t[k]) for k in ['tag','consumed_tokens','wall_seconds','wall_tokens_per_second','peak_allocated_bytes','peak_reserved_bytes'])+'|')
 lines+=['Q7：PROJECT_INDEPENDENCE=UNKNOWN。统计单位为文档，目标N=5；不同raw hash不证明项目独立。不把384×5位置、layer/head当独立样本，不作正式项目级显著性宣称。若5/5，名义单侧sign概率1/32仅在五个方向独立且交换性等假设成立时适用；本轮未知项目独立性，不能包装成confirmatory显著性。',
  'Q8：'+('若本轮有完整正向复制，则可为另行预注册和授权的C6R×Dual-M交互实验提供探索动机；当前codec比较仍不构成互补性证据。' if complete and winners['H_DUAL_M']>=4 else '当前完整性/方向不足以据此断言交互收益；后续交互若研究仍须独立预注册授权。')+'本任务不执行该实验。',
  '## NOT_ESTABLISHED',
  'Q9：尚未建立learned rotation互补性、row-scale/query-readout/scale-feedback唯一中介、共同FP状态上的codec因果净误差、FP_NATIVE/FP_H等价、物理packed INT8、部署速度收益、AIME/free-generation提升或跨分布泛化。local-QDQ MSE来自两臂各自历史状态，不能替代共同输入因果对照。',
  '## FAIL_AND_STOP / COMPLETE_AND_STOP',
  '首失败：'+json.dumps(failure,ensure_ascii=False),
  '```json',json.dumps(status,ensure_ascii=False,indent=2),'```',
  '报告绝对路径：'+str(r/'FINAL_SCIENTIFIC_REPORT.md'),
  '最终逐文件SHA及永久归档独立验证回执见hashes/。完成seal后STOP；不启动下一科研实验、自动git push或新GPU工作。']
 with (r/'FINAL_SCIENTIFIC_REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 for a,b in [('gpu_resource_ledger.json','GPU_RESOURCE_LEDGER.json'),('science_results.json','SCIENCE_RESULTS.json'),('science_results.csv','SCIENCE_RESULTS.csv'),('window_results.csv','WINDOW_RESULTS.csv'),('runtime_cost.json','RUNTIME_COST.json'),('analysis/SCORE_ARTIFACT_VERIFICATION.json','SCORE_ARTIFACT_VERIFICATION.json')]:
  if (r/a).exists():
   if (r/b).exists():need((r/a).read_bytes()==(r/b).read_bytes(),'Conflicting output alias: '+b)
   else:
    with (r/b).open('xb') as out:out.write((r/a).read_bytes())
 return status

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(run(a.root),ensure_ascii=False))
