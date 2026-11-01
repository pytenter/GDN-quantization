#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original Frozen V4, only after all60 valid payloads and true GPU release."""
import argparse,csv,io,json,statistics,time,subprocess
from pathlib import Path
import runtime as R

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;R.verify(r)
 R.check(not (r/'AIME26_FROZEN_V4_SCORES.json').exists(),'Do not rerun completed scientific score')
 ledger=R.read(r/'RESOURCE_LEDGER.json');R.check(R.read(r/'execution/FORMAL_GPU_COMPLETE.json')['all_own_GPUs_released'] and all(d.get('process_exited') and d.get('NVML_released') for d in ledger['dispatches']),'GPU release incomplete')
 live=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
 R.check(not {d['pid'] for d in ledger['dispatches']}.intersection(int(x.split(',')[0]) for x in live.splitlines() if x.strip()),'Own GPU PID still active')
 jobs=R.read(r/'DATASET_AND_JOB_MANIFEST.json')['jobs'];R.check(len(jobs)==60 and len({j['problem_id'] for j in jobs})==30 and len({j['canonical_job_id'] for j in jobs})==60,'Dataset roster invalid')
 paths=list((r/'raw_results').glob('*.json'));R.check(len(paths)==60,'No full accuracy until all60 exact unique legalcompletions')
 rows=[R.validate_record(R.read(r/'raw_results'/(j['canonical_job_id']+'.json')),j,r) for j in jobs]
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
 scorer=R.load_module('aime60_exact_frozen_v4',r/'protocol/aime26_scorer_v4.py');R.check(R.sha(r/'protocol/aime26_scorer_v4.py')=='fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b','Scorer SHA changed')
 for row,job in zip(rows,jobs):
  R.check(tok.decode(row['generated_token_ids'],skip_special_tokens=True)==row['response'],'Raw text/tokens mismatch')
  x=scorer.extract_v4(row['response'],row['finish_reason'],job['problem']);correct=bool(scorer.compare_with_gold(x,job['answer']));record=dict(row);record.update(scoring_status='COMPLETE_FROZEN_V4',extracted_answer=x.normalized_value,correct=correct,frozen_v4_extraction=x.to_dict(),raw_generation_payload_sha256=row['output_payload_sha256']);record.pop('output_payload_sha256');record['scored_payload_sha256']=R.canonical_sha(record);R.atomic_new(r/'results'/(job['canonical_job_id']+'.json'),record)
  row.update(extracted_answer=x.normalized_value,correct=correct,extraction=x.to_dict())
 seed={str(s):dict(correct=sum(z['correct'] for z in rows if z['seed']==s),total=30) for s in (1,2)};correct=sum(x['correct'] for x in rows)
 summary=dict(task_id=R.TASK,status='FORMAL_COMPLETE_60_60_AND_STOP',method=R.METHOD,total=60,correct=correct,accuracy=correct/60,seeds=seed,scorer_sha256=R.sha(r/'protocol/aime26_scorer_v4.py'),EOS_count=sum(x['eos_seen'] for x in rows),length_limit_count=sum(x['truncated'] for x in rows),runtime_errors=0,nonfinite_errors=0,median_generated_length=statistics.median(x['generated_token_count'] for x in rows),total_generated_tokens=sum(x['generated_token_count'] for x in rows),source_manifest_sha256=R.sha(r/'SOURCE_SHA256_MANIFEST.json'),historical_comparison='Reference only: historical41/60 etc differ in prefill QDQ timing/deterministic runtime; no strictmatched claim')
 R.atomic_new(r/'AIME26_FROZEN_V4_SCORES.json',summary)
 with (r/'FORMAL_RESULTS_60.jsonl').open('x') as f:
  for job in jobs:f.write(json.dumps(R.read(r/'results'/(job['canonical_job_id']+'.json')),ensure_ascii=False,sort_keys=True)+'\n')
 fields=['canonical_job_id','problem_id','problem_idx','seed','extracted_answer','gold_answer','correct','generated_token_count','finish_reason','gpu_uuid','latency_seconds','tokens_per_second','peak_allocated_bytes','peak_reserved_bytes']
 with (r/'AIME26_PER_PROBLEM_RESULTS.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:z.get(k) for k in fields} for z in rows)
 formal=[x for x in ledger['dispatches'] if x['mode']=='formal'];start=min(x['dispatch_unix'] for x in formal);end=max(x['exit_and_release_unix'] for x in formal);wall=end-start
 per={str(g):dict(gpu_uuid=jobs[g]['gpu_uuid'],jobs=sum(x['gpu']==g for x in rows),median_tokens_per_second=statistics.median(x['tokens_per_s'] for x in rows if x['gpu']==g),weighted_single_request_TPS=sum(x['generated_token_count'] for x in rows if x['gpu']==g)/sum(x['latency_s'] for x in rows if x['gpu']==g)) for g in range(4)}
 cost=dict(task_id=R.TASK,total_GPU_hours=ledger['cumulative_GPU_hours'],formal_phase_GPU_hours=sum(x['occupied_seconds'] for x in formal)/3600,formal_wall_seconds=wall,overall_GPU_stages_wall_seconds=max(x['exit_and_release_unix'] for x in ledger['dispatches'])-min(x['dispatch_unix'] for x in ledger['dispatches']),per_GPU=per,aggregate_generated_tokens_per_formal_phase_wall_second=summary['total_generated_tokens']/wall,aggregate_scope='4GPU sumtokens / actual formalphasewall including model load/idle/cleanup; not singlecardTPS',peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in rows),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in rows))
 R.atomic_new(r/'THROUGHPUT_AND_GPU_COST.json',cost)
 report=f'''Hadamard + Dual-M INT8\nQwen3.5-9B / AIME26\n\nPrimary:\nCorrect / 60 = {correct}/60 ({correct/60:.4%})\n\nSeed 1:\nCorrect / 30 = {seed['1']['correct']}/30\n\nSeed 2:\nCorrect / 30 = {seed['2']['correct']}/30\n\nTotal GPUh = {cost['total_GPU_hours']:.9f}\nTotal wall time = {wall/3600:.4f}小时（正式GPU阶段；含加载至释放）\nMedian tokens/s per GPU = {json.dumps({g:x['median_tokens_per_second'] for g,x in per.items()})}\n\n终态：FORMAL_COMPLETE_60_60_AND_STOP。全部60个problem×seed唯一，原Canonical30与seeds1/2，原采样warper顺序及Top-K tie/EOS规则，原FrozenV4统一评分，没有中途按正确率改计划。EOS {summary['EOS_count']}，length {summary['length_limit_count']}，runtime/nonfinite 0/0。中位输出{summary['median_generated_length']} tokens，总输出{summary['total_generated_tokens']} tokens。四卡合计阶段吞吐{cost['aggregate_generated_tokens_per_formal_phase_wall_second']:.6f} token/s，不能当单卡吞吐。详细峰值/分GPU成本见THROUGHPUT_AND_GPU_COST。\n\n数值路径为原32K Dual-M Reference、H128合格Q/K位置、all8 signedINT8/ones/noFP16pivot/FP16row+column/一次columnLS/最终reencode，保留floor/ceiling/guard/舍入与signed-zero审计。GDN/H/QDQ deterministicTrue、原采样函数执行期间临时native许可并在finally恢复True、TF32False、原compensated chunk，BF16权重。只对完整prefill末尾做一次QDQ，再每次decode后QDQ；不是prefill内部每prompt token都QDQ，且不同于旧AIME首次prefill未QDQ。Attention KV未量化。FP32 QDQ carrier不是物理packedINT8常驻部署；瞬时编码tensor不等于常驻压缩。\n\n每卡64非AIME固定输入（16prefill+48cachedforward）两freshprocess同GPU全trace精确；官方all8/ones仅作原codec算子参照，非另一个正式实验臂。未声称跨GPUbitwise或81920全程数值等价已获短canary证明。正式每token保留数值/scale风险、24层QDQ与next-consumption审计。已知official signed-zero byte差异在canary原始stat保留，未放宽门禁。\n\n本轮只有一个正式臂。历史H+C128的41/60或C6R40/60只作背景，因prefillQDQ/数值runtime等合同不同，不能宣称严格同协议提升或下降。60次生成只有30个不同问题，两个seed并非60个独立问题；不将该样本外推为普适算法结论。\n\n原始tokens/text保存在raw_results，评分后完整单条在results，完整60条FORMAL_RESULTS_60.jsonl。原始未完整attempt永久保留且不评分为数学错误。模型权重/大tensor不下载，本轮未训练、未跑FP/H_C128/STEPQuant/C6R或新32K，未优化或gitpush。全部own进程与NVML已释放。运行费用含资格/加载/正式至真实释放，不套用性能诊断预算。\n'''
 with (r/'FINAL_SCIENTIFIC_REPORT.md').open('x') as f:f.write(report)
 R.atomic_status(r/'GATE_STATUS_FINAL.json',dict(task_id=R.TASK,status='FORMAL_COMPLETE_60_60_AND_STOP',GateA='PASS',GateB='PASS',GateC='PASS_PREFIX_ONLY',legal_complete=60,scored=60,all_own_GPUs_released=True,source_unchanged=True))
 R.atomic_new(r/'FIRST_FAILURE.json',dict(task_id=R.TASK,status='NO_TRUE_FAILURE',first_failure=None))
 R.atomic_status(r/'FORMAL_PROGRESS.json',dict(task_id=R.TASK,status='FORMAL_COMPLETE_60_60_AND_STOP',total_jobs=60,completed=60,running=0,pending=0,failed=0,actual_GPU_hours=ledger['cumulative_GPU_hours']))
 catalog=[]
 for p in sorted(r.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.name!='DELIVERY_MANIFEST.json':catalog.append(dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=R.sha(p)))
 R.atomic_new(r/'DELIVERY_MANIFEST.json',dict(task_id=R.TASK,status='FORMAL_COMPLETE_60_60_AND_STOP',files=catalog,report_sha256=R.sha(r/'FINAL_SCIENTIFIC_REPORT.md')))
 print(json.dumps(dict(status='FORMAL_COMPLETE_60_60_AND_STOP',correct=correct,total=60)),flush=True)
if __name__=='__main__':main()
