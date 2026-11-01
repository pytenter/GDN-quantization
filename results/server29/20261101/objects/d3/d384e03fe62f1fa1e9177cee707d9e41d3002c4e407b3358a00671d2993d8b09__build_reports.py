# -*- coding: utf-8 -*-
"""CPU reporting from actual immutable measurements; never invent formal rows."""
import argparse,hashlib,json,os,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import execution_common as C

def read(name):return json.loads((ROOT/name).read_text())
def write(name,value):
    p=ROOT/name;tmp=p.with_name('.'+p.name+'.report_tmp')
    assert not tmp.exists()
    with tmp.open('x') as f:
        if isinstance(value,str):f.write(value)
        else:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
    tmp.replace(p)

def reports(terminal=False,source_sha=None,permit_sha=None,partial=False):
    if source_sha is not None:
        C.source(ROOT,source_sha);assert C.sha(ROOT/'GPU_PERMIT.json')==permit_sha
    perf=read('performance/FAST_MATCHED_THROUGHPUT_V2.json')
    groups={tag:[x for x in perf['measurements'] if (x['path'],x['decode_period'])==pair] for tag,pair in [('replay8',('replay',8)),('direct8',('direct',8)),('direct16',('direct',16))]}
    med={k:statistics.median(x['decode_TPS'] for x in rr) for k,rr in groups.items()}
    improvements={k:med[k]/med['replay8']-1 for k in ['direct8','direct16']}
    kernel=read('performance/KERNEL_MICROBENCH_V1.json')
    numeric=read('validation/FAST_BATCH_KEYS_V5.json');cpu=read('validation/CPU_FULL_V3.json')
    downstream=read('validation/FULL_MODEL_PARITY_V1.json');lineage=read('validation/LINEAGE_COUNTERFACTUAL_V1.json')
    formal=[]
    if terminal:
        C.source(ROOT,source_sha);assert C.sha(ROOT/'GPU_PERMIT.json')==permit_sha
        score=read('SCORED_RESULTS.json');assert len(score['rows'])==60
        for job in read('FORMAL_JOBS.json')['jobs']:
            p=ROOT/'formal'/job['id'];closed=json.loads((p/'OWNING_CLOSED.json').read_text());value=json.loads((p/'RESULT.json').read_text())
            assert closed['owning_Popen_exit']==0 and closed['release_observed'] and closed['RESULT_sha256']==C.sha(p/'RESULT.json')
            formal.append(dict(job=job,closed=closed,result=value['result'],model_load_wall_s=value['model_load_wall_s']))
    memory=dict(scope='Actual tensor payload; CUDA allocator peaks reported separately. Capacity is never substituted for terminal live occupancy.',
        layers=24,logical_values=24*32*128*128,codes_only_bpv=5,
        delta_record_formula=dict(packed_K4_codes=1024,FP16_K_scales=128,packed_K_zero_points=32,u_FP32=16384,log_decay_FP32=128,total_per_layer_token=17696),
        capacity_bytes={str(p):17696*24*p for p in [8,16]},
        actual_fixed_workload={k:[dict(peak_allocated_bytes=x['peak_allocated_bytes'],peak_reserved_bytes=x['peak_reserved_bytes'],layers=x['quantizer_audit']['storage_by_layer']) for x in rr] for k,rr in groups.items()},
        formal_terminal=[dict(unit=x['job']['id'],period=x['job']['period'],memory=x['result']['memory'],layers=x['result']['quantizer_audit']['storage_by_layer']) for x in formal],
        transient_dense_boundary_shape=[1,32,128,128],transient_dense_boundary_dtype='FP32',transient_dense_one_layer_bytes=2097152,
        transient_vector_scratch='Batched decoded K FP32, stacked u FP32, exp(g) FP32; temporary, not persistent dense State.',
        native_marker_bytes_all_layers=192,persistent_runtime_head_maps_bytes_all_layers=3072,
        unmeasured=['Pure SM active memory/time','Separate full-attention KV peak attribution','Allocator block lifetime after logical tensor release'])
    for x in formal:
        inventory=x['result']['quantizer_audit']['storage_by_layer']
        for row in inventory.values():
            assert row['buffer_nbytes']==row['pending_tokens']*17696
            assert row['decode_buffer_capacity_nbytes']==row['decode_period']*17696
    write('memory_accounting.json',memory)
    runtime=dict(fixed_workload_median_TPS=med,relative_to_replay8=improvements,
        same_GPU_repeats=3,prompt_tokens=16,warmup_decode_forwards=32,measured_decode_forwards=128,
        generation_lengths_do_not_affect_fixed_workload=True,kernel_measurements=kernel,
        formal=[dict(unit=x['job']['id'],period=x['job']['period'],generation_wall_s=x['result']['wall_seconds'],decode_and_sampling_s=x['result']['decode_and_sampling_seconds'],prefill_s=x['result']['prompt_seconds'],dispatch_wall_s=x['closed']['dispatch_wall_s'],model_load_wall_s=x['model_load_wall_s']) for x in formal],
        GPU_active_SM_seconds=None,GPU_active_measurement_status='NOT_MEASURED',dispatch_wall_is_not_GPU_active=True,
        engineering_failure_history=['CUDA_KERNEL_V1 actual compile failure, corrected Python global binding','CPU_CONTROL_FRESH1 exact historical proc cause UNRESOLVED; FRESH3 Live argv mismatch retained; empty-argv observation race inferred; no GPU credit; see PRE_FORMAL_CPU_SCOPE_NOTES'],
        reference_old_key_preparation='performance/MATCHED_THROUGHPUT_V1.json; dP16 slower, retained unchanged')
    write('runtime_accounting.json',runtime)
    write('source_identity.json',dict(formal_source_manifest_sha256=source_sha,GPU_permit_sha256=permit_sha,
        baseline_source_sha256='d70a1b7a7d54a59d12bba3ad33b60b495af787295629744990543a899f947267',
        unchanged_quantizer_sha256=C.sha(ROOT/'src/pace_quantizer.py'),unchanged_head_mask_sha256=C.sha(ROOT/'calibration/head_mask.json'),
        identity_basis='Frozen per-file SHA256 snapshot, not unverified current HEAD.',
        git_commit_status='No experiment git commit is asserted; original frozen delivery snapshot is authoritative.',
        configurations={str(p):read('configs/DP'+str(p)+'.json') for p in [8,16]}))
    perftext=f'''# 吞吐报告\n\n固定工作量：55 原 GPU1，TP1/BF16，同一9B模型、16-token非AIME prompt、32 warmup、128 decode forwards，原sampler和检查实际执行，三轮轮换顺序；同步计时，模型加载/编译/prefill不计入decode TPS。\n\n|路径|三轮中位 tokens/s|相对原replay8|\n|---|---:|---:|\n|原replay dP8|{med['replay8']:.6f}|基准|\n|高效Direct dP8|{med['direct8']:.6f}|{improvements['direct8']:+.2%}|\n|高效Direct dP16|{med['direct16']:.6f}|{improvements['direct16']:+.2%}|\n\ndP16相对新dP8为{med['direct16']/med['direct8']-1:+.2%}。这是短固定工作量结果，不能代替长上下文正式轨迹吞吐。历史不同长度/检查/硬件结果不用于本加速比分母。原逐条K准备的4.926/5.308/4.421 TPS仍保存在MATCHED_THROUGHPUT_V1；负结果未删除。\n\nRUNTIME_VERIFIED：FAST_MATCHED_THROUGHPUT_V2.json、对应OWNING_WAIT_AND_RELEASE.json。Kernel event图计时约19.10/27.95微秒，普通Python提交约173/174微秒；不含完整K准备和全模型，不等于模型token latency。旧单次boundary观测15.09/23.05ms不是稳定统计。\n'''
    write('THROUGHPUT_REPORT.md',perftext)
    write('IMPLEMENTATION_AUDIT.md','''# 实现审计\n\nCODE_VERIFIED：类型B，低比特base + packed K4/FP32 u/FP32 log-decay。src/lq_unroll_buffer.py::step在decode每步调用packed_readout；src/packed_direct_read.py::_kernel融合INT4/INT8分组解码、base两个matvec、pending向量读出和本步u/output，没有全局dense State输出。正式生成只有UnrollEngine。replay8只作明确要求的固定工作量reference。\n\nToken1…P−1：packed_base不变，追加actual u；每步临时批解码K，BF16 native norm后FP32向量。TokenP先算u和output，追加本条，然后_delta_state用实际u一次顺序重建FP32矩阵，原PackedAffineState.from_tensor提交并清空引用。未满窗口EOS/cap不补forward、不flush，preserve尾部。Prefill完全委托原PACEBuffer，256/prompt_end原语义。最后sampled token未forward，decode_forwards=generated−1。\n\n没有持久dense FP32 State；None slot与Python引用无tensor payload。临时boundary State与allocator reserved必须分别解释。批处理K与逐条原codec/native BF16norm逐元素一致，验证和四卡真实9B smoke保留。\n\n两配置json只差decode_period=8/16；量化器、K4、head mask、sampler、FrozenV4均复用冻结字节。非官方LeapQuant kernel复现声明：仅借鉴可核验公式，自研Triton实现，不声称官方实现相同。\n''')
    write('DIRECT_READ_KERNEL_REPORT.md','''# Packed Direct Read核验\n\nCODE_VERIFIED：src/packed_direct_read.py；INT4/INT8 heads分别launch，沿Key维对packed codes、整数zero、FP16 mantissa和optional INT8 exponent解码；FP32 accumulation；enable_fp_fusion=False。未保存全部Query/原V/beta，pending只存压缩K和actualu/g。\n\nRUNTIME_VERIFIED：FAST_BATCH_KEYS_V5 40 cases、CUDA_FRESH_V4独立fresh进程、SAME_STATE_ORACLE_V1 300独立FP64对照、四卡FAST_MODEL_SMOKE_V2、FULL_MODEL_PARITY_V1。实际loaded model flags在各receipt中。CUDA_KERNEL_V1真实编译失败及exit1/release仍保留。\n\nKernel与全模型吞吐分开，图重放CUDA event核时间不包含Python dispatch和K批准备。无官方kernel吞吐或SM active时间断言。\n''')
    write('NUMERICAL_PARITY_REPORT.md',f'''# 数值审计\n\nCPU prototype原跨轨迹screen仍FAIL：worst output NRMSE={cpu['worst']['output']['normalized_RMSE']:.8%}，不改0.5%阈值、不把CPU数值路径当成GPU数值路径。正式CPU fallback被代码拒绝。\n\n目标CUDA两fresh完整case与fast40case：output worst={numeric['worst']['output']['normalized_RMSE']:.8%}，delta={numeric['worst']['delta']['normalized_RMSE']:.8%}，boundary={numeric['worst']['boundary_state']['normalized_RMSE']:.8%}；原0.5%筛查数值不变。same-actual-state output={numeric['worst']['actual_delta_state_readout']['normalized_RMSE']:.8%}，小于原0.1%。无非有限值、错计数或tail隐式flush。\n\n旧局部rounding heuristic的687个UNEXPLAINED标签仍在原工件；实际同窗口base替换反事实验证见LINEAGE_COUNTERFACTUAL_V1：1077个跨轨迹code差异在换为同base的sequential replay后消失，另2个是同base局部adjacent rounding；同base所有本地差异均可由原量化编码逐值重现且局部未解释数0。samebase u/boundary worst NRMSE={lineage['local_max_NRMSE']['same_window_base_delta']:.9g}/{lineage['local_max_NRMSE']['same_window_base_boundary']:.9g}。这是继承base分叉的运行证据，不是把所有不同code都要求处于同一个当步舍入边界。\n\n独立FP64 oracle使用真实packed输入/实际u/g，BF16输出对同样BF16舍入oracle最坏约0.01915%；该证据说明局部算子，不保证80K生成等价。\n\n真实32层/24GDN非AIME teacher-forced模型：prefill bitwise相等、marker相等；decode开始有差异，logits worst NRMSE={downstream['max_logits_normalized_RMSE']:.6%}，GDN module输出worst={downstream['max_layer_output_normalized_RMSE']:.6%}。下游逐层传播并非单个相同输入算子的0.5%筛查对象；未声明模型输出等价。新两配置作为独立FP32数值实现，历史21/30与0/30不复用、不继承。\n\nCODE_VERIFIED / RUNTIME_VERIFIED见各脚本与原始JSON；80K稳定性、准确率和长上下文吞吐在真实formal完成前UNRESOLVED。科学准入最终判定另见validation/FINAL_NUMERICAL_AND_RUNTIME_ADMISSION.json，不把CPU原FAIL改写为PASS。\n''')
    status='COMPLETE_BOTH_CONFIGS' if terminal and all(v>=0 for v in improvements.values()) else 'COMPLETE_WITH_PERFORMANCE_NEGATIVE' if terminal else 'PARTIAL_WITH_VALIDATED_ARTIFACTS' if partial else 'ENGINEERING_VALIDATED_FORMAL_NOT_COMPLETE'
    summary=read('SCORED_RESULTS.json')['summary'] if terminal else None
    write('EXPERIMENT_SUMMARY.md','# 实验总结\n\n状态：'+status+'\n\n'+('正式60条全新轨迹、两fresh FrozenV4 CPU同payload评分完成。统计仅限seed1 canonical30。\n\n'+json.dumps(summary,ensure_ascii=False,indent=2) if terminal else '正式尚未完整交付，不报告新AIME准确率。工程吞吐和数值偏差见专门报告。')+'\n')
    write('README.md','# PACE48 LeapQuant-style Window Unrolling dP8/dP16\n\n'+status+'\n\n本目录为独立新实现。原冻结PACE目录只读。正式只运行Direct8/16，原replay8仅为工程测速reference。详见IMPLEMENTATION_AUDIT.md、NUMERICAL_PARITY_REPORT.md、DIRECT_READ_KERNEL_REPORT.md、THROUGHPUT_REPORT.md、memory_accounting.json、runtime_accounting.json、source_identity.json和operations/HUMAN_REQUEST.txt。\n')
    print(json.dumps(dict(status=status,formal_completed=len(formal),fixed_workload_median_TPS=med)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--terminal',action='store_true');p.add_argument('--partial',action='store_true');p.add_argument('--source-sha');p.add_argument('--permit-sha');a=p.parse_args();assert not(a.terminal and a.partial);reports(a.terminal,a.source_sha,a.permit_sha,a.partial)
