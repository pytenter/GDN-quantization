# -*- coding: utf-8 -*-
"""CPU-only reporting of actual new measurements, including negative outcomes."""
import csv,json,statistics,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def write(p,x):
    with (ROOT/p).open('x') as f:
        if isinstance(x,str):f.write(x)
        else:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
def main():
    terminal=read('runtime/COMPUTE_TERMINAL.json');rows=[];groups={};status=terminal['status']
    for p in sorted((ROOT/'measurements').glob('*.json')) if (ROOT/'measurements').exists() else []:rows.append(json.loads(p.read_text()))
    for mode in ['short','long']:
        groups[mode]={}
        for period in [8,16,32]:
            rr=[x for x in rows if x['mode']==mode and x['decode_period']==period]
            if rr:groups[mode][str(period)]=dict(N=len(rr),TPS=[x['decode_TPS'] for x in rr],median_TPS=statistics.median(x['decode_TPS'] for x in rr),min_TPS=min(x['decode_TPS'] for x in rr),max_TPS=max(x['decode_TPS'] for x in rr),decode_wall_s=[x['decode_wall_s'] for x in rr],median_TPOT_ms=statistics.median(x['mean_decode_token_ms'] for x in rr))
    complete=terminal['status']=='BENCHMARKS_COMPLETE_PENDING_CPU_REPORT' and len(rows)==18 and all(len(v)==3 for v in groups.values())
    comparisons={}
    if complete:
        for mode in groups:
            g=groups[mode];a,b,c=g['8'],g['16'],g['32']
            comparisons[mode]=dict(speedup32_vs16=c['median_TPS']/b['median_TPS'],delta32_vs16_percent=100*(c['median_TPS']/b['median_TPS']-1),speedup32_vs8=c['median_TPS']/a['median_TPS'],delta32_vs8_percent=100*(c['median_TPS']/a['median_TPS']-1),observed_ranges32_16_overlap=not(c['min_TPS']>b['max_TPS'] or b['min_TPS']>c['max_TPS']),not_a_statistical_significance_test=True)
        clear=all(groups[m]['32']['min_TPS']>groups[m]['16']['max_TPS'] for m in groups)
        status='COMPLETE' if clear else 'COMPLETE_NO_CLEAR_SPEEDUP'
    write('RESULT.json',dict(status=status,all_requested_benchmarks_completed=complete,groups=groups,comparisons=comparisons,source_sha256=terminal['source_sha256'],new_AIME_samples=0,new_accuracy_evaluations=0,original_experiments_modified=False))
    if rows:
        fields=['mode','repeat','position','decode_period','prompt_tokens','warmup_decode_forwards','measured_decode_forwards','decode_TPS','mean_decode_token_ms','decode_wall_s','prefill_latency_s','peak_allocated_bytes','peak_reserved_bytes']
        with (ROOT/'BENCHMARK_RAW.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:x[k] for k in fields} for x in rows);f.flush();os.fsync(f.fileno())
    numeric=[]
    for p in sorted((ROOT/'validation').glob('DP32_FRESH*.json')):
        x=json.loads(p.read_text());numeric.append(dict(path=str(p.relative_to(ROOT)),status=x['status'],worst_NRMSE=x.get('worst_NRMSE'),same_base_worst_NRMSE=x.get('same_base_worst_NRMSE'),local_unexplained=x.get('local_unexplained_code_mismatches'),cases=len(x.get('cases',[]))))
    write('NUMERICAL_VALIDATION.md','# dP32 数值与生命周期验证\n\n'+json.dumps(numeric,ensure_ascii=False,indent=2)+'\n\n沿用原0.5%跨 replay、0.1%同 actual-state 读出的工程门禁，不改阈值。覆盖31/32/33、63/64/65、prompt_end、未满窗口preserve、强弱衰减、exponent、实际32 heads及257步弱衰减；原跨轨迹局部标签保留，同窗口base反事实单独解释继承分叉。验证使用dense reference，吞吐路径的capture_validation=False。验证失败后没有启动性能测试。原代码和旧正式输出均未修改。\n')
    memory=[]
    for x in rows:
        layers=x['quantizer_audit']['storage_by_layer'];totals={k:sum(v[k] for v in layers.values()) for k in ['base_nbytes','buffer_nbytes','peak_decode_buffer_nbytes','decode_buffer_capacity_nbytes','vector_scratch_peak_payload_bytes','decode_dense_materializations','packed_kernel_launches']}
        exponents=sum(z['nbytes'] for v in layers.values() for z in v['inventory'] if 'scale_exponents' in z['name'])
        memory.append(dict(mode=x['mode'],repeat=x['repeat'],period=x['decode_period'],totals=totals,terminal_optional_exponent_bytes=exponents,terminal_pending_by_layer={k:v['pending_tokens'] for k,v in layers.items()},allocated_peak=x['peak_allocated_bytes'],reserved_peak=x['peak_reserved_bytes']))
    write('MEMORY_ACCOUNTING.json',dict(record_components=dict(K4_codes=1024,K_FP16_scales=128,K_packed_zeros=32,u_FP32=16384,logg_FP32=128,total=17696),theoretical_capacity_24layers={str(p):17696*24*p for p in [8,16,32]},actual_full_model_measurements=memory,actual_full_pending_tensor_inventory='KERNEL_RAW.json rows[*].boundaries[*].actual_precommit_inventory',scratch_scope='Runtime-reported live vector scratch peak is an actual named subset, not an allocator-wide temporary peak. Full dense boundary tensor [1,32,128,128] FP32=2097152 bytes per layer, transient. Encoder FP64/layout temporaries also exist; allocator-wide scratch attribution UNMEASURED.',payload_is_not_allocator_reserved=True,SM_occupancy='UNMEASURED',weights_copied=False))
    kernel=read('KERNEL_RAW.json') if (ROOT/'KERNEL_RAW.json').exists() else None
    text='# dP8 / dP16 / dP32 同卡吞吐比较\n\n状态：'+status+'\n\n仅新目录、物理GPU3/RTX3090、TP1、BF16。主结论使用本轮同卡重测；原GPU1的6.65/6.79仅作历史辅助。原非AIME16-token输入、原sampler与检查、32 warmup、128实测decode forward、每配置3次且顺序轮换。Short直接调用字节未改的model_engineering.fixed_run。Long使用同一token流和相同逐token forward，计数为真实decode forward，非并行prefill。每次重建独立请求cache；没有自由生成、AIME答案或评分。\n\n'
    for mode in ['short','long']:
        text+='## '+mode+'\n\n|配置|每次 token/s|中位数|最小/最大|中位 TPOT ms|\n|---|---|---:|---|---:|\n'
        for period,v in groups[mode].items():text+=f"|dP{period}|{', '.join(f'{x:.6f}' for x in v['TPS'])}|{v['median_TPS']:.6f}|{v['min_TPS']:.6f}/{v['max_TPS']:.6f}|{v['median_TPOT_ms']:.6f}|\n"
        if mode in comparisons:text+='\n'+json.dumps(comparisons[mode],ensure_ascii=False)+'\n\n'
    if complete:
        text+='## 相同上下文区间\n\n|Decode forward区间|dP8中位TPS|dP16中位TPS|dP32中位TPS|\n|---|---:|---:|---:|\n'
        for lo,hi in [(2048,8192),(8192,16384)]:
            values=[]
            for period in [8,16,32]:values.append(statistics.median(z['TPS'] for x in rows if x['mode']=='long' and x['decode_period']==period for z in x['intervals'] if z['start_decode_forward']==lo and z['end_decode_forward']==hi))
            text+=f'|{lo}–{hi}|'+ '|'.join(f'{x:.6f}' for x in values)+'|\n'
    text+='\n重复间范围接近或重叠时不宣称可明确区分快慢；区间分离也不是总体统计显著性证明。长测试历时较长，其他卡上的原实验照常运行，CPU/温度/功耗环境见逐15秒telemetry，不能冒充隔离服务吞吐或SM occupancy。写回减少和pending增加的净效应依据本轮真实测量判断，不通过改kernel优化dP32。\n'
    write('THROUGHPUT_COMPARISON.md',text)
    kt='# Kernel 与边界时间\n\nPacked base两个matvec、pending向量读出、delta与output融合在原同一个Triton实现中，实际分两种位宽各launch一次。不能不改kernel就分离各内部组件的排他时间：这些分项为UNMEASURED；提供合并时间、不同pending数量和独立host dispatch时间。prep的Torch kernel总数未profile，不把2个fused launch冒充全步总launch数。\n\n边界的重建、原encoder总时间和packing分别使用相同诊断wrapper测量，诊断不进入主模型吞吐计时。encoder扣除packing的剩余为INFERRED，含scale/zp/exponent/RNE/layout/checks/dispatch，不能称纯scale算子时间。CUDA event为stream elapsed，可能包含host发射空隙，不是SM active时间。每组10次，原始值在KERNEL_RAW.json。\n\n'
    if kernel:
        kt+='|周期|最大非边界pending|fused中位µs|边界全步中位µs|重建中位µs|encoder中位µs|packing中位µs|\n|---|---:|---:|---:|---:|---:|---:|\n'
        for x in kernel['rows']:
            rr=next(z for z in x['readouts'] if z['pending_records']==x['period']-2);bb=x['boundaries']
            kt+=f"|{x['period']}|{rr['pending_records']}|{rr['median_fused_us']:.6f}|{statistics.median(z['whole_boundary_host_synchronized_us'] for z in bb):.6f}|{statistics.median(z['components']['state_reconstruction']['event_stream_elapsed_us'] for z in bb):.6f}|{statistics.median(z['components']['state_encode_total_including_packing']['event_stream_elapsed_us'] for z in bb):.6f}|{statistics.median(z['packing_event_stream_us'] for z in bb):.6f}|\n"
    write('KERNEL_LATENCY_REPORT.md',kt)
    receipts={str(p.relative_to(ROOT)):json.loads(p.read_text()) for p in (ROOT/'runtime').glob('*/OWNING_EXIT_AND_RELEASE.json')}
    write('EXECUTION_LOG.md','# 执行与资源记录\n\n'+json.dumps(dict(compute_terminal=terminal,actual_receipts=receipts),ensure_ascii=False,indent=2)+'\n\nGPU active/SM seconds未测，不能用dispatch wall冒充。所有旧进程只读观察，没有信号、暂停、迁移、优先级更改或输出覆盖。最终CPU报告进程、parent与Bash wait收据晚于本文件，由独立终态交付封存入MANIFEST。\n')
    print(json.dumps(dict(status=status,measurements=len(rows),complete=complete)),flush=True)
if __name__=='__main__':main()
