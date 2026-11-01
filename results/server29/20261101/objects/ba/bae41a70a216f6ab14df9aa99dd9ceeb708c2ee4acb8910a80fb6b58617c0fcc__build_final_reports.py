"""CPU final reports from two byte-equal frozen scoring passes, no experiments."""
import argparse,csv,hashlib,json,os,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES={'C0':'FP_STATE','C1':'LQUnroll+dP8','C2':'H+LQUnroll+dP8','C3':'LQUnroll+dP16'}
def write_json(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def write_csv(p,rows):
    assert rows,'Never emit an empty scientific CSV'
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def build(a):
    first=Path(a.score1).read_bytes();second=Path(a.score2).read_bytes();assert first==second,'Two fresh CPU scores differ'
    score=json.loads(first);assert len(score['rows'])==99
    payload=Path(a.payload_root);rows=score['rows'];summary=score['summary'];configs=summary['arms']
    by={(r['problem_idx'],r['config_id']):r for r in rows};pairs=[]
    for i in range(1,34):
        row=dict(problem_idx=i,problem_type=json.dumps(by[i,'C0']['problem_type'],ensure_ascii=False))
        for c in NAMES:
            r=by.get((i,c));row.update({c+'_correct':None if r is None else r['correct'],c+'_generated_tokens':None if r is None else r['generated_token_count'],
                c+'_termination':'NOT_RUN' if r is None else r['finish_reason'],c+'_final_answer':None if r is None else r['score']['extraction']['raw'],
                c+'_parser_status':'NOT_RUN_STOP_ROTATION_PARITY_FAILED' if r is None else r['parser_status'],c+'_source_artifact':None if r is None else r['source_artifact']})
        pairs.append(row)
    write_csv(ROOT/'paired_results.csv',pairs)
    main=[]
    for c in NAMES:
        z=configs[c];main.append(dict(configuration=NAMES[c],config_id=c,N=z['N'],correct=z['correct'],accuracy=z['accuracy'],
            delta_correct_vs_FP=z.get('delta_correct_vs_FP'),mean_tokens=z.get('mean_tokens'),median_tokens=z.get('median_tokens'),EOS=z.get('EOS'),length_cap=z.get('length_cap'),status=z['status']))
    write_csv(ROOT/'accuracy_summary.csv',main)
    write_json(ROOT/'PAIR_STATISTICS.json',{k:v for k,v in summary.items() if k!='arms'})
    out=ROOT/'per_question_results';out.mkdir(exist_ok=True)
    for r in rows:write_json(out/(r['unit']+'.json'),r)
    memory=[];runtime=[]
    for r in rows:
        x=json.loads((payload/'formal'/r['unit']/'RESULT.json').read_text());v=x['result'];q=v['quantizer_audit'];close=json.loads((payload/'formal'/r['unit']/'OWNING_CLOSED.json').read_text())
        if r['config_id']=='C0':
            m=dict(full_FP32_state_persistent_bytes=q['recurrent_state_nbytes'],packed_base_bytes=0,live_update_payload_bytes=0,update_capacity_bytes=0,peak_decode_update_payload_bytes=0,writebacks=0)
        else:
            layers=q['storage_by_layer'];m=dict(full_FP32_state_persistent_bytes=0,packed_base_bytes=sum(l['base_nbytes'] for l in layers.values()),
                live_update_payload_bytes=sum(l['buffer_nbytes'] for l in layers.values()),update_capacity_bytes=sum(l['decode_buffer_capacity_nbytes'] for l in layers.values()),
                peak_decode_update_payload_bytes=sum(l['peak_decode_buffer_nbytes'] for l in layers.values()),
                peak_prefill_payload_bytes=sum(l['peak_prefill_buffer_nbytes'] for l in layers.values()),
                per_layer_writebacks={k:l['writeback_count'] for k,l in layers.items()},
                per_layer_pending={k:l['pending_tokens'] for k,l in layers.items()},
                exponent_sideinfo_live_bytes=sum(t['nbytes'] for l in layers.values() for t in l['inventory'] if 'scale_exponents' in t['name']),
                temporary_vector_scratch_layer_max=max(l['vector_scratch_peak_payload_bytes'] for l in layers.values()),
                dense_decode_reconstructions_per_layer={k:l['decode_dense_materializations'] for k,l in layers.items()})
        m.update(unit=r['unit'],config_id=r['config_id'],peak_allocated_bytes=v['peak_allocated_bytes'],peak_reserved_bytes=v['peak_reserved_bytes'],
            full_attention_KV_peak_observed_payload_bytes=v['full_attention_KV_peak_observed_payload_bytes'],measurement='RUNTIME_VERIFIED_tensor_metadata_and_allocator_counters',
            sum_of_layer_payload_peaks_scope='Layer peaks need not occur at identical moments; excludes transient complete State and allocator overhead',
            allocator_lifetime_profile='UNMEASURED',SM_active='UNMEASURED')
        memory.append(m)
        runtime.append(dict(unit=r['unit'],config_id=r['config_id'],gpu=x['gpu'],gpu_uuid=x['runtime']['driver_uuid'],model_load_wall_s=x['model_load_wall_s'],
            generated_tokens=v['generated_token_count'],prompt_tokens=v['input_tokens'],actual_decode_forwards=v['actual_decode_forwards'],
            generation_wall_s=v['wall_seconds'],prefill_wall_s=v['prompt_seconds'],decode_and_sampling_s=v['decode_and_sampling_seconds'],decode_forward_TPS=v['decode_forward_TPS'],
            dispatch_wall_s=close['dispatch_wall_s'],dispatch_wall_GPU_accounting_hours=close['dispatch_wall_s']/3600,
            physical_GPU_active_hours=None,physical_GPU_active_measurement='UNMEASURED',runtime=x['runtime']))
    write_json(ROOT/'memory_accounting.json',dict(rows=memory,C2='NOT_RUN',scope='Base, sideinfo and pending ring are live tensor payloads; capacity is a configured bound; temporary FP32 State is boundary scratch, never persistent in C1/C3. Full attention KV and allocator are separate.',
        code_formula_record_bytes=17696,decode_capacity_formula='24 layers * period * (1024 K4 code bytes +128 FP16 scales +32 packed integer ZP +16384 FP32 u +128 FP32 log-decay)',
        expected_capacity_dp8=3397632,expected_capacity_dp16=6795264,formula_status='CODE_VERIFIED; verify actual inventory per trajectory'))
    write_json(ROOT/'runtime_accounting.json',dict(rows=runtime,arms=configs,scope='Each GPU is one TP1 model. Pooled per-config generation/decode TPS is affected by trajectory length and hardware load; not four-GPU aggregate or fixed-work kernel speedup.',physical_GPU_active_hours='UNMEASURED'))
    write_json(ROOT/'generation_diagnostics.json',dict(arms=configs,rows=[{k:v for k,v in r.items() if k not in ['score','source_artifact']} for r in rows],nonfinite=0,runtime_error=0,
        repetition_definition='Overlapping32-token grams: (totalgrams-uniquegrams)/totalgrams;0 below32tokens',C2='NOT_RUN'))
    table='| 配置 | 正确 | 准确率 | 相对FP题数 | 平均token | EOS | cap |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in main:
        f=lambda v:'NA' if v is None else str(v)
        table+='| '+r['configuration']+' | '+f(r['correct'])+'/'+str(r['N'])+' | '+('NA' if r['accuracy'] is None else f"{100*r['accuracy']:.2f}%")+' | '+f(r['delta_correct_vs_FP'])+' | '+f(r['mean_tokens'])+' | '+f(r['EOS'])+' | '+f(r['length_cap'])+' |\n'
    text='# HMMT February 2026 正式评测报告\n\n最终科学状态：**PARTIAL_WITH_VALIDATED_ARTIFACTS**。99条全新、同seed1、合法exit0/release且29持久化的轨迹完成；C2因正式生成前旋转FP门禁失败未运行，其准确率不可用，不能当成0/33。\n\n'+table
    text+='\nC2：V1归一化前BF16旋转输出NRMSE1.05379%超过0.5%；V2改为原归一化后的FP32 H，算子与Direct Read门禁通过，但49前向真实9B FP-H logits最坏4.20688%超过预先固定1%。原容差未提高，所有失败保留，不据此推断HMMT准确率。其余三组源码与模型资格独立通过。\n\n'
    text+='主要配对比较（探索性，三个预定主比较；C2比较不可用）：\n\n'+''.join('- '+k+': '+json.dumps(summary[k],ensure_ascii=False)+'\n' for k in ['primary_A','primary_B','primary_C'])
    text+='\nHMMT是新数据集，新协议temperature0.6、maxnew65536；历史AIME结果不复用。仅seed1、33题小样本；未作多重检验校正的p值应探索性解释，未显著不证明生成等价。与DAMP多次采样协议不同，不声称复现其论文准确率，也不能用跨数据集分数作量化因果改进证据。C2缺失使Hadamard作用不可判定。\n\n'
    text+='C0实际FP32 State常驻50,331,648 bytes，无State/K4量化；C1/C3为packed mixed base加K4/u_FP32/log-decay_FP32记录，每层每token17696 bytes，24层dP8容量3,397,632、dP16容量6,795,264 bytes，实际live随pending变化。完整FP32 State只作周期边界临时重建；EOS保留尾部。KV、临时scratch、allocator allocated/reserved单独记录，不将容量称实占。详见memory_accounting.json。\n\n'
    text+='吞吐口径见runtime_accounting.json：HMMT输出长度不同，实际TPS不是严格同工作量算法性能证明。既有同卡固定快路径6.653195/6.794332 TPS仅作历史独立口径，不与本轮总wall混为因果加速。物理GPU active、SM occupancy/active未测，不写成0。\n\n'
    text+='数据与规则：官方MathArena/hmmt_feb_2026 revision589a53fe152febc0f4f99f3e561007bf8da8e664；33题升序、四配置原计划同prompt，C2禁入。评分为正式前冻结的精确数学适配协议，非AIME整数评分，也不声称逐字使用官方宽松grader。两个fresh本地CPU评分各自以原55空CVD tokenizer重解码同payload后评分，结果字节一致。PARSE_FAILED/TIMEOUT/UNRESOLVED按预定规则计错，原始提取与诊断全部保留。\n'
    with (ROOT/'HMMT26_EXPERIMENT_REPORT.md').open('x') as f:f.write(text)
    write_json(ROOT/'FINAL_SCIENTIFIC_RESULT.json',dict(status='PARTIAL_WITH_VALIDATED_ARTIFACTS',requested=132,completed_valid=99,excluded_config='C2',reason='STOP_ROTATION_PARITY_FAILED_PREDECLARED_FULL_MODEL_GATE',source_sha256=score['source_sha256'],two_fresh_CPU_scores_byte_equal=True,summary=summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--score1',required=True);p.add_argument('--score2',required=True);p.add_argument('--payload-root',required=True);build(p.parse_args())
