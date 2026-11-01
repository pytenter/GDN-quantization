#!/usr/bin/env python3
"""One terminal CPU scoring/seal/archive. No CUDA API, model load, or sampling."""
import argparse
import collections
import csv
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pace_control as C
from pace_four_gpu_results import completed_rows as _four_completed_rows

PYTHON = '/data/ydai/miniconda3/envs/bitdecode/bin/python'
MODEL = '/data/zypan/modelscope_models/Qwen3.5-9B'
SSH_SOCKET = '/tmp/qwen_replication_archive29.sock'
STORAGE = 'zypan@202.38.247.29'
METHOD_VARIANT = 'PACE48_FP16_EXPONENT_V1'
STATE_SCALE_FORMAT = 'FP16_MANTISSA_PLUS_OPTIONAL_INT8_EXPONENT'
STATE_VALUES = 24 * 32 * 128 * 128
MAX_STATE_EXPONENT_BYTES = STATE_VALUES // 32


def method_labels():
    return dict(execution_revision=C.REVISION, method_variant=METHOD_VARIANT, state_scale_format=STATE_SCALE_FORMAT,
                original_unmodified_R2_R3=False,
                inherited_payload_scope='ORIGINAL3_E0_EXACT_AND_CLOSED_EXPONENT_EXECUTION_PAYLOADS')


def complete_rows(root, invalid=None):
    # Only the strict fixed-origin admission may route the three old payloads.
    # Their original wrappers, identities and intervals are never rewritten.
    return _four_completed_rows(Path(root), invalid=invalid)


def terminal_guard(root):
    C.authorize(root)
    result = C.read(root / 'SESSION_RESULT.json')
    C.require(result['task_id'] == C.TASK and result['own_workers_exit_and_NVML_release'] is True and
              result['source_manifest_sha256'] == C.sha(root / C.SOURCE) and
              result['permit_sha256'] == C.sha(root / 'GPU_PERMIT.json') and
              not C.same_process(result['controller']), 'Controller/worker terminal release unresolved')
    intervals = C.read(root / 'RESOURCE_LEDGER.json')['intervals']
    observation = C.query_gpu()  # .55 only. No query or signal is sent to archive29.
    for x in intervals:
        C.require(x['released'] is True and x['returncode'] is not None and
                  not (x['identity'] is not None and C.same_process(x['identity'])) and
                  C.release_state(True, observation, x['pid'], x['identity']),
                  'Owned worker identity/NVML release not independently confirmed')
    return dict(status='PASS_OWN_TRACKED_IDENTITIES_GONE_AND_NVML_RELEASED', NVML55=observation,
                interval_count=len(intervals), other_GPU_owners_foreign=True,
                archive29_GPU_queries=0, source_manifest_sha256=C.verify_sources(root),
                model17_stat=C.model_guard(root), external_fresh_SHA=C.verify_external(root, full_model=False))


def score_pass(root, out):
    C.authorize(root); C.verify_science(root); C.model_guard(root)
    C.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Scorer must be CPU-only')
    pairs = complete_rows(root)
    C.require(len(pairs) == 60, 'FrozenV4 full scoring requires exactly60 legal trajectories')
    identity = C.process_identity(os.getpid())
    C.atomic_new(out.with_suffix('.STARTED.json'), dict(status='SCORER_STARTED_ONCE_CPU',
                 identity=identity, output=str(out), source_manifest_sha256=C.sha(root / C.SOURCE),
                 permit_sha256=C.sha(root / 'GPU_PERMIT.json'), CUDA_VISIBLE_DEVICES='', unix=time.time()))
    spec = importlib.util.spec_from_file_location('pace_frozen_v4', root / 'protocol/aime26_scorer_v4.py')
    scorer = importlib.util.module_from_spec(spec); sys.modules[spec.name] = scorer; spec.loader.exec_module(scorer)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    gold = {x['problem_id']: x for x in (json.loads(s) for s in (root / 'protocol/aime26_frozen.jsonl').read_text().splitlines())}
    rows = []
    for job, wrapper, target in pairs:
        r = wrapper['result']
        C.require(tokenizer.decode(r['generated_token_ids'], skip_special_tokens=True) == r['response'],
                  'Fresh tokenizer re-decode differs from saved text')
        ex = scorer.extract_v4(r['response'], r['finish_reason'], gold[job['problem_id']]['problem'])
        tokens = r['generated_token_ids']
        grams = [tuple(tokens[i:i + 32]) for i in range(max(0, len(tokens) - 31))]
        rows.append(dict(problem_id=job['problem_id'], condition=job['condition'], seed=1,
            correct=bool(scorer.compare_with_gold(ex, gold[job['problem_id']]['answer'])),
            extraction=asdict(ex), generated_token_count=len(tokens), finish_reason=r['finish_reason'],
            wall_seconds=r['wall_seconds'], TPS=r['TPS'], prompt_hash=r['prompt_text_sha256'],
            raw_result_path=str(target.relative_to(root) / 'RESULT.json'), raw_result_sha256=C.sha(target / 'RESULT.json'),
            evidence_source_manifest_sha256=wrapper['source_manifest_sha256'],
            evidence_permit_sha256=wrapper['permit_sha256'],
            evidence_origin='CURRENT_SCALE_EXPONENT_REPAIR' if wrapper['source_manifest_sha256'] == C.sha(root / C.SOURCE) else 'REUSED_CLOSED_PRIOR_EXECUTION',
            method_variant=METHOD_VARIANT, state_scale_format=STATE_SCALE_FORMAT,
            evidence_stored_format='LEGACY_FP16_ONLY_E0_EQUIVALENT' if wrapper['source_manifest_sha256'] == C.ORIGINAL_SOURCE_SHA else STATE_SCALE_FORMAT,
            token_sha256=r['generated_token_ids_sha256'],
            repeated_32gram_fraction=0 if not grams else 1 - len(set(grams)) / len(grams),
            peak_allocated_bytes=r['peak_allocated_bytes'], peak_reserved_bytes=r['peak_reserved_bytes']))
    C.model_guard(root); C.verify_sources(root)
    C.atomic_new(out, dict(task_id=C.TASK, status='PASS_FRESH_CPU_FROZEN_V4_EXACT_SAVED_PAYLOAD',
                 identity=identity, payload=dict(rows=rows,
                 scorer_sha256=C.sha(root / 'protocol/aime26_scorer_v4.py'),
                 dataset_sha256=C.sha(root / 'protocol/aime26_frozen.jsonl'),
                 source_manifest_sha256=C.sha(root / C.SOURCE), science_freeze_sha256=C.verify_science(root),
                 **method_labels())))


def run_score_child(root, index, env):
    target = root / ('FROZEN_V4_FRESH_PASS_' + str(index) + '.json')
    prefix = root / ('FROZEN_V4_FRESH_PASS_' + str(index))
    argv = [PYTHON, str(root / 'scripts/finalize_four_gpu_cpu.py'), '--root', str(root),
            '--score-pass', str(target)]
    C.atomic_new(prefix.with_suffix('.PROCESS_DISPATCH.json'), dict(argv=argv,
                 parent=C.process_identity(os.getpid()), source_manifest_sha256=C.sha(root / C.SOURCE),
                 permit_sha256=C.sha(root / 'GPU_PERMIT.json'), CUDA_VISIBLE_DEVICES='', unix=time.time()))
    identity = None
    identity_error = None
    with prefix.with_suffix('.log').open('xb', buffering=0) as log:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    identity = C.process_identity(proc.pid)
                    C.require(identity['cmdline'] == argv, 'Fresh scorer full argv differs')
                    break
                except FileNotFoundError:
                    if proc.poll() is not None:
                        break
                    time.sleep(.01)
            if identity is not None:
                C.atomic_new(prefix.with_suffix('.PROCESS_STARTED.json'), dict(identity=identity,
                             source_manifest_sha256=C.sha(root / C.SOURCE), permit_sha256=C.sha(root / 'GPU_PERMIT.json')))
        except BaseException as exc:
            identity_error = repr(exc)
        finally:
            # The owning CPU parent always reaps its one direct scorer, including
            # an identity-registration failure. No guessed exit status is used.
            returncode = proc.wait()
    if identity is None and target.with_suffix('.STARTED.json').exists():
        candidate = C.read(target.with_suffix('.STARTED.json'))['identity']
        if candidate['pid'] == proc.pid and candidate['cmdline'] == argv:
            identity = candidate
    C.atomic_new(prefix.with_suffix('.PROCESS_EXIT.json'), dict(identity=identity, pid=proc.pid,
                 returncode=returncode, actual_exited=True,
                 identity_gone=identity is not None and not C.same_process(identity),
                 identity_error=identity_error,
                 source_manifest_sha256=C.sha(root / C.SOURCE), permit_sha256=C.sha(root / 'GPU_PERMIT.json'), unix=time.time()))
    C.require(identity_error is None and returncode == 0 and identity is not None and not C.same_process(identity),
              'Original FrozenV4 fresh CPU scoring exit/identity failed')
    value = C.read(target)
    C.require(value['identity'] == identity, 'Fresh score result/actual child identity differs')
    return value


def write_bytes(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def summarize(root, rows):
    summaries = {}
    for condition, period in C.CONDITIONS.items():
        arm = [x for x in rows if x['condition'] == condition]
        C.require(len(arm) == 30, 'Per-arm denominator must be30')
        directory = root / (condition + '_PACE48_DP' + str(period))
        write_bytes(directory / 'results.jsonl', b''.join(C.canonical(x) + b'\n' for x in arm))
        summary = dict(condition=condition, **method_labels(), N=30, correct=sum(x['correct'] for x in arm),
                       mean_tokens=math.fsum(x['generated_token_count'] for x in arm) / 30,
                       EOS=sum(x['finish_reason'] == 'eos' for x in arm),
                       length_capped=sum(x['finish_reason'] == 'length' for x in arm),
                       total_runtime_seconds=math.fsum(x['wall_seconds'] for x in arm),
                       mean_runtime_seconds=math.fsum(x['wall_seconds'] for x in arm) / 30,
                       peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in arm),
                       peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in arm),
                       mean_repeated_32gram_fraction=math.fsum(x['repeated_32gram_fraction'] for x in arm) / 30,
                       nonfinite=0, runtime_errors=0)
        summary['accuracy'] = summary['correct'] / 30
        summary['EOS_rate'] = summary['EOS'] / 30
        C.atomic_new(directory / 'summary.json', summary)
        summaries[condition] = summary
    by = {(x['problem_id'], x['condition']): x for x in rows}
    paired = []
    counts = collections.Counter()
    for problem in [r['problem_id'] for r in C.read(root / 'AIME26_DATASET_FREEZE.json')['records']]:
        a, b = by[problem, 'R2'], by[problem, 'R3']
        category = ('both_correct' if a['correct'] and b['correct'] else 'R2_only_correct' if a['correct'] else
                    'R3_only_correct' if b['correct'] else 'both_wrong')
        counts[category] += 1
        paired.append(dict(problem_id=problem, R2_correct=a['correct'], R3_correct=b['correct'],
                           category=category, R2_tokens=a['generated_token_count'], R3_tokens=b['generated_token_count'],
                           R2_finish=a['finish_reason'], R3_finish=b['finish_reason'],
                           R2_seconds=a['wall_seconds'], R3_seconds=b['wall_seconds']))
    directory = root / 'comparison'; directory.mkdir(exist_ok=True)
    with (directory / 'paired_results.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    b, c = counts['R2_only_correct'], counts['R3_only_correct']
    discordant = b + c
    p = 1.0 if not discordant else min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / 2**discordant)
    comparison = dict(N_pairs=30, seed=1, categories=dict(counts), **method_labels(),
        correct_difference=summaries['R2']['correct'] - summaries['R3']['correct'],
        delta_accuracy=summaries['R2']['accuracy'] - summaries['R3']['accuracy'],
        exact_McNemar_two_sided_p=p, R2_rescued=[x['problem_id'] for x in paired if x['category'] == 'R2_only_correct'],
        R2_regressed=[x['problem_id'] for x in paired if x['category'] == 'R3_only_correct'],
        statistical_limit='One seed and30 paired already-known AIME questions; exploratory only, no universal losslessness claim.')
    C.atomic_new(directory / 'paired_summary.json', comparison)
    return summaries, comparison


def memory_accounting(root, pairs):
    rows = []
    for job, wrapper, target in pairs:
        r = wrapper['result']; audit = r['quantizer_audit']; stores = list(audit['storage_by_layer'].values())
        inventory = [x for s in stores for x in s['inventory']]
        categories = collections.Counter()
        for x in inventory:
            name = x['name']
            key = ('state_scale_exponent' if name.startswith('base.') and name.endswith('.scale_exponents') else
                   'state_codes' if name.startswith('base.') and name.endswith('.codes') else
                   'state_scale' if name.startswith('base.') and name.endswith('.scales') else
                   'state_zero_point' if name.startswith('base.') and name.endswith('.zero_points') else
                   'buffer_keys' if '.key.' in name or name.endswith('raw_bf16_key') else
                   'buffer_values' if name.endswith('.value') else 'buffer_beta' if name.endswith('.beta') else
                   'buffer_log_decay' if name.endswith('.log_decay') else 'other_tensor')
            categories[key] += x['nbytes']
            if key == 'state_scale_exponent':
                C.require(x['dtype'] in ('torch.int8', 'int8'), 'State exponent is not actual INT8 storage')
            if '.key.' in name:
                C.require(not name.endswith('.scale_exponents'), 'Key buffer exponent extension is forbidden')
        values = STATE_VALUES
        exponent_bytes = categories['state_scale_exponent']
        C.require(0 <= exponent_bytes <= MAX_STATE_EXPONENT_BYTES,
                  'Actual state exponent allocation exceeds per-G32 INT8 maximum')
        historical = wrapper['source_manifest_sha256'] == C.ORIGINAL_SOURCE_SHA
        C.require(not historical or exponent_bytes == 0, 'Original stored-format evidence was altered')
        actual = sum(x['nbytes'] for x in inventory) + audit['native_marker_tensor_nbytes']
        rows.append(dict(unit=job['id'], condition=job['condition'], state_values=values,
            **method_labels(), evidence_stored_format='LEGACY_FP16_ONLY_E0_EQUIVALENT' if historical else STATE_SCALE_FORMAT,
            actual_state_exponent_nbytes=exponent_bytes,
            exponent_allocation='NONE_ZERO_BYTES' if exponent_bytes == 0 else 'DENSE_INT8_ONLY_IN_OVERFLOWING_HEAD_PARTS',
            actual_base_bits_per_value=8 * sum(categories[k] for k in
                 ('state_codes', 'state_scale', 'state_zero_point', 'state_scale_exponent')) / values,
            theoretical_state_payload_bits_per_value=5.0, final_live_tensor_nbytes=dict(categories),
            native_marker_nbytes=audit['native_marker_tensor_nbytes'], final_resident_tensor_nbytes=actual,
            final_resident_tensor_bits_per_state_value=actual * 8 / values,
            peak_prefill_buffer_payload_nbytes=sum(x['peak_prefill_buffer_nbytes'] for x in stores),
            peak_decode_buffer_payload_nbytes=sum(x['peak_decode_buffer_nbytes'] for x in stores),
            decode_buffer_capacity_nbytes=sum(x['decode_buffer_capacity_nbytes'] for x in stores),
            peak_allocated_bytes=r['peak_allocated_bytes'], peak_reserved_bytes=r['peak_reserved_bytes']))
    value = dict(status='PASS_ACTUAL_PACKED_TENSOR_INVENTORIES', rows=rows,
        scope='Actual persistent tensor payload at terminal plus measured buffer payload peaks. Python indices/object overhead, allocator reserved, attention KV and transient FP32 replay scratch are separate; peak CUDA allocator includes their observed device effects.',
        no_full_dense_recurrent_carrier=True, theoretical_state_payload_bpv=5.0,
        ordinary_mixed_state_with_scale_and_zero_point_bpv=5.65625,
        maximum_mixed_state_with_exponents_bpv=5.90625,
        maximum_additional_state_exponent_nbytes=MAX_STATE_EXPONENT_BYTES,
        key_buffer_exponent_nbytes=0, **method_labels())
    C.atomic_new(root / 'comparison/memory_accounting.json', value)
    return value


def analysis_text(summaries, comparison, memory, ledger):
    lines = ['# PACE FP16 mantissa + optional INT8 exponent：dP8 / dP1 seed1 正式对照', '',
             '新格式 PACE48_FP16_EXPONENT_V1，原 R2/R3 仅作为固定任务编号。正常 FP16 可表示路径保持 e=0；真正溢出的 state G32 组按已授权指数格式存储，不是未修改的原 R2/R3。', '',
             '| Configuration | Correct | Accuracy | Mean Tokens | EOS | Runtime |',
             '|---|---:|---:|---:|---:|---:|']
    for c in ('R2', 'R3'):
        x = summaries[c]
        lines.append(f"| {c} dP{C.CONDITIONS[c]} | {x['correct']}/30 | {100*x['accuracy']:.2f}% | {x['mean_tokens']:.2f} | {x['EOS']}/30 | {x['total_runtime_seconds']:.2f}s |")
    a, b = summaries['R2'], summaries['R3']
    direction = '优于' if comparison['correct_difference'] > 0 else '低于' if comparison['correct_difference'] < 0 else '等于'
    def movement(delta, lower, higher):
        return lower if delta < 0 else higher if delta > 0 else '相同'

    def memory_stats(condition):
        rows = [x for x in memory['rows'] if x['condition'] == condition]
        def mean(values):
            return math.fsum(values) / len(rows)
        final = [x['final_resident_tensor_nbytes'] for x in rows]
        codes = mean([x['final_live_tensor_nbytes'].get('state_codes', 0) for x in rows])
        scales = mean([x['final_live_tensor_nbytes'].get('state_scale', 0) for x in rows])
        zeros = mean([x['final_live_tensor_nbytes'].get('state_zero_point', 0) for x in rows])
        exponents = mean([x['actual_state_exponent_nbytes'] for x in rows])
        base_bpv = mean([8 * sum(x['final_live_tensor_nbytes'].get(k, 0) for k in
                                ('state_codes', 'state_scale', 'state_zero_point', 'state_scale_exponent')) / x['state_values'] for x in rows])
        return dict(capacity=max(x['decode_buffer_capacity_nbytes'] for x in rows),
                    final_mean=mean(final), final_min=min(final), final_max=max(final),
                    codes=codes, scales=scales, zeros=zeros, exponents=exponents, base_bpv=base_bpv,
                    buffer_mean=mean([sum(value for key, value in x['final_live_tensor_nbytes'].items()
                                          if key.startswith('buffer_')) for x in rows]),
                    marker_mean=mean([x['native_marker_nbytes'] for x in rows]),
                    prefill_peak=max(x['peak_prefill_buffer_payload_nbytes'] for x in rows),
                    decode_peak=max(x['peak_decode_buffer_payload_nbytes'] for x in rows))

    ma, mb = memory_stats('R2'), memory_stats('R3')
    capped_delta = a['length_capped'] - b['length_capped']
    eos_delta = a['EOS'] - b['EOS']
    repeated_delta = a['mean_repeated_32gram_fraction'] - b['mean_repeated_32gram_fraction']
    tokens_delta = a['mean_tokens'] - b['mean_tokens']
    research_basis = ('观察到dP8正确题数净增加，可作为独立复核周期设置的线索' if comparison['correct_difference'] > 0 else
                      '观察到dP8正确题数净减少，本次不支持把延长周期视为可靠准确率增益' if comparison['correct_difference'] < 0 else
                      '未观察到dP8正确题数净增益，逐题与生成行为仍需分别解读')
    lines += ['', f"1. 本次seed1中，dP8正确题数{direction}dP1：差{comparison['correct_difference']:+d}题，准确率差{100*comparison['delta_accuracy']:+.2f}个百分点。",
              '逐题配对分组：' + json.dumps(comparison['categories'], ensure_ascii=False) + '。',
              '2. dP8独自答对（rescue）：' + (', '.join(comparison['R2_rescued']) or '无') +
              '；dP1独自答对（dP8 regression）：' + (', '.join(comparison['R2_regressed']) or '无') + '。完整30题配对见paired_results.csv。',
              f"双侧exact McNemar p={comparison['exact_McNemar_two_sided_p']:.8g}；仅一seed、30道既有AIME题，不能推断普遍无损或把相同总分称为生成等价。",
              '', f"3. 生成行为（以下差值均为dP8−dP1）：length-cap截断{a['length_capped']}/{b['length_capped']}题，差{capped_delta:+d}题，dP8截断{movement(capped_delta, '减少', '增加')}；EOS为{a['EOS']}/{b['EOS']}题，差{eos_delta:+d}题、EOS率差{100*eos_delta/30:+.2f}个百分点，dP8 EOS{movement(eos_delta, '减少', '增加')}。",
              f"平均token数{a['mean_tokens']:.2f}/{b['mean_tokens']:.2f}，差{tokens_delta:+.2f}；平均32-gram重复比例{a['mean_repeated_32gram_fraction']:.6f}/{b['mean_repeated_32gram_fraction']:.6f}，差{100*repeated_delta:+.6f}个百分点，dP8重复比例{movement(repeated_delta, '降低', '升高')}。重复比例按每条轨迹的重复32-token窗口占比取均值，不代表重复轨迹条数。",
              f"Nonfinite分别{a['nonfinite']}/{b['nonfinite']}，runtime errors分别{a['runtime_errors']}/{b['runtime_errors']}。这些是本次已生成轨迹的描述性诊断，没有触发重采样、改mask或调参。",
              f"4. 以下tensor内存均按每条轨迹全部24个GDN层合计。dP8/dP1实际decode buffer容量分别{ma['capacity']}/{mb['capacity']} bytes，差{ma['capacity']-mb['capacity']:+d} bytes；实测decode buffer payload峰值分别{ma['decode_peak']}/{mb['decode_peak']} bytes，差{ma['decode_peak']-mb['decode_peak']:+d} bytes。容量包含处理中的边界token；返回后pending最多分别7/0，None slots不计tensor字节。",
              f"终态真实live tensor总量（含native marker）均值为{ma['final_mean']:.2f}/{mb['final_mean']:.2f} bytes，差{ma['final_mean']-mb['final_mean']:+.2f} bytes；dP8范围[{ma['final_min']}, {ma['final_max']}]，dP1范围[{mb['final_min']}, {mb['final_max']}] bytes。终态buffer均值{ma['buffer_mean']:.2f}/{mb['buffer_mean']:.2f} bytes、差{ma['buffer_mean']-mb['buffer_mean']:+.2f}；marker均值{ma['marker_mean']:.2f}/{mb['marker_mean']:.2f} bytes。终态占用由实际pending决定，不能用满容量代替。",
              f"纯mixed INT4/INT8 codes理论为5.00000 bits/value；真实packed base包含FP16 mantissa、整数zero point及实际可选INT8 exponent，inventory计算为{ma['base_bpv']:.5f}/{mb['base_bpv']:.5f} bits/value（普通路径5.65625，全指数分配最大5.90625）。终态state codes均值{ma['codes']:.2f}/{mb['codes']:.2f} bytes、差{ma['codes']-mb['codes']:+.2f}；FP16 mantissa均值{ma['scales']:.2f}/{mb['scales']:.2f} bytes、差{ma['scales']-mb['scales']:+.2f}；整数zero points均值{ma['zeros']:.2f}/{mb['zeros']:.2f} bytes、差{ma['zeros']-mb['zeros']:+.2f}。含指数的side information合计{ma['scales']+ma['zeros']+ma['exponents']:.2f}/{mb['scales']+mb['zeros']+mb['exponents']:.2f} bytes。",
              f"Prefill BF16 buffer payload实测峰值{ma['prefill_peak']}/{mb['prefill_peak']} bytes，差{ma['prefill_peak']-mb['prefill_peak']:+d}。CUDA allocator峰值allocated/reserved：dP8 {a['peak_allocated_bytes']}/{a['peak_reserved_bytes']} bytes，dP1 {b['peak_allocated_bytes']}/{b['peak_reserved_bytes']} bytes。这些allocator峰值含模型、attention KV和临时FP32 replay scratch的设备影响；live inventory仅计持久tensor，不含Python对象或allocator reserved。逐轨迹库存保存在memory_accounting.json。",
              f"表中Runtime是逐题记录的生成wall合计，含prefill/采样/审计，不含该进程模型加载；dP8/dP1每题均值{a['mean_runtime_seconds']:.2f}/{b['mean_runtime_seconds']:.2f}s。资源计账另按每次own worker Popen dispatch→实际exit+NVMLrelease的wall累加，含加载、IO及非AIME资格/校准：{ledger['cumulative_GPUh']:.6f} dispatch-wall GPU-accounting hours。物理GPU active/SM时间未测（UNRESOLVED_NOT_MEASURED），不得把dispatch-wall计账称为实测GPU active时间。",
              '', '本实现基于论文及明确公开的量化约定，未核到可验证官方代码，标记UNVERIFIED_AGAINST_OFFICIAL_CODE。校准corpus为声明的8个RepoBench-P前缀，未使用AIME；其repository/project独立性UNKNOWN。',
              '两配置同一mask、相同问题及原FrozenV4协议。实际冻结历史deterministic_algorithms=False，依用户协议优先条款保留。',
              f"5. 是否支持继续研究mixed precision：{research_basis}。本次60条轨迹及数值门禁支持这套声明实现的运行可行性；本轮仅比较混合精度下的周期，不能单独确证混合精度相对FP或全INT8的收益。一次seed1、30题的小样本只提供探索性证据；若继续，应另行预注册独立数据或seed的验证，本轮不自动安排或扩展任何实验。",
              '6. 是否需要改进head importance或quantizer：本轮两组共享这两项，没有分离它们的贡献，因此无法判断应优先修改哪项或修改是否必要。逐题分歧、截断和重复可作为诊断线索；后续方法变更需独立非AIME校准和新的预注册对照，不依据本次AIME成绩现场调整。本轮封存并停止。',
              f"state 指数实际分配均值 dP8/dP1={ma['exponents']:.2f}/{mb['exponents']:.2f} bytes；无溢出的 head part 不分配指数 tensor。24 层全部 state G32 组分配时最多额外393216 bytes（384 KiB），普通混合 state 含 scale/ZP 为5.65625 bits/value，指数全部分配为5.90625 bits/value；均不含 buffer/marker/allocator/model/KV/scratch。key buffer 格式与指数字节数0保持原样。",
              '历史背景数字来源为用户给定，本轮未重新运行或独立复核，属于非同时对照：G32+Sym8 20/30；H+G32+Sym8 21/30；G32+Asym8 25/30；H+G32+Asym8 25/30。它们不纳入本轮60条轨迹、配对统计或性能比较。', '']
    return '\n'.join(lines).encode()


def archive(root):
    # The only remote mutation is a new, exact archive subtree on .29; no GPU call there.
    exclude = {'DELIVERY_MANIFEST.json', 'ARCHIVE_VERIFICATION.json', 'DELIVERY_READY.json', 'FINISHER_EXIT.json'}
    rows = []
    for p in sorted(root.rglob('*')):
        C.require(not p.is_symlink(), 'No symlink may enter permanent archive')
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in exclude or rel.endswith('.tmp') or rel == 'finisher.log':
            continue
        C.require('\n' not in rel and '\r' not in rel and '\x00' not in rel, 'Unsafe archive file name')
        rows.append(dict(path=rel, bytes=p.stat().st_size, sha256=C.sha(p),
                         download_small=not p.suffix.lower() in ('.pt', '.bin', '.safetensors', '.so', '.cubin')))
    manifest = dict(task_id=C.TASK, status='FRESH_SHA_BYTES_TERMINAL_ASSETS', files=rows,
                    count=len(rows), total_bytes=sum(x['bytes'] for x in rows),
                    source_manifest_sha256=C.sha(root / C.SOURCE), permit_sha256=C.sha(root / 'GPU_PERMIT.json'))
    C.atomic_new(root / 'DELIVERY_MANIFEST.json', manifest)
    manifest_sha = C.sha(root / 'DELIVERY_MANIFEST.json')
    final = C.read(root / 'ROOT.json')['archive_root']
    C.require(final == C.ARCHIVE, 'Exact new permanent archive path required')
    stage = final + '.staging_' + manifest_sha
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlPath=' + SSH_SOCKET, STORAGE]
    def remote(script, payload):
        command = '/usr/bin/python3 -c ' + shlex.quote(script)
        p = subprocess.run(ssh + [command], input=C.canonical(payload), capture_output=True, timeout=7200)
        C.require(p.returncode == 0, 'Archive29 CPU action failed: ' + p.stderr.decode(errors='replace'))
        return json.loads(p.stdout)
    remote("import json,os,sys;from pathlib import Path;p=json.load(sys.stdin);f=Path(p['final']);s=Path(p['stage']);assert not f.exists() and not s.exists();s.parent.mkdir(parents=True,exist_ok=True);s.mkdir();print(json.dumps({'status':'CREATED_NEW_EXCLUSIVE_STAGE'}))", dict(final=final, stage=stage))
    names = [x['path'] for x in rows] + ['DELIVERY_MANIFEST.json']
    with tempfile.NamedTemporaryFile(mode='w', prefix='pace48_archive_files_', delete=True) as listing:
        listing.write('\n'.join(names) + '\n'); listing.flush()
        p = subprocess.run(['rsync', '-rlt', '--files-from=' + listing.name, '-e',
                            'ssh -o BatchMode=yes -o ControlPath=' + SSH_SOCKET,
                            str(root) + '/', STORAGE + ':' + stage + '/'], capture_output=True, timeout=7200)
        C.require(p.returncode == 0, 'Archive copy failed: ' + p.stderr.decode(errors='replace'))
    verifier = '''import ctypes,hashlib,json,os,sys
from pathlib import Path
p=json.load(sys.stdin);s=Path(p['stage']);f=Path(p['final'])
assert not f.exists()
def digest(q):
 h=hashlib.sha256()
 with q.open('rb') as z:
  for b in iter(lambda:z.read(4194304),b''):h.update(b)
 return h.hexdigest()
assert digest(s/'DELIVERY_MANIFEST.json')==p['manifest_sha256']
m=json.loads((s/'DELIVERY_MANIFEST.json').read_text())
for x in m['files']:
 q=s/x['path'];a=q.stat();assert q.is_file() and not q.is_symlink() and a.st_size==x['bytes'] and digest(q)==x['sha256']
r={'status':'PASS_INDEPENDENT_ARCHIVE29_ALL_SHA_BYTES','manifest_sha256':p['manifest_sha256'],'files':m['count'],'bytes':m['total_bytes'],'archive_root':str(f),'GPU_calls':0}
with (s/'ARCHIVE_VERIFICATION.json').open('x') as z:json.dump(r,z,sort_keys=True);z.flush();os.fsync(z.fileno())
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rc=fn(-100,os.fsencode(s),-100,os.fsencode(f),1)
if rc:raise OSError(ctypes.get_errno(),'renameat2 NOREPLACE failed')
print(json.dumps(r,sort_keys=True))
'''
    proof = remote(verifier, dict(stage=stage, final=final, manifest_sha256=manifest_sha))
    # Verify originals again after transfer; immutable archive verification receipt is mirrored byte-for-byte.
    for x in rows:
        p = root / x['path']
        C.require(p.stat().st_size == x['bytes'] and C.sha(p) == x['sha256'], 'Source changed during archive')
    raw = subprocess.run(ssh + ['/bin/cat ' + shlex.quote(final + '/ARCHIVE_VERIFICATION.json')], capture_output=True, timeout=60)
    C.require(raw.returncode == 0 and json.loads(raw.stdout) == proof, 'Archive receipt re-read differs')
    write_bytes(root / 'ARCHIVE_VERIFICATION.json', raw.stdout)
    C.atomic_new(root / 'DELIVERY_READY.json', dict(task_id=C.TASK, status='VERIFIED_COMPLETE_OR_REAL_STOP',
                 final_status=C.read(root / 'final_verdict.json')['status'], archive_root=final,
                 manifest_sha256=manifest_sha, archive_verification_sha256=C.sha(root / 'ARCHIVE_VERIFICATION.json'),
                 no_GPU_or_model_tensor_download=True))
    return proof


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--score-pass')
    a = p.parse_args(); root = Path(a.root).resolve()
    if a.score_pass:
        score_pass(root, Path(a.score_pass)); return 0
    return finalize(root)



def completion_accounting(root,pairs):
    reuse=C.read(Path(root)/'REUSE_ADMISSION.json');current=[j['id'] for j,w,t in pairs if w['source_manifest_sha256']==C.sha(Path(root)/C.SOURCE)];inherited=[j['id'] for j,w,t in pairs if w['source_manifest_sha256']!=C.sha(Path(root)/C.SOURCE)]
    C.require(len(set(current+inherited))==len(pairs),'Duplicate completion')
    return dict(formal_total_unique=len(pairs),reused_formal_completed=len(inherited),current_formal_completed=len(current),reused_units=inherited,current_units=current,mixed_independent_executions=True,original_uninterrupted_60=False,physical_GPUs=[0,1,2,3],TP=1,prior_costs_separate=True,**method_labels())


def full_completion(root,session,pairs,invalid):
    accounting=completion_accounting(root,pairs);count=collections.Counter(j['condition'] for j,w,t in pairs);ledger=C.read(Path(root)/'RESOURCE_LEDGER.json');plan=C.read(Path(root)/'FOUR_GPU_PLAN.json');reuse=C.read(Path(root)/'REUSE_ADMISSION.json')
    intervals=ledger['intervals'];current=[x for x in intervals if x['stage']=='formal'];gates=[x for x in intervals if x['stage']=='scale_repair_gates']
    full=session['status']=='GPU_COMPLETE_FOURGPU_REUSE_PLUS_REMAINING60' and len(pairs)==60 and not invalid and count=={'R2':30,'R3':30} and accounting['reused_formal_completed']==len(reuse['reused_ids']) and accounting['current_formal_completed']==len(plan['jobs']) and len(current)==len(plan['jobs']) and len(gates)==4 and {x['gpu'] for x in gates}=={0,1,2,3} and all(x['returncode']==0 and x['released'] for x in intervals)
    full=full and {x['unit'] for x in current}=={x['unit'] for x in plan['jobs']} and {x['unit'] for x in gates}=={'SCALE_REPAIR_GATES_GPU'+str(g) for g in range(4)} and len(intervals)==len(current)+4 and all(w.get('method_variant')==METHOD_VARIANT and w.get('state_scale_format')==STATE_SCALE_FORMAT for j,w,t in pairs if w['source_manifest_sha256']==C.sha(Path(root)/C.SOURCE))
    return full,accounting,count


def finalize(root):
    root=Path(root).resolve();C.require(str(root)==C.REMOTE,'Exact fourGPU root required')
    C.atomic_new(root/'FINISHER_STARTED.json',dict(identity=C.process_identity(os.getpid()),source_manifest_sha256=C.sha(root/C.SOURCE),permit_sha256=C.sha(root/'GPU_PERMIT.json'),unix=time.time(),**method_labels()))
    try:
        release=terminal_guard(root);C.atomic_new(root/'FINAL_IDENTITY_RELEASE_VERIFICATION.json',release)
        invalid=[];session=C.read(root/'SESSION_RESULT.json');pairs=complete_rows(root,invalid);full,accounting,count=full_completion(root,session,pairs,invalid)
        C.atomic_new(root/'CONTINUATION_COMPLETION_ACCOUNTING.json',accounting)
        if full:
            for gpu in range(4):C.verify_result(root,'scale_repair_gates','SCALE_REPAIR_GATES_GPU'+str(gpu))
            env=C.worker_environment();env['CUDA_VISIBLE_DEVICES']='';outputs=[run_score_child(root,i,env) for i in (0,1)]
            C.require(outputs[0]['identity']['pid']!=outputs[1]['identity']['pid'] and outputs[0]['payload']==outputs[1]['payload'],'Two fresh FrozenV4 passes differ')
            summaries,comparison=summarize(root,outputs[0]['payload']['rows']);memory=memory_accounting(root,pairs)
            text=analysis_text(summaries,comparison,memory,C.read(root/'RESOURCE_LEDGER.json'))
            text+=('\n\n本报告保留此前合法闭合的'+str(accounting['reused_formal_completed'])+'条原payload，四卡TP1独立执行补齐'+str(accounting['current_formal_completed'])+'条，合计唯一60条；不是未中断连续60条。原来源身份、GPU与成本单独保留；四卡未改变数值公式、mask、周期、seed、生成上限或评分规则。\n').encode()
            write_bytes(root/'comparison/scientific_analysis.md',text)
            verdict=dict(status='FORMAL_COMPLETE_60_60_AND_STOP',task_id=C.TASK,formal_completed=60,R2_completed=30,R3_completed=30,FrozenV4_two_fresh_exact=True,summaries=summaries,comparison=comparison,continuation_accounting=accounting,mixed_independent_executions=True,original_uninterrupted_60=False,failure=None,**method_labels())
        else:
            verdict=dict(status='FAIL_CLOSED_STOP',task_id=C.TASK,formal_completed=len(pairs),R2_completed=count['R2'],R3_completed=count['R3'],planned_formal=60,full_accuracy_reported=False,partial_scientific_CSV_created=False,failure=session['failure'],invalid_completion_receipts=invalid,continuation_accounting=accounting,no_automatic_retry=True,**method_labels())
        C.atomic_new(root/'final_verdict.json',verdict);archive(root);C.atomic_new(root/'FINISHER_EXIT.json',dict(status='PASS_TERMINAL_DELIVERY',returncode=0,unix=time.time()));return 0
    except BaseException as e:
        if not (root/'FINISHER_FAILURE.json').exists():C.atomic_new(root/'FINISHER_FAILURE.json',dict(status='FAIL_CLOSED_DELIVERY',error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True,unix=time.time()))
        return 1

if __name__=='__main__':raise SystemExit(main())
