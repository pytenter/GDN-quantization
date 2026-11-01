#!/usr/bin/env python3
import argparse, gzip, hashlib, json, math, os, re, statistics, time
from collections import Counter, defaultdict
from pathlib import Path

TASK = 'GDN_LONG_HORIZON_FAILURE_MECHANISM_CAUSAL_AUDIT_V1'
ROOT = Path('/data/zypan')
EXP = ROOT / 'experiments/qwen35_gdn_quant'
RES = ROOT / 'results'
REP = ROOT / 'reports'
FIG = ROOT / 'figures/gdn_long_horizon_failure_mechanism_v1'
PREV_RESULT = RES / 'gdn_multi_timescale_stability_v1.json'
PREV_SCRIPT = EXP / 'run_multi_timescale_stability_validation.py'
RESULT = RES / 'gdn_long_horizon_failure_mechanism_v1.json'
RECORDS = RES / 'gdn_long_horizon_failure_mechanism_v1_records.jsonl'
METRIC_AUDIT = RES / 'gdn_long_horizon_failure_mechanism_v1_metric_audit.json'
FAILURE_ALIGNED = RES / 'gdn_long_horizon_failure_mechanism_v1_failure_aligned.jsonl.gz'
L25_HEADS = RES / 'gdn_long_horizon_failure_mechanism_v1_l25_heads.jsonl.gz'
INTERVENTIONS = RES / 'gdn_long_horizon_failure_mechanism_v1_interventions.jsonl'
RESIDUAL_PROP = RES / 'gdn_long_horizon_failure_mechanism_v1_residual_propagation.jsonl'
REPORT = REP / 'gdn_long_horizon_failure_mechanism_v1.md'
OFFSETS = [-512, -256, -128, -64, -32, -16, -8, -4, -1, 0]

def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')

def isnum(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))

def median(xs):
    xs = sorted(float(x) for x in xs if isnum(x))
    return statistics.median(xs) if xs else None

def pct(xs, q):
    xs = sorted(float(x) for x in xs if isnum(x))
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

def sha(path):
    path = Path(path)
    if not path.exists():
        return 'MISSING'
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(str(tmp), str(path))

def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))

def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n')

def write_gzip_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')

def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def iter_glob(pattern):
    for path in sorted(RES.glob(pattern)):
        yield from iter_jsonl(path)

def touch_required():
    FIG.mkdir(parents=True, exist_ok=True)
    for path in [RECORDS, INTERVENTIONS, RESIDUAL_PROP]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    for path in [FAILURE_ALIGNED, L25_HEADS]:
        if not path.exists():
            write_gzip_jsonl(path, [])

def source_hashes():
    files = [
        PREV_SCRIPT,
        EXP / 'state_fake_quant.py',
        EXP / 'run_protocol_consistency_audit.py',
        EXP / 'run_state_sensitivity_structure_audit.py',
        Path(__file__),
        PREV_RESULT,
    ]
    return {str(p): sha(p) for p in files}

def old_metric_source_audit():
    src = PREV_SCRIPT.read_text(encoding='utf-8') if PREV_SCRIPT.exists() else ''
    m = re.search(r'def state_metrics\(.*?return out', src, re.S)
    snippet = m.group(0) if m else ''
    found = 'delta-qdelta' in snippet and 'lost_update_energy_ratio' in snippet
    return {
        'source_file': str(PREV_SCRIPT),
        'source_sha256': sha(PREV_SCRIPT),
        'old_metric_definition_found': found,
        'same_grid_fraction_found': 'same_grid_swamped_fraction' in snippet,
        'old_metric_expression': (
            'sum((Delta - Delta_hat)^2) / sum(Delta^2)'
            if found else 'UNKNOWN'
        ),
        'root_cause': 'METRIC_NAMING_ERROR' if found else 'UNRESOLVED',
        'reason': (
            'Prior lost_update_energy_ratio is update reconstruction error '
            'energy ratio, not bounded same-grid lost-update fraction.'
        ),
    }

def old_metric_value_audit():
    vals = []
    same = []
    rows = 0
    gt1 = 0
    maxrow = None
    patterns = [
        'gdn_multi_timescale_stability_v1_records_shard_*.jsonl',
        'gdn_multi_timescale_stability_v1_layer_records_shard_*.jsonl.gz',
        'gdn_multi_timescale_stability_v1_token_records_shard_*.jsonl.gz',
    ]
    for pattern in patterns:
        for row in iter_glob(pattern):
            rows += 1
            v = row.get('lost_update_energy_ratio')
            if isnum(v):
                fv = float(v)
                vals.append(fv)
                if fv > 1.0:
                    gt1 += 1
                    if maxrow is None or fv > maxrow['value']:
                        maxrow = {
                            'value': fv,
                            'track': row.get('track'),
                            'prompt_id': row.get('prompt_id'),
                            'quantizer': row.get('quantizer'),
                            'checkpoint': row.get('checkpoint'),
                            'token_idx': row.get('token_idx'),
                            'layer_idx': row.get('layer_idx'),
                        }
            s = row.get('same_grid_swamped_fraction')
            if isnum(s):
                same.append(float(s))
    return {
        'rows_scanned': rows,
        'old_lost_update_ratio_count': len(vals),
        'old_lost_update_ratio_gt_1_count': gt1,
        'old_lost_update_ratio_max': max(vals) if vals else None,
        'old_lost_update_ratio_p50': median(vals),
        'old_lost_update_ratio_p95': pct(vals, 0.95),
        'old_lost_update_ratio_max_row': maxrow,
        'same_grid_swamped_fraction_count': len(same),
        'same_grid_swamped_fraction_min': min(same) if same else None,
        'same_grid_swamped_fraction_max': max(same) if same else None,
    }

def stage_metric_audit(args):
    audit = {
        'task': TASK,
        'created_at': now(),
        'stage': 'metric-audit',
        'execution_mode': 'offline_existing_records_only',
        'source_hashes': source_hashes(),
        'old_metric_source_audit': old_metric_source_audit(),
        'old_metric_value_audit': old_metric_value_audit(),
        'corrected_bounded_metric_definition': {
            'S_in': 'state before recurrent update',
            'S_pre': 'state after recurrent update before quantization',
            'Delta': 'S_pre - S_in',
            'scale': 'computed from S_pre',
            'q_in': 'Q_s(S_in)',
            'q_pre': 'Q_s(S_pre)',
            'R_lost': 'sum_{q_pre == q_in} Delta_i^2 / total_delta_energy',
            'R_element': 'count(q_pre != q_in) / N',
        },
        'corrected_bounded_metric_recomputable_from_existing_records': False,
        'missing_state_for_recompute': [
            'S_in', 'S_pre', 'q_in', 'q_pre', 'per-head integer codes'
        ],
        'corrected_lost_update_metrics': {
            'status': 'NOT_RECOMPUTABLE_OFFLINE_FROM_PRIOR_RECORDS',
            'bounded_ratio_violations': 'UNKNOWN_WITHOUT_STATE_SNAPSHOTS',
            'note': 'Prior records contain aggregates, not tensors/codes.',
        },
        'metric_gate': 'BLOCKED_FOR_CAUSAL_SWAMPING_CLAIMS',
        'gpu_runs_started': False,
        'why_not_pass': (
            'Offline-only constraint prevents new instrumented state replay; '
            'Stage 0 cannot certify corrected bounded metrics.'
        ),
    }
    write_json(METRIC_AUDIT, audit)
    append_jsonl(RECORDS, {
        'stage': 'metric-audit',
        'created_at': audit['created_at'],
        'metric_gate': audit['metric_gate'],
    })
    touch_required()
    return audit

def previous_failures():
    rows = []
    seen = set()
    pattern = 'gdn_multi_timescale_stability_v1_failure_traces_shard_*.jsonl.gz'
    for row in iter_glob(pattern):
        key = (
            row.get('track'), row.get('prompt_id'), row.get('quantizer'),
            row.get('token'), row.get('layer'), row.get('head')
        )
        if key not in seen:
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda r: (
        str(r.get('track')),
        str(r.get('quantizer')),
        int(r.get('prompt_id', -1)),
        int(r.get('token', 10**9)),
    ))
    return rows

def failure_site_aggregation(failures):
    counts = Counter(
        (r.get('layer'), r.get('head'))
        for r in failures
        if r.get('quantizer') == 'INT4-column'
    )
    total = sum(counts.values())
    focal = sum(
        n for (layer, head), n in counts.items()
        if layer == 25 and head in (14, 15)
    )
    return {
        'int4_column_failure_count': total,
        'site_counts': [
            {'layer': layer, 'head': head, 'count': n}
            for (layer, head), n in counts.most_common()
        ],
        'l25_h14_h15_count': focal,
        'l25_h14_h15_fraction': focal / total if total else None,
    }

def row_column_checkpoint_comparison(prev):
    table = {}
    for row in prev.get('aggregate_results', []):
        key = (row.get('track'), row.get('quantizer'), row.get('checkpoint'))
        table[key] = row
    checkpoints = sorted({
        key[2] for key in table
        if key[0] == 'teacher_forced'
        and key[1] in ('INT8-column', 'INT4-row', 'INT4-column')
    })
    out = []
    for ck in checkpoints:
        item = {'track': 'teacher_forced', 'checkpoint': ck}
        for quant in ['INT8-column', 'INT4-row', 'INT4-column']:
            row = table.get(('teacher_forced', quant, ck), {})
            item[quant] = {
                'prompt_count': row.get('prompt_count'),
                'KL': row.get('KL'),
                'global_divergence': row.get('global_divergence'),
                'finite_failures': row.get('finite_failures'),
                'old_lost_update_energy_ratio': row.get('lost_update_energy_ratio'),
                'same_grid_swamped_fraction': row.get('same_grid_swamped_fraction'),
            }
        out.append(item)
    return out

def token_index():
    idx = defaultdict(dict)
    pattern = 'gdn_multi_timescale_stability_v1_token_records_shard_*.jsonl.gz'
    for row in iter_glob(pattern):
        key = (row.get('track'), row.get('prompt_id'), row.get('quantizer'))
        idx[key][row.get('token_idx')] = row
    return idx

def failure_aligned_rows(failures):
    idx = token_index()
    out = []
    for fail in failures:
        if fail.get('quantizer') != 'INT4-column':
            continue
        key = (fail.get('track'), fail.get('prompt_id'), fail.get('quantizer'))
        failure_token = int(fail.get('token', 0))
        for off in OFFSETS:
            target = failure_token + off - 1 if off else failure_token - 1
            row = idx.get(key, {}).get(target)
            out.append({
                'stage': 'forensic',
                'status': 'EXACT_TOKEN_AVAILABLE' if row else 'MISSING_IN_PRIOR_TOKEN_RECORDS',
                'track': fail.get('track'),
                'prompt_id': fail.get('prompt_id'),
                'quantizer': fail.get('quantizer'),
                'failure_token': failure_token,
                'failure_layer': fail.get('layer'),
                'failure_head': fail.get('head'),
                'offset': off,
                'target_token_idx': target,
                'KL': row.get('KL') if row else None,
                'global_divergence': row.get('global_divergence') if row else None,
                'old_lost_update_energy_ratio': (
                    row.get('lost_update_energy_ratio') if row else None
                ),
                'same_grid_swamped_fraction': (
                    row.get('same_grid_swamped_fraction') if row else None
                ),
                'update_relative_error': (
                    row.get('update_relative_error') if row else None
                ),
                'update_cosine': row.get('update_cosine') if row else None,
            })
    return out

def l25_head_rows(prev):
    rows = []
    for row in prev.get('per_head_summary', []):
        if row.get('layer_idx') == 25:
            rec = dict(row)
            rec['stage'] = 'l25-existing-summary'
            rec['diagnostic_set_status'] = 'EXISTING_AGGREGATE_NOT_STAGE2_RUN'
            rows.append(rec)
    rows.sort(key=lambda r: (
        str(r.get('track')),
        str(r.get('quantizer')),
        int(r.get('head_idx', -1)),
    ))
    return rows

def rank_l25_heads(rows):
    by_quant = defaultdict(list)
    for row in rows:
        if row.get('track') == 'teacher_forced':
            by_quant[row.get('quantizer')].append(row)
    out = {}
    for quant, vals in by_quant.items():
        vals = sorted(
            vals,
            key=lambda r: r.get('lost_update_energy_ratio') or -1,
            reverse=True,
        )
        out[quant] = []
        for row in vals[:8]:
            out[quant].append({
                'head_idx': row.get('head_idx'),
                'old_lost_update_energy_ratio': row.get('lost_update_energy_ratio'),
                'same_grid_swamped_fraction': row.get('same_grid_swamped_fraction'),
                'mean_relative_error': row.get('mean_relative_error'),
                'state_norm_max': row.get('state_norm_max'),
            })
    return out

def failure_aggregation_audit(prev):
    out = []
    for row in prev.get('aggregate_results', []):
        if row.get('track') != 'teacher_forced':
            continue
        if row.get('quantizer') not in ['INT8-column', 'INT4-row', 'INT4-column']:
            continue
        out.append({
            'track': row.get('track'),
            'quantizer': row.get('quantizer'),
            'checkpoint': row.get('checkpoint'),
            'prompt_count_in_aggregate': row.get('prompt_count'),
            'finite_failures': row.get('finite_failures'),
            'KL': row.get('KL'),
            'note': (
                'Checkpoint aggregates are available-record summaries; '
                'late checkpoints may underrepresent prompts failed earlier.'
            ),
        })
    return out

def earliest_precursor(aligned):
    candidates = [
        row for row in aligned
        if row['status'] == 'EXACT_TOKEN_AVAILABLE' and isnum(row.get('KL'))
    ]
    if not candidates:
        return {
            'status': 'NOT_IDENTIFIABLE_OFFLINE',
            'reason': 'Dense failure-aligned token offsets are mostly absent.',
        }
    worst = max(candidates, key=lambda r: r.get('global_divergence') or -1)
    return {
        'status': 'PARTIAL_PRIOR_TOKEN_RECORDS',
        'strongest_available_signal': worst,
    }

def stage_forensic(args):
    prev = read_json(PREV_RESULT, {})
    fails = previous_failures()
    aligned = failure_aligned_rows(fails)
    heads = l25_head_rows(prev)
    forensic = {
        'task': TASK,
        'created_at': now(),
        'stage': 'forensic',
        'execution_mode': 'offline_existing_records_only',
        'source_hashes': source_hashes(),
        'previous_final_classification': prev.get('final_classification'),
        'failure_site_aggregation': failure_site_aggregation(fails),
        'failure_traces': fails,
        'row_column_checkpoint_comparison': row_column_checkpoint_comparison(prev),
        'failure_aggregation_audit': failure_aggregation_audit(prev),
        'failure_aligned_forensics': {
            'rows': len(aligned),
            'exact_rows': sum(
                1 for r in aligned if r['status'] == 'EXACT_TOKEN_AVAILABLE'
            ),
            'missing_rows': sum(
                1 for r in aligned if r['status'] != 'EXACT_TOKEN_AVAILABLE'
            ),
            'note': 'Prior token records are sparse.',
        },
        'earliest_precursor_signal': earliest_precursor(aligned),
        'l25_full_head_map_existing_summary': {
            'rows': len(heads),
            'status': 'EXISTING_AGGREGATE_ONLY_NOT_STAGE2_DIAGNOSTIC_RERUN',
            'ranked_by_old_lost_update_ratio': rank_l25_heads(heads),
        },
    }
    write_gzip_jsonl(FAILURE_ALIGNED, aligned)
    write_gzip_jsonl(L25_HEADS, heads)
    append_jsonl(RECORDS, {
        'stage': 'forensic',
        'created_at': forensic['created_at'],
        'failures': len(fails),
        'aligned_rows': len(aligned),
    })
    out_path = RES / 'gdn_long_horizon_failure_mechanism_v1_forensic.json'
    write_json(out_path, forensic)
    touch_required()
    return forensic

def write_report(result):
    metric = result['metric_audit']
    forensic = result['forensic']
    site = forensic.get('failure_site_aggregation', {})
    old = metric.get('old_metric_value_audit', {})
    lines = [
        '# ' + TASK,
        '',
        '- FINAL_CLASSIFICATION: `' + result['FINAL_CLASSIFICATION'] + '`',
        '- METHOD_DESIGN_READY: `NO`',
        '- MECHANISM_CLOSURE_STATUS: `' + result['MECHANISM_CLOSURE_STATUS'] + '`',
        '- METRIC_GATE: `' + result['METRIC_GATE'] + '`',
        '- OLD_LOST_UPDATE_RATIO_ROOT_CAUSE: `'
        + result['OLD_LOST_UPDATE_RATIO_ROOT_CAUSE'] + '`',
        '',
        '## Metric Audit',
        '',
        'Previous lost_update_energy_ratio is update reconstruction error ratio.',
        'It is not a bounded same-grid lost-update fraction.',
        'Old metric count: ' + str(old.get('old_lost_update_ratio_count')),
        'Old metric > 1 count: ' + str(old.get('old_lost_update_ratio_gt_1_count')),
        'Old metric max: ' + str(old.get('old_lost_update_ratio_max')),
        'Corrected bounded metrics are not recomputable from prior records.',
        '',
        '## Failure Forensics',
        '',
        'INT4-column failures: ' + str(site.get('int4_column_failure_count')),
        'L25 H14/H15 failures: ' + str(site.get('l25_h14_h15_count')),
        'L25 H14/H15 fraction: ' + str(site.get('l25_h14_h15_fraction')),
        'Failure-aligned rows: '
        + str(forensic['failure_aligned_forensics']['rows']),
        'Exact aligned rows: '
        + str(forensic['failure_aligned_forensics']['exact_rows']),
        '',
        '## Causal Status',
        '',
        'L25 H14/H15 is strong correlational evidence, not causal closure.',
        'Rescue, rounding, scale, and residual interventions were not run offline.',
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def analyze(args):
    metric = read_json(METRIC_AUDIT, None)
    if metric is None:
        metric = stage_metric_audit(args)
    forensic_path = RES / 'gdn_long_horizon_failure_mechanism_v1_forensic.json'
    forensic = read_json(forensic_path, None)
    if forensic is None:
        forensic = stage_forensic(args)
    root = metric['old_metric_source_audit']['root_cause']
    result = {
        'TASK': TASK,
        'task': TASK,
        'created_at': now(),
        'FINAL_CLASSIFICATION': 'LONG_HORIZON_MECHANISM_INCONCLUSIVE',
        'final_classification': 'LONG_HORIZON_MECHANISM_INCONCLUSIVE',
        'METHOD_DESIGN_READY': 'NO',
        'method_design_ready': 'NO',
        'MECHANISM_CLOSURE_STATUS': (
            'OFFLINE_FORENSIC_PARTIAL; '
            'GPU_CAUSAL_INTERVENTIONS_NOT_STARTED'
        ),
        'mechanism_closure_status': (
            'OFFLINE_FORENSIC_PARTIAL; '
            'GPU_CAUSAL_INTERVENTIONS_NOT_STARTED'
        ),
        'METRIC_GATE': metric['metric_gate'],
        'metric_gate': metric['metric_gate'],
        'SOURCE_HASHES': source_hashes(),
        'source_hashes': source_hashes(),
        'METRIC_AUDIT_RESULT': metric,
        'metric_audit': metric,
        'OLD_LOST_UPDATE_RATIO_ROOT_CAUSE': root,
        'old_lost_update_ratio_root_cause': root,
        'CORRECTED_LOST_UPDATE_METRICS': metric['corrected_lost_update_metrics'],
        'forensic': forensic,
        'FAILURE_AGGREGATION_AUDIT': forensic['failure_aggregation_audit'],
        'FAILURE_ALIGNED_FORENSICS': forensic['failure_aligned_forensics'],
        'EARLIEST_PRECURSOR_SIGNAL': forensic['earliest_precursor_signal'],
        'L25_FULL_HEAD_MAP': forensic['l25_full_head_map_existing_summary'],
        'H14_H15_STATUS': {
            'status': 'STRONG_CORRELATIONAL_FAILURE_SITE_NOT_CAUSALLY_CLOSED',
            'data': forensic['failure_site_aggregation'],
        },
        'PRECISION_RESCUE_RESULT': 'NOT_RUN_BLOCKED_BY_METRIC_GATE_AND_OFFLINE_CONSTRAINT',
        'ORIENTATION_RESCUE_RESULT': 'NOT_RUN_BLOCKED_BY_METRIC_GATE_AND_OFFLINE_CONSTRAINT',
        'L25_RESCUE_RESULT': 'NOT_RUN_BLOCKED_BY_METRIC_GATE_AND_OFFLINE_CONSTRAINT',
        'SUFFICIENCY_TEST_RESULT': 'NOT_RUN_BLOCKED_BY_METRIC_GATE_AND_OFFLINE_CONSTRAINT',
        'ROUNDING_INTERVENTION_RESULT': 'NOT_RUN_BLOCKED_BY_METRIC_GATE_AND_OFFLINE_CONSTRAINT',
        'STOCHASTIC_ROUNDING_SEED_STABILITY': 'NOT_RUN',
        'SCALE_DRIFT_RESULT': 'NOT_RUN; PRIOR RECORDS LACK SCALE TIME SERIES',
        'ORACLE_SCALE_RESULT': 'NOT_RUN',
        'ACTUAL_RESIDUAL_PROPAGATION': 'NOT_RUN; PRIOR RECORDS LACK RESIDUAL TENSORS',
        'EQUAL_NORM_RESIDUAL_RESULT': 'NOT_RUN',
        'GDN_INTERNAL_TERM_STATUS': 'NOT_AUDITED_IN_THIS_OFFLINE_PASS',
        'FAILURE_SURVIVAL_ANALYSIS': forensic['failure_site_aggregation'],
        'WINDOWED_KL_RESULT': forensic['row_column_checkpoint_comparison'],
        'H1_SWAMPING': (
            'NOT_SUPPORTED_AS_CAUSAL; OLD METRIC WAS MISNAMED AND '
            'CORRECTED BOUNDED METRIC IS NOT RECOMPUTABLE OFFLINE'
        ),
        'H2_L25_H14_H15_CAUSALITY': (
            'CORRELATIONAL_ONLY_UNTIL_RESCUE_OR_RESIDUAL_INTERVENTION'
        ),
        'H3_PRECISION_VS_ORIENTATION': 'NOT_TESTED_OFFLINE',
        'H4_ROUNDING_BIAS': 'NOT_TESTED_OFFLINE',
        'H5_SCALE_DRIFT': 'NOT_TESTED_OFFLINE',
        'H6_RESIDUAL_GEOMETRY': 'NOT_TESTED_OFFLINE',
        'WHAT_IS_CAUSALLY_SUPPORTED': [
            'Old lost_update_energy_ratio > 1 is caused by metric naming/definition.'
        ],
        'WHAT_IS_CORRELATIONAL_ONLY': [
            'INT4-column long-horizon failure concentration at L25 H14/H15.',
            'INT4-column short/long reversal versus INT4-row.',
        ],
        'NEGATIVE_RESULTS': [
            'No offline evidence closes swamping, precision rescue, rounding, scale drift, or residual propagation causality.'
        ],
        'WHAT_IS_NOT_SUPPORTED': [
            'A final quantization method.',
            'A causal H14/H15 rescue claim.',
            'A causal swamping claim based on prior lost_update_energy_ratio.',
        ],
        'CURRENT_STRONGEST_RESEARCH_CONCLUSION': (
            'INT4-column has robust long-horizon failure association at L25 H14/H15; '
            'the prior swamping metric is misnamed, so causal closure requires new '
            'instrumented intervention runs.'
        ),
        'RECOMMENDED_NEXT_EXPERIMENT': (
            'Run a small Stage 0 instrumented replay that records S_in, S_pre, '
            'q_in, and q_pre before any rescue experiments.'
        ),
        'EXACT_RERUN_COMMANDS': [
            'cd /data/zypan/experiments/qwen35_gdn_quant',
            'python3 run_long_horizon_failure_mechanism_audit.py --stage metric-audit --resume',
            'python3 run_long_horizon_failure_mechanism_audit.py --stage forensic --resume',
            'python3 run_long_horizon_failure_mechanism_audit.py --stage analyze --resume',
        ],
        'FILES_CREATED': [
            str(p) for p in [
                RESULT, RECORDS, METRIC_AUDIT, FAILURE_ALIGNED, L25_HEADS,
                INTERVENTIONS, RESIDUAL_PROP, REPORT,
            ]
        ],
        'FILES_MODIFIED': [str(Path(__file__))],
        'ENVIRONMENT': {
            'mode': 'offline_existing_records_only',
            'hostname': os.uname().nodename if hasattr(os, 'uname') else None,
            'python': os.popen('python3 --version').read().strip(),
        },
    }
    write_json(RESULT, result)
    write_report(result)
    append_jsonl(RECORDS, {
        'stage': 'analyze',
        'created_at': result['created_at'],
        'final_classification': result['FINAL_CLASSIFICATION'],
    })
    return result

def placeholder(stage):
    obj = {
        'task': TASK,
        'stage': stage,
        'created_at': now(),
        'status': 'NOT_RUN_BLOCKED_BY_METRIC_GATE_AND_OFFLINE_CONSTRAINT',
        'gpu_runs_started': False,
    }
    target = INTERVENTIONS if stage in ['rescue', 'rounding', 'scale'] else RESIDUAL_PROP
    append_jsonl(target, obj)
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=[
        'metric-audit', 'forensic', 'l25-map', 'rescue', 'rounding',
        'scale', 'residual', 'analyze', 'all'
    ])
    parser.add_argument('--prompt-ids', default='')
    parser.add_argument('--max-steps', type=int, default=2048)
    parser.add_argument('--interventions', default='')
    parser.add_argument('--rounding', default='')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    touch_required()
    if args.stage == 'metric-audit':
        out = stage_metric_audit(args)
    elif args.stage == 'forensic':
        out = stage_forensic(args)
    elif args.stage == 'analyze':
        out = analyze(args)
    elif args.stage == 'all':
        stage_metric_audit(args)
        stage_forensic(args)
        out = analyze(args)
    else:
        out = placeholder(args.stage)
    print(json.dumps(out, indent=2, sort_keys=True))



def l25_head_rows(prev):
    buckets = defaultdict(lambda: defaultdict(list))
    for rec in iter_glob('gdn_multi_timescale_stability_v1_records_shard_*.jsonl'):
        for row in rec.get('per_head', []):
            if row.get('layer_idx') != 25:
                continue
            key = (rec.get('track'), rec.get('quantizer'), row.get('head_idx'))
            for name in [
                'lost_update_energy_ratio',
                'same_grid_swamped_fraction',
                'mean_relative_error',
                'state_norm_max',
                'total_normalized_injection_energy',
            ]:
                val = row.get(name)
                if isnum(val):
                    buckets[key][name].append(float(val))
            buckets[key]['count'].append(1.0)
    rows = []
    for (track, quant, head), vals in sorted(buckets.items()):
        out = {
            'stage': 'l25-existing-summary',
            'diagnostic_set_status': 'EXISTING_RECORD_AGGREGATE_NOT_STAGE2_RUN',
            'track': track,
            'quantizer': quant,
            'layer_idx': 25,
            'head_idx': head,
            'record_count': int(sum(vals.get('count', []))),
        }
        for name, arr in vals.items():
            if name != 'count':
                out[name] = sum(arr) / len(arr) if arr else None
        rows.append(out)
    return rows

if __name__ == '__main__':
    main()
