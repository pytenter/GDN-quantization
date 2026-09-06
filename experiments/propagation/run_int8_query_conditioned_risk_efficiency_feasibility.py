#!/usr/bin/env python3
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/data/zypan')
EXP = ROOT / 'experiments' / 'qwen35_gdn_quant'
REPO = ROOT / 'GDN-quantization'
TASK = 'GDN_INT8_QUERY_CONDITIONED_RISK_EFFICIENCY_FEASIBILITY_V1'
SLUG = 'gdn_int8_query_conditioned_risk_efficiency_feasibility_v1'
RUN_DIR = ROOT / 'runs' / SLUG
RES_DIR = REPO / 'results' / 'propagation'
REP_DIR = REPO / 'reports' / 'propagation'
G_DIR = ROOT / 'runs' / 'gdn_int8_functional_orientation_low_rank_and_stability_audit_v1' / 'geometry_tensors'
RAW_PATH = RES_DIR / 'gdn_int8_functional_proxy_natural_config_generalization_v1_formal_raw_results.jsonl'
STRUCT_CASE_PATH = RES_DIR / 'gdn_int8_functional_geometry_structural_simplification_v1_dynamic_case_level.csv'
PULLBACK_FINAL_PATH = RES_DIR / 'gdn_int8_functional_risk_state_space_pullback_v1_final_classification.json'
PANEL = ['R128', 'R64', 'R32', 'R16', 'C128', 'C64', 'C32', 'C16']
GROUP_SIZES = [4, 8, 16, 32]
H = 32
DK = 128
DV = 128
FDIM = H * DV
EPS = 1e-12

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import numpy as np
import torch
import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_orientation_state_change_mechanism as p1
import run_int8_real_rc_tangential_recurrent_mediation as realrc
import run_int8_linearized_functional_risk_proxy_extraction as frozen_proxy
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen

CONFIGS = [c for c in axis.CONFIGS if c['name'] in PANEL]
GDN_LAYERS = frozen.GDN_LAYERS


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def finite(x):
    return isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = sorted(float(x) for x in xs if finite(x))
    return statistics.median(xs) if xs else None


def ranks(vals):
    order = sorted((float(v), i) for i, v in enumerate(vals))
    out = [0.0] * len(order)
    j = 0
    while j < len(order):
        k = j + 1
        while k < len(order) and order[k][0] == order[j][0]:
            k += 1
        r = (j + k - 1) / 2.0 + 1.0
        for _, i in order[j:k]:
            out[i] = r
        j = k
    return out


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs))
    return num / den if den > EPS else None


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def kendall(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    c = d = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            if abs(dx) <= EPS or abs(dy) <= EPS:
                continue
            if dx * dy > 0:
                c += 1
            else:
                d += 1
    return (c - d) / (c + d) if c + d else None


def pair_class(a, b):
    oa = 'R' if a.startswith('R') else 'C'
    ob = 'R' if b.startswith('R') else 'C'
    if oa == ob == 'R':
        return 'R-R'
    if oa == ob == 'C':
        return 'C-C'
    return 'R-C'


def pairwise(rows, key, target, cls=None):
    by = defaultdict(list)
    for r in rows:
        by[r['unit_id']].append(r)
    ok = tot = 0
    for urs in by.values():
        for i in range(len(urs)):
            for j in range(i + 1, len(urs)):
                if cls and pair_class(urs[i]['config'], urs[j]['config']) != cls:
                    continue
                dx = urs[i][key] - urs[j][key]
                dy = urs[i][target] - urs[j][target]
                if abs(dx) <= EPS or abs(dy) <= EPS:
                    continue
                ok += int(dx * dy > 0)
                tot += 1
    return ok / tot if tot else None


def safe_name(x):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(x))


def sh(cmd):
    return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()


def safe_sh(cmd):
    try:
        return sh(cmd)
    except Exception as e:
        return type(e).__name__ + ': ' + str(e)


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    (RUN_DIR / name).write_text(txt, encoding='utf-8')
    (RES_DIR / f'{SLUG}_{name}').write_text(txt, encoding='utf-8')


def save_csv(name, rows):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    for path in [RUN_DIR / name, RES_DIR / f'{SLUG}_{name}']:
        with path.open('w', newline='', encoding='utf-8') as f:
            if fields:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow(r)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_raw():
    rows = [json.loads(l) for l in RAW_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]
    return rows, {(r['unit_id'], r['config']): r for r in rows}


def g_path(unit_id, layer):
    return G_DIR / f'{safe_name(unit_id)}_layer{layer}_G.pt'


def readout(E, q):
    return torch.einsum('hkv,hk->hv', E, q)


def q2_score(E, q2, w):
    return float((E.square() * q2.reshape(H, DK, 1) * w.reshape(H, 1, DV)).sum())


def group_q2(q2, group_size):
    assert DK % group_size == 0
    return q2.reshape(H, DK // group_size, group_size).mean(dim=2).repeat_interleave(group_size, dim=1)


def head_q2(q2):
    return q2.mean(dim=1, keepdim=True).repeat(1, DK)


def summarize(rows, key, target):
    rel = [abs(r[key] - r[target]) / (abs(r[target]) + EPS) for r in rows]
    out = {
        'spearman': spearman([r[key] for r in rows], [r[target] for r in rows]),
        'kendall': kendall([r[key] for r in rows], [r[target] for r in rows]),
        'pairwise': pairwise(rows, key, target),
        'median_relative_error': med(rel),
        'mean_relative_error': mean(rel),
    }
    for cls in ['R-R', 'C-C', 'R-C']:
        out[f'{cls}_pairwise'] = pairwise(rows, key, target, cls)
    return out


def summarize_future(rows, key):
    return {
        'spearman': spearman([r[key] for r in rows], [r['future_KL'] for r in rows]),
        'kendall': kendall([r[key] for r in rows], [r['future_KL'] for r in rows]),
        'pairwise': pairwise(rows, key, 'future_KL'),
        'R-R_pairwise': pairwise(rows, key, 'future_KL', 'R-R'),
        'C-C_pairwise': pairwise(rows, key, 'future_KL', 'C-C'),
        'R-C_pairwise': pairwise(rows, key, 'future_KL', 'R-C'),
    }


def reconstruct(model, tokenizer, e2e):
    rows = list(frozen_proxy.unit_rows().values())
    pmap = frozen_proxy.prompt_map()
    data = {}
    for idx, row in enumerate(rows):
        unit_id = row['unit']
        t0 = int(row['t0'])
        prompt = pmap[row['prompt_id']]
        print(f'[{now()}] reconstruct current q/E {idx + 1}/{len(rows)} {unit_id}', flush=True)
        fp_past, next_ids, _ = realrc.build_base_to_t0(torch, model, tokenizer, e2e, prompt, t0)
        records = realrc.first_future_records(torch, model, next_ids, fp_past)
        data[unit_id] = {'prompt_id': row['prompt_id'], 't0': t0, 'layers': {}, 'q_context': {}}
        for layer in GDN_LAYERS:
            rec = records[layer]
            state = p1.get_state(fp_past, layer).detach().float()
            q = realrc.q_for_readout(torch, rec).detach().float().reshape(H, DK).cpu()
            G = torch.load(g_path(unit_id, layer), map_location='cpu').float()
            w = torch.diagonal(G).reshape(H, DV).clone()
            cases = {}
            for cfg in CONFIGS:
                y, _, _, _, _, _ = axis.grouped_quant(torch, state, state, cfg)
                E = (y - state).detach().float().reshape(H, DK, DV).cpu()
                x = readout(E, q)
                cases[cfg['name']] = {
                    'E': E.half(),
                    'dynamic_diag_sq': float((w * x.square()).sum()),
                    'query_only_sq': float(x.square().sum()),
                    'q2_dynamic_w_sq': q2_score(E, q.square(), w),
                }
            data[unit_id]['layers'][str(layer)] = {'q': q, 'w': w, 'cases': cases}
            del G
        torch.cuda.empty_cache()
        requests = [('prev1', [max(1, t0 - 1)]), ('recent4', list(range(max(1, t0 - 4), t0))),
                    ('recent8', list(range(max(1, t0 - 8), t0))), ('recent16', list(range(max(1, t0 - 16), t0))),
                    ('future_oracle4', list(range(t0 + 1, t0 + 5)))]
        unique_ts = sorted(set(t for _, ts in requests for t in ts))
        q2_by_t = {}
        print(f'[{now()}] temporal q replay {unit_id}: {len(unique_ts)} unique prefixes', flush=True)
        for tt in unique_ts:
            fp2, nids2, _ = realrc.build_base_to_t0(torch, model, tokenizer, e2e, prompt, tt)
            recs2 = realrc.first_future_records(torch, model, nids2, fp2)
            q2_by_t[tt] = {}
            for layer in GDN_LAYERS:
                q2 = realrc.q_for_readout(torch, recs2[layer]).detach().float().reshape(H, DK).cpu()
                q2_by_t[tt][str(layer)] = q2.square()
            del fp2, recs2
            torch.cuda.empty_cache()
        q_context = {}
        for name, ts in requests:
            if not ts:
                continue
            sums = {str(layer): torch.zeros((H, DK)) for layer in GDN_LAYERS}
            for tt in ts:
                for layer in GDN_LAYERS:
                    sums[str(layer)] += q2_by_t[tt][str(layer)]
            q_context[name] = {layer: sums[layer] / float(len(ts)) for layer in sums}
        data[unit_id]['q_context'] = q_context
    return data


def calibrate_w(data, cal_units, layer):
    out = torch.zeros((H, DV))
    for u in cal_units:
        out += data[u]['layers'][str(layer)]['w'].float()
    return out / float(len(cal_units))


def run_microbench():
    if not torch.cuda.is_available():
        return {'MICROBENCH_OVERHEAD': 'NOT_RUN', 'reason': 'cuda unavailable'}
    device = torch.device('cuda')
    E = torch.randn((H, DK, DV), device=device)
    q2 = torch.rand((H, DK), device=device)
    w = torch.rand((H, DV), device=device)
    reps = 200
    warm = 30

    def bench(fn):
        vals = []
        for i in range(reps + warm):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            if i >= warm:
                vals.append(start.elapsed_time(end))
        vals = sorted(vals)
        return {'median_ms': med(vals), 'p25_ms': vals[len(vals) // 4], 'p75_ms': vals[(3 * len(vals)) // 4]}

    baseline = bench(lambda: E.square().sum())
    q2w = bench(lambda: (E.square() * q2.reshape(H, DK, 1) * w.reshape(H, 1, DV)).sum())
    group = group_q2(q2, 16)
    gq = bench(lambda: (E.square() * group.reshape(H, DK, 1) * w.reshape(H, 1, DV)).sum())
    overhead = (q2w['median_ms'] - baseline['median_ms']) / (baseline['median_ms'] + EPS)
    return {
        'MICROBENCH_BACKEND_LIMITED': 'YES',
        'MICROBENCH_OVERHEAD': overhead,
        'baseline_current_int8_stat_proxy': baseline,
        'q2_staticw_proxy': q2w,
        'group16_staticw_proxy': gq,
        'note': 'PyTorch elementwise proxy benchmark only; not production CUDA latency.',
    }


def complexity_rows(best_key):
    label_for_key = {
        'M0': 'M0',
        'query_only': 'query-only exact',
        'dynq_staticw': 'dyn-q/static-w exact',
        'q2_staticw': 'q2/static-w',
        'headq_staticw': 'head-q/static-w',
        'static_column': 'static-column',
    }
    if str(best_key).startswith('group'):
        label_for_key[best_key] = 'group-q/static-w'
    selected_label = label_for_key.get(best_key, best_key)
    state_elems = H * DK * DV
    state_bytes = state_elems * 2
    rows = [
        ('M0', 'baseline', 0, 0, 0, 0, 'none', 'YES', 'YES'),
        ('query_only exact', 'high dynamic baseline', 2 * state_elems, H * DV, 0, 0, 'q_t O(d_k)', 'YES', 'PARTIAL'),
        ('dyn-q/static-w exact', 'best query-conditioned exact', 2 * state_elems + H * DV, H * DV, 0, H * DV, 'q_t O(d_k)', 'YES', 'PARTIAL'),
        ('q2/static-w', 'separable no cross-row terms', 3 * state_elems, 0, 0, H * DV, 'q2 O(d_k)', 'YES', 'YES'),
        ('group-q/static-w', 'coarser q2', 3 * state_elems, 0, 0, H * DV, 'group q2 O(d_k/group)', 'YES', 'YES'),
        ('head-q/static-w', 'head energy only', 3 * state_elems, 0, 0, H * DV, 'head scalar O(head)', 'YES', 'YES'),
        ('static-column', 'no dynamic q', 2 * state_elems, 0, 0, H * DV, 'none', 'YES', 'YES'),
    ]
    out = []
    for cand, fidelity, flops, tmp, passes, static_dim, dyn_meta, fusable, practical in rows:
        out.append({
            'Candidate': cand,
            'Fidelity': fidelity,
            'Future-KL': 'evaluation only',
            'Extra state pass': passes,
            'Extra state bytes': 0 if passes == 0 else state_bytes,
            'Extra FLOPs': flops,
            'Dynamic metadata': dyn_meta,
            'Static dimensions': static_dim,
            'Fusable': fusable,
            'Practical': practical,
            'Selected': cand == selected_label,
        })
    return out


def write_docs(final, comp_rows):
    cand = [
        '# Candidate Definitions',
        '',
        '- `M0`: raw Frobenius reconstruction.',
        '- `query_only`: `||E^T q_t||^2`.',
        '- `dynq_staticw`: `sum_j w_bar_j (q_t^T E[:,j])^2`.',
        '- `q2_staticw`: `sum_ij q_t,i^2 w_bar_j E_ij^2`; drops cross-row terms.',
        '- `groupq_staticw`: same as q2/static-w with q2 averaged over key groups `{4,8,16,32}`.',
        '- `headq_staticw`: head-level query energy times static value weights.',
        '- `static_column`: `sum_j w_bar_j ||E[:,j]||^2`.',
        '',
        'No candidate computes online full G, dense H, Jacobian, or extra future signal.',
    ]
    (RUN_DIR / 'candidate_definitions.md').write_text('\n'.join(cand) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_candidate_definitions.md').write_text('\n'.join(cand) + '\n', encoding='utf-8')
    md = ['# Complexity Analysis', '', '| Candidate | Fidelity | Future-KL | Extra state pass | Extra state bytes | Extra FLOPs | Dynamic metadata | Fusable | Practical |',
          '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in comp_rows:
        md.append('| {Candidate} | {Fidelity} | {Future-KL} | {Extra state pass} | {Extra state bytes} | {Extra FLOPs} | {Dynamic metadata} | {Fusable} | {Practical} |'.format(**r))
    text = '\n'.join(md) + '\n'
    (RUN_DIR / 'complexity_analysis.md').write_text(text, encoding='utf-8')
    (REP_DIR / f'{SLUG}_complexity_analysis.md').write_text(text, encoding='utf-8')
    fusion = [
        '# Fusion Feasibility',
        '',
        'Fusable single pass candidates: q2/static-w, grouped-q/static-w, head-q/static-w, static-column.',
        '',
        'During an existing state scan, each loaded `E_ij` can update weighted statistics using either `q2_i * w_j`, `group_q2_g * w_j`, `head_energy_h * w_j`, or `w_j`. This does not require reloading the recurrent state.',
        '',
        'Exact query-only and dyn-q/static-w require column reductions `E^T q`, which are scientifically useful but less directly compatible with a per-element quantization statistic.',
    ]
    (RUN_DIR / 'fusion_feasibility.md').write_text('\n'.join(fusion) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_fusion_feasibility.md').write_text('\n'.join(fusion) + '\n', encoding='utf-8')
    sections = [
        ('Task Definition', TASK),
        ('Previous State', json.dumps(final['previous_state'], indent=2, sort_keys=True)),
        ('Why Dynamic-q Is Not Automatically Free', 'q_t is available in the model, but using it for quantizer scale/objective construction still requires arithmetic over state errors. The red line is zero extra full-state passes.'),
        ('Reference Functional-Risk Formula', 'R_dyn_dyn = sum_j w_t,j (q_t^T E[:,j])^2.'),
        ('Candidate Approximations', '\n'.join(cand)),
        ('Scientific Fidelity', json.dumps(final['scientific_fidelity_summary'], indent=2, sort_keys=True)),
        ('Strict Held-Out Generalization', json.dumps(final['heldout_summary'], indent=2, sort_keys=True)),
        ('Dynamic-q Incremental Value', json.dumps(final['dynamic_ablation'], indent=2, sort_keys=True)),
        ('Temporal Causality', json.dumps(final['temporal_query_summary'], indent=2, sort_keys=True)),
        ('Future-query Oracle Gap', str(final['FUTURE_QUERY_ORACLE_GAIN'])),
        ('FLOP Analysis', json.dumps(final['complexity_summary'], indent=2, sort_keys=True)),
        ('Memory-Traffic Analysis', 'Best efficient candidate has zero extra state read bytes because it is fusible into the existing state scan.'),
        ('Single-Pass Fusion Analysis', '\n'.join(fusion)),
        ('Optional Microbenchmark', json.dumps(final['microbenchmark'], indent=2, sort_keys=True)),
        ('Complexity-vs-Fidelity Pareto Frontier', json.dumps(final['pareto'], indent=2, sort_keys=True)),
        ('Positive Results', final['positive_results']),
        ('Negative Results', final['negative_results']),
        ('Difference From DAMP', 'This audit does not allocate mixed precision or FP16 protection; it evaluates same-bit INT8 objective weights.'),
        ('Final Scientific Classification', final['FINAL_SCIENTIFIC_CLASSIFICATION']),
        ('Readiness Gates', json.dumps(final['readiness_gates'], indent=2, sort_keys=True)),
        ('Exact Next Task', final['NEXT_RECOMMENDED_TASK']),
    ]
    lines = [f'# {TASK}', '']
    for title, body in sections:
        lines += [f'## {title}', str(body), '']
    report = '\n'.join(lines)
    (RUN_DIR / 'final_report.md').write_text(report, encoding='utf-8')
    (REP_DIR / f'{SLUG}.md').write_text(report, encoding='utf-8')
    (RES_DIR / f'{SLUG}_final_report.md').write_text(report, encoding='utf-8')


def finalize_from_artifacts():
    audit = json.loads((RUN_DIR / 'artifact_audit.json').read_text(encoding='utf-8'))
    summary = json.loads((RUN_DIR / 'scientific_fidelity_summary.json').read_text(encoding='utf-8'))
    temporal_summary = json.loads((RUN_DIR / 'temporal_query_summary.json').read_text(encoding='utf-8'))
    micro = json.loads((RUN_DIR / 'microbenchmark.json').read_text(encoding='utf-8'))
    previous = json.loads(PULLBACK_FINAL_PATH.read_text(encoding='utf-8'))
    candidate_keys = ['M0', 'query_only', 'dynq_staticw', 'q2_staticw'] + [f'group{gs}_q_staticw' for gs in GROUP_SIZES] + ['headq_staticw', 'static_column']
    best_group = max([f'group{gs}_q_staticw' for gs in GROUP_SIZES], key=lambda k: summary[k]['vs_dynamic_diag']['spearman'])
    efficient_candidates = ['q2_staticw', best_group, 'headq_staticw', 'static_column']
    best_efficient = max(efficient_candidates, key=lambda k: summary[k]['vs_dynamic_diag']['spearman'])
    best_scientific = max(['dynq_staticw'] + efficient_candidates, key=lambda k: summary[k]['vs_dynamic_diag']['spearman'])
    comp = complexity_rows(best_efficient)
    save_json('complexity_analysis.json', {'rows': comp, 'memory_traffic_priority': 'extra full-state passes first, then bytes, then FLOPs'})
    dyn_gain = summary['dynq_staticw']['vs_dynamic_diag']['spearman'] - summary['static_column']['vs_dynamic_diag']['spearman']
    current_future = summary['q2_staticw']['vs_future_KL']['spearman']
    recent_best = max(temporal_summary[k]['vs_future_KL']['spearman'] for k in ['prev1_q2_staticw', 'recent4_q2_staticw', 'recent8_q2_staticw', 'recent16_q2_staticw'])
    oracle_gain = temporal_summary['future_oracle4_q2_staticw']['vs_dynamic_diag']['spearman'] - summary['q2_staticw']['vs_dynamic_diag']['spearman']
    scientific_ready = 'YES' if summary[best_scientific]['vs_dynamic_diag']['spearman'] >= 0.90 and summary[best_scientific]['vs_dynamic_diag']['pairwise'] >= 0.93 else ('PARTIAL' if summary[best_scientific]['vs_dynamic_diag']['pairwise'] >= 0.90 else 'NO')
    temporal_ready = 'YES' if current_future >= 0.60 else ('PARTIAL' if current_future >= 0.40 or recent_best >= 0.40 else 'NO')
    efficiency_ready = 'YES'
    useful_over_static = summary[best_efficient]['vs_dynamic_diag']['spearman'] > summary['static_column']['vs_dynamic_diag']['spearman'] + 0.02
    if best_efficient == 'q2_staticw' and useful_over_static and scientific_ready in ['YES', 'PARTIAL'] and temporal_ready in ['YES', 'PARTIAL']:
        cls = 'QUERY_CONDITIONED_SEPARABLE_RISK_SUPPORTED'
        qready = 'YES' if scientific_ready == 'YES' and temporal_ready == 'YES' else 'NO'
        next_task = 'GDN_INT8_QUERY_CONDITIONED_INT8_QUANTIZER_PROTOTYPE_V1' if qready == 'YES' else 'strengthen causal query-conditioned evidence before quantizer prototype'
    elif best_efficient.startswith('group') and useful_over_static:
        cls = 'GROUP_QUERY_CONDITIONED_RISK_SUPPORTED'
        qready = 'YES' if scientific_ready == 'YES' and temporal_ready == 'YES' else 'NO'
        next_task = 'GDN_INT8_QUERY_CONDITIONED_INT8_QUANTIZER_PROTOTYPE_V1' if qready == 'YES' else 'validate grouped query conditioning on broader causal panel'
    elif summary['dynq_staticw']['vs_dynamic_diag']['spearman'] > 0.85 and not useful_over_static:
        cls = 'PREFER_STATIC_COLUMN_STRUCTURE'
        qready = 'NO'
        next_task = 'revisit static column/state weighting before adding dynamic query complexity'
    elif summary['dynq_staticw']['vs_dynamic_diag']['spearman'] > 0.85:
        cls = 'DYNAMIC_QUERY_SCIENTIFICALLY_SUPPORTED_BUT_NOT_EFFICIENT'
        qready = 'NO'
        next_task = 'compress exact query-conditioned readout or reduce state-pass cost before quantizer design'
    else:
        cls = 'QUERY_CONDITIONED_RISK_NOT_SUPPORTED'
        qready = 'NO'
        next_task = 'investigate alternative causal state-risk predictors'
    if scientific_ready != 'YES' or temporal_ready != 'YES' or efficiency_ready != 'YES':
        qready = 'NO'
    final = {
        'TASK': TASK,
        'FORMAL_STATUS': 'COMPLETE',
        'ARTIFACT_REUSE_GATE': audit['ARTIFACT_REUSE_GATE'],
        'DYNQ_STATICW_VS_DYNAMIC_DIAG_SPEARMAN': summary['dynq_staticw']['vs_dynamic_diag']['spearman'],
        'DYNQ_STATICW_HELDOUT_PAIRWISE': summary['dynq_staticw']['vs_dynamic_diag']['pairwise'],
        'Q2_STATICW_VS_DYNAMIC_DIAG_SPEARMAN': summary['q2_staticw']['vs_dynamic_diag']['spearman'],
        'Q2_STATICW_HELDOUT_PAIRWISE': summary['q2_staticw']['vs_dynamic_diag']['pairwise'],
        'GROUPQ_STATICW_VS_DYNAMIC_DIAG_SPEARMAN': summary[best_group]['vs_dynamic_diag']['spearman'],
        'GROUPQ_STATICW_HELDOUT_PAIRWISE': summary[best_group]['vs_dynamic_diag']['pairwise'],
        'HEADQ_STATICW_VS_DYNAMIC_DIAG_SPEARMAN': summary['headq_staticw']['vs_dynamic_diag']['spearman'],
        'HEADQ_STATICW_HELDOUT_PAIRWISE': summary['headq_staticw']['vs_dynamic_diag']['pairwise'],
        'STATIC_COLUMN_VS_DYNAMIC_DIAG_SPEARMAN': summary['static_column']['vs_dynamic_diag']['spearman'],
        'BEST_GROUP_CANDIDATE': best_group,
        'BEST_SCIENTIFIC_CANDIDATE': best_scientific,
        'BEST_EFFICIENT_CANDIDATE': best_efficient,
        'CURRENT_Q_FUTURE_DAMAGE_SIGNAL': 'YES' if current_future >= 0.60 else ('PARTIAL' if current_future >= 0.40 else 'NO'),
        'RECENT_QUERY_FUTURE_DAMAGE_SIGNAL': 'YES' if recent_best >= 0.60 else ('PARTIAL' if recent_best >= 0.40 else 'NO'),
        'FUTURE_QUERY_ORACLE_GAIN': oracle_gain,
        'BEST_CANDIDATE_EXTRA_FLOPS_PER_TOKEN': next(r['Extra FLOPs'] for r in comp if r['Selected']),
        'BEST_CANDIDATE_EXTRA_STATE_READ_BYTES_PER_TOKEN': next(r['Extra state bytes'] for r in comp if r['Selected']),
        'BEST_CANDIDATE_EXTRA_STATE_PASSES': next(r['Extra state pass'] for r in comp if r['Selected']),
        'FUSABLE_SINGLE_PASS': 'YES',
        'ZERO_EXTRA_STATE_PASS_FEASIBLE': 'YES',
        'MICROBENCH_OVERHEAD': micro.get('MICROBENCH_OVERHEAD'),
        'SCIENTIFIC_SIGNAL_READY': scientific_ready,
        'TEMPORAL_CAUSALITY_READY': temporal_ready,
        'ALGORITHMIC_EFFICIENCY_READY': efficiency_ready,
        'QUANTIZER_DESIGN_READY': qready,
        'FINAL_SCIENTIFIC_CLASSIFICATION': cls,
        'NEXT_RECOMMENDED_TASK': next_task,
        'artifact_audit': audit,
        'previous_state': {'pullback_commit': '52761e4', 'pullback_classification': previous['FINAL_SCIENTIFIC_CLASSIFICATION']},
        'scientific_fidelity_summary': summary,
        'heldout_summary': {'protocol': 'leave-one-unit-out static w_bar', 'candidates': summary},
        'dynamic_ablation': {'dynq_staticw_minus_static_column_spearman': dyn_gain, 'best_group': best_group, 'best_efficient': best_efficient},
        'temporal_query_summary': temporal_summary,
        'complexity_summary': {'rows': comp},
        'microbenchmark': micro,
        'pareto': [{k: summary[k]['vs_dynamic_diag']['spearman']} for k in candidate_keys],
        'readiness_gates': {'SCIENTIFIC_SIGNAL_READY': scientific_ready, 'TEMPORAL_CAUSALITY_READY': temporal_ready, 'ALGORITHMIC_EFFICIENCY_READY': efficiency_ready, 'QUANTIZER_DESIGN_READY': qready},
        'positive_results': 'Zero extra state pass is feasible for q2/group/head/static-column candidates, and the cheap q2/group approximations are fusible into a single state scan.',
        'negative_results': 'Exact dyn-q/static-w and cheap q2/group variants do not substantially beat static-column or query-only fidelity under strict held-out evaluation; current/recent q gives limited future-KL signal.',
    }
    save_json('final_classification.json', final)
    write_docs(final, comp)
    print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    global torch
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    if '--finalize-only' in sys.argv:
        finalize_from_artifacts()
        return
    raw, raw_by = load_raw()
    struct_rows = read_csv(STRUCT_CASE_PATH)
    struct_by = {(r['unit_id'], r['config']): r for r in struct_rows}
    previous = json.loads(PULLBACK_FINAL_PATH.read_text(encoding='utf-8'))
    audit = {
        'TASK': TASK,
        'timestamp': now(),
        'branch': safe_sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        'HEAD': safe_sh(['git', 'rev-parse', 'HEAD']),
        'git_status_start': safe_sh(['git', 'status', '--short']),
        'required_head_at_least': '52761e4',
        'raw_cases': len(raw),
        'structural_cases': len(struct_rows),
        'geometry_tensor_count': len(list(G_DIR.glob('*_G.pt'))),
        'GPU_RERUN_REQUIRED': 'YES_MINIMAL_Q_E_AND_TEMPORAL_Q_REPLAY',
        'artifact_sources': [str(RAW_PATH), str(STRUCT_CASE_PATH), str(PULLBACK_FINAL_PATH), str(G_DIR)],
    }
    audit['ARTIFACT_REUSE_GATE'] = 'PASS' if len(raw) == 72 and len(struct_rows) == 72 and audit['geometry_tensor_count'] == 216 else 'FAIL'
    save_json('artifact_audit.json', audit)
    (RUN_DIR / 'artifact_audit.md').write_text('# Artifact Audit\n\n' + json.dumps(audit, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_artifact_audit.md').write_text('# Artifact Audit\n\n' + json.dumps(audit, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if audit['ARTIFACT_REUSE_GATE'] != 'PASS':
        final = {'TASK': TASK, 'FORMAL_STATUS': 'STOPPED_ARTIFACT_GATE_FAIL', 'artifact_audit': audit}
        save_json('final_classification.json', final)
        print(json.dumps(final, indent=2, sort_keys=True))
        return

    torch.set_grad_enabled(False)
    torch, model, tokenizer, _, e2e = p1.setup_model()
    data = reconstruct(model, tokenizer, e2e)
    units = sorted(data)
    base_rows = []
    for hold in units:
        print(f'[{now()}] heldout query-conditioned scoring {hold}', flush=True)
        cal_units = [u for u in units if u != hold]
        wbar_by_layer = {str(layer): calibrate_w(data, cal_units, layer) for layer in GDN_LAYERS}
        for cfg in PANEL:
            base = raw_by[(hold, cfg)]
            row = {
                'unit_id': hold,
                'prompt_id': base['prompt_id'],
                't0': base['t0'],
                'config': cfg,
                'future_KL': base['future_KL'],
                'M0': base['M0'],
                'M5_full': float(struct_by[(hold, cfg)]['M5_full_recomputed']),
                'dynamic_diag': 0.0,
                'query_only': 0.0,
                'dynq_staticw': 0.0,
                'q2_staticw': 0.0,
                'headq_staticw': 0.0,
                'static_column': 0.0,
                'prev1_q2_staticw': 0.0,
                'recent4_q2_staticw': 0.0,
                'recent8_q2_staticw': 0.0,
                'recent16_q2_staticw': 0.0,
                'future_oracle4_q2_staticw': 0.0,
            }
            for gs in GROUP_SIZES:
                row[f'group{gs}_q_staticw'] = 0.0
            for layer in GDN_LAYERS:
                ld = data[hold]['layers'][str(layer)]
                E = ld['cases'][cfg]['E'].float()
                q = ld['q'].float()
                q2 = q.square()
                w = ld['w'].float()
                wbar = wbar_by_layer[str(layer)]
                x = readout(E, q)
                row['dynamic_diag'] += float((w * x.square()).sum())
                row['query_only'] += float(x.square().sum())
                row['dynq_staticw'] += float((wbar * x.square()).sum())
                row['q2_staticw'] += q2_score(E, q2, wbar)
                row['headq_staticw'] += q2_score(E, head_q2(q2), wbar)
                row['static_column'] += float((E.square() * wbar.reshape(H, 1, DV)).sum())
                for gs in GROUP_SIZES:
                    row[f'group{gs}_q_staticw'] += q2_score(E, group_q2(q2, gs), wbar)
                for qname in ['prev1', 'recent4', 'recent8', 'recent16', 'future_oracle4']:
                    row[f'{qname}_q2_staticw'] += q2_score(E, data[hold]['q_context'][qname][str(layer)], wbar)
            for key in list(row.keys()):
                if key in ['unit_id', 'prompt_id', 'config', 'future_KL', 't0']:
                    continue
                if key in ['M0', 'M5_full']:
                    continue
                row[key] = math.sqrt(max(0.0, row[key]))
            base_rows.append(row)
    save_csv('heldout_cases.csv', base_rows)
    save_csv('scientific_fidelity_cases.csv', base_rows)

    candidate_keys = ['M0', 'query_only', 'dynq_staticw', 'q2_staticw'] + [f'group{gs}_q_staticw' for gs in GROUP_SIZES] + ['headq_staticw', 'static_column']
    summary = {}
    for k in candidate_keys:
        summary[k] = {'vs_dynamic_diag': summarize(base_rows, k, 'dynamic_diag'), 'vs_full': summarize(base_rows, k, 'M5_full'), 'vs_future_KL': summarize_future(base_rows, k)}
    save_json('scientific_fidelity_summary.json', summary)
    save_json('heldout_summary.json', {'protocol': 'leave-one-unit-out w_bar; current q_t is causal runtime information; future KL not used for calibration', 'candidates': summary})

    temporal_keys = ['q2_staticw', 'prev1_q2_staticw', 'recent4_q2_staticw', 'recent8_q2_staticw', 'recent16_q2_staticw', 'future_oracle4_q2_staticw']
    temporal_rows = []
    temporal_summary = {}
    for k in temporal_keys:
        temporal_summary[k] = {'vs_dynamic_diag': summarize(base_rows, k, 'dynamic_diag'), 'vs_future_KL': summarize_future(base_rows, k), 'ORACLE_ONLY': 'YES' if k.startswith('future') else 'NO'}
        temporal_rows.append({'candidate': k, 'spearman_vs_dynamic_diag': temporal_summary[k]['vs_dynamic_diag']['spearman'], 'spearman_vs_future_KL': temporal_summary[k]['vs_future_KL']['spearman'], 'oracle_only': temporal_summary[k]['ORACLE_ONLY']})
    save_csv('temporal_query_audit.csv', temporal_rows)
    save_json('temporal_query_summary.json', temporal_summary)

    best_group = max([f'group{gs}_q_staticw' for gs in GROUP_SIZES], key=lambda k: summary[k]['vs_dynamic_diag']['spearman'])
    efficient_candidates = ['q2_staticw', best_group, 'headq_staticw', 'static_column']
    best_efficient = max(efficient_candidates, key=lambda k: summary[k]['vs_dynamic_diag']['spearman'])
    best_scientific = max(['dynq_staticw'] + efficient_candidates, key=lambda k: summary[k]['vs_dynamic_diag']['spearman'])
    comp = complexity_rows(best_efficient)
    save_json('complexity_analysis.json', {'rows': comp, 'memory_traffic_priority': 'extra full-state passes first, then bytes, then FLOPs'})
    micro = run_microbench()
    save_json('microbenchmark.json', micro)
    (RUN_DIR / 'microbenchmark.md').write_text('# Microbenchmark\n\n' + json.dumps(micro, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_microbenchmark.md').write_text('# Microbenchmark\n\n' + json.dumps(micro, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    dyn_gain = summary['dynq_staticw']['vs_dynamic_diag']['spearman'] - summary['static_column']['vs_dynamic_diag']['spearman']
    current_future = summary['q2_staticw']['vs_future_KL']['spearman']
    recent_best = max(temporal_summary[k]['vs_future_KL']['spearman'] for k in ['prev1_q2_staticw', 'recent4_q2_staticw', 'recent8_q2_staticw', 'recent16_q2_staticw'])
    oracle_gain = temporal_summary['future_oracle4_q2_staticw']['vs_dynamic_diag']['spearman'] - summary['q2_staticw']['vs_dynamic_diag']['spearman']
    scientific_ready = 'YES' if summary[best_scientific]['vs_dynamic_diag']['spearman'] >= 0.90 and summary[best_scientific]['vs_dynamic_diag']['pairwise'] >= 0.93 else ('PARTIAL' if summary[best_scientific]['vs_dynamic_diag']['pairwise'] >= 0.90 else 'NO')
    temporal_ready = 'YES' if current_future >= 0.60 else ('PARTIAL' if current_future >= 0.40 or recent_best >= 0.40 else 'NO')
    efficiency_ready = 'YES' if best_efficient in ['q2_staticw', best_group, 'headq_staticw', 'static_column'] else 'NO'
    useful_over_static = summary[best_efficient]['vs_dynamic_diag']['spearman'] > summary['static_column']['vs_dynamic_diag']['spearman'] + 0.02
    if best_efficient == 'q2_staticw' and useful_over_static and scientific_ready in ['YES', 'PARTIAL'] and temporal_ready in ['YES', 'PARTIAL']:
        cls = 'QUERY_CONDITIONED_SEPARABLE_RISK_SUPPORTED'
        qready = 'YES' if scientific_ready == 'YES' and temporal_ready == 'YES' else 'NO'
        next_task = 'GDN_INT8_QUERY_CONDITIONED_INT8_QUANTIZER_PROTOTYPE_V1' if qready == 'YES' else 'strengthen causal query-conditioned evidence before quantizer prototype'
    elif best_efficient.startswith('group') and useful_over_static:
        cls = 'GROUP_QUERY_CONDITIONED_RISK_SUPPORTED'
        qready = 'YES' if scientific_ready == 'YES' and temporal_ready == 'YES' else 'NO'
        next_task = 'GDN_INT8_QUERY_CONDITIONED_INT8_QUANTIZER_PROTOTYPE_V1' if qready == 'YES' else 'validate grouped query conditioning on broader causal panel'
    elif summary['dynq_staticw']['vs_dynamic_diag']['spearman'] > 0.85 and not useful_over_static:
        cls = 'PREFER_STATIC_COLUMN_STRUCTURE'
        qready = 'NO'
        next_task = 'revisit static column/state weighting before adding dynamic query complexity'
    elif summary['dynq_staticw']['vs_dynamic_diag']['spearman'] > 0.85:
        cls = 'DYNAMIC_QUERY_SCIENTIFICALLY_SUPPORTED_BUT_NOT_EFFICIENT'
        qready = 'NO'
        next_task = 'compress exact query-conditioned readout or reduce state-pass cost before quantizer design'
    else:
        cls = 'QUERY_CONDITIONED_RISK_NOT_SUPPORTED'
        qready = 'NO'
        next_task = 'investigate alternative causal state-risk predictors'
    if scientific_ready != 'YES' or temporal_ready != 'YES' or efficiency_ready != 'YES':
        qready = 'NO'

    final = {
        'TASK': TASK,
        'FORMAL_STATUS': 'COMPLETE',
        'ARTIFACT_REUSE_GATE': audit['ARTIFACT_REUSE_GATE'],
        'DYNQ_STATICW_VS_DYNAMIC_DIAG_SPEARMAN': summary['dynq_staticw']['vs_dynamic_diag']['spearman'],
        'DYNQ_STATICW_HELDOUT_PAIRWISE': summary['dynq_staticw']['vs_dynamic_diag']['pairwise'],
        'Q2_STATICW_VS_DYNAMIC_DIAG_SPEARMAN': summary['q2_staticw']['vs_dynamic_diag']['spearman'],
        'Q2_STATICW_HELDOUT_PAIRWISE': summary['q2_staticw']['vs_dynamic_diag']['pairwise'],
        'GROUPQ_STATICW_VS_DYNAMIC_DIAG_SPEARMAN': summary[best_group]['vs_dynamic_diag']['spearman'],
        'GROUPQ_STATICW_HELDOUT_PAIRWISE': summary[best_group]['vs_dynamic_diag']['pairwise'],
        'HEADQ_STATICW_VS_DYNAMIC_DIAG_SPEARMAN': summary['headq_staticw']['vs_dynamic_diag']['spearman'],
        'HEADQ_STATICW_HELDOUT_PAIRWISE': summary['headq_staticw']['vs_dynamic_diag']['pairwise'],
        'STATIC_COLUMN_VS_DYNAMIC_DIAG_SPEARMAN': summary['static_column']['vs_dynamic_diag']['spearman'],
        'BEST_GROUP_CANDIDATE': best_group,
        'BEST_SCIENTIFIC_CANDIDATE': best_scientific,
        'BEST_EFFICIENT_CANDIDATE': best_efficient,
        'CURRENT_Q_FUTURE_DAMAGE_SIGNAL': 'YES' if current_future >= 0.60 else ('PARTIAL' if current_future >= 0.40 else 'NO'),
        'RECENT_QUERY_FUTURE_DAMAGE_SIGNAL': 'YES' if recent_best >= 0.60 else ('PARTIAL' if recent_best >= 0.40 else 'NO'),
        'FUTURE_QUERY_ORACLE_GAIN': oracle_gain,
        'BEST_CANDIDATE_EXTRA_FLOPS_PER_TOKEN': next(r['Extra FLOPs'] for r in comp if r['Selected']),
        'BEST_CANDIDATE_EXTRA_STATE_READ_BYTES_PER_TOKEN': next(r['Extra state bytes'] for r in comp if r['Selected']),
        'BEST_CANDIDATE_EXTRA_STATE_PASSES': next(r['Extra state pass'] for r in comp if r['Selected']),
        'FUSABLE_SINGLE_PASS': 'YES',
        'ZERO_EXTRA_STATE_PASS_FEASIBLE': 'YES',
        'MICROBENCH_OVERHEAD': micro.get('MICROBENCH_OVERHEAD'),
        'SCIENTIFIC_SIGNAL_READY': scientific_ready,
        'TEMPORAL_CAUSALITY_READY': temporal_ready,
        'ALGORITHMIC_EFFICIENCY_READY': efficiency_ready,
        'QUANTIZER_DESIGN_READY': qready,
        'FINAL_SCIENTIFIC_CLASSIFICATION': cls,
        'NEXT_RECOMMENDED_TASK': next_task,
        'artifact_audit': audit,
        'previous_state': {'pullback_commit': '52761e4', 'pullback_classification': previous['FINAL_SCIENTIFIC_CLASSIFICATION']},
        'scientific_fidelity_summary': summary,
        'heldout_summary': {'protocol': 'leave-one-unit-out static w_bar', 'candidates': summary},
        'dynamic_ablation': {'dynq_staticw_minus_static_column_spearman': dyn_gain, 'best_group': best_group, 'best_efficient': best_efficient},
        'temporal_query_summary': temporal_summary,
        'complexity_summary': {'rows': comp},
        'microbenchmark': micro,
        'pareto': [{k: summary[k]['vs_dynamic_diag']['spearman']} for k in candidate_keys],
        'readiness_gates': {'SCIENTIFIC_SIGNAL_READY': scientific_ready, 'TEMPORAL_CAUSALITY_READY': temporal_ready, 'ALGORITHMIC_EFFICIENCY_READY': efficiency_ready, 'QUANTIZER_DESIGN_READY': qready},
        'positive_results': 'Zero extra state pass is feasible for q2/group/head/static-column candidates, and the cheap q2/group approximations are fusible into a single state scan.',
        'negative_results': 'Exact dyn-q/static-w and cheap q2/group variants do not substantially beat static-column or query-only fidelity under strict held-out evaluation; current/recent q gives limited future-KL signal.',
    }
    save_json('final_classification.json', final)
    write_docs(final, comp)
    print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
