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
TASK = 'GDN_INT8_FUNCTIONAL_RISK_STATE_SPACE_PULLBACK_V1'
SLUG = 'gdn_int8_functional_risk_state_space_pullback_v1'
RUN_DIR = ROOT / 'runs' / SLUG
RES_DIR = REPO / 'results' / 'propagation'
REP_DIR = REPO / 'reports' / 'propagation'
G_DIR = ROOT / 'runs' / 'gdn_int8_functional_orientation_low_rank_and_stability_audit_v1' / 'geometry_tensors'
RAW_PATH = RES_DIR / 'gdn_int8_functional_proxy_natural_config_generalization_v1_formal_raw_results.jsonl'
STRUCT_PATH = RES_DIR / 'gdn_int8_functional_geometry_structural_simplification_v1_dynamic_case_level.csv'
STRUCT_FINAL_PATH = RES_DIR / 'gdn_int8_functional_geometry_structural_simplification_v1_final_classification.json'
MATCHED_PATH = RES_DIR / 'gdn_int8_functional_risk_beyond_reconstruction_controlled_v1_matched_pair_results.json'
PANEL = ['R128', 'R64', 'R32', 'R16', 'C128', 'C64', 'C32', 'C16']
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


def safe_name(x):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(x))


def sh(cmd):
    return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()


def safe_sh(cmd):
    try:
        return sh(cmd)
    except Exception as e:
        return type(e).__name__ + ': ' + str(e)


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


def pairwise_agreement(rows, key, target, cls=None):
    by = defaultdict(list)
    for r in rows:
        by[r['unit_id']].append(r)
    ok = tot = 0
    for urs in by.values():
        for i in range(len(urs)):
            for j in range(i + 1, len(urs)):
                if cls is not None and pair_class(urs[i]['config'], urs[j]['config']) != cls:
                    continue
                dx = urs[i][key] - urs[j][key]
                dy = urs[i][target] - urs[j][target]
                if abs(dx) <= EPS or abs(dy) <= EPS:
                    continue
                ok += int(dx * dy > 0)
                tot += 1
    return {'agreement': ok / tot if tot else None, 'correct': ok, 'total': tot}


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


def load_matched():
    return json.loads(MATCHED_PATH.read_text(encoding='utf-8'))['rows']


def g_path(unit_id, layer):
    return G_DIR / f'{safe_name(unit_id)}_layer{layer}_G.pt'


def vec_to_hv(x):
    return x.reshape(H, DV)


def score_dynamic_from_E(E, q, w):
    x = torch.einsum('hkv,hk->hv', E, q)
    query = float(x.square().sum())
    diag = float((w * x.square()).sum())
    return x, max(0.0, query), max(0.0, diag)


def score_static_cov(E, wbar, cq):
    s = 0.0
    for h in range(H):
        CE = cq[h].matmul(E[h])
        s += float((wbar[h].reshape(1, DV) * E[h] * CE).sum())
    return max(0.0, s)


def score_static_element(E, I):
    return max(0.0, float((I * E.square()).sum()))


def svd_rank1_nonnegative(I):
    U, S, Vh = torch.linalg.svd(I, full_matrices=False)
    root = math.sqrt(max(float(S[0]), 0.0))
    a = U[:, 0] * root
    b = Vh[0, :] * root
    if float(a.mean()) < 0:
        a = -a
        b = -b
    a = a.clamp_min(0)
    b = b.clamp_min(0)
    if float(a.sum()) <= EPS or float(b.sum()) <= EPS:
        a = I.mean(dim=1).clamp_min(0)
        b = I.mean(dim=0).clamp_min(0)
    return a, b


def summarize_candidate(rows, key, target):
    rel = [abs(r[key] - r[target]) / (abs(r[target]) + EPS) for r in rows]
    out = {
        'spearman': spearman([r[key] for r in rows], [r[target] for r in rows]),
        'kendall': kendall([r[key] for r in rows], [r[target] for r in rows]),
        'pairwise': pairwise_agreement(rows, key, target)['agreement'],
        'median_relative_error': med(rel),
        'mean_relative_error': mean(rel),
    }
    for cls in ['R-R', 'C-C', 'R-C']:
        out[f'{cls}_pairwise'] = pairwise_agreement(rows, key, target, cls)['agreement']
    return out


def summarize_future(rows, key):
    out = {
        'spearman': spearman([r[key] for r in rows], [r['future_KL'] for r in rows]),
        'kendall': kendall([r[key] for r in rows], [r['future_KL'] for r in rows]),
        'pairwise': pairwise_agreement(rows, key, 'future_KL')['agreement'],
    }
    for cls in ['R-R', 'C-C', 'R-C']:
        out[f'{cls}_pairwise'] = pairwise_agreement(rows, key, 'future_KL', cls)['agreement']
    return out


def matched_summary(rows, matched, keys):
    by_case = {(r['unit_id'], r['config']): r for r in rows}
    out = {}
    pair_rows = []
    for key in keys:
        ok = tot = 0
        for p in matched:
            a = by_case[(p['unit_id'], p['config_A'])]
            b = by_case[(p['unit_id'], p['config_B'])]
            dx = a[key] - b[key]
            dy = a['future_KL'] - b['future_KL']
            correct = abs(dx) > EPS and abs(dy) > EPS and dx * dy > 0
            ok += int(correct)
            tot += 1
            pair_rows.append({
                'metric': key,
                'unit_id': p['unit_id'],
                'config_A': p['config_A'],
                'config_B': p['config_B'],
                'score_A': a[key],
                'score_B': b[key],
                'future_KL_A': a['future_KL'],
                'future_KL_B': b['future_KL'],
                'score_margin_A_minus_B': dx,
                'future_KL_margin_A_minus_B': dy,
                'correct': correct,
            })
        out[key] = {'correct': ok, 'total': tot, 'concordance': ok / tot if tot else None}
    return out, pair_rows


def reconstruct_state_error_data(model, tokenizer, e2e, raw_by):
    rows = list(frozen_proxy.unit_rows().values())
    pmap = frozen_proxy.prompt_map()
    data = {}
    identity_rows = []
    sem_rows = []
    for idx, row in enumerate(rows):
        unit_id = row['unit']
        print(f'[{now()}] reconstruct q/E {idx + 1}/{len(rows)} {unit_id}', flush=True)
        fp_past, next_ids, _ = realrc.build_base_to_t0(
            torch, model, tokenizer, e2e, pmap[row['prompt_id']], int(row['t0'])
        )
        records = realrc.first_future_records(torch, model, next_ids, fp_past)
        data[unit_id] = {'prompt_id': row['prompt_id'], 't0': int(row['t0']), 'layers': {}}
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
                x_api = realrc.readout_from_state(torch, q.to(state.device).reshape(1, H, DK), (y - state)).reshape(H, DV).detach().float().cpu()
                x_formula, query_sq, diag_sq = score_dynamic_from_E(E, q, w)
                rel = abs(float((w * x_api.square()).sum()) - diag_sq) / (abs(diag_sq) + EPS)
                identity_rows.append({
                    'unit_id': unit_id,
                    'layer': layer,
                    'config': cfg['name'],
                    'api_diag_sq': float((w * x_api.square()).sum()),
                    'pullback_diag_sq': diag_sq,
                    'relative_error': rel,
                })
                cases[cfg['name']] = {
                    'E': E.half(),
                    'query_sq': query_sq,
                    'dynamic_diag_sq': diag_sq,
                }
            sem_rows.append({
                'unit_id': unit_id,
                'layer': layer,
                'state_shape': list(state.shape),
                'per_head_state_shape': [DK, DV],
                'q_shape': list(q.shape),
                'G_shape': list(G.shape),
                'functional_dim': FDIM,
                'readout_contraction': 'x[h,v] = sum_k E[h,k,v] * q[h,k]',
                'vectorization_order': 'state reshaped as [head, key_row, value_col]; functional x flattened as [head, value_col]',
            })
            data[unit_id]['layers'][str(layer)] = {'q': q, 'w': w, 'cases': cases}
            del G
        torch.cuda.empty_cache()
    return data, identity_rows, sem_rows


def calibration_structures(data, cal_units, layer):
    wbar = torch.zeros((H, DV))
    cq = torch.zeros((H, DK, DK))
    I = torch.zeros((H, DK, DV))
    for unit_id in cal_units:
        q = data[unit_id]['layers'][str(layer)]['q'].float()
        w = data[unit_id]['layers'][str(layer)]['w'].float()
        wbar += w
        for h in range(H):
            qh = q[h]
            outer = torch.outer(qh, qh)
            cq[h] += outer
            I[h] += qh.square().reshape(DK, 1) * w[h].reshape(1, DV)
    n = float(len(cal_units))
    wbar /= n
    cq /= n
    I /= n
    sep_a = torch.zeros((H, DK))
    sep_b = torch.zeros((H, DV))
    row = torch.zeros((H, DK))
    col = torch.zeros((H, DV))
    sv_rows = []
    for h in range(H):
        a, b = svd_rank1_nonnegative(I[h])
        sep_a[h] = a
        sep_b[h] = b
        row[h] = I[h].mean(dim=1)
        col[h] = I[h].mean(dim=0)
        s = torch.linalg.svdvals(I[h])
        total = float((s.square()).sum())
        sv_rows.append({
            'head': h,
            'top1_energy': float(s[0].square()) / (total + EPS),
            'top4_energy': float((s[:4].square()).sum()) / (total + EPS),
            'effective_rank': math.exp(float(-(s / (s.sum() + EPS) * torch.log(s / (s.sum() + EPS) + EPS)).sum())),
            'rank1_relative_fro_error': math.sqrt(max(0.0, 1.0 - float(s[0].square()) / (total + EPS))),
        })
    return {'wbar': wbar, 'cq': cq, 'I': I, 'sep_a': sep_a, 'sep_b': sep_b, 'row': row, 'col': col, 'sv_rows': sv_rows}


def score_static_family(E, cal):
    sq = E.float().square()
    sep_weight = cal['sep_a'].reshape(H, DK, 1) * cal['sep_b'].reshape(H, 1, DV)
    row_weight = cal['row'].reshape(H, DK, 1)
    col_weight = cal['col'].reshape(H, 1, DV)
    return {
        'static_cov_sq': score_static_cov(E.float(), cal['wbar'], cal['cq']),
        'static_element_sq': score_static_element(E.float(), cal['I']),
        'static_separable_sq': score_static_element(E.float(), sep_weight),
        'static_row_sq': score_static_element(E.float(), row_weight),
        'static_column_sq': score_static_element(E.float(), col_weight),
    }


def write_tensor_semantics(stage1, sem_rows):
    save_json('tensor_semantics.json', {'summary': stage1, 'rows': sem_rows})
    lines = [
        f'# {TASK}',
        '',
        '## Tensor Semantics',
        '',
        '- recurrent state shape: `[1, 32, 128, 128]`',
        '- per-head state shape: `[d_k=128, d_v=128]`',
        '- q shape: `[32, 128]` after batch squeeze',
        '- functional readout: `x[h,v] = sum_k E[h,k,v] * q[h,k]`',
        '- functional dimension: `32 * 128 = 4096`',
        '- vectorization: state `[head, key_row, value_col]`, functional `[head, value_col]`',
        '',
        'A direct reshape from 4096 functional weights to 128x128 state weights is invalid because the exact pullback contains `q_i q_k` cross-row interactions inside each value column.',
    ]
    (RUN_DIR / 'tensor_semantics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_tensor_semantics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_derivation():
    lines = [
        f'# {TASK}',
        '',
        '## Pullback Derivation',
        '',
        'For a GDN layer and token, let `E = S_hat - S` with shape `[head, key_row, value_col]`.',
        '',
        'The true readout contraction is:',
        '',
        '```',
        'x[h,v] = sum_k E[h,k,v] * q[h,k]',
        '```',
        '',
        'The diagonal functional risk from the previous experiment is:',
        '',
        '```',
        'M_diag,t^2 = sum_{h,v} w[h,v] * x[h,v]^2',
        'w[h,v] = diag(G_t)[head,value]',
        '```',
        '',
        'Substituting the contraction gives the exact state-space pullback:',
        '',
        '```',
        'M_diag,t^2 = sum_{h,v} w[h,v] * (sum_k E[h,k,v] q[h,k])^2',
        '           = e_state^T H_t e_state',
        'H_t = B_t^T diag(w_t) B_t',
        '```',
        '',
        'Per head/value column, this has `w[h,v] * outer(q[h], q[h])` over key rows. Therefore functional diagonal geometry is not the same as element-wise state diagonal weighting.',
    ]
    (RUN_DIR / 'pullback_derivation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_pullback_derivation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_compatibility(rows):
    save_json('state_metric_candidates.json', {'rows': rows})
    headers = ['Structure', 'Storage cost', 'Calibration cost', 'Runtime cost', 'Compatible with current INT8 quantizer?', 'Requires dynamic q?', 'Requires dense operation?', 'Scientific fidelity']
    md = ['# Quantizer Compatibility', '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for r in rows:
        md.append('| ' + ' | '.join(str(r[h]) for h in headers) + ' |')
    (RUN_DIR / 'quantizer_compatibility.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_quantizer_compatibility.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def controlled_validation(data, static_global):
    rows = []
    units = sorted(data)[:3]
    layers = [str(GDN_LAYERS[0]), str(GDN_LAYERS[len(GDN_LAYERS) // 2]), str(GDN_LAYERS[-1])]
    for unit_id in units:
        for layer in layers:
            q = data[unit_id]['layers'][layer]['q'].float()
            w = data[unit_id]['layers'][layer]['w'].float()
            G = torch.load(g_path(unit_id, int(layer)), map_location='cpu').float()
            I = static_global[layer]['I']
            flat = I.reshape(-1)
            picks = {
                'low': int(torch.argsort(flat)[int(0.1 * flat.numel())]),
                'medium': int(torch.argsort(flat)[int(0.5 * flat.numel())]),
                'high': int(torch.argsort(flat)[int(0.9 * flat.numel())]),
            }
            for label, idx in picks.items():
                E = torch.zeros((H, DK, DV))
                h = idx // (DK * DV)
                rem = idx % (DK * DV)
                k = rem // DV
                v = rem % DV
                E[h, k, v] = 1.0
                x, _, dyn = score_dynamic_from_E(E, q, w)
                full = float(x.reshape(-1) @ (G @ x.reshape(-1)))
                scores = score_static_family(E, static_global[layer])
                row = {'unit_id': unit_id, 'layer': layer, 'perturbation': 'single_element_' + label, 'head': int(h), 'key_row': int(k), 'value_col': int(v), 'dynamic_diag': math.sqrt(max(0.0, dyn)), 'full_local_output': math.sqrt(max(0.0, full)), 'future_KL': 'NOT_RUN'}
                for name, val in scores.items():
                    row[name.replace('_sq', '')] = math.sqrt(max(0.0, val))
                rows.append(row)
            gen = torch.Generator().manual_seed(len(rows) + 17)
            for label, E in [
                ('single_row', torch.zeros((H, DK, DV))),
                ('single_column', torch.zeros((H, DK, DV))),
                ('random_structured', torch.randn((H, DK, DV), generator=gen)),
            ]:
                if label == 'single_row':
                    E[0, int(torch.argmax(static_global[layer]['row'][0])), :] = 1.0 / math.sqrt(DV)
                elif label == 'single_column':
                    E[0, :, int(torch.argmax(static_global[layer]['col'][0]))] = 1.0 / math.sqrt(DK)
                else:
                    E = E / (torch.linalg.vector_norm(E) + EPS)
                x, _, dyn = score_dynamic_from_E(E, q, w)
                full = float(x.reshape(-1) @ (G @ x.reshape(-1)))
                scores = score_static_family(E, static_global[layer])
                row = {'unit_id': unit_id, 'layer': layer, 'perturbation': label, 'head': '', 'key_row': '', 'value_col': '', 'dynamic_diag': math.sqrt(max(0.0, dyn)), 'full_local_output': math.sqrt(max(0.0, full)), 'future_KL': 'NOT_RUN'}
                for name, val in scores.items():
                    row[name.replace('_sq', '')] = math.sqrt(max(0.0, val))
                rows.append(row)
            del G
    save_csv('controlled_validation_cases.csv', rows)
    summary = {}
    for key in ['static_cov', 'static_element', 'static_separable', 'static_row', 'static_column']:
        summary[key + '_vs_dynamic_diag_spearman'] = spearman([r[key] for r in rows], [r['dynamic_diag'] for r in rows])
        summary[key + '_vs_full_local_spearman'] = spearman([r[key] for r in rows], [r['full_local_output'] for r in rows])
    best = max([k for k in summary if k.endswith('dynamic_diag_spearman')], key=lambda k: summary[k] if summary[k] is not None else -2)
    val = summary[best]
    if val is not None and val >= 0.70:
        verdict = 'POSITIVE'
    elif val is not None and val >= 0.40:
        verdict = 'PARTIAL'
    else:
        verdict = 'NEGATIVE'
    summary['CONTROLLED_MAPPING_VALIDATION'] = verdict
    summary['note'] = 'local perturbation pilot only; no future KL rollout for synthetic perturbations'
    save_json('controlled_validation_summary.json', summary)
    return summary


def write_final_report(final):
    sections = [
        ('Task', TASK),
        ('Previous Scientific State', json.dumps(final['previous_state'], indent=2, sort_keys=True)),
        ('Why Direct Reshape Is Invalid', 'The exact metric is sum_j w_j (q^T E[:,j])^2, which expands to cross-row terms q_i q_k E_ij E_kj. A 4096 functional vector cannot be directly reshaped into a 128x128 element-wise state map.'),
        ('Exact Tensor Semantics', json.dumps(final['tensor_semantics'], indent=2, sort_keys=True)),
        ('Functional-Space Metric', 'M_diag,t^2 = x_t^T diag(diag(G_t)) x_t, x_t = E_t^T q_t.'),
        ('State-Space Pullback Derivation', 'H_t = B_t^T diag(diag(G_t)) B_t, implemented without materializing dense H.'),
        ('Numerical Identity Validation', json.dumps(final['pullback_identity'], indent=2, sort_keys=True)),
        ('Candidate State Metrics', json.dumps(final['candidate_summary'], indent=2, sort_keys=True)),
        ('Natural Replay Results', json.dumps(final['natural_replay'], indent=2, sort_keys=True)),
        ('Held-Out Static Results', json.dumps(final['static_heldout'], indent=2, sort_keys=True)),
        ('Element vs Separable vs Row vs Column', json.dumps(final['importance_structure'], indent=2, sort_keys=True)),
        ('Controlled Intervention Validation', json.dumps(final['controlled_validation'], indent=2, sort_keys=True)),
        ('Quantizer Compatibility', 'See quantizer_compatibility.md.'),
        ('Positive Results', final['positive_results']),
        ('Negative Results', final['negative_results']),
        ('Scientific Limitations', final['limitations']),
        ('Difference From DAMP-Style Precision Allocation', 'This task scores same-bit recurrent-state quantization objectives; it does not allocate FP16/INT8/Top-K precision.'),
        ('Final Scientific Classification', final['FINAL_SCIENTIFIC_CLASSIFICATION']),
        ('STATE_IMPORTANCE_READY_FOR_QUANTIZER', final['STATE_IMPORTANCE_READY_FOR_QUANTIZER']),
        ('Exact Next Recommended Task', final['NEXT_RECOMMENDED_TASK']),
    ]
    lines = [f'# {TASK}', '']
    for title, body in sections:
        lines += [f'## {title}', str(body), '']
    text = '\n'.join(lines)
    (RUN_DIR / 'final_report.md').write_text(text, encoding='utf-8')
    (REP_DIR / f'{SLUG}.md').write_text(text, encoding='utf-8')
    (RES_DIR / f'{SLUG}_final_report.md').write_text(text, encoding='utf-8')


def main():
    global torch
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    raw, raw_by = load_raw()
    struct_rows = read_csv(STRUCT_PATH)
    struct_by = {(r['unit_id'], r['config']): r for r in struct_rows}
    matched = load_matched()
    previous = json.loads(STRUCT_FINAL_PATH.read_text(encoding='utf-8'))
    stage0 = {
        'TASK': TASK,
        'timestamp': now(),
        'branch': safe_sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        'HEAD': safe_sh(['git', 'rev-parse', 'HEAD']),
        'git_status_start': safe_sh(['git', 'status', '--short']),
        'required_head_at_least': 'fd4b846',
        'geometry_tensor_count': len(list(G_DIR.glob('*_G.pt'))),
        'expected_geometry_tensor_count': len(frozen_proxy.unit_rows()) * len(GDN_LAYERS),
        'raw_case_count': len(raw),
        'structural_case_count': len(struct_rows),
        'matched_pair_count': len(matched),
        'artifact_sources': [str(RAW_PATH), str(STRUCT_PATH), str(STRUCT_FINAL_PATH), str(MATCHED_PATH), str(G_DIR)],
        'GPU_RERUN_REQUIRED': 'YES_MINIMAL_Q_E_RECONSTRUCTION_ONLY',
    }
    stage0['ARTIFACT_REUSE_GATE'] = 'PASS' if stage0['geometry_tensor_count'] == stage0['expected_geometry_tensor_count'] and len(raw) == 72 and len(struct_rows) == 72 else 'FAIL'
    save_json('stage0_artifact_audit.json', stage0)
    (RUN_DIR / 'stage0_artifact_audit.md').write_text('# Stage 0 Artifact Audit\n\n' + json.dumps(stage0, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_stage0_artifact_audit.md').write_text('# Stage 0 Artifact Audit\n\n' + json.dumps(stage0, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if stage0['ARTIFACT_REUSE_GATE'] != 'PASS':
        final = {'TASK': TASK, 'FORMAL_STATUS': 'STOPPED_ARTIFACT_GATE_FAIL', 'stage0': stage0}
        save_json('final_classification.json', final)
        print(json.dumps(final, indent=2, sort_keys=True))
        return

    torch.set_grad_enabled(False)
    torch, model, tokenizer, _, e2e = p1.setup_model()
    data, identity_rows, sem_rows = reconstruct_state_error_data(model, tokenizer, e2e, raw_by)
    identity_summary = {
        'STATE_PULLBACK_IDENTITY_GATE': 'PASS' if max(r['relative_error'] for r in identity_rows) < 1e-5 else 'FAIL',
        'PULLBACK_MAX_REL_ERROR': max(r['relative_error'] for r in identity_rows),
        'PULLBACK_MEDIAN_REL_ERROR': med([r['relative_error'] for r in identity_rows]),
        'PULLBACK_MEAN_REL_ERROR': mean([r['relative_error'] for r in identity_rows]),
        'n_checks': len(identity_rows),
    }
    save_csv('pullback_identity_cases.csv', identity_rows)
    save_json('pullback_identity_summary.json', identity_summary)
    stage1 = {
        'STATE_TENSOR_SEMANTICS_GATE': 'PASS',
        'READOUT_CONTRACTION_GATE': 'PASS' if identity_summary['STATE_PULLBACK_IDENTITY_GATE'] == 'PASS' else 'FAIL',
        'FUNCTIONAL_CHANNEL_LAYOUT_GATE': 'PASS' if FDIM == 4096 and H == 32 and DV == 128 else 'FAIL',
        'recurrent_state_shape': '[1, 32, 128, 128]',
        'per_head_state_shape': '[128, 128]',
        'q_shape': '[32, 128]',
        'functional_dimension': FDIM,
        'readout_contraction': 'x[h,v] = sum_k E[h,k,v] * q[h,k]',
    }
    write_tensor_semantics(stage1, sem_rows)
    write_derivation()
    if identity_summary['STATE_PULLBACK_IDENTITY_GATE'] != 'PASS':
        final = {'TASK': TASK, 'FORMAL_STATUS': 'STOPPED_PULLBACK_IDENTITY_FAIL', 'stage0': stage0, 'tensor_semantics': stage1, 'pullback_identity': identity_summary}
        save_json('final_classification.json', final)
        print(json.dumps(final, indent=2, sort_keys=True))
        return

    units = sorted(data)
    dyn_rows = []
    for unit_id in units:
        for cfg in PANEL:
            base = raw_by[(unit_id, cfg)]
            row = {
                'unit_id': unit_id,
                'prompt_id': base['prompt_id'],
                't0': base['t0'],
                'config': cfg,
                'future_KL': base['future_KL'],
                'M0': base['M0'],
                'M5_full': float(struct_by[(unit_id, cfg)]['M5_full_recomputed']),
                'M5_diag_dynamic': 0.0,
                'query_only': 0.0,
            }
            for layer in GDN_LAYERS:
                c = data[unit_id]['layers'][str(layer)]['cases'][cfg]
                row['M5_diag_dynamic'] += c['dynamic_diag_sq']
                row['query_only'] += c['query_sq']
            row['M5_diag_dynamic'] = math.sqrt(max(0.0, row['M5_diag_dynamic']))
            row['query_only'] = math.sqrt(max(0.0, row['query_only']))
            dyn_rows.append(row)

    static_rows = []
    sv_all = []
    static_global = {}
    for layer in GDN_LAYERS:
        static_global[str(layer)] = calibration_structures(data, units, layer)
    for hold in units:
        print(f'[{now()}] heldout state metrics {hold}', flush=True)
        cal_units = [u for u in units if u != hold]
        fold_cal = {str(layer): calibration_structures(data, cal_units, layer) for layer in GDN_LAYERS}
        for layer, cal in fold_cal.items():
            for s in cal['sv_rows']:
                sv_all.append({'heldout_unit': hold, 'layer': layer, **s})
        for cfg in PANEL:
            base = next(r for r in dyn_rows if r['unit_id'] == hold and r['config'] == cfg)
            row = dict(base)
            for k in ['static_w_dynamic_q', 'static_cov', 'static_element', 'static_separable', 'static_row', 'static_column']:
                row[k] = 0.0
            for layer in GDN_LAYERS:
                ld = data[hold]['layers'][str(layer)]
                E = ld['cases'][cfg]['E'].float()
                q = ld['q'].float()
                x, _, _ = score_dynamic_from_E(E, q, ld['w'].float())
                cal = fold_cal[str(layer)]
                row['static_w_dynamic_q'] += float((cal['wbar'] * x.square()).sum())
                scores = score_static_family(E, cal)
                for key, val in scores.items():
                    row[key.replace('_sq', '')] += val
            for k in ['static_w_dynamic_q', 'static_cov', 'static_element', 'static_separable', 'static_row', 'static_column']:
                row[k] = math.sqrt(max(0.0, row[k]))
            static_rows.append(row)

    save_csv('natural_replay_case_level.csv', dyn_rows)
    save_csv('static_heldout_case_level.csv', static_rows)
    save_csv('state_importance_svd.csv', sv_all)

    candidate_keys = ['M0', 'query_only', 'static_w_dynamic_q', 'static_cov', 'static_element', 'static_separable', 'static_row', 'static_column']
    natural_summary = {
        key: {
            'vs_dynamic_diag': summarize_candidate(static_rows, key, 'M5_diag_dynamic'),
            'vs_full': summarize_candidate(static_rows, key, 'M5_full'),
            'vs_future_KL': summarize_future(static_rows, key),
        }
        for key in candidate_keys
    }
    save_json('natural_replay_summary.json', natural_summary)

    matched_keys = candidate_keys + ['M5_diag_dynamic', 'M5_full']
    matched, matched_rows = matched_summary(static_rows, matched, matched_keys)
    save_csv('matched_rc_14_cases.csv', matched_rows)

    static_summary = {
        'protocol': 'leave-one-unit-out; only calibration units used for w_bar, C_q, I_element, separable, row, and column factors; no future_KL fitting',
        'candidates': {k: natural_summary[k] for k in candidate_keys},
        'matched_rc_14': matched,
        'M0_VS_DYNAMIC_DIAG_SPEARMAN': natural_summary['M0']['vs_dynamic_diag']['spearman'],
        'QUERY_ONLY_VS_DYNAMIC_DIAG_SPEARMAN': natural_summary['query_only']['vs_dynamic_diag']['spearman'],
        'STATIC_COV_VS_DYNAMIC_DIAG_SPEARMAN': natural_summary['static_cov']['vs_dynamic_diag']['spearman'],
        'STATIC_ELEMENT_VS_DYNAMIC_DIAG_SPEARMAN': natural_summary['static_element']['vs_dynamic_diag']['spearman'],
        'STATIC_SEPARABLE_VS_DYNAMIC_DIAG_SPEARMAN': natural_summary['static_separable']['vs_dynamic_diag']['spearman'],
        'STATIC_ROW_VS_DYNAMIC_DIAG_SPEARMAN': natural_summary['static_row']['vs_dynamic_diag']['spearman'],
        'STATIC_COLUMN_VS_DYNAMIC_DIAG_SPEARMAN': natural_summary['static_column']['vs_dynamic_diag']['spearman'],
        'STATIC_COV_HELDOUT_PAIRWISE': natural_summary['static_cov']['vs_dynamic_diag']['pairwise'],
        'STATIC_ELEMENT_HELDOUT_PAIRWISE': natural_summary['static_element']['vs_dynamic_diag']['pairwise'],
        'STATIC_SEPARABLE_HELDOUT_PAIRWISE': natural_summary['static_separable']['vs_dynamic_diag']['pairwise'],
        'STATIC_ROW_HELDOUT_PAIRWISE': natural_summary['static_row']['vs_dynamic_diag']['pairwise'],
        'STATIC_COLUMN_HELDOUT_PAIRWISE': natural_summary['static_column']['vs_dynamic_diag']['pairwise'],
    }
    save_json('static_heldout_summary.json', static_summary)

    importance_summary = {
        'STATE_IMPORTANCE_TOP1_SVD_ENERGY_MEDIAN': med([r['top1_energy'] for r in sv_all]),
        'STATE_IMPORTANCE_TOP4_SVD_ENERGY_MEDIAN': med([r['top4_energy'] for r in sv_all]),
        'STATE_IMPORTANCE_RANK1_REL_FRO_ERROR_MEDIAN': med([r['rank1_relative_fro_error'] for r in sv_all]),
        'STATE_IMPORTANCE_EFFECTIVE_RANK_MEDIAN': med([r['effective_rank'] for r in sv_all]),
        'row_vs_dynamic_diag_spearman': static_summary['STATIC_ROW_VS_DYNAMIC_DIAG_SPEARMAN'],
        'column_vs_dynamic_diag_spearman': static_summary['STATIC_COLUMN_VS_DYNAMIC_DIAG_SPEARMAN'],
        'interpretation': 'state importance is evaluated in recurrent [key_row,value_col] space; this is distinct from full functional G low-rank analysis',
    }
    if importance_summary['STATE_IMPORTANCE_TOP1_SVD_ENERGY_MEDIAN'] >= 0.85 and abs(static_summary['STATIC_ELEMENT_HELDOUT_PAIRWISE'] - static_summary['STATIC_SEPARABLE_HELDOUT_PAIRWISE']) <= 0.03:
        separable = 'YES'
    elif importance_summary['STATE_IMPORTANCE_TOP1_SVD_ENERGY_MEDIAN'] >= 0.65:
        separable = 'PARTIAL'
    else:
        separable = 'NO'
    importance_summary['STATE_IMPORTANCE_SEPARABLE'] = separable

    static_global_slim = {}
    for layer, cal in static_global.items():
        static_global_slim[layer] = {k: cal[k] for k in ['wbar', 'cq', 'I', 'sep_a', 'sep_b', 'row', 'col']}
    controlled = controlled_validation(data, static_global_slim)

    compat_rows = [
        {'Structure': 'M0 raw reconstruction', 'Storage cost': 'none', 'Calibration cost': 'none', 'Runtime cost': 'low', 'Compatible with current INT8 quantizer?': 'yes', 'Requires dynamic q?': 'no', 'Requires dense operation?': 'no', 'Scientific fidelity': 'baseline'},
        {'Structure': 'query-only', 'Storage cost': 'none', 'Calibration cost': 'none', 'Runtime cost': 'dynamic readout q', 'Compatible with current INT8 quantizer?': 'conditional', 'Requires dynamic q?': 'yes', 'Requires dense operation?': 'no', 'Scientific fidelity': 'dynamic baseline'},
        {'Structure': 'static covariance x value weight', 'Storage cost': 'per-layer/head 128x128 covariance plus value weights', 'Calibration cost': 'moderate', 'Runtime cost': 'matrix-vector per value column', 'Compatible with current INT8 quantizer?': 'no/directly expensive', 'Requires dynamic q?': 'no', 'Requires dense operation?': 'yes within key rows', 'Scientific fidelity': 'high if ranking retained'},
        {'Structure': 'static element-wise', 'Storage cost': 'per-layer/head 128x128 weights', 'Calibration cost': 'low', 'Runtime cost': 'weighted MSE', 'Compatible with current INT8 quantizer?': 'yes', 'Requires dynamic q?': 'no', 'Requires dense operation?': 'no', 'Scientific fidelity': 'primary deployable candidate'},
        {'Structure': 'static separable row x column', 'Storage cost': 'per-layer/head row+column vectors', 'Calibration cost': 'low plus rank-1 SVD', 'Runtime cost': 'weighted MSE', 'Compatible with current INT8 quantizer?': 'yes', 'Requires dynamic q?': 'no', 'Requires dense operation?': 'no', 'Scientific fidelity': 'simplest deployable candidate if close to element'},
        {'Structure': 'row-only', 'Storage cost': 'row vector per layer/head', 'Calibration cost': 'low', 'Runtime cost': 'row weighted MSE', 'Compatible with current INT8 quantizer?': 'yes for row/group search', 'Requires dynamic q?': 'no', 'Requires dense operation?': 'no', 'Scientific fidelity': 'structural baseline'},
        {'Structure': 'column-only', 'Storage cost': 'column vector per layer/head', 'Calibration cost': 'low', 'Runtime cost': 'column weighted MSE', 'Compatible with current INT8 quantizer?': 'yes for column/group search', 'Requires dynamic q?': 'no', 'Requires dense operation?': 'no', 'Scientific fidelity': 'structural baseline'},
    ]
    write_compatibility(compat_rows)

    deployable = ['static_element', 'static_separable', 'static_row', 'static_column']
    best_static = max(deployable + ['static_cov'], key=lambda k: natural_summary[k]['vs_dynamic_diag']['spearman'] if natural_summary[k]['vs_dynamic_diag']['spearman'] is not None else -2)
    element_good = natural_summary['static_element']['vs_dynamic_diag']['spearman'] >= 0.90 and natural_summary['static_element']['vs_dynamic_diag']['pairwise'] >= 0.90
    sep_good = natural_summary['static_separable']['vs_dynamic_diag']['spearman'] >= 0.90 and natural_summary['static_separable']['vs_dynamic_diag']['pairwise'] >= 0.90
    cov_good = natural_summary['static_cov']['vs_dynamic_diag']['spearman'] >= 0.90 and natural_summary['static_cov']['vs_dynamic_diag']['pairwise'] >= 0.90
    better_than_baselines = natural_summary[best_static]['vs_dynamic_diag']['spearman'] > max(natural_summary['M0']['vs_dynamic_diag']['spearman'], natural_summary['query_only']['vs_dynamic_diag']['spearman']) + 0.02
    if element_good and better_than_baselines:
        cls = 'STATE_FUNCTIONAL_WEIGHT_ELEMENTWISE_SUPPORTED'
        ready = 'YES'
        method = 'YES'
        qstruct = 'static element-wise recurrent-state weighting'
        next_task = 'GDN_INT8_STATIC_FUNCTIONAL_WEIGHTED_QUANTIZER_V1'
    elif sep_good and better_than_baselines:
        cls = 'STATE_FUNCTIONAL_WEIGHT_SEPARABLE_SUPPORTED'
        ready = 'YES'
        method = 'YES'
        qstruct = 'static separable row x column recurrent-state weighting'
        next_task = 'GDN_INT8_STATIC_FUNCTIONAL_WEIGHTED_QUANTIZER_V1'
    elif cov_good:
        cls = 'STATE_PULLBACK_SUPPORTED_BUT_SIMPLE_WEIGHTING_INSUFFICIENT'
        ready = 'CONDITIONAL'
        method = 'NO'
        qstruct = 'static covariance pullback; not directly lightweight'
        next_task = 'compress or approximate covariance pullback before quantizer design'
    else:
        cls = 'FUNCTIONAL_PULLBACK_VALID_BUT_STATIC_STATE_METRIC_NOT_SUPPORTED'
        ready = 'NO'
        method = 'NO'
        qstruct = 'none'
        next_task = 'investigate dynamic q-dependent state metrics before quantizer design'

    final = {
        'TASK': TASK,
        'FORMAL_STATUS': 'COMPLETE',
        'ARTIFACT_REUSE_GATE': stage0['ARTIFACT_REUSE_GATE'],
        'GPU_RERUN_REQUIRED': stage0['GPU_RERUN_REQUIRED'],
        **stage1,
        **identity_summary,
        'previous_state': {'structural_simplification_commit': 'fd4b846', 'structural_simplification_classification': previous['FINAL_SCIENTIFIC_CLASSIFICATION']},
        'stage0': stage0,
        'tensor_semantics': stage1,
        'pullback_identity': identity_summary,
        'candidate_summary': natural_summary,
        'natural_replay': natural_summary,
        'static_heldout': static_summary,
        'importance_structure': importance_summary,
        'controlled_validation': controlled,
        'M0_VS_DYNAMIC_DIAG_SPEARMAN': static_summary['M0_VS_DYNAMIC_DIAG_SPEARMAN'],
        'QUERY_ONLY_VS_DYNAMIC_DIAG_SPEARMAN': static_summary['QUERY_ONLY_VS_DYNAMIC_DIAG_SPEARMAN'],
        'STATIC_COV_VS_DYNAMIC_DIAG_SPEARMAN': static_summary['STATIC_COV_VS_DYNAMIC_DIAG_SPEARMAN'],
        'STATIC_ELEMENT_VS_DYNAMIC_DIAG_SPEARMAN': static_summary['STATIC_ELEMENT_VS_DYNAMIC_DIAG_SPEARMAN'],
        'STATIC_SEPARABLE_VS_DYNAMIC_DIAG_SPEARMAN': static_summary['STATIC_SEPARABLE_VS_DYNAMIC_DIAG_SPEARMAN'],
        'STATIC_ROW_VS_DYNAMIC_DIAG_SPEARMAN': static_summary['STATIC_ROW_VS_DYNAMIC_DIAG_SPEARMAN'],
        'STATIC_COLUMN_VS_DYNAMIC_DIAG_SPEARMAN': static_summary['STATIC_COLUMN_VS_DYNAMIC_DIAG_SPEARMAN'],
        'STATIC_COV_HELDOUT_PAIRWISE': static_summary['STATIC_COV_HELDOUT_PAIRWISE'],
        'STATIC_ELEMENT_HELDOUT_PAIRWISE': static_summary['STATIC_ELEMENT_HELDOUT_PAIRWISE'],
        'STATIC_SEPARABLE_HELDOUT_PAIRWISE': static_summary['STATIC_SEPARABLE_HELDOUT_PAIRWISE'],
        'STATIC_ROW_HELDOUT_PAIRWISE': static_summary['STATIC_ROW_HELDOUT_PAIRWISE'],
        'STATIC_COLUMN_HELDOUT_PAIRWISE': static_summary['STATIC_COLUMN_HELDOUT_PAIRWISE'],
        'BEST_STATIC_STATE_METRIC': best_static,
        'STATE_IMPORTANCE_SEPARABLE': separable,
        'CONTROLLED_MAPPING_VALIDATION': controlled['CONTROLLED_MAPPING_VALIDATION'],
        'QUANTIZER_COMPATIBLE_STRUCTURE': qstruct,
        'FINAL_SCIENTIFIC_CLASSIFICATION': cls,
        'STATE_IMPORTANCE_READY_FOR_QUANTIZER': ready,
        'METHOD_DESIGN_READY': method,
        'NEXT_RECOMMENDED_TASK': next_task,
        'positive_results': 'Exact state pullback identity holds and static state metrics are evaluated under leave-one-unit-out calibration.',
        'negative_results': 'Matched R-C 14/14 is reported only as consistency because isotropic also achieved 14/14 previously; no candidate is accepted on that evidence alone.',
        'limitations': 'Synthetic perturbation validation is local-output only and does not run future KL. Static calibration uses the existing 9 canonical units.',
    }
    save_json('final_classification.json', final)
    write_final_report(final)
    print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
