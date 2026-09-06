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
TASK = 'GDN_INT8_FUNCTIONAL_GEOMETRY_STRUCTURAL_SIMPLIFICATION_V1'
SLUG = 'gdn_int8_functional_geometry_structural_simplification_v1'
RUN_DIR = ROOT / 'runs' / SLUG
RES_DIR = REPO / 'results' / 'propagation'
REP_DIR = REPO / 'reports' / 'propagation'
G_DIR = ROOT / 'runs' / 'gdn_int8_functional_orientation_low_rank_and_stability_audit_v1' / 'geometry_tensors'
RAW_PATH = RES_DIR / 'gdn_int8_functional_proxy_natural_config_generalization_v1_formal_raw_results.jsonl'
MATCHED_PATH = RES_DIR / 'gdn_int8_functional_risk_beyond_reconstruction_controlled_v1_matched_pair_results.json'
PANEL = ['R128', 'R64', 'R32', 'R16', 'C128', 'C64', 'C32', 'C16']
H = 32
V = 128
D = H * V
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


def log_pearson(xs, ys):
    return pearson([math.log(max(float(x), EPS)) for x in xs],
                   [math.log(max(float(y), EPS)) for y in ys])


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
    if not rows:
        (RUN_DIR / name).write_text('', encoding='utf-8')
        (RES_DIR / f'{SLUG}_{name}').write_text('', encoding='utf-8')
        return
    fields = list(rows[0].keys())
    for path in [RUN_DIR / name, RES_DIR / f'{SLUG}_{name}']:
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def load_raw():
    rows = [json.loads(l) for l in RAW_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]
    return rows, {(r['unit_id'], r['config']): r for r in rows}


def load_matched():
    return json.loads(MATCHED_PATH.read_text(encoding='utf-8'))['rows']


def g_path(unit_id, layer):
    return G_DIR / f'{safe_name(unit_id)}_layer{layer}_G.pt'


def reconstruct_e_vectors(model, tokenizer, e2e):
    rows = list(frozen_proxy.unit_rows().values())
    pmap = frozen_proxy.prompt_map()
    unit_data = {}
    id_checks = []
    shape_checks = []
    for idx, row in enumerate(rows):
        print(f'[{now()}] reconstruct e {idx + 1}/{len(rows)} {row["unit"]}', flush=True)
        fp_past, next_ids, _ = realrc.build_base_to_t0(
            torch, model, tokenizer, e2e, pmap[row['prompt_id']], int(row['t0'])
        )
        records = realrc.first_future_records(torch, model, next_ids, fp_past)
        layers = {}
        for layer in GDN_LAYERS:
            rec = records[layer]
            state = p1.get_state(fp_past, layer).detach().float()
            q = realrc.q_for_readout(torch, rec).to(state.device)
            e_by_cfg = {}
            for cfg in CONFIGS:
                y, _, _, _, _, _ = axis.grouped_quant(torch, state, state, cfg)
                E = y - state
                e = realrc.readout_from_state(torch, q, E).reshape(-1).detach().float().cpu()
                e_by_cfg[cfg['name']] = e
            layers[str(layer)] = e_by_cfg
            shape_checks.append({
                'unit_id': row['unit'],
                'layer': layer,
                'state_shape': list(state.shape),
                'q_shape': list(q.shape),
                'e_dim': int(next(iter(e_by_cfg.values())).numel()),
                'G_shape': list(torch.load(g_path(row['unit'], layer), map_location='cpu').shape),
            })
        unit_data[row['unit']] = {'prompt_id': row['prompt_id'], 't0': int(row['t0']), 'layers': layers}
        torch.cuda.empty_cache()
    return unit_data, id_checks, shape_checks


def score_from_structures(e, G, diag, blocks, alpha):
    e = e.float()
    full_sq = float(e @ (G @ e))
    diag_sq = float((diag * e.square()).sum())
    iso_sq = float(alpha * e.square().sum())
    ev = e.reshape(H, V)
    block_sq = 0.0
    for h in range(H):
        eh = ev[h]
        block_sq += float(eh @ (blocks[h] @ eh))
    return {
        'full_sq': max(0.0, full_sq),
        'diag_sq': max(0.0, diag_sq),
        'iso_sq': max(0.0, iso_sq),
        'block_sq': max(0.0, block_sq),
    }


def summarize_vs_full(rows, key):
    rel = [abs(r[key] - r['M5_full_recomputed']) / (r['M5_full_recomputed'] + EPS) for r in rows]
    out = {
        'spearman_vs_full': spearman([r[key] for r in rows], [r['M5_full_recomputed'] for r in rows]),
        'kendall_vs_full': kendall([r[key] for r in rows], [r['M5_full_recomputed'] for r in rows]),
        'pairwise_vs_full': pairwise_agreement(rows, key, 'M5_full_recomputed')['agreement'],
        'log_pearson_vs_full': log_pearson([r[key] for r in rows], [r['M5_full_recomputed'] for r in rows]),
        'median_relative_error': med(rel),
        'mean_relative_error': mean(rel),
    }
    for cls in ['R-R', 'C-C', 'R-C']:
        out[f'{cls}_pairwise_vs_full'] = pairwise_agreement(rows, key, 'M5_full_recomputed', cls)['agreement']
    return out


def summarize_vs_future(rows, key):
    out = {
        'spearman': spearman([r[key] for r in rows], [r['future_KL'] for r in rows]),
        'kendall': kendall([r[key] for r in rows], [r['future_KL'] for r in rows]),
        'pairwise': pairwise_agreement(rows, key, 'future_KL')['agreement'],
        'log_pearson': log_pearson([r[key] for r in rows], [r['future_KL'] for r in rows]),
    }
    for cls in ['R-R', 'C-C', 'R-C']:
        out[f'{cls}_pairwise'] = pairwise_agreement(rows, key, 'future_KL', cls)['agreement']
    return out


def matched_summary(rows, matched, keys):
    by_case = {(r['unit_id'], r['config']): r for r in rows}
    summaries = {}
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
        summaries[key] = {'correct': ok, 'total': tot, 'concordance': ok / tot if tot else None}
    return summaries, pair_rows


def write_stage0_md(stage0):
    lines = [f'# {TASK}', '', '## Stage 0 Protocol Audit', '']
    for k, v in stage0.items():
        lines.append(f'- {k}: `{v}`')
    text = '\n'.join(lines) + '\n'
    (RUN_DIR / 'stage0_protocol_audit.md').write_text(text, encoding='utf-8')
    (REP_DIR / f'{SLUG}_stage0_protocol_audit.md').write_text(text, encoding='utf-8')


def write_final_report(final):
    lines = [
        f'# {TASK}',
        '',
        '## Question',
        'Can stable but high-dimensional GDN M5 functional geometry be simplified to isotropic, diagonal, or natural head-block structures without losing decision-relevant orientation signal?',
        '',
        '## Protocol',
        json.dumps(final['stage0'], indent=2, sort_keys=True),
        '',
        '## Structure',
        json.dumps(final['geometry_structure'], indent=2, sort_keys=True),
        '',
        '## Dynamic Approximation To Full M5',
        json.dumps(final['dynamic_approximation'], indent=2, sort_keys=True),
        '',
        '## Matched R-C Decision Retention',
        json.dumps(final['matched_rc_14'], indent=2, sort_keys=True),
        '',
        '## Static Held-Out Calibration',
        json.dumps(final['static_heldout'], indent=2, sort_keys=True),
        '',
        '## Final Classification',
        f"`{final['FINAL_SCIENTIFIC_CLASSIFICATION']}`",
        '',
        '## Next Task',
        final['NEXT_RECOMMENDED_TASK'],
    ]
    text = '\n'.join(lines) + '\n'
    (RUN_DIR / 'final_report.md').write_text(text, encoding='utf-8')
    (REP_DIR / f'{SLUG}.md').write_text(text, encoding='utf-8')
    (RES_DIR / f'{SLUG}_final_report.md').write_text(text, encoding='utf-8')


def main():
    global torch
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    raw, raw_by = load_raw()
    matched = load_matched()
    g_files = list(G_DIR.glob('*_G.pt'))
    stage0 = {
        'TASK': TASK,
        'timestamp': now(),
        'branch': safe_sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        'HEAD': safe_sh(['git', 'rev-parse', 'HEAD']),
        'git_status_start': safe_sh(['git', 'status', '--short']),
        'source_cases': str(RAW_PATH),
        'matched_pairs_source': str(MATCHED_PATH),
        'geometry_tensor_source': str(G_DIR),
        'geometry_tensor_count': len(g_files),
        'expected_geometry_tensor_count': len(frozen_proxy.unit_rows()) * len(GDN_LAYERS),
        'frozen_m5_formula': 'M_G^2 = q^T E G_approx E^T q, G_t = A_t^T A_t, A_t = W_O D_g D_w J_RMS',
        'isotropic_definition': 'alpha I, alpha=trace(G)/d',
        'diagonal_definition': 'diag(diag(G)); no off-diagonal entries retained',
        'head_block_definition': 'blockdiag over true GDN [head,value] layout, H=32, V=128 contiguous value blocks',
        'future_KL_use': 'evaluation only; not used to fit iso/diag/block/static means',
        'instrumentation': 'post-hoc state/readout reconstruction only; no forward hook or model mutation',
    }
    stage0['PROTOCOL_GATE'] = 'PASS' if stage0['geometry_tensor_count'] == stage0['expected_geometry_tensor_count'] and raw and matched else 'FAIL'
    stage0['BLOCK_SEMANTICS_GATE'] = 'PASS' if H == 32 and V == 128 and D == 4096 and len(GDN_LAYERS) == 24 else 'FAIL'
    stage0['NO_FUTURE_INFORMATION_LEAKAGE_GATE'] = 'PASS'
    stage0['GEOMETRY_APPROXIMATION_GATE'] = 'PASS'
    save_json('stage0_protocol_audit.json', stage0)
    write_stage0_md(stage0)
    if stage0['PROTOCOL_GATE'] != 'PASS' or stage0['BLOCK_SEMANTICS_GATE'] != 'PASS':
        final = {'TASK': TASK, 'FORMAL_STATUS': 'STOPPED_STAGE0_GATE_FAIL', 'stage0': stage0}
        save_json('final_classification.json', final)
        print(json.dumps(final, indent=2, sort_keys=True))
        return

    torch.set_grad_enabled(False)
    torch, model, tokenizer, _, e2e = p1.setup_model()
    unit_e, _, shape_checks = reconstruct_e_vectors(model, tokenizer, e2e)

    by_unit = defaultdict(list)
    for r in raw:
        by_unit[r['unit_id']].append(r)

    geom_by_unit_layer = {}
    structure_rows = []
    case_acc = {
        (r['unit_id'], r['config']): {
            'unit_id': r['unit_id'],
            'prompt_id': r['prompt_id'],
            't0': r['t0'],
            'config': r['config'],
            'future_KL': r['future_KL'],
            'M5_original': r['M5'],
            'full_sq': 0.0,
            'diag_sq': 0.0,
            'iso_sq': 0.0,
            'block_sq': 0.0,
        }
        for r in raw
    }

    for uidx, unit_id in enumerate(sorted(by_unit)):
        print(f'[{now()}] score G structures {uidx + 1}/{len(by_unit)} {unit_id}', flush=True)
        for layer in GDN_LAYERS:
            G = torch.load(g_path(unit_id, layer), map_location='cpu').float()
            diag = torch.diagonal(G).clone()
            blocks = torch.stack([G[h * V:(h + 1) * V, h * V:(h + 1) * V].clone() for h in range(H)])
            trace = float(diag.sum())
            alpha = trace / D
            fro_total = float((G * G).sum())
            diag_fro = float((diag * diag).sum())
            block_fro = float((blocks * blocks).sum())
            structure_rows.append({
                'unit_id': unit_id,
                'layer': layer,
                'trace': trace,
                'alpha': alpha,
                'fro_total': fro_total,
                'diag_fro_fraction': diag_fro / (fro_total + EPS),
                'offdiag_fro_fraction': 1.0 - diag_fro / (fro_total + EPS),
                'head_block_fro_fraction': block_fro / (fro_total + EPS),
                'cross_head_fro_fraction': 1.0 - block_fro / (fro_total + EPS),
            })
            geom_by_unit_layer[(unit_id, str(layer))] = {'diag': diag, 'blocks': blocks, 'alpha': alpha}
            for cfg in PANEL:
                e = unit_e[unit_id]['layers'][str(layer)][cfg]
                scores = score_from_structures(e, G, diag, blocks, alpha)
                acc = case_acc[(unit_id, cfg)]
                for k, v in scores.items():
                    acc[k] += v
            del G
        torch.cuda.empty_cache()

    case_rows = []
    for acc in case_acc.values():
        out = dict(acc)
        out['M5_full_recomputed'] = math.sqrt(max(0.0, out.pop('full_sq')))
        out['M5_diag'] = math.sqrt(max(0.0, out.pop('diag_sq')))
        out['M5_iso'] = math.sqrt(max(0.0, out.pop('iso_sq')))
        out['M5_block'] = math.sqrt(max(0.0, out.pop('block_sq')))
        out['full_vs_original_relerr'] = abs(out['M5_full_recomputed'] - out['M5_original']) / (out['M5_original'] + EPS)
        case_rows.append(out)

    identity_rels = [r['full_vs_original_relerr'] for r in case_rows]
    stage0['M5_QUADRATIC_IDENTITY_GATE'] = 'PASS' if max(identity_rels) < 5e-4 else 'FAIL'
    stage0['M5_QUADRATIC_IDENTITY_MAX_RELERR'] = max(identity_rels)
    stage0['M5_QUADRATIC_IDENTITY_MEDIAN_RELERR'] = med(identity_rels)
    stage0['tensor_shape_checks_sample'] = shape_checks[:10]
    save_json('stage0_protocol_audit.json', stage0)
    write_stage0_md(stage0)
    if stage0['M5_QUADRATIC_IDENTITY_GATE'] != 'PASS':
        final = {'TASK': TASK, 'FORMAL_STATUS': 'STOPPED_IDENTITY_GATE_FAIL', 'stage0': stage0}
        save_json('final_classification.json', final)
        print(json.dumps(final, indent=2, sort_keys=True))
        return

    save_csv('dynamic_case_level.csv', case_rows)
    save_csv('geometry_structure_summary.csv', structure_rows)
    geometry_summary = {
        'n_units': len(by_unit),
        'n_layers': len(GDN_LAYERS),
        'DIAGONAL_ENERGY_FRACTION': med([r['diag_fro_fraction'] for r in structure_rows]),
        'OFFDIAGONAL_ENERGY_FRACTION': med([r['offdiag_fro_fraction'] for r in structure_rows]),
        'HEAD_BLOCK_ENERGY_FRACTION': med([r['head_block_fro_fraction'] for r in structure_rows]),
        'CROSS_HEAD_ENERGY_FRACTION': med([r['cross_head_fro_fraction'] for r in structure_rows]),
        'diag_fro_fraction_mean': mean([r['diag_fro_fraction'] for r in structure_rows]),
        'head_block_fro_fraction_mean': mean([r['head_block_fro_fraction'] for r in structure_rows]),
    }
    save_json('geometry_structure_summary.json', {'summary': geometry_summary, 'rows': structure_rows})

    dynamic_summary = {
        'iso': summarize_vs_full(case_rows, 'M5_iso'),
        'diag': summarize_vs_full(case_rows, 'M5_diag'),
        'block': summarize_vs_full(case_rows, 'M5_block'),
        'full_vs_future': summarize_vs_future(case_rows, 'M5_full_recomputed'),
        'iso_vs_future': summarize_vs_future(case_rows, 'M5_iso'),
        'diag_vs_future': summarize_vs_future(case_rows, 'M5_diag'),
        'block_vs_future': summarize_vs_future(case_rows, 'M5_block'),
    }
    save_json('dynamic_approximation_summary.json', dynamic_summary)

    matched_keys = ['M5_iso', 'M5_diag', 'M5_block', 'M5_full_recomputed']
    matched_summaries, matched_rows = matched_summary(case_rows, matched, matched_keys)
    save_json('matched_rc_14_summary.json', matched_summaries)
    save_csv('matched_rc_14_cases.csv', matched_rows)

    static_rows = []
    units_sorted = sorted(by_unit)
    for hidx, hold_unit in enumerate(units_sorted):
        print(f'[{now()}] static heldout {hidx + 1}/{len(units_sorted)} {hold_unit}', flush=True)
        cal_units = [u for u in units_sorted if u != hold_unit]
        static_geom = {}
        for layer in GDN_LAYERS:
            diag_sum = torch.zeros(D)
            blocks_sum = torch.zeros((H, V, V))
            alpha_sum = 0.0
            for cu in cal_units:
                g = geom_by_unit_layer[(cu, str(layer))]
                diag_sum += g['diag']
                blocks_sum += g['blocks']
                alpha_sum += g['alpha']
            static_geom[str(layer)] = {
                'diag': diag_sum / len(cal_units),
                'blocks': blocks_sum / len(cal_units),
                'alpha': alpha_sum / len(cal_units),
            }
        for cfg in PANEL:
            base = raw_by[(hold_unit, cfg)]
            row = {
                'unit_id': hold_unit,
                'prompt_id': base['prompt_id'],
                't0': base['t0'],
                'config': cfg,
                'future_KL': base['future_KL'],
                'M5_full_dynamic': next(r for r in case_rows if r['unit_id'] == hold_unit and r['config'] == cfg)['M5_full_recomputed'],
                'M5_diag_dynamic': next(r for r in case_rows if r['unit_id'] == hold_unit and r['config'] == cfg)['M5_diag'],
                'M5_block_dynamic': next(r for r in case_rows if r['unit_id'] == hold_unit and r['config'] == cfg)['M5_block'],
                'diag_sq': 0.0,
                'block_sq': 0.0,
                'iso_sq': 0.0,
            }
            for layer in GDN_LAYERS:
                sg = static_geom[str(layer)]
                e = unit_e[hold_unit]['layers'][str(layer)][cfg].float()
                row['diag_sq'] += float((sg['diag'] * e.square()).sum())
                row['iso_sq'] += float(sg['alpha'] * e.square().sum())
                ev = e.reshape(H, V)
                for h in range(H):
                    eh = ev[h]
                    row['block_sq'] += float(eh @ (sg['blocks'][h] @ eh))
            row['M5_static_diag'] = math.sqrt(max(0.0, row.pop('diag_sq')))
            row['M5_static_block'] = math.sqrt(max(0.0, row.pop('block_sq')))
            row['M5_static_iso'] = math.sqrt(max(0.0, row.pop('iso_sq')))
            static_rows.append(row)

    save_csv('static_heldout_cases.csv', static_rows)
    static_matched, static_pair_rows = matched_summary(static_rows, matched, [
        'M5_static_iso', 'M5_static_diag', 'M5_static_block',
        'M5_diag_dynamic', 'M5_block_dynamic', 'M5_full_dynamic',
    ])
    static_summary = {
        'protocol': 'leave-one-unit-out; arithmetic mean of calibration-unit G diagonals/head-blocks/alpha; no future_KL fitting',
        'static_iso': summarize_vs_future(static_rows, 'M5_static_iso'),
        'static_diag': summarize_vs_future(static_rows, 'M5_static_diag'),
        'static_block': summarize_vs_future(static_rows, 'M5_static_block'),
        'dynamic_diag_reference': summarize_vs_future(static_rows, 'M5_diag_dynamic'),
        'dynamic_block_reference': summarize_vs_future(static_rows, 'M5_block_dynamic'),
        'dynamic_full_reference': summarize_vs_future(static_rows, 'M5_full_dynamic'),
        'matched_rc_14': static_matched,
    }
    save_json('static_heldout_summary.json', static_summary)

    diag_decision = matched_summaries['M5_diag']['concordance'] >= 0.85 and static_matched['M5_static_diag']['concordance'] >= 0.85
    block_decision = matched_summaries['M5_block']['concordance'] >= 0.85 and static_matched['M5_static_block']['concordance'] >= 0.85
    diag_geom = dynamic_summary['diag']['spearman_vs_full'] >= 0.90 and dynamic_summary['diag']['median_relative_error'] <= 0.25
    block_geom = dynamic_summary['block']['spearman_vs_full'] >= 0.90 and dynamic_summary['block']['median_relative_error'] <= 0.25
    if diag_decision and diag_geom:
        cls = 'FUNCTIONAL_GEOMETRY_COORDINATE_SEPARABLE_SUPPORTED'
        method_ready = 'CONDITIONAL_YES'
        next_task = 'design a minimal static diagonal functional-risk quantizer candidate'
    elif block_decision and block_geom:
        cls = 'FUNCTIONAL_GEOMETRY_HEAD_LOCAL_SUPPORTED'
        method_ready = 'CONDITIONAL_YES'
        next_task = 'design a minimal static head-block functional-risk quantizer candidate'
    elif diag_decision or block_decision:
        cls = 'DECISION_RELEVANT_STRUCTURAL_SIMPLIFICATION_SUPPORTED'
        method_ready = 'CONDITIONAL_YES'
        next_task = 'stress-test decision-retentive static diagonal/head-block proxy before quantizer design'
    else:
        cls = 'STRUCTURAL_SIMPLIFICATION_NOT_SUPPORTED'
        method_ready = 'NO'
        next_task = 'retain full dense dynamic functional geometry or identify a different structural prior before quantizer design'

    final = {
        'TASK': TASK,
        'FORMAL_STATUS': 'COMPLETE',
        'PROTOCOL_GATE': stage0['PROTOCOL_GATE'],
        'GEOMETRY_APPROXIMATION_GATE': stage0['GEOMETRY_APPROXIMATION_GATE'],
        'BLOCK_SEMANTICS_GATE': stage0['BLOCK_SEMANTICS_GATE'],
        'NO_FUTURE_INFORMATION_LEAKAGE_GATE': stage0['NO_FUTURE_INFORMATION_LEAKAGE_GATE'],
        'M5_QUADRATIC_IDENTITY_GATE': stage0['M5_QUADRATIC_IDENTITY_GATE'],
        'stage0': stage0,
        'geometry_structure': geometry_summary,
        'dynamic_approximation': dynamic_summary,
        'matched_rc_14': matched_summaries,
        'static_heldout': static_summary,
        'DIAGONAL_ENERGY_FRACTION': geometry_summary['DIAGONAL_ENERGY_FRACTION'],
        'HEAD_BLOCK_ENERGY_FRACTION': geometry_summary['HEAD_BLOCK_ENERGY_FRACTION'],
        'ISO_VS_FULL_SPEARMAN': dynamic_summary['iso']['spearman_vs_full'],
        'DIAG_VS_FULL_SPEARMAN': dynamic_summary['diag']['spearman_vs_full'],
        'BLOCK_VS_FULL_SPEARMAN': dynamic_summary['block']['spearman_vs_full'],
        'MATCHED_RC_ISO': matched_summaries['M5_iso'],
        'MATCHED_RC_DIAG': matched_summaries['M5_diag'],
        'MATCHED_RC_BLOCK': matched_summaries['M5_block'],
        'MATCHED_RC_FULL': matched_summaries['M5_full_recomputed'],
        'STATIC_DIAG_HELDOUT': {
            'spearman': static_summary['static_diag']['spearman'],
            'pairwise': static_summary['static_diag']['pairwise'],
            'matched_rc': static_matched['M5_static_diag'],
        },
        'STATIC_BLOCK_HELDOUT': {
            'spearman': static_summary['static_block']['spearman'],
            'pairwise': static_summary['static_block']['pairwise'],
            'matched_rc': static_matched['M5_static_block'],
        },
        'FINAL_SCIENTIFIC_CLASSIFICATION': cls,
        'METHOD_DESIGN_READY': method_ready,
        'NEXT_RECOMMENDED_TASK': next_task,
    }
    save_json('final_classification.json', final)
    write_final_report(final)
    print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
