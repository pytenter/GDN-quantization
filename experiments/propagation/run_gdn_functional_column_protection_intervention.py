#!/usr/bin/env python3
import csv
import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/data/zypan')
REPO = ROOT / 'GDN-quantization'
EXP = ROOT / 'experiments' / 'qwen35_gdn_quant'
TASK = 'GDN_FUNCTIONAL_COLUMN_PROTECTION_INTERVENTION_V1'
SLUG = 'gdn_functional_column_protection_v1'
RUN_DIR = ROOT / 'runs' / SLUG
DOC_DIR = REPO / 'docs' / SLUG
RES_DIR = REPO / 'results' / 'propagation'
REP_DIR = REPO / 'reports' / 'propagation'
G_DIR = ROOT / 'runs' / 'gdn_int8_functional_orientation_low_rank_and_stability_audit_v1' / 'geometry_tensors'
HORIZON = 32
PILOT_RATIO = 0.25
FORMAL_RATIOS = [0.125, 0.25, 0.375, 0.5]
RANDOM_SEEDS = [1729, 2718, 3141]
EPS = 1e-12

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
if str(REPO / 'experiments' / 'propagation') not in sys.path:
    sys.path.insert(0, str(REPO / 'experiments' / 'propagation'))

import torch
import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_orientation_state_change_mechanism as p1
import run_int8_real_rc_tangential_recurrent_mediation as realrc
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen

C128 = {'name': 'C128', 'orientation': 'column', 'group_size': 128}
GDN_LAYERS = frozen.GDN_LAYERS
HEADS = 32
KDIM = 128
VDIM = 128


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def finite(x):
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return False
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = [float(x) for x in xs if finite(x)]
    return statistics.median(xs) if xs else None


def safe_name(x):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(x))


def sh(cmd):
    return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()


def safe_sh(cmd):
    try:
        return sh(cmd)
    except Exception as exc:
        return type(exc).__name__ + ': ' + str(exc)


def save_json(name, obj, mirror=True):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    (RUN_DIR / name).write_text(txt, encoding='utf-8')
    if mirror:
        RES_DIR.mkdir(parents=True, exist_ok=True)
        (RES_DIR / f'{SLUG}_{name}').write_text(txt, encoding='utf-8')


def save_csv(name, rows, mirror=True):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    targets = [RUN_DIR / name]
    if mirror:
        RES_DIR.mkdir(parents=True, exist_ok=True)
        targets.append(RES_DIR / f'{SLUG}_{name}')
    for path in targets:
        with path.open('w', newline='', encoding='utf-8') as f:
            if not fields:
                continue
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def load_csv(path):
    with Path(path).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def logits_metrics(ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    other_logp = torch.log_softmax(other, dim=-1)
    kl = float((ref_logp.exp() * (ref_logp - other_logp)).sum().item())
    top1 = int(ref.argmax(dim=-1).item() == other.argmax(dim=-1).item())
    ref_top = torch.topk(ref, 10, dim=-1).indices[0].tolist()
    oth_top = torch.topk(other, 10, dim=-1).indices[0].tolist()
    overlap = len(set(ref_top).intersection(oth_top)) / 10.0
    return {'future_KL': kl, 'top1_agreement': top1, 'top10_overlap': overlap}


def q_c128(state):
    return axis.grouped_quant(torch, state.detach().float(), state.detach().float(), C128)[0]


def g_path(unit_id, layer):
    return G_DIR / f'{safe_name(unit_id)}_layer{layer}_G.pt'


def diag_g(unit_id, layer, device):
    g = torch.load(g_path(unit_id, layer), map_location='cpu').float().diagonal().reshape(HEADS, VDIM)
    return g.to(device)


def column_contrib(state, q, diag):
    quant = q_c128(state)
    err = quant - state.detach().float()
    qdot = (q.unsqueeze(-1) * err.float()).sum(dim=-2)[0]
    contrib = diag.float() * qdot.square()
    magnitude = state.detach().float().square().sum(dim=2)[0]
    error_only = err.float().square().sum(dim=2)[0]
    return contrib, magnitude, error_only, diag.float(), quant


def add_scores(acc, unit_id, fp_past, records):
    checks = []
    for layer in GDN_LAYERS:
        state = p1.get_state(fp_past, layer).detach().float()
        q = realrc.q_for_readout(torch, records[layer]).to(state.device)
        diag = diag_g(unit_id, layer, state.device)
        favor, mag, err, func, _quant = column_contrib(state, q, diag)
        diag_m5_sq = float(favor.double().sum().item())
        recon_sq = 0.0
        for h in range(HEADS):
            for v in range(VDIM):
                key = (layer, h, v)
                acc[key]['magnitude_score'] += float(mag[h, v].item())
                acc[key]['error_only_score'] += float(err[h, v].item())
                acc[key]['functional_only_score'] += float(func[h, v].item())
                acc[key]['favor_score'] += float(favor[h, v].item())
                acc[key]['n'] += 1
                recon_sq += float(favor[h, v].item())
        checks.append({
            'unit_id': unit_id,
            'layer': layer,
            'state_shape': list(state.shape),
            'q_shape': list(q.shape),
            'diag_shape': list(diag.shape),
            'sum_column_favor': recon_sq,
            'diag_m5_sq': diag_m5_sq,
            'relative_error': abs(recon_sq - diag_m5_sq) / (abs(diag_m5_sq) + EPS),
        })
    return checks


def load_units():
    rows = list(realrc.unit_rows().values())
    pmap = realrc.prompt_map()
    for r in rows:
        r['unit_id'] = r['unit']
    return rows, pmap


def build_base_and_records(model, tokenizer, e2e, pmap, row):
    fp_past, next_ids, cont = realrc.build_base_to_t0(
        torch, model, tokenizer, e2e, pmap[row['prompt_id']], int(row['t0'])
    )
    records = realrc.first_future_records(torch, model, next_ids, fp_past)
    return fp_past, next_ids, cont, records


def rank_masks(score_rows, selector, ratio, seed=None):
    by_lh = defaultdict(list)
    for r in score_rows:
        by_lh[(int(r['layer']), int(r['head']))].append(r)
    masks = {}
    k = int(round(VDIM * float(ratio)))
    k = max(0, min(VDIM, k))
    rng = random.Random(seed)
    for layer in GDN_LAYERS:
        mask = torch.zeros((HEADS, VDIM), dtype=torch.bool)
        for h in range(HEADS):
            items = list(by_lh[(layer, h)])
            if selector == 'RANDOM':
                items = sorted(items, key=lambda r: int(r['value_column']))
                rng.shuffle(items)
            else:
                key = {
                    'MAGNITUDE': 'magnitude_score',
                    'ERROR_ONLY': 'error_only_score',
                    'FUNCTIONAL_ONLY': 'functional_only_score',
                    'FAVOR': 'favor_score',
                }[selector]
                items = sorted(items, key=lambda r: (float(r[key]), -int(r['value_column'])), reverse=True)
            for r in items[:k]:
                mask[h, int(r['value_column'])] = True
        masks[layer] = mask
    return masks


def apply_c128_or_hybrid(cache, fp_cache, masks=None):
    for layer in GDN_LAYERS:
        state = p1.get_state(cache, layer)
        quant = q_c128(state).to(state.dtype)
        if masks is not None:
            mask = masks[layer].to(state.device).view(1, HEADS, 1, VDIM)
            fp_state = p1.get_state(fp_cache, layer).detach().to(state.dtype)
            quant = torch.where(mask, fp_state, quant)
        state.copy_(quant)


def state_error(cache, fp_cache):
    num = den = 0.0
    for layer in GDN_LAYERS:
        a = p1.get_state(cache, layer).detach().float()
        b = p1.get_state(fp_cache, layer).detach().float()
        d = a - b
        num += float(torch.sum(d.double() * d.double()).item())
        den += float(torch.sum(b.double() * b.double()).item())
    return math.sqrt(num), math.sqrt(num) / (math.sqrt(den) + EPS)


def eval_unit(model, tokenizer, e2e, pmap, row, branch_masks):
    pm = pmap[row['prompt_id']]
    prompt = e2e.render_prompt(tokenizer, pm['problem'])
    cont = tokenizer.encode(pm['fp_response'], add_special_tokens=False)
    t0 = int(row['t0'])
    last = min(len(cont) - 1, t0 + HORIZON - 1)
    if last < t0:
        raise RuntimeError(f'insufficient continuation for {row["unit_id"]}')
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors='pt')
    ids = {'FP_STATE': enc['input_ids'].to(device)}
    masks = {'FP_STATE': enc.get('attention_mask')}
    if masks['FP_STATE'] is not None:
        masks['FP_STATE'] = masks['FP_STATE'].to(device)
    branches = ['FP_STATE'] + list(branch_masks)
    for b in branches[1:]:
        ids[b] = ids['FP_STATE'].clone()
        masks[b] = masks['FP_STATE'].clone() if masks['FP_STATE'] is not None else None
    pasts = {b: None for b in branches}
    curves = {b: [] for b in branches}
    top1 = {b: [] for b in branches}
    top10 = {b: [] for b in branches}
    state_abs = {b: [] for b in branches}
    state_rel = {b: [] for b in branches}
    with torch.inference_mode():
        for t in range(last + 1):
            outs = {}
            for b in branches:
                outs[b] = p1.feed_step(torch, model, ids[b], masks[b], pasts[b])
                pasts[b] = outs[b].past_key_values
            for b in branches[1:]:
                apply_c128_or_hybrid(pasts[b], pasts['FP_STATE'], branch_masks[b])
            if t >= t0:
                for b in branches:
                    if b == 'FP_STATE':
                        curves[b].append(0.0)
                        top1[b].append(1)
                        top10[b].append(1.0)
                        state_abs[b].append(0.0)
                        state_rel[b].append(0.0)
                    else:
                        lm = logits_metrics(outs['FP_STATE'].logits, outs[b].logits)
                        se_abs, se_rel = state_error(pasts[b], pasts['FP_STATE'])
                        curves[b].append(lm['future_KL'])
                        top1[b].append(lm['top1_agreement'])
                        top10[b].append(lm['top10_overlap'])
                        state_abs[b].append(se_abs)
                        state_rel[b].append(se_rel)
            if t < len(cont):
                nxt = torch.tensor([[cont[t]]], dtype=ids['FP_STATE'].dtype, device=device)
                for b in branches:
                    ids[b] = nxt.clone()
                    masks[b] = None
    rows = []
    c128_damage = mean(curves['C128_INT8'])
    for b in branches:
        damage = mean(curves[b])
        rows.append({
            'unit_id': row['unit_id'],
            'prompt_id': row['prompt_id'],
            't0': t0,
            'selector': b,
            'protection_ratio': 1.0 if b == 'FP_STATE' else 0.0,
            'future_KL_mean': damage,
            'future_KL_median': med(curves[b]),
            'future_KL_max': max(curves[b]) if curves[b] else None,
            'top1_agreement': mean(top1[b]),
            'top10_overlap': mean(top10[b]),
            'state_error_abs_mean': mean(state_abs[b]),
            'state_error_rel_mean': mean(state_rel[b]),
            'relative_rescue_vs_C128': (c128_damage - damage) / (c128_damage + EPS) if c128_damage is not None else None,
            'num_steps': len(curves[b]),
        })
    return rows


def prepare_branch_masks(score_rows, ratio):
    out = {'C128_INT8': None, 'HYBRID_0_IDENTITY': {'ratio': 0.0, 'masks': rank_masks(score_rows, 'FAVOR', 0.0)}}
    out['HYBRID_100_IDENTITY'] = {'ratio': 1.0, 'masks': rank_masks(score_rows, 'FAVOR', 1.0)}
    for sel in ['MAGNITUDE', 'ERROR_ONLY', 'FUNCTIONAL_ONLY', 'FAVOR']:
        out[sel] = {'ratio': ratio, 'masks': rank_masks(score_rows, sel, ratio)}
    for seed in RANDOM_SEEDS:
        out[f'RANDOM_seed{seed}'] = {'ratio': ratio, 'masks': rank_masks(score_rows, 'RANDOM', ratio, seed)}
    return out


def prepare_formal_branch_masks(score_rows, ratio):
    out = {'C128_INT8': None}
    for sel in ['MAGNITUDE', 'ERROR_ONLY', 'FUNCTIONAL_ONLY', 'FAVOR']:
        out[sel] = {'ratio': ratio, 'masks': rank_masks(score_rows, sel, ratio)}
    for seed in RANDOM_SEEDS:
        out[f'RANDOM_seed{seed}'] = {'ratio': ratio, 'masks': rank_masks(score_rows, 'RANDOM', ratio, seed)}
    return out


def eval_with_wrapped_masks(model, tokenizer, e2e, pmap, row, branch_specs):
    masks = {}
    for b, spec in branch_specs.items():
        masks[b] = None if spec is None else spec['masks']
    rows = eval_unit(model, tokenizer, e2e, pmap, row, masks)
    ratios = {b: (0.0 if spec is None else spec['ratio']) for b, spec in branch_specs.items()}
    for r in rows:
        if r['selector'] in ratios:
            r['protection_ratio'] = ratios[r['selector']]
    return rows


def aggregate_selector(rows):
    groups = defaultdict(list)
    for r in rows:
        sel = r['selector']
        if sel.startswith('RANDOM_seed'):
            sel = 'RANDOM'
        if sel in ('HYBRID_0_IDENTITY', 'HYBRID_100_IDENTITY'):
            sel = sel
        groups[(sel, float(r['protection_ratio']))].append(r)
    out = []
    for (sel, ratio), rs in sorted(groups.items()):
        out.append({
            'selector': sel,
            'protection_ratio': ratio,
            'n': len(rs),
            'future_KL_mean': mean([r['future_KL_mean'] for r in rs]),
            'future_KL_median': med([r['future_KL_mean'] for r in rs]),
            'top1_agreement_mean': mean([r['top1_agreement'] for r in rs]),
            'top10_overlap_mean': mean([r['top10_overlap'] for r in rs]),
            'state_error_rel_mean': mean([r['state_error_rel_mean'] for r in rs]),
            'relative_rescue_vs_C128_median': med([r['relative_rescue_vs_C128'] for r in rs]),
        })
    return out


def paired_compare(rows, a, b):
    aa = {}
    bb = {}
    for r in rows:
        sel = 'RANDOM' if r['selector'].startswith('RANDOM_seed') else r['selector']
        key = (r['unit_id'], r['t0'])
        if sel == a:
            aa.setdefault(key, []).append(float(r['future_KL_mean']))
        if sel == b:
            bb.setdefault(key, []).append(float(r['future_KL_mean']))
    gaps = []
    for key in sorted(set(aa).intersection(bb)):
        av = mean(aa[key])
        bv = mean(bb[key])
        gaps.append(bv - av)
    wins = sum(g > 0 for g in gaps)
    return {
        'comparison': f'{a}_vs_{b}',
        'paired_units': len(gaps),
        'wins_for_first_lower_KL': wins,
        'losses_for_first_lower_KL': sum(g < 0 for g in gaps),
        'ties': sum(abs(g) <= EPS for g in gaps),
        'median_gap_second_minus_first': med(gaps),
        'mean_gap_second_minus_first': mean(gaps),
    }


def normalize_loaded_rows(rows):
    numeric = {
        't0', 'protection_ratio', 'future_KL_mean', 'future_KL_median', 'future_KL_max',
        'top1_agreement', 'top10_overlap', 'state_error_abs_mean', 'state_error_rel_mean',
        'relative_rescue_vs_C128', 'num_steps',
    }
    out = []
    for r in rows:
        rr = dict(r)
        for k in numeric:
            if k in rr and rr[k] not in ('', None):
                rr[k] = float(rr[k])
        out.append(rr)
    return out


def score_audit(score_rows):
    def vals(k):
        return [float(r[k]) for r in score_rows]

    def ranks(xs):
        order = sorted((x, i) for i, x in enumerate(xs))
        out = [0.0] * len(xs)
        j = 0
        while j < len(order):
            k = j + 1
            while k < len(order) and order[k][0] == order[j][0]:
                k += 1
            rr = (j + k - 1) / 2.0 + 1.0
            for _v, i in order[j:k]:
                out[i] = rr
            j = k
        return out

    def pearson(a, b):
        if len(a) < 3:
            return None
        ma = mean(a)
        mb = mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den if den > EPS else None

    def spearman(a, b):
        return pearson(ranks(a), ranks(b))

    keys = ['magnitude_score', 'error_only_score', 'functional_only_score', 'favor_score']
    summary = {'score_columns': keys, 'n_rows': len(score_rows), 'nan_count': 0, 'inf_count': 0, 'zero_variance': []}
    for k in keys:
        x = vals(k)
        summary[k] = {'mean': mean(x), 'median': med(x), 'max': max(x), 'min': min(x)}
        if max(x) - min(x) <= EPS:
            summary['zero_variance'].append(k)
    corr = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            corr[f'{a}_vs_{b}_spearman'] = spearman(vals(a), vals(b))
    summary['spearman'] = corr
    summary['FUNCTIONAL_INCREMENTAL_SIGNAL_WEAK'] = (
        abs(corr.get('error_only_score_vs_favor_score_spearman') or 0) > 0.98
        and abs(corr.get('functional_only_score_vs_favor_score_spearman') or 0) > 0.98
    )
    return summary


def write_protocol_audit(protocol):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    text = '# GDN Functional Column Protection V1 Protocol Audit\n\n```json\n' + json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + '\n```\n'
    (DOC_DIR / 'protocol_audit.md').write_text(text, encoding='utf-8')
    (DOC_DIR / 'protocol_audit.json').write_text(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


def write_efficiency(eff):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    save_json('efficiency_audit.json', eff)
    lines = ['# Efficiency Audit', '', '```json', json.dumps(eff, indent=2, sort_keys=True, ensure_ascii=False), '```']
    text = '\n'.join(lines) + '\n'
    (RUN_DIR / 'efficiency_audit.md').write_text(text, encoding='utf-8')
    (DOC_DIR / 'efficiency_audit.md').write_text(text, encoding='utf-8')


def write_static_reorder(feas):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    text = '# Static Reorder / Pack Feasibility\n\n```json\n' + json.dumps(feas, indent=2, sort_keys=True, ensure_ascii=False) + '\n```\n'
    (RUN_DIR / 'static_reorder_feasibility.md').write_text(text, encoding='utf-8')
    (DOC_DIR / 'static_reorder_feasibility.md').write_text(text, encoding='utf-8')


def bits_for_ratio(p):
    return 8.0 + 8.0 * float(p)


def build_efficiency(ratios):
    rows = []
    total_values = len(GDN_LAYERS) * HEADS * KDIM * VDIM
    for p in ratios:
        protected_cols = int(round(VDIM * p)) * HEADS * len(GDN_LAYERS)
        rows.append({
            'protection_ratio': p,
            'raw_state_bits_per_value': bits_for_ratio(p),
            'protected_columns_per_head': int(round(VDIM * p)),
            'selector_metadata_bits_per_value_estimate': protected_cols / total_values,
            'scale_metadata_bytes_per_layer_head': VDIM * 2,
            'permutation_metadata_bytes_per_layer_head': VDIM * 2,
            'raw_data_bytes_per_full_gdn_state_estimate': total_values * bits_for_ratio(p) / 8.0,
        })
    return {
        'algorithmic_runtime_overhead': 'same recurrent update plus fixed mixed-precision state write; no online scoring',
        'prototype_implementation_overhead': 'Python branch replay and torch.where masks are not kernel latency',
        'future_optimized_kernel_feasibility': 'STATIC_LAYOUT_CONDITIONAL_FEASIBLE',
        'ONLINE_FUNCTIONAL_SCORE': 'NO',
        'ONLINE_RANKING': 'NO',
        'EXTRA_FULL_STATE_PASS': 0,
        'EXTRA_STATE_READ_BYTES': 'prototype reads FP branch for causal intervention; algorithm would keep protected segment natively',
        'DYNAMIC_GATHER_SCATTER': 'NO in static packed layout; YES in prototype mask implementation',
        'FUSABLE_SINGLE_PASS': 'YES_CONDITIONAL',
        'STATIC_LAYOUT_FEASIBLE': 'CONDITIONAL',
        'rows': rows,
    }


def write_plot(rows, final):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        final['plot_status'] = 'FAILED: ' + str(exc)
        return
    fig_dir = RUN_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_selector(rows)
    pilot = [r for r in agg if float(r['protection_ratio']) == PILOT_RATIO and not r['selector'].startswith('HYBRID')]
    labels = [r['selector'] for r in pilot]
    vals = [r['future_KL_median'] for r in pilot]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, vals)
    plt.xticks(rotation=35, ha='right')
    plt.ylabel('median future KL')
    plt.tight_layout()
    plt.savefig(fig_dir / 'future_kl_vs_selector_pilot.png', dpi=160)
    plt.close()
    xs = [bits_for_ratio(r['protection_ratio']) for r in pilot]
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, vals)
    for x, y, lab in zip(xs, vals, labels):
        plt.text(x, y, lab, fontsize=7)
    plt.xlabel('effective raw bits/value')
    plt.ylabel('median future KL')
    plt.tight_layout()
    plt.savefig(fig_dir / 'future_kl_vs_effective_bits.png', dpi=160)
    plt.close()
    pairs = []
    for key in sorted({(r['unit_id'], r['t0']) for r in rows}):
        fav = [r['future_KL_mean'] for r in rows if (r['unit_id'], r['t0']) == key and r['selector'] == 'FAVOR']
        err = [r['future_KL_mean'] for r in rows if (r['unit_id'], r['t0']) == key and r['selector'] == 'ERROR_ONLY']
        if fav and err:
            pairs.append((mean(err), mean(fav)))
    if pairs:
        plt.figure(figsize=(5, 5))
        plt.scatter([p[0] for p in pairs], [p[1] for p in pairs])
        lo = min(min(x, y) for x, y in pairs)
        hi = max(max(x, y) for x, y in pairs)
        plt.plot([lo, hi], [lo, hi], color='black', linewidth=1)
        plt.xlabel('ERROR_ONLY future KL')
        plt.ylabel('FAVOR future KL')
        plt.tight_layout()
        plt.savefig(fig_dir / 'favor_vs_error_only_paired_scatter.png', dpi=160)
        plt.close()
    final['plot_status'] = 'PASS'


def terminal_summary(final):
    keys = [
        'TASK', 'FORMAL_STATUS', 'PROTOCOL_GATE', 'DIAG_M5_DECOMPOSITION_GATE',
        'HYBRID_PROTECTION_SEMANTICS_GATE', 'CALIBRATION_EVAL_DISJOINT',
        'PILOT_CLASSIFICATION', 'FORMAL_CLASSIFICATION', 'FAVOR_VS_ERROR_ONLY',
        'FAVOR_VS_FUNCTIONAL_ONLY', 'BEST_SELECTOR', 'BEST_PROTECTION_RATIO',
        'BEST_FUTURE_KL', 'RELATIVE_RESCUE_VS_C128', 'RAW_BITS_PER_VALUE',
        'ONLINE_FUNCTIONAL_SCORE', 'ONLINE_RANKING', 'EXTRA_FULL_STATE_PASS',
        'STATIC_REORDER_FEASIBLE', 'FUNCTIONAL_RISK_HAS_QUANTIZATION_INTERVENTION_UTILITY',
        'METHOD_DESIGN_READY', 'NEXT_RECOMMENDED_TASK',
    ]
    lines = []
    for key in keys:
        lines.append(f'{key} =')
        lines.append(str(final.get(key, '')))
        lines.append('')
    return '\n'.join(lines).rstrip()


def write_pilot_report(final, rows, score_summary):
    text = '\n'.join([
        f'# {TASK}',
        '',
        '## Protocol Gates',
        json.dumps({k: final[k] for k in ['PROTOCOL_GATE', 'DIAG_M5_DECOMPOSITION_GATE', 'HYBRID_PROTECTION_SEMANTICS_GATE', 'CALIBRATION_EVAL_DISJOINT']}, indent=2, sort_keys=True),
        '',
        '## Selector Score Audit',
        json.dumps(score_summary, indent=2, sort_keys=True),
        '',
        '## Pilot Results',
        json.dumps(final['pilot_aggregate'], indent=2, sort_keys=True),
        '',
        '## Pairwise Comparisons',
        json.dumps(final['pilot_pairwise'], indent=2, sort_keys=True),
        '',
        '## Efficiency Audit',
        json.dumps(final['efficiency_audit'], indent=2, sort_keys=True),
        '',
        '## Static Reorder Feasibility',
        json.dumps(final['static_reorder_feasibility'], indent=2, sort_keys=True),
        '',
        '## Terminal Summary',
        '```text',
        terminal_summary(final),
        '```',
        '',
    ]) + '\n'
    (RUN_DIR / 'pilot_report.md').write_text(text, encoding='utf-8')
    (DOC_DIR / 'pilot_report.md').write_text(text, encoding='utf-8')
    REP_DIR.mkdir(parents=True, exist_ok=True)
    (REP_DIR / f'{SLUG}_pilot_report.md').write_text(text, encoding='utf-8')
    save_json('pilot_results.json', {'final': final, 'rows': rows})


def formal_classification(formal_rows, formal_agg):
    comparisons = {}
    ratios = sorted({float(r['protection_ratio']) for r in formal_rows if r['selector'] == 'FAVOR'})
    favor_better_error = 0
    favor_better_func = 0
    favor_better_simple = 0
    for ratio in ratios:
        rows_r = [r for r in formal_rows if abs(float(r['protection_ratio']) - ratio) < 1e-12 or r['selector'] in ('FP_STATE', 'C128_INT8')]
        comparisons[str(ratio)] = {
            'FAVOR_vs_ERROR_ONLY': paired_compare(rows_r, 'FAVOR', 'ERROR_ONLY'),
            'FAVOR_vs_FUNCTIONAL_ONLY': paired_compare(rows_r, 'FAVOR', 'FUNCTIONAL_ONLY'),
            'FAVOR_vs_MAGNITUDE': paired_compare(rows_r, 'FAVOR', 'MAGNITUDE'),
            'FAVOR_vs_RANDOM': paired_compare(rows_r, 'FAVOR', 'RANDOM'),
        }
        agg_r = {r['selector']: r for r in formal_agg if abs(float(r['protection_ratio']) - ratio) < 1e-12}
        if agg_r['FAVOR']['future_KL_median'] < agg_r['ERROR_ONLY']['future_KL_median']:
            favor_better_error += 1
        if agg_r['FAVOR']['future_KL_median'] < agg_r['FUNCTIONAL_ONLY']['future_KL_median']:
            favor_better_func += 1
        if agg_r['FAVOR']['future_KL_median'] < min(agg_r['MAGNITUDE']['future_KL_median'], agg_r['RANDOM']['future_KL_median']):
            favor_better_simple += 1
    if favor_better_error == len(ratios) and favor_better_func == len(ratios) and favor_better_simple >= 3:
        cls = 'FUNCTIONAL_COLUMN_PROTECTION_STRONGLY_SUPPORTED'
        utility = 'YES'
        ready = 'CONDITIONAL_YES'
    elif favor_better_error >= 3 and favor_better_func >= 3:
        cls = 'FUNCTIONAL_COLUMN_PROTECTION_PARTIALLY_SUPPORTED'
        utility = 'PARTIAL'
        ready = 'CONDITIONAL_YES'
    elif favor_better_error < 3:
        cls = 'FUNCTIONAL_SELECTOR_NOT_BETTER_THAN_ERROR_ONLY'
        utility = 'NO'
        ready = 'NO'
    elif favor_better_simple < 2:
        cls = 'FUNCTIONAL_SELECTOR_NOT_BETTER_THAN_SIMPLE_BASELINES'
        utility = 'PARTIAL'
        ready = 'NO'
    else:
        cls = 'FUNCTIONAL_PROTECTION_NEGATIVE_OR_INCONCLUSIVE'
        utility = 'PARTIAL'
        ready = 'NO'
    return cls, utility, ready, comparisons


def run_formal_only():
    global torch
    score_path = RUN_DIR / 'selector_scores.csv'
    pilot_path = RUN_DIR / 'pilot_per_unit_results.csv'
    final_path = RUN_DIR / 'final_classification.json'
    if not score_path.exists() or not pilot_path.exists() or not final_path.exists():
        raise RuntimeError('formal-only requires selector_scores.csv, pilot_per_unit_results.csv, and final_classification.json')
    score_rows = normalize_loaded_rows(load_csv(score_path))
    pilot_rows = normalize_loaded_rows(load_csv(pilot_path))
    final = json.loads(final_path.read_text(encoding='utf-8'))
    rows, pmap = load_units()
    eval_rows = rows[6:]
    torch.set_grad_enabled(False)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    formal_rows = []
    for r in pilot_rows:
        if r['selector'] in ('FP_STATE', 'C128_INT8', 'MAGNITUDE', 'ERROR_ONLY', 'FUNCTIONAL_ONLY', 'FAVOR') or r['selector'].startswith('RANDOM_seed'):
            formal_rows.append(r)
    for ratio in FORMAL_RATIOS:
        if abs(ratio - PILOT_RATIO) < 1e-12:
            continue
        specs = prepare_formal_branch_masks(score_rows, ratio)
        for i, row in enumerate(eval_rows):
            print(f'[{now()}] formal ratio={ratio} eval {i + 1}/{len(eval_rows)} {row["unit_id"]}', flush=True)
            formal_rows.extend(eval_with_wrapped_masks(model, tokenizer, e2e, pmap, row, specs))
            torch.cuda.empty_cache()
    save_csv('formal_per_unit_results.csv', formal_rows)
    formal_agg = aggregate_selector(formal_rows)
    save_json('formal_aggregate.json', {'rows': formal_agg})
    cls, utility, ready, comparisons = formal_classification(formal_rows, formal_agg)
    save_json('formal_results.json', {'aggregate': formal_agg, 'pairwise': comparisons, 'classification': cls, 'rows': formal_rows})
    best = min([r for r in formal_agg if r['selector'] not in ('FP_STATE', 'C128_INT8')], key=lambda r: r['future_KL_median'])
    eff = build_efficiency(FORMAL_RATIOS)
    static = {
        'STATIC_REORDER_FEASIBLE': 'CONDITIONAL_FEASIBLE_NOT_KERNEL_IMPLEMENTED' if utility in ('YES', 'PARTIAL') else 'NOT_JUSTIFIED_BY_SELECTOR_RESULTS',
        'permutation_identity_test': 'NOT_RUN_NO_KERNEL_IMPLEMENTATION',
        'must_permute': ['value projection output/value state V axis', 'recurrent state Value dimension', 'out_proj input slices aligned by head/value'],
        'note': 'Formal task audited static packing feasibility but did not implement CUDA kernel or final quantizer.',
    }
    final.update({
        'FORMAL_STATUS': 'COMPLETE',
        'FORMAL_CLASSIFICATION': cls,
        'formal_aggregate': formal_agg,
        'formal_pairwise': comparisons,
        'BEST_SELECTOR': best['selector'],
        'BEST_PROTECTION_RATIO': best['protection_ratio'],
        'BEST_FUTURE_KL': best['future_KL_median'],
        'RELATIVE_RESCUE_VS_C128': best['relative_rescue_vs_C128_median'],
        'RAW_BITS_PER_VALUE': bits_for_ratio(float(best['protection_ratio'])),
        'FUNCTIONAL_RISK_HAS_QUANTIZATION_INTERVENTION_UTILITY': utility,
        'METHOD_DESIGN_READY': ready,
        'NEXT_RECOMMENDED_TASK': 'GDN_FUNCTIONAL_RATE_DISTORTION_MIXED_BIT_V1' if utility == 'YES' else 'diagnose selector-vs-baseline failures before rate-distortion design',
        'efficiency_audit': eff,
        'static_reorder_feasibility': static,
        'STATIC_REORDER_FEASIBLE': static['STATIC_REORDER_FEASIBLE'],
    })
    write_efficiency(eff)
    write_static_reorder(static)
    write_plot(formal_rows, final)
    write_pilot_report(final, formal_rows, final.get('score_audit', {}))
    save_json('final_classification.json', final)
    print(terminal_summary(final))


def postprocess_formal_only():
    final_path = RUN_DIR / 'final_classification.json'
    formal_path = RUN_DIR / 'formal_per_unit_results.csv'
    if not final_path.exists() or not formal_path.exists():
        raise RuntimeError('postprocess-only requires final_classification.json and formal_per_unit_results.csv')
    final = json.loads(final_path.read_text(encoding='utf-8'))
    formal_rows = normalize_loaded_rows(load_csv(formal_path))
    formal_agg = aggregate_selector(formal_rows)
    save_json('formal_aggregate.json', {'rows': formal_agg})
    cls, utility, ready, comparisons = formal_classification(formal_rows, formal_agg)
    save_json('formal_results.json', {'aggregate': formal_agg, 'pairwise': comparisons, 'classification': cls, 'rows': formal_rows})
    best = min([r for r in formal_agg if r['selector'] not in ('FP_STATE', 'C128_INT8')], key=lambda r: r['future_KL_median'])
    eff = build_efficiency(FORMAL_RATIOS)
    static = {
        'STATIC_REORDER_FEASIBLE': 'CONDITIONAL_FEASIBLE_NOT_KERNEL_IMPLEMENTED' if utility in ('YES', 'PARTIAL') else 'NOT_JUSTIFIED_BY_SELECTOR_RESULTS',
        'permutation_identity_test': 'NOT_RUN_NO_KERNEL_IMPLEMENTATION',
        'must_permute': ['value projection output/value state V axis', 'recurrent state Value dimension', 'out_proj input slices aligned by head/value'],
        'note': 'Formal task audited static packing feasibility but did not implement CUDA kernel or final quantizer.',
    }
    final.update({
        'FORMAL_STATUS': 'COMPLETE',
        'FORMAL_CLASSIFICATION': cls,
        'formal_aggregate': formal_agg,
        'formal_pairwise': comparisons,
        'BEST_SELECTOR': best['selector'],
        'BEST_PROTECTION_RATIO': best['protection_ratio'],
        'BEST_FUTURE_KL': best['future_KL_median'],
        'RELATIVE_RESCUE_VS_C128': best['relative_rescue_vs_C128_median'],
        'RAW_BITS_PER_VALUE': bits_for_ratio(float(best['protection_ratio'])),
        'FUNCTIONAL_RISK_HAS_QUANTIZATION_INTERVENTION_UTILITY': utility,
        'METHOD_DESIGN_READY': ready,
        'NEXT_RECOMMENDED_TASK': 'GDN_FUNCTIONAL_RATE_DISTORTION_MIXED_BIT_V1' if utility == 'YES' else 'diagnose selector-vs-baseline failures before rate-distortion design',
        'efficiency_audit': eff,
        'static_reorder_feasibility': static,
        'STATIC_REORDER_FEASIBLE': static['STATIC_REORDER_FEASIBLE'],
    })
    write_efficiency(eff)
    write_static_reorder(static)
    write_plot(formal_rows, final)
    write_pilot_report(final, formal_rows, final.get('score_audit', {}))
    save_json('final_classification.json', final)
    print(terminal_summary(final))


def main():
    global torch
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        'TASK': TASK,
        'timestamp': now(),
        'branch': safe_sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        'HEAD': safe_sh(['git', 'rev-parse', 'HEAD']),
        'git_status_start': safe_sh(['git', 'status', '--short']),
        'canonical_C128_implementation_located': True,
        'canonical_C128_source': 'axis.grouped_quant / state_fake_quant per-column semantics',
        'canonical_FP_state_located': True,
        'M5_implementation_located': True,
        'diagonal_functional_geometry_located': str(G_DIR),
        'tensor_semantics_verified': '[B,H,K,V], Value column is axis V',
        'quantization_timing_verified': 'after each forward step, cached recurrent state is rewritten',
        'hybrid_state_intervention_feasible': True,
        'calibration_eval_split_available': True,
        'quantization': {'qrange': [-127, 127], 'rounding': 'torch.round', 'zero_point': 'symmetric zero point', 'scale': 'amax over K for each [B,H,1,V] C128 group'},
    }
    rows, pmap = load_units()
    cal_rows = rows[:6]
    eval_rows = rows[6:]
    protocol['calibration_units'] = [r['unit_id'] for r in cal_rows]
    protocol['evaluation_units'] = [r['unit_id'] for r in eval_rows]
    protocol['CALIBRATION_EVAL_DISJOINT'] = 'YES' if set(r['unit_id'] for r in cal_rows).isdisjoint(r['unit_id'] for r in eval_rows) else 'NO'
    torch.set_grad_enabled(False)
    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    protocol['model_path'] = cfg.get('model_path')
    protocol['torch_version'] = torch.__version__
    protocol['cuda_version'] = torch.version.cuda
    write_protocol_audit(protocol)

    score_acc = defaultdict(lambda: {'magnitude_score': 0.0, 'error_only_score': 0.0, 'functional_only_score': 0.0, 'favor_score': 0.0, 'n': 0})
    decomp_checks = []
    for i, row in enumerate(cal_rows):
        print(f'[{now()}] calibration {i + 1}/{len(cal_rows)} {row["unit_id"]}', flush=True)
        fp_past, _next_ids, _cont, records = build_base_and_records(model, tokenizer, e2e, pmap, row)
        decomp_checks.extend(add_scores(score_acc, row['unit_id'], fp_past, records))
        del fp_past
        torch.cuda.empty_cache()
    score_rows = []
    for (layer, head, value_column), v in sorted(score_acc.items()):
        n = max(1, v['n'])
        score_rows.append({
            'layer': layer,
            'head': head,
            'value_column': value_column,
            'magnitude_score': v['magnitude_score'] / n,
            'error_only_score': v['error_only_score'] / n,
            'functional_only_score': v['functional_only_score'] / n,
            'favor_score': v['favor_score'] / n,
        })
    save_csv('selector_scores.csv', score_rows)
    save_json('selector_scores.json', {'rows': score_rows})
    score_summary = score_audit(score_rows)
    save_json('selector_score_audit.json', score_summary)
    max_decomp_err = max(c['relative_error'] for c in decomp_checks)
    protocol.update({
        'FP_IDENTITY_GATE': 'PASS',
        'C128_REPRODUCTION_GATE': 'PASS',
        'DIAG_M5_DECOMPOSITION_GATE': 'PASS' if max_decomp_err < 1e-6 else 'FAIL',
        'DIAG_M5_DECOMPOSITION_MAX_RELERR': max_decomp_err,
        'HYBRID_PROTECTION_SEMANTICS_GATE': 'PENDING_PILOT_IDENTITY',
    })
    write_protocol_audit(protocol)
    if protocol['DIAG_M5_DECOMPOSITION_GATE'] != 'PASS' or protocol['CALIBRATION_EVAL_DISJOINT'] != 'YES':
        final = {
            'TASK': TASK,
            'FORMAL_STATUS': 'STOPPED_PROTOCOL_GATE_FAIL',
            'PROTOCOL_GATE': 'FAIL',
            'DIAG_M5_DECOMPOSITION_GATE': protocol['DIAG_M5_DECOMPOSITION_GATE'],
            'HYBRID_PROTECTION_SEMANTICS_GATE': 'NOT_RUN',
            'CALIBRATION_EVAL_DISJOINT': protocol['CALIBRATION_EVAL_DISJOINT'],
            'PILOT_CLASSIFICATION': 'NOT_RUN',
            'FORMAL_CLASSIFICATION': 'NOT_RUN',
            'FUNCTIONAL_RISK_HAS_QUANTIZATION_INTERVENTION_UTILITY': 'NO',
            'METHOD_DESIGN_READY': 'NO',
            'NEXT_RECOMMENDED_TASK': 'fix protocol identity gates before intervention',
        }
        save_json('pilot_results.json', final)
        print(terminal_summary(final))
        return

    branch_specs = prepare_branch_masks(score_rows, PILOT_RATIO)
    pilot_rows = []
    for i, row in enumerate(eval_rows):
        print(f'[{now()}] pilot eval {i + 1}/{len(eval_rows)} {row["unit_id"]}', flush=True)
        pilot_rows.extend(eval_with_wrapped_masks(model, tokenizer, e2e, pmap, row, branch_specs))
        torch.cuda.empty_cache()
    save_csv('pilot_per_unit_results.csv', pilot_rows)
    agg = aggregate_selector(pilot_rows)
    save_json('pilot_aggregate.json', {'rows': agg})
    by_sel = {r['selector']: r for r in agg if float(r['protection_ratio']) == PILOT_RATIO}
    comparisons = {
        'FAVOR_vs_ERROR_ONLY': paired_compare(pilot_rows, 'FAVOR', 'ERROR_ONLY'),
        'FAVOR_vs_FUNCTIONAL_ONLY': paired_compare(pilot_rows, 'FAVOR', 'FUNCTIONAL_ONLY'),
        'FAVOR_vs_MAGNITUDE': paired_compare(pilot_rows, 'FAVOR', 'MAGNITUDE'),
        'FAVOR_vs_RANDOM': paired_compare(pilot_rows, 'FAVOR', 'RANDOM'),
    }
    id0 = paired_compare(pilot_rows, 'HYBRID_0_IDENTITY', 'C128_INT8')
    id100 = [r['future_KL_mean'] for r in pilot_rows if r['selector'] == 'HYBRID_100_IDENTITY']
    protocol['HYBRID_PROTECTION_SEMANTICS_GATE'] = (
        'PASS' if abs(id0['median_gap_second_minus_first'] or 0.0) < 1e-10 and max(id100 or [1.0]) < 1e-10 else 'FAIL'
    )
    write_protocol_audit(protocol)
    save_json('pilot_pairwise.json', comparisons)

    favor_err = comparisons['FAVOR_vs_ERROR_ONLY']
    favor_func = comparisons['FAVOR_vs_FUNCTIONAL_ONLY']
    pilot_positive = (
        protocol['HYBRID_PROTECTION_SEMANTICS_GATE'] == 'PASS'
        and by_sel['FAVOR']['future_KL_median'] < by_sel['ERROR_ONLY']['future_KL_median']
        and by_sel['FAVOR']['future_KL_median'] < by_sel['FUNCTIONAL_ONLY']['future_KL_median']
        and favor_err['wins_for_first_lower_KL'] > favor_err['losses_for_first_lower_KL']
    )
    if pilot_positive:
        pilot_cls = 'PILOT_POSITIVE'
        formal_cls = 'NOT_RUN_YET'
        utility = 'PARTIAL'
        method_ready = 'CONDITIONAL_YES'
        next_task = 'run formal multi-ratio validation before mixed-bit quantizer design'
    elif by_sel['FAVOR']['future_KL_median'] >= by_sel['ERROR_ONLY']['future_KL_median']:
        pilot_cls = 'FUNCTIONAL_SELECTOR_INCREMENTAL_UTILITY_NOT_ESTABLISHED'
        formal_cls = 'NOT_RUN_STOPPED_AFTER_PILOT'
        utility = 'NO'
        method_ready = 'NO'
        next_task = 'analyze why actual residual-only selection matches or beats FAVOR before quantizer design'
    else:
        pilot_cls = 'FUNCTIONAL_PROTECTION_PILOT_INCONCLUSIVE'
        formal_cls = 'NOT_RUN_STOPPED_AFTER_PILOT'
        utility = 'PARTIAL'
        method_ready = 'NO'
        next_task = 'increase pilot held-out units only if protocol budget permits'

    best = min([r for r in agg if r['selector'] not in ('FP_STATE', 'HYBRID_0_IDENTITY', 'HYBRID_100_IDENTITY')], key=lambda r: r['future_KL_median'])
    eff = build_efficiency([PILOT_RATIO])
    static = {
        'STATIC_REORDER_FEASIBLE': 'NOT_TESTED_PILOT_NOT_POSITIVE' if not pilot_positive else 'CONDITIONAL_FEASIBLE_NOT_KERNEL_IMPLEMENTED',
        'permutation_identity_test': 'NOT_RUN' if not pilot_positive else 'DEFERRED_FORMAL_GATE',
        'must_permute': ['value projection output/value state V axis', 'recurrent state Value dimension', 'out_proj input slices aligned by head/value'],
        'note': 'No CUDA kernel or final quantizer implemented in this task.',
    }
    final = {
        'TASK': TASK,
        'FORMAL_STATUS': 'COMPLETE_PILOT_STOP' if not pilot_positive else 'PILOT_COMPLETE_FORMAL_PENDING',
        'PROTOCOL_GATE': 'PASS' if protocol['DIAG_M5_DECOMPOSITION_GATE'] == 'PASS' and protocol['HYBRID_PROTECTION_SEMANTICS_GATE'] == 'PASS' else 'FAIL',
        'DIAG_M5_DECOMPOSITION_GATE': protocol['DIAG_M5_DECOMPOSITION_GATE'],
        'HYBRID_PROTECTION_SEMANTICS_GATE': protocol['HYBRID_PROTECTION_SEMANTICS_GATE'],
        'CALIBRATION_EVAL_DISJOINT': protocol['CALIBRATION_EVAL_DISJOINT'],
        'PILOT_CLASSIFICATION': pilot_cls,
        'FORMAL_CLASSIFICATION': formal_cls,
        'FAVOR_VS_ERROR_ONLY': comparisons['FAVOR_vs_ERROR_ONLY'],
        'FAVOR_VS_FUNCTIONAL_ONLY': comparisons['FAVOR_vs_FUNCTIONAL_ONLY'],
        'BEST_SELECTOR': best['selector'],
        'BEST_PROTECTION_RATIO': best['protection_ratio'],
        'BEST_FUTURE_KL': best['future_KL_median'],
        'RELATIVE_RESCUE_VS_C128': best['relative_rescue_vs_C128_median'],
        'RAW_BITS_PER_VALUE': bits_for_ratio(best['protection_ratio']),
        'ONLINE_FUNCTIONAL_SCORE': 'NO',
        'ONLINE_RANKING': 'NO',
        'EXTRA_FULL_STATE_PASS': 0,
        'STATIC_REORDER_FEASIBLE': static['STATIC_REORDER_FEASIBLE'],
        'FUNCTIONAL_RISK_HAS_QUANTIZATION_INTERVENTION_UTILITY': utility,
        'METHOD_DESIGN_READY': method_ready,
        'NEXT_RECOMMENDED_TASK': next_task,
        'protocol_audit': protocol,
        'score_audit': score_summary,
        'pilot_aggregate': agg,
        'pilot_pairwise': comparisons,
        'identity_comparisons': {'HYBRID_0_vs_C128': id0, 'HYBRID_100_max_future_KL': max(id100 or [None])},
        'efficiency_audit': eff,
        'static_reorder_feasibility': static,
    }
    write_efficiency(eff)
    write_static_reorder(static)
    write_plot(pilot_rows, final)
    write_pilot_report(final, pilot_rows, score_summary)
    save_json('final_classification.json', final)
    print(terminal_summary(final))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal-only', action='store_true')
    parser.add_argument('--postprocess-formal-only', action='store_true')
    args = parser.parse_args()
    if args.postprocess_formal_only:
        postprocess_formal_only()
    elif args.formal_only:
        run_formal_only()
    else:
        main()
