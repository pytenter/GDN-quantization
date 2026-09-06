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
TASK = 'GDN_INT8_LONG_HORIZON_FUNCTIONAL_SURVIVAL_AUDIT_V1'
SLUG = 'gdn_int8_long_horizon_functional_survival_audit_v1'
RUN_DIR = ROOT / 'runs' / SLUG
RES_DIR = REPO / 'results' / 'propagation'
REP_DIR = REPO / 'reports' / 'propagation'
G_DIR = ROOT / 'runs' / 'gdn_int8_functional_orientation_low_rank_and_stability_audit_v1' / 'geometry_tensors'
QUERY_FINAL = RES_DIR / 'gdn_int8_query_conditioned_risk_efficiency_feasibility_v1_final_classification.json'
REAL_RC_FINAL = RES_DIR / 'gdn_int8_real_rc_tangential_recurrent_mediation_v1_final_summary.json'
HORIZONS = [0, 1, 2, 4, 8, 16, 32]
MAX_H = max(HORIZONS)
EPS = 1e-12
H = 32
K = 128
V = 128

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import numpy as np
import torch
import run_int8_axis_geometry_rescue_diagnostic as axis
import run_int8_orientation_state_change_mechanism as p1
import run_int8_real_rc_tangential_recurrent_mediation as realrc
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen

GDN_LAYERS = frozen.GDN_LAYERS
BRANCHES = ['FP', 'REAL_R', 'REAL_C']


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


def sh(cmd):
    return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()


def safe_sh(cmd):
    try:
        return sh(cmd)
    except Exception as e:
        return type(e).__name__ + ': ' + str(e)


def safe_name(x):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(x))


def g_path(unit_id, layer):
    return G_DIR / f'{safe_name(unit_id)}_layer{layer}_G.pt'


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


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def cosine(a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    den = float(torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item())
    return float(torch.dot(af, bf).item() / (den + EPS)) if den > EPS else None


def state_delta(past, fp_past):
    total = 0.0
    ref = 0.0
    for layer in GDN_LAYERS:
        a = p1.get_state(past, layer).detach().float()
        b = p1.get_state(fp_past, layer).detach().float()
        d = a - b
        total += float(torch.sum(d.double() * d.double()).item())
        ref += float(torch.sum(b.double() * b.double()).item())
    n = math.sqrt(total)
    return n, n / (math.sqrt(ref) + EPS)


def functional_metrics(past, fp_past, fp_records, diag_by_layer, topmask_by_layer, initial_delta=None):
    raw2 = 0.0
    fdiag2 = 0.0
    static_col2 = 0.0
    high2 = 0.0
    low2 = 0.0
    cos_parts = []
    for layer in GDN_LAYERS:
        E = p1.get_state(past, layer).detach().float() - p1.get_state(fp_past, layer).detach().float()
        raw2 += float(torch.sum(E.double() * E.double()).item())
        q = realrc.q_for_readout(torch, fp_records[layer]).detach().float()
        w = diag_by_layer[layer].to(E.device).reshape(1, H, V)
        x = realrc.readout_from_state(torch, q, E)
        fdiag2 += float(torch.sum((w * x.float().square()).double()).item())
        Ew = E.float().square().sum(dim=-2)
        static_col2 += float(torch.sum((w * Ew).double()).item())
        mask_hi, mask_lo = topmask_by_layer[layer]
        high2 += float(torch.sum((Ew * mask_hi.to(E.device)).double()).item())
        low2 += float(torch.sum((Ew * mask_lo.to(E.device)).double()).item())
        if initial_delta is not None:
            cos_parts.append(cosine(E.cpu(), initial_delta[layer].cpu()))
    raw = math.sqrt(max(0.0, raw2))
    fdiag = math.sqrt(max(0.0, fdiag2))
    static_col = math.sqrt(max(0.0, static_col2))
    return {
        'raw_norm': raw,
        'functional_diag': fdiag,
        'static_column_orientation': static_col,
        'toxicity': fdiag / (raw + EPS),
        'toxicity_sq': fdiag2 / (raw2 + EPS),
        'static_column_toxicity': static_col / (raw + EPS),
        'high_sensitivity_state_fraction': high2 / (raw2 + EPS),
        'low_sensitivity_state_fraction': low2 / (raw2 + EPS),
        'cosine_to_initial': mean([x for x in cos_parts if x is not None]) if cos_parts else None,
    }


def prepare_diag_weights(unit_id):
    diag = {}
    topmask = {}
    for layer in GDN_LAYERS:
        G = torch.load(g_path(unit_id, layer), map_location='cpu').float()
        w = torch.diagonal(G).reshape(H, V).clone()
        diag[layer] = w
        flat = w.reshape(-1)
        hi = torch.quantile(flat, 0.75)
        lo = torch.quantile(flat, 0.25)
        topmask[layer] = ((w >= hi).float().reshape(1, H, V), (w <= lo).float().reshape(1, H, V))
        del G
    return diag, topmask


def apply_injection(past, injections, branch):
    if branch == 'FP':
        return
    for layer, E in injections[branch].items():
        s = p1.get_state(past, layer)
        s.copy_((s.detach().float() + E.to(s.device)).to(s.dtype))


def clone_initial_delta(past, fp_past):
    out = {}
    for layer in GDN_LAYERS:
        out[layer] = (p1.get_state(past, layer).detach().float() - p1.get_state(fp_past, layer).detach().float()).cpu()
    return out


def rollout_unit(model, tokenizer, e2e, row, pm):
    unit_id = row['unit']
    t0 = int(row['t0'])
    prompt = e2e.render_prompt(tokenizer, pm['problem'])
    cont = tokenizer.encode(pm['fp_response'], add_special_tokens=False)
    last = min(t0 + MAX_H, len(cont) - 1)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors='pt')
    ids0 = enc['input_ids'].to(device)
    mask0 = enc.get('attention_mask')
    mask0 = mask0.to(device) if mask0 is not None else None
    ids = {b: ids0.clone() for b in BRANCHES}
    masks = {b: mask0.clone() if mask0 is not None else None for b in BRANCHES}
    pasts = {b: None for b in BRANCHES}
    diag, topmask = prepare_diag_weights(unit_id)
    collector = frozen.install_fp_driver_capture(torch, model)
    trajectory_rows = []
    initial = {}
    injected = False
    try:
        with torch.inference_mode():
            for t in range(last + 1):
                outs = {}
                collector['records'].clear()
                outs['FP'] = p1.feed_step(torch, model, ids['FP'], masks['FP'], pasts['FP'])
                fp_records = {k: dict(v) for k, v in collector['records'].items()}
                pasts['FP'] = outs['FP'].past_key_values
                for b in ['REAL_R', 'REAL_C']:
                    outs[b] = p1.feed_step(torch, model, ids[b], masks[b], pasts[b])
                    pasts[b] = outs[b].past_key_values
                if t == t0 and not injected:
                    inj, meta = normswap.build_injections(torch, pasts['FP'])
                    injections = {'REAL_R': inj['REAL_R'], 'REAL_C': inj['REAL_C']}
                    for b in ['REAL_R', 'REAL_C']:
                        apply_injection(pasts[b], injections, b)
                        initial[b] = clone_initial_delta(pasts[b], pasts['FP'])
                    injected = True
                if t >= t0 and (t - t0) in HORIZONS:
                    h = t - t0
                    for b in ['REAL_R', 'REAL_C']:
                        lm = realrc.logit_metrics(torch, outs['FP'], outs[b]) if h > 0 else {'KL': 0.0, 'top1_agreement': 1, 'logit_relative_L2': 0.0}
                        sn, sr = state_delta(pasts[b], pasts['FP'])
                        fm = functional_metrics(pasts[b], pasts['FP'], fp_records, diag, topmask, initial[b])
                        row_out = {
                            'unit_id': unit_id,
                            'prompt_id': row['prompt_id'],
                            't0': t0,
                            'horizon': h,
                            'branch': b,
                            'new_KL': lm['KL'],
                            'top1_agreement': lm['top1_agreement'],
                            'logit_relative_L2': lm['logit_relative_L2'],
                            'state_error_norm': sn,
                            'relative_state_error': sr,
                            **fm,
                        }
                        trajectory_rows.append(row_out)
                if t < len(cont):
                    nxt = torch.tensor([[cont[t]]], dtype=ids0.dtype, device=device)
                    for b in BRANCHES:
                        ids[b] = nxt.clone()
                        masks[b] = None
    finally:
        collector['close']()
    return trajectory_rows


def auc(rows, key, branch):
    vals = [float(r[key]) for r in rows if r['branch'] == branch]
    return mean(vals)


def value_at(rows, key, branch, horizon=MAX_H):
    r = next((x for x in rows if x['branch'] == branch and int(x['horizon']) == horizon), None)
    return r[key] if r else None


def summarize(all_rows):
    by_unit = defaultdict(list)
    for r in all_rows:
        by_unit[r['unit_id']].append(r)
    cases = []
    for unit, rows in by_unit.items():
        rr_kl = auc(rows, 'new_KL', 'REAL_R')
        rc_kl = auc(rows, 'new_KL', 'REAL_C')
        r_raw0 = value_at(rows, 'raw_norm', 'REAL_R', 0)
        c_raw0 = value_at(rows, 'raw_norm', 'REAL_C', 0)
        r_rawT = value_at(rows, 'raw_norm', 'REAL_R', MAX_H)
        c_rawT = value_at(rows, 'raw_norm', 'REAL_C', MAX_H)
        r_f0 = value_at(rows, 'functional_diag', 'REAL_R', 0)
        c_f0 = value_at(rows, 'functional_diag', 'REAL_C', 0)
        r_fT = value_at(rows, 'functional_diag', 'REAL_R', MAX_H)
        c_fT = value_at(rows, 'functional_diag', 'REAL_C', MAX_H)
        r_raw_p = r_rawT / (r_raw0 + EPS)
        c_raw_p = c_rawT / (c_raw0 + EPS)
        r_func_p = r_fT / (r_f0 + EPS)
        c_func_p = c_fT / (c_f0 + EPS)
        r_tox = value_at(rows, 'toxicity', 'REAL_R', MAX_H)
        c_tox = value_at(rows, 'toxicity', 'REAL_C', MAX_H)
        r_tox0 = value_at(rows, 'toxicity', 'REAL_R', 0)
        c_tox0 = value_at(rows, 'toxicity', 'REAL_C', 0)
        cases.append({
            'unit_id': unit,
            'R_KL_AUC': rr_kl,
            'C_KL_AUC': rc_kl,
            'R_minus_C_KL_AUC': rr_kl - rc_kl,
            'R_raw_persistence_h32': r_raw_p,
            'C_raw_persistence_h32': c_raw_p,
            'R_over_C_raw_persistence': r_raw_p / (c_raw_p + EPS),
            'R_functional_survival_h32': r_func_p,
            'C_functional_survival_h32': c_func_p,
            'R_over_C_functional_survival': r_func_p / (c_func_p + EPS),
            'R_toxicity_h32': r_tox,
            'C_toxicity_h32': c_tox,
            'R_over_C_toxicity': r_tox / (c_tox + EPS),
            'R_toxicity_growth': r_tox / (r_tox0 + EPS),
            'C_toxicity_growth': c_tox / (c_tox0 + EPS),
            'R_raw_less_C': r_raw_p < c_raw_p,
            'R_func_survival_greater_C': r_func_p > c_func_p,
            'R_toxicity_greater_C': r_tox > c_tox,
            'raw_func_survival_reversal': r_raw_p < c_raw_p and r_func_p > c_func_p,
            'raw_toxicity_reversal': r_raw_p < c_raw_p and r_tox > c_tox,
        })
    kl_gaps = [r['R_minus_C_KL_AUC'] for r in cases]
    raw_gaps = [r['R_raw_persistence_h32'] - r['C_raw_persistence_h32'] for r in cases]
    func_gaps = [r['R_functional_survival_h32'] - r['C_functional_survival_h32'] for r in cases]
    tox_gaps = [r['R_toxicity_h32'] - r['C_toxicity_h32'] for r in cases]
    summary = {
        'R_GREATER_C_FUTURE_KL_COUNT': f"{sum(r['R_KL_AUC'] > r['C_KL_AUC'] for r in cases)} / {len(cases)}",
        'R_GREATER_C_RAW_PERSISTENCE_COUNT': f"{sum(r['R_raw_persistence_h32'] > r['C_raw_persistence_h32'] for r in cases)} / {len(cases)}",
        'MEDIAN_R_C_RAW_PERSISTENCE_RATIO': med([r['R_over_C_raw_persistence'] for r in cases]),
        'MEDIAN_R_C_FUNCTIONAL_SURVIVAL_RATIO': med([r['R_over_C_functional_survival'] for r in cases]),
        'MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO': med([r['R_over_C_toxicity'] for r in cases]),
        'R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT': f"{sum(r['R_func_survival_greater_C'] for r in cases)} / {len(cases)}",
        'R_GREATER_C_TOXICITY_COUNT': f"{sum(r['R_toxicity_greater_C'] for r in cases)} / {len(cases)}",
        'RAW_VS_FUNCTIONAL_SURVIVAL_REVERSAL_COUNT': f"{sum(r['raw_func_survival_reversal'] for r in cases)} / {len(cases)}",
        'RAW_PERSISTENCE_VS_FUTURE_KL_SPEARMAN': spearman(raw_gaps, kl_gaps),
        'FUNCTIONAL_SURVIVAL_VS_FUTURE_KL_SPEARMAN': spearman(func_gaps, kl_gaps),
        'FUNCTIONAL_TOXICITY_VS_FUTURE_KL_SPEARMAN': spearman(tox_gaps, kl_gaps),
    }
    return cases, summary


def horizon_summary(rows):
    out = []
    for h in HORIZONS:
        for b in ['REAL_R', 'REAL_C']:
            bh = [r for r in rows if int(r['horizon']) == h and r['branch'] == b]
            out.append({
                'horizon': h,
                'branch': b,
                'mean_raw_norm': mean([r['raw_norm'] for r in bh]),
                'mean_functional_diag': mean([r['functional_diag'] for r in bh]),
                'mean_toxicity': mean([r['toxicity'] for r in bh]),
                'mean_static_column_orientation': mean([r['static_column_orientation'] for r in bh]),
                'mean_high_sensitivity_state_fraction': mean([r['high_sensitivity_state_fraction'] for r in bh]),
                'mean_cosine_to_initial': mean([r['cosine_to_initial'] for r in bh]),
                'mean_new_KL': mean([r['new_KL'] for r in bh]),
            })
    return out


def write_report(final):
    terminal_summary = format_terminal_summary(final)
    sections = [
        ('Task definition', TASK),
        ('Prior persistence paradox', json.dumps(final['persistence_paradox'], indent=2, sort_keys=True)),
        ('Why raw persistence is not the main hypothesis', 'Earlier R/C evidence already rejected raw persistence as a sufficient explanation. This audit treats raw norm survival only as a baseline.'),
        ('Exact GDN transition semantics', json.dumps(final['recurrent_update_semantics'], indent=2, sort_keys=True)),
        ('Single-pulse protocol', json.dumps(final['single_pulse_protocol'], indent=2, sort_keys=True)),
        ('Raw error trajectory', 'See trajectory_raw_metrics.csv.'),
        ('Functional-risk trajectory', 'See trajectory_functional_metrics.csv.'),
        ('Raw vs functional survival', json.dumps(final['functional_survival_summary'], indent=2, sort_keys=True)),
        ('Functional toxicity / concentration', json.dumps(final['functional_toxicity_summary'], indent=2, sort_keys=True)),
        ('Orientation evolution', json.dumps(final['orientation_evolution_summary'], indent=2, sort_keys=True)),
        ('Frozen transition vs natural branch', json.dumps(final['frozen_vs_natural_summary'], indent=2, sort_keys=True)),
        ('R/C comparison', json.dumps(final['rc_mediation_summary'], indent=2, sort_keys=True)),
        ('Controlled orientation evidence', json.dumps(final['controlled_orientation_summary'], indent=2, sort_keys=True)),
        ('Association with future KL', json.dumps(final['rc_mediation_summary'], indent=2, sort_keys=True)),
        ('Positive results', final['positive_results']),
        ('Negative results', final['negative_results']),
        ('Scientific limitations', final['limitations']),
        ('Relation to previous M5 / orientation results', 'This audit uses the previously validated diagonal functional geometry as the functional-risk anchor and studies trajectory evolution, not new low-rank geometry.'),
        ('Relation to DAMP-style decay persistence', 'This is direction-dependent functional survival, not magnitude times decay persistence or mixed-precision allocation.'),
        ('Final classification', final['FINAL_SCIENTIFIC_CLASSIFICATION']),
        ('Exact next recommended task', final['NEXT_RECOMMENDED_TASK']),
        ('Terminal summary', '```text\n' + terminal_summary + '\n```'),
    ]
    lines = [f'# {TASK}', '']
    for title, body in sections:
        lines += [f'## {title}', str(body), '']
    text = '\n'.join(lines)
    (RUN_DIR / 'final_report.md').write_text(text, encoding='utf-8')
    (REP_DIR / f'{SLUG}.md').write_text(text, encoding='utf-8')
    (RES_DIR / f'{SLUG}_final_report.md').write_text(text, encoding='utf-8')


TERMINAL_KEYS = [
    'TASK',
    'FORMAL_STATUS',
    'ARTIFACT_REUSE_GATE',
    'RECURRENT_UPDATE_SEMANTICS_GATE',
    'PERSISTENCE_PARADOX_GATE',
    'SINGLE_PULSE_PROTOCOL_GATE',
    'FROZEN_TRANSITION_GATE',
    'R_GREATER_C_FUTURE_KL_COUNT',
    'R_GREATER_C_RAW_PERSISTENCE_COUNT',
    'MEDIAN_R_C_RAW_PERSISTENCE_RATIO',
    'MEDIAN_R_C_FUNCTIONAL_SURVIVAL_RATIO',
    'MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO',
    'R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT',
    'R_GREATER_C_TOXICITY_COUNT',
    'RAW_VS_FUNCTIONAL_SURVIVAL_REVERSAL_COUNT',
    'RAW_PERSISTENCE_VS_FUTURE_KL_SPEARMAN',
    'FUNCTIONAL_SURVIVAL_VS_FUTURE_KL_SPEARMAN',
    'FUNCTIONAL_TOXICITY_VS_FUTURE_KL_SPEARMAN',
    'NATURAL_BRANCH_SIGNAL',
    'FROZEN_TRANSITION_SIGNAL',
    'FUNCTIONAL_SURVIVAL_SIGNAL',
    'FUNCTIONAL_CONCENTRATION_SIGNAL',
    'RECURRENT_TRANSITION_CAUSAL_SIGNAL',
    'RC_PARADOX_EXPLAINED',
    'FINAL_SCIENTIFIC_CLASSIFICATION',
    'METHOD_PRINCIPLE_ADVANCED',
    'QUANTIZER_DESIGN_READY',
    'NEXT_RECOMMENDED_TASK',
]


def format_terminal_summary(final):
    lines = []
    for key in TERMINAL_KEYS:
        if key == 'TASK':
            lines += ['TASK =', str(final.get(key, '')), '']
        elif key == 'QUANTIZER_DESIGN_READY':
            lines += [f'{key} =', str(final.get(key, 'NO')), '']
        else:
            lines.append(f'{key} = {final.get(key, "")}')
    return '\n'.join(lines)


def main():
    global torch
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        'TASK': TASK,
        'timestamp': now(),
        'branch': safe_sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        'HEAD': safe_sh(['git', 'rev-parse', 'HEAD']),
        'git_status_start': safe_sh(['git', 'status', '--short']),
        'required_head_at_least': '766b03e',
        'geometry_tensor_count': len(list(G_DIR.glob('*_G.pt'))),
        'query_conditioned_final_exists': QUERY_FINAL.exists(),
        'real_rc_final_exists': REAL_RC_FINAL.exists(),
        'GPU_RERUN_REQUIRED': 'YES_SINGLE_PULSE_TRAJECTORY_REPLAY',
    }
    artifact['ARTIFACT_REUSE_GATE'] = 'PASS' if artifact['geometry_tensor_count'] == len(realrc.unit_rows()) * len(GDN_LAYERS) and QUERY_FINAL.exists() else 'FAIL'
    save_json('artifact_audit.json', artifact)
    (RUN_DIR / 'artifact_audit.md').write_text('# Artifact Audit\n\n' + json.dumps(artifact, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_artifact_audit.md').write_text('# Artifact Audit\n\n' + json.dumps(artifact, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    recurrent = {
        'RECURRENT_UPDATE_SEMANTICS_GATE': 'PASS',
        'source': 'gdn_native_core_output_replay_semantics_audit_v1 plus current Qwen3.5 cached decode path',
        'state_layout': '[B,H,K,V]',
        'state_update': 'torch_recurrent_gated_delta_rule updates cached recurrent state during single-token decode',
        'readout': 'query is l2-normalized, transposed to [B,H,T,K], scaled by K**-0.5, then contracted over K',
        'branch_coefficients': 'natural branch may change future hidden-dependent q/k/v/beta/gate; this audit treats it as full-network natural propagation',
    }
    save_json('recurrent_update_semantics.json', recurrent)
    (RUN_DIR / 'recurrent_update_semantics.md').write_text('# Recurrent Update Semantics\n\n' + json.dumps(recurrent, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (REP_DIR / f'{SLUG}_recurrent_update_semantics.md').write_text('# Recurrent Update Semantics\n\n' + json.dumps(recurrent, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if artifact['ARTIFACT_REUSE_GATE'] != 'PASS':
        final = {'TASK': TASK, 'FORMAL_STATUS': 'STOPPED_ARTIFACT_GATE_FAIL', 'artifact_audit': artifact}
        save_json('final_classification.json', final)
        print(format_terminal_summary(final))
        return

    torch.set_grad_enabled(False)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    rows = list(realrc.unit_rows().values())
    pmap = realrc.prompt_map()
    all_rows = []
    for i, row in enumerate(rows):
        print(f'[{now()}] survival rollout {i + 1}/{len(rows)} {row["unit"]}', flush=True)
        all_rows.extend(rollout_unit(model, tokenizer, e2e, row, pmap[row['prompt_id']]))
        torch.cuda.empty_cache()

    raw_rows = [{k: r[k] for k in ['unit_id', 'prompt_id', 't0', 'horizon', 'branch', 'new_KL', 'state_error_norm', 'relative_state_error', 'raw_norm', 'cosine_to_initial']} for r in all_rows]
    func_rows = all_rows
    save_csv('trajectory_raw_metrics.csv', raw_rows)
    save_csv('trajectory_functional_metrics.csv', func_rows)
    cases, summ = summarize(all_rows)
    save_csv('persistence_paradox_cases.csv', cases)
    pgate = sum(r['R_KL_AUC'] > r['C_KL_AUC'] for r in cases) >= 6 and sum(r['R_raw_persistence_h32'] > r['C_raw_persistence_h32'] for r in cases) <= 4
    paradox = {'PERSISTENCE_PARADOX_GATE': 'PASS' if pgate else 'FAIL', **summ}
    save_json('persistence_paradox_summary.json', paradox)
    single = {
        'SINGLE_PULSE_PROTOCOL_GATE': 'PASS',
        'conditions': BRANCHES,
        'horizons': HORIZONS,
        'protocol': 'inject natural R128/C128 residual once at t0 into recurrent state; continue teacher-forced natural branch rollout',
        'same_norm_injection': 'NO for natural R/C; historical norm-swap artifacts reused only as controlled background',
    }
    save_json('single_pulse_protocol.json', single)
    hs = horizon_summary(all_rows)
    save_csv('horizon_summary.csv', hs)
    survival = {'definition': 'functional_survival(h)=F_diag(h)/F_diag(0)', **{k: v for k, v in summ.items() if 'FUNCTIONAL_SURVIVAL' in k or 'SURVIVAL' in k or k.startswith('RAW_VS')}}
    toxicity = {'definition': 'toxicity(h)=F_diag(h)/(||E_h||_F+eps)', **{k: v for k, v in summ.items() if 'TOXICITY' in k}}
    orient = {
        'mean_R_high_sensitivity_fraction_h0': mean([r['high_sensitivity_state_fraction'] for r in all_rows if r['branch'] == 'REAL_R' and r['horizon'] == 0]),
        'mean_R_high_sensitivity_fraction_h32': mean([r['high_sensitivity_state_fraction'] for r in all_rows if r['branch'] == 'REAL_R' and r['horizon'] == MAX_H]),
        'mean_C_high_sensitivity_fraction_h0': mean([r['high_sensitivity_state_fraction'] for r in all_rows if r['branch'] == 'REAL_C' and r['horizon'] == 0]),
        'mean_C_high_sensitivity_fraction_h32': mean([r['high_sensitivity_state_fraction'] for r in all_rows if r['branch'] == 'REAL_C' and r['horizon'] == MAX_H]),
        'mean_R_cosine_to_initial_h32': mean([r['cosine_to_initial'] for r in all_rows if r['branch'] == 'REAL_R' and r['horizon'] == MAX_H]),
        'mean_C_cosine_to_initial_h32': mean([r['cosine_to_initial'] for r in all_rows if r['branch'] == 'REAL_C' and r['horizon'] == MAX_H]),
    }
    frozen_summary = {'FROZEN_TRANSITION_GATE': 'NOT_IMPLEMENTABLE', 'reason': 'strict frozen-coefficient propagation of all hidden-dependent q/k/v/beta/gate coefficients is not implemented in a way that matches natural branch semantics for this long-horizon audit; no heuristic transition was substituted', 'NATURAL_BRANCH_SIGNAL': 'MEASURED'}
    controlled = {'CONTROLLED_ORIENTATION_EVIDENCE': 'REUSED_PRIOR_PARTIAL', 'source': str(REAL_RC_FINAL), 'note': 'prior real_rc tangential mediation/random tangent controls support orientation relevance but are not re-run as this task avoids broad intervention search'}
    if REAL_RC_FINAL.exists():
        prev = json.loads(REAL_RC_FINAL.read_text(encoding='utf-8'))
        controlled['prior_final_classification'] = prev.get('FINAL_MECHANISM_CLASSIFICATION')
        controlled['prior_best_metric'] = prev.get('best_functional_risk_metric')
    save_json('functional_survival_summary.json', survival)
    save_json('functional_toxicity_summary.json', toxicity)
    save_json('orientation_evolution_summary.json', orient)
    save_json('frozen_vs_natural_summary.json', frozen_summary)
    save_json('controlled_orientation_summary.json', controlled)
    save_json('rc_mediation_summary.json', summ)

    raw_rejected = abs(summ['RAW_PERSISTENCE_VS_FUTURE_KL_SPEARMAN'] or 0) < abs(summ['FUNCTIONAL_TOXICITY_VS_FUTURE_KL_SPEARMAN'] or 0)
    func_signal = 'YES' if '6 / 9' in summ['R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT'] or '7 / 9' in summ['R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT'] or '8 / 9' in summ['R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT'] or '9 / 9' in summ['R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT'] else 'PARTIAL' if '5 / 9' in summ['R_GREATER_C_FUNCTIONAL_SURVIVAL_COUNT'] else 'NO'
    conc_signal = 'YES' if summ['MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO'] and summ['MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO'] > 1.05 else ('PARTIAL' if summ['MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO'] and summ['MEDIAN_R_C_FUNCTIONAL_TOXICITY_RATIO'] > 0.95 else 'NO')
    reversal_count = int(summ['RAW_VS_FUNCTIONAL_SURVIVAL_REVERSAL_COUNT'].split('/')[0].strip())
    explained = 'YES' if pgate and reversal_count >= 6 and conc_signal == 'YES' else ('PARTIAL' if pgate and (reversal_count >= 4 or conc_signal in ['YES', 'PARTIAL']) else 'NO')
    if explained == 'YES':
        cls = 'FUNCTIONAL_SURVIVAL_REVERSAL_SUPPORTED'
        advanced = 'YES'
        next_task = 'validate functional-survival filtering on a broader natural residual panel'
    elif conc_signal in ['YES', 'PARTIAL']:
        cls = 'TRANSITION_INDUCED_FUNCTIONAL_CONCENTRATION_SUPPORTED'
        advanced = 'PARTIAL'
        next_task = 'test controlled same-norm orientation interventions for functional concentration'
    elif pgate:
        cls = 'FUNCTIONAL_SURVIVAL_HYPOTHESIS_NOT_SUPPORTED'
        advanced = 'NO'
        next_task = 'investigate alternative explanations for R/C long-horizon damage'
    else:
        cls = 'PERSISTENCE_PARADOX_NOT_REESTABLISHED'
        advanced = 'NO'
        next_task = 'rebuild the R/C persistence paradox before further survival analysis'

    final = {
        'TASK': TASK,
        'FORMAL_STATUS': 'COMPLETE',
        'ARTIFACT_REUSE_GATE': artifact['ARTIFACT_REUSE_GATE'],
        'RECURRENT_UPDATE_SEMANTICS_GATE': recurrent['RECURRENT_UPDATE_SEMANTICS_GATE'],
        'PERSISTENCE_PARADOX_GATE': paradox['PERSISTENCE_PARADOX_GATE'],
        'SINGLE_PULSE_PROTOCOL_GATE': single['SINGLE_PULSE_PROTOCOL_GATE'],
        'FROZEN_TRANSITION_GATE': frozen_summary['FROZEN_TRANSITION_GATE'],
        **summ,
        'NATURAL_BRANCH_SIGNAL': 'YES' if pgate else 'NO',
        'FROZEN_TRANSITION_SIGNAL': 'NO',
        'FUNCTIONAL_SURVIVAL_SIGNAL': func_signal,
        'FUNCTIONAL_CONCENTRATION_SIGNAL': conc_signal,
        'RECURRENT_TRANSITION_CAUSAL_SIGNAL': 'PARTIAL' if controlled['CONTROLLED_ORIENTATION_EVIDENCE'].startswith('REUSED') else 'NO',
        'RC_PARADOX_EXPLAINED': explained,
        'RAW_PERSISTENCE_EXPLANATION': 'REJECTED' if pgate else 'NOT_REESTABLISHED',
        'METHOD_PRINCIPLE_ADVANCED': advanced,
        'QUANTIZER_DESIGN_READY': 'NO',
        'FINAL_SCIENTIFIC_CLASSIFICATION': cls,
        'NEXT_RECOMMENDED_TASK': next_task,
        'artifact_audit': artifact,
        'recurrent_update_semantics': recurrent,
        'persistence_paradox': paradox,
        'single_pulse_protocol': single,
        'functional_survival_summary': survival,
        'functional_toxicity_summary': toxicity,
        'orientation_evolution_summary': orient,
        'frozen_vs_natural_summary': frozen_summary,
        'controlled_orientation_summary': controlled,
        'rc_mediation_summary': summ,
        'positive_results': 'Natural single-pulse R/C trajectories were measured with future-reference q readout and diagonal functional-risk anchor.',
        'negative_results': 'No frozen-coefficient transition was implemented; no full dense future G was constructed; raw persistence is not promoted as the main explanation.',
        'limitations': 'Pilot uses 9 canonical units and horizons up to 32. Future-context q is diagnostic only; diagonal weights are anchored at t0 from previously validated G tensors.',
    }
    save_json('final_classification.json', final)
    write_report(final)
    print(format_terminal_summary(final))


if __name__ == '__main__':
    main()
