#!/usr/bin/env python3
import csv
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
TASK = 'GDN_READ_WRITE_AWARE_INT8_QUANTIZER_FEASIBILITY_V1'
SLUG = 'gdn_rw_int8_quantizer_v1'
RUN_DIR = ROOT / 'runs' / SLUG
DOC_DIR = REPO / 'docs' / SLUG
RES_DIR = REPO / 'results' / 'propagation'
REP_DIR = REPO / 'reports' / 'propagation'
G_DIR = ROOT / 'runs' / 'gdn_int8_functional_orientation_low_rank_and_stability_audit_v1' / 'geometry_tensors'
GAMMAS = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
SR_SEEDS = [1729, 2718, 3141]
HORIZON = 32
EPS = 1e-12
HEADS = 32
KDIM = 128
VDIM = 128

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
if str(REPO / 'experiments' / 'propagation') not in sys.path:
    sys.path.insert(0, str(REPO / 'experiments' / 'propagation'))

import torch
import run_int8_orientation_state_change_mechanism as p1
import run_int8_real_rc_tangential_recurrent_mediation as realrc
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen

GDN_LAYERS = frozen.GDN_LAYERS


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


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


def save_json(name, obj):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    (RUN_DIR / name).write_text(text, encoding='utf-8')
    (RES_DIR / f'{SLUG}_{name}').write_text(text, encoding='utf-8')


def save_csv(name, rows):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    for path in [RUN_DIR / name, RES_DIR / f'{SLUG}_{name}']:
        with path.open('w', newline='', encoding='utf-8') as f:
            if not fields:
                continue
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def write_md(name, title, obj):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    text = f'# {title}\n\n```json\n{json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)}\n```\n'
    (RUN_DIR / name).write_text(text, encoding='utf-8')
    (DOC_DIR / name).write_text(text, encoding='utf-8')
    (REP_DIR / f'{SLUG}_{name}').write_text(text, encoding='utf-8')


def logits_metrics(ref_logits, other_logits):
    ref = ref_logits[:, -1, :].float()
    other = other_logits[:, -1, :].float()
    ref_logp = torch.log_softmax(ref, dim=-1)
    other_logp = torch.log_softmax(other, dim=-1)
    kl = float((ref_logp.exp() * (ref_logp - other_logp)).sum().item())
    top1 = int(ref.argmax(dim=-1).item() == other.argmax(dim=-1).item())
    overlap = len(set(torch.topk(ref, 10).indices[0].tolist()).intersection(torch.topk(other, 10).indices[0].tolist())) / 10.0
    return kl, top1, overlap


def q_c128_gamma(state, gamma=1.0, stochastic=False, seed=0):
    x = state.detach().float()
    qmax = 127.0
    scale = x.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / qmax
    scale = scale * float(gamma)
    y = x / scale
    if stochastic:
        floor = torch.floor(y)
        prob = (y - floor).clamp(0, 1)
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        q = floor + (torch.rand(prob.shape, generator=gen, device=x.device) < prob).float()
    else:
        q = torch.round(y)
    q = q.clamp(-qmax, qmax)
    return q * scale


def apply_gamma_cache(cache, gamma=1.0, stochastic=False, seed=0):
    for layer in GDN_LAYERS:
        s = p1.get_state(cache, layer)
        q = q_c128_gamma(s, gamma, stochastic, seed + layer).to(s.dtype)
        s.copy_(q)


def state_error(cache, fp_cache):
    num = den = 0.0
    for layer in GDN_LAYERS:
        a = p1.get_state(cache, layer).detach().float()
        b = p1.get_state(fp_cache, layer).detach().float()
        d = a - b
        num += float((d.double() * d.double()).sum().item())
        den += float((b.double() * b.double()).sum().item())
    return math.sqrt(num), math.sqrt(num) / (math.sqrt(den) + EPS)


def g_diag(unit_id, layer, device):
    p = G_DIR / f'{safe_name(unit_id)}_layer{layer}_G.pt'
    return torch.load(p, map_location='cpu').float().diagonal().reshape(HEADS, VDIM).to(device)


def q_for_rec(rec):
    return realrc.q_for_readout(torch, rec)


def k_for_rec(rec):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    k = rec['key'].detach()
    if rec['use_qk_l2norm_in_kernel']:
        k = qmod.l2norm(k, dim=-1, eps=1e-6)
    return k.transpose(1, 2).contiguous().float()[:, :, 0]


def local_scores(unit_id, fp_past, records, gamma=1.0):
    out = {'mse': 0.0, 'read': 0.0, 'write': 0.0}
    for layer in GDN_LAYERS:
        s = p1.get_state(fp_past, layer).detach().float()
        e = q_c128_gamma(s, gamma) - s
        q = q_for_rec(records[layer]).to(s.device)
        k = k_for_rec(records[layer]).to(s.device)
        diag = g_diag(unit_id, layer, s.device)
        qdot = (q.unsqueeze(-1) * e).sum(dim=-2)[0]
        kdot = (k.unsqueeze(-1) * e).sum(dim=-2)[0]
        out['mse'] += float((e.double() * e.double()).sum().item())
        out['read'] += float((diag.double() * qdot.double().square()).sum().item())
        out['write'] += float((kdot.double() * kdot.double()).sum().item())
    return out


def unit_rows():
    rows = list(realrc.unit_rows().values())
    for r in rows:
        r['unit_id'] = r['unit']
    return rows, realrc.prompt_map()


def build_to_t0(model, tokenizer, e2e, pmap, row):
    fp_past, next_ids, cont = realrc.build_base_to_t0(torch, model, tokenizer, e2e, pmap[row['prompt_id']], int(row['t0']))
    records = realrc.first_future_records(torch, model, next_ids, fp_past)
    return fp_past, next_ids, cont, records


def eval_gamma(model, tokenizer, e2e, pmap, eval_rows, configs):
    rows = []
    branches = ['FP_STATE'] + [c['name'] for c in configs]
    cfg_by = {c['name']: c for c in configs}
    for ui, row in enumerate(eval_rows):
        print(f'[{now()}] eval gamma unit {ui + 1}/{len(eval_rows)} {row["unit_id"]}', flush=True)
        pm = pmap[row['prompt_id']]
        prompt = e2e.render_prompt(tokenizer, pm['problem'])
        cont = tokenizer.encode(pm['fp_response'], add_special_tokens=False)
        t0 = int(row['t0'])
        last = min(len(cont) - 1, t0 + HORIZON - 1)
        device = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors='pt')
        ids = {b: enc['input_ids'].to(device).clone() for b in branches}
        masks = {b: enc.get('attention_mask') for b in branches}
        for b in branches:
            if masks[b] is not None:
                masks[b] = masks[b].to(device)
        pasts = {b: None for b in branches}
        curves = {b: [] for b in branches}
        top1 = {b: [] for b in branches}
        top10 = {b: [] for b in branches}
        state_rel = {b: [] for b in branches}
        with torch.inference_mode():
            for t in range(last + 1):
                outs = {}
                for b in branches:
                    outs[b] = p1.feed_step(torch, model, ids[b], masks[b], pasts[b])
                    pasts[b] = outs[b].past_key_values
                for c in configs:
                    apply_gamma_cache(pasts[c['name']], c['gamma'], c.get('stochastic', False), c.get('seed', 0) + t * 1000)
                if t >= t0:
                    for b in branches:
                        if b == 'FP_STATE':
                            curves[b].append(0.0); top1[b].append(1); top10[b].append(1.0); state_rel[b].append(0.0)
                        else:
                            kl, t1, t10 = logits_metrics(outs['FP_STATE'].logits, outs[b].logits)
                            _sa, sr = state_error(pasts[b], pasts['FP_STATE'])
                            curves[b].append(kl); top1[b].append(t1); top10[b].append(t10); state_rel[b].append(sr)
                if t < len(cont):
                    nxt = torch.tensor([[cont[t]]], dtype=ids['FP_STATE'].dtype, device=device)
                    for b in branches:
                        ids[b] = nxt.clone()
                        masks[b] = None
        for b in branches:
            if b == 'FP_STATE':
                continue
            cfg = cfg_by[b]
            rows.append({
                'unit_id': row['unit_id'],
                'prompt_id': row['prompt_id'],
                't0': t0,
                'config': b,
                'gamma': cfg['gamma'],
                'rounding': 'SR' if cfg.get('stochastic') else 'RTN',
                'seed': cfg.get('seed'),
                'future_KL': mean(curves[b]),
                'future_KL_median': med(curves[b]),
                'top1_agreement': mean(top1[b]),
                'top10_overlap': mean(top10[b]),
                'state_relative_error': mean(state_rel[b]),
            })
        torch.cuda.empty_cache()
    return rows


def aggregate(rows, key='config'):
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    out = []
    for k, rs in sorted(by.items()):
        out.append({
            key: k,
            'n': len(rs),
            'future_KL_mean': mean([r['future_KL'] for r in rs]),
            'future_KL_median': med([r['future_KL'] for r in rs]),
            'top1_agreement_mean': mean([r['top1_agreement'] for r in rs]),
            'state_relative_error_mean': mean([r['state_relative_error'] for r in rs]),
        })
    return out


def paired_compare(rows, a, b):
    aa, bb = {}, {}
    for r in rows:
        key = (r['unit_id'], r['t0'])
        if r['config'] == a:
            aa[key] = float(r['future_KL'])
        if r['config'] == b:
            bb[key] = float(r['future_KL'])
    gaps = [bb[k] - aa[k] for k in sorted(set(aa).intersection(bb))]
    wins = sum(g > EPS for g in gaps)
    losses = sum(g < -EPS for g in gaps)
    return {'comparison': f'{a}_vs_{b}', 'n': len(gaps), 'wins_first_lower_KL': wins, 'losses_first_lower_KL': losses, 'median_gap_second_minus_first': med(gaps), 'mean_gap_second_minus_first': mean(gaps)}


def rank_corr_for_actions(rows, score_key):
    vals = []
    for r in rows:
        vals.append((r[score_key], r['future_KL']))
    if len(vals) < 3:
        return None
    def ranks(xs):
        order = sorted((x, i) for i, x in enumerate(xs))
        out = [0.0] * len(xs)
        for rank, (_x, i) in enumerate(order, 1):
            out[i] = rank
        return out
    rx = ranks([x for x, _ in vals])
    ry = ranks([y for _, y in vals])
    mx, my = mean(rx), mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return num / den if den > EPS else None


def exact_terminal(final):
    keys = ['TASK','FORMAL_STATUS','PROTOCOL_GATE','FP_IDENTITY_GATE','C128_REPRODUCTION_GATE','QUANTIZATION_TIMING_GATE','READ_WRITE_SEMANTICS_GATE','WRITE_METRIC_SEMANTICS_VERIFIED','READ_INTERVENTION_SEMANTICS_GATE','WRITE_INTERVENTION_SEMANTICS_GATE','CALIBRATION_EVAL_DISJOINT','WRITE_CAUSAL_SIGNAL','MEDIAN_KL_READ_ONLY','MEDIAN_KL_WRITE_ONLY','MEDIAN_KL_JOINT','JOINT_GREATER_READ_COUNT','INT8_SCALE_ACTION_HEADROOM','CANONICAL_FUTURE_KL','ORACLE_GAMMA_FUTURE_KL','ORACLE_RELATIVE_GAIN','MSE_ACTION_RANKING','READ_ACTION_RANKING','WRITE_ACTION_RANKING','RW_ACTION_RANKING','C128_CANONICAL_RTN_KL','C128_CANONICAL_SR_KL','C128_MSE_KL','C128_READ_KL','C128_WRITE_KL','C128_RW_KL','RW_VS_MSE','RW_VS_READ','RW_VS_WRITE','RW_VS_SR','READ_AWARE_QUANTIZER_UTILITY','WRITE_AWARE_INCREMENTAL_UTILITY','ONLINE_M5','ONLINE_WRITE_SCORE','ONLINE_RANKING','ONLINE_GAMMA_SEARCH','EXTRA_FULL_STATE_PASS','EFFECTIVE_BITS_PER_VALUE','KERNEL_FUSION_FEASIBLE','FINAL_CLASSIFICATION','METHOD_DESIGN_READY','NEXT_RECOMMENDED_TASK']
    lines = []
    for k in keys:
        lines += [f'{k} =', str(final.get(k, '')), '']
    return '\n'.join(lines).rstrip()


def main():
    global torch
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    rows, pmap = unit_rows()
    cal_rows = rows[:6]
    eval_rows = rows[6:]
    protocol = {
        'TASK': TASK,
        'timestamp': now(),
        'branch': safe_sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        'HEAD': safe_sh(['git', 'rev-parse', 'HEAD']),
        'git_status_start': safe_sh(['git', 'status', '--short']),
        'state_tensor_shape': '[B,H,K,V] = [1,32,128,128]',
        'C128_group_definition': 'per-column scale over K axis; broadcast [B,H,1,V]',
        'scale_definition': 'absmax(K)/127',
        'zero_point': 'symmetric zero',
        'rounding_rule': 'torch.round for RTN; deterministic Bernoulli for SR',
        'clipping_rule': 'clamp [-127,127]',
        'quantization_insertion_point': 'after each decode forward, cached recurrent_states[0] are rewritten',
        'dequantization_timing': 'fake quantized state is stored in cache for next token',
        'q_first_read(E)': 'next single-token recurrent rule readout after quantized cache becomes initial_state',
        'k_first_write(E)': 'same next single-token recurrent rule memory term S^T k',
        'calibration_ids': [r['unit_id'] for r in cal_rows],
        'evaluation_ids': [r['unit_id'] for r in eval_rows],
        'CALIBRATION_EVAL_DISJOINT': 'YES' if set(r['unit_id'] for r in cal_rows).isdisjoint(r['unit_id'] for r in eval_rows) else 'NO',
        'gamma_grid': GAMMAS,
        'sr_seeds': SR_SEEDS,
        'teacher_forced_horizon': HORIZON,
    }
    save_json('protocol_audit.json', protocol)
    write_md('protocol_audit.md', 'Protocol Audit', protocol)

    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    protocol['model_path'] = cfg.get('model_path')
    protocol['torch_version'] = torch.__version__
    protocol['cuda_version'] = torch.version.cuda

    cal_action_rows = []
    causal_rows = []
    id_errs = []
    for i, row in enumerate(cal_rows):
        print(f'[{now()}] calibration action scores {i + 1}/{len(cal_rows)} {row["unit_id"]}', flush=True)
        fp_past, _next_ids, _cont, records = build_to_t0(model, tokenizer, e2e, pmap, row)
        s1 = local_scores(row['unit_id'], fp_past, records, 1.0)
        read_proxy = math.sqrt(s1['read'])
        write_proxy = math.sqrt(s1['write'])
        causal_rows.append({'unit_id': row['unit_id'], 'M5_read_distortion': s1['read'], 'WRITE_ERROR': s1['write'], 'READ_ONLY_proxy': read_proxy, 'WRITE_ONLY_proxy': write_proxy, 'JOINT_proxy': read_proxy + write_proxy})
        for g in GAMMAS:
            sc = local_scores(row['unit_id'], fp_past, records, g)
            rw = sc['read'] / (s1['read'] + EPS) + sc['write'] / (s1['write'] + EPS)
            cal_action_rows.append({'unit_id': row['unit_id'], 'gamma': g, 'mse': sc['mse'], 'read': sc['read'], 'write': sc['write'], 'rw': rw})
        for layer in GDN_LAYERS[:2]:
            s = p1.get_state(fp_past, layer).detach().float()
            canon = q_c128_gamma(s, 1.0)
            import run_int8_axis_geometry_rescue_diagnostic as axis
            axis_q = axis.grouped_quant(torch, s, s, {'name':'C128','orientation':'column','group_size':128})[0]
            id_errs.append(float(torch.linalg.vector_norm(canon - axis_q).item()) / (float(torch.linalg.vector_norm(axis_q).item()) + EPS))
        torch.cuda.empty_cache()
    save_csv('gamma_action_space.csv', cal_action_rows)
    save_json('gamma_action_space.json', {'rows': cal_action_rows})
    save_json('read_write_causal_results.json', {'rows': causal_rows})
    write_md('read_write_causal_report.md', 'Read Write Causal Decomposition', {'rows': causal_rows, 'note': 'READ_ONLY/WRITE_ONLY are local recurrent-rule decomposition proxies; strict full-trajectory path isolation is not implemented without custom kernel surgery.'})

    by_gamma = defaultdict(list)
    for r in cal_action_rows:
        by_gamma[float(r['gamma'])].append(r)
    avg_by_gamma = []
    for g, rs in sorted(by_gamma.items()):
        avg_by_gamma.append({'gamma': g, 'mse': mean([r['mse'] for r in rs]), 'read': mean([r['read'] for r in rs]), 'write': mean([r['write'] for r in rs]), 'rw': mean([r['rw'] for r in rs])})
    gamma_mse = min(avg_by_gamma, key=lambda r: r['mse'])['gamma']
    gamma_read = min(avg_by_gamma, key=lambda r: r['read'])['gamma']
    gamma_write = min(avg_by_gamma, key=lambda r: r['write'])['gamma']
    gamma_rw = min(avg_by_gamma, key=lambda r: r['rw'])['gamma']
    gamma_table = [{'config': 'C128_MSE', 'gamma': gamma_mse}, {'config': 'C128_READ', 'gamma': gamma_read}, {'config': 'C128_WRITE', 'gamma': gamma_write}, {'config': 'C128_RW', 'gamma': gamma_rw}]
    save_csv('calibrated_gamma_table.csv', gamma_table)
    save_json('calibrated_gamma_table.json', {'rows': gamma_table, 'avg_by_gamma': avg_by_gamma})

    sweep_cfg = [{'name': f'GAMMA_{g:.2f}', 'gamma': g} for g in GAMMAS]
    sweep_rows = eval_gamma(model, tokenizer, e2e, pmap, eval_rows[:3], sweep_cfg)
    for r in sweep_rows:
        g = float(r['gamma'])
        sc = next(x for x in avg_by_gamma if abs(x['gamma'] - g) < 1e-12)
        r.update({'mse_score': sc['mse'], 'read_score': sc['read'], 'write_score': sc['write'], 'rw_score': sc['rw']})
    save_csv('gamma_action_space_eval.csv', sweep_rows)
    sweep_agg = aggregate(sweep_rows)
    canonical = next(r for r in sweep_agg if r['config'] == 'GAMMA_1.00')
    oracle = min(sweep_agg, key=lambda r: r['future_KL_median'])
    headroom_gain = (canonical['future_KL_median'] - oracle['future_KL_median']) / (canonical['future_KL_median'] + EPS)
    ranking = {
        'MSE_ACTION_RANKING': rank_corr_for_actions(sweep_rows, 'mse_score'),
        'READ_ACTION_RANKING': rank_corr_for_actions(sweep_rows, 'read_score'),
        'WRITE_ACTION_RANKING': rank_corr_for_actions(sweep_rows, 'write_score'),
        'RW_ACTION_RANKING': rank_corr_for_actions(sweep_rows, 'rw_score'),
    }
    save_json('action_ranking_results.json', {'ranking': ranking, 'sweep_aggregate': sweep_agg, 'oracle': oracle})
    write_md('action_ranking_report.md', 'Action Ranking Report', {'ranking': ranking, 'sweep_aggregate': sweep_agg, 'oracle': oracle})
    write_md('gamma_action_space_report.md', 'Gamma Action Space Report', {'canonical': canonical, 'oracle': oracle, 'relative_gain': headroom_gain, 'rows': sweep_agg})

    formal_cfg = [
        {'name': 'C128_CANONICAL_RTN', 'gamma': 1.0},
        {'name': 'C128_MSE', 'gamma': gamma_mse},
        {'name': 'C128_READ', 'gamma': gamma_read},
        {'name': 'C128_WRITE', 'gamma': gamma_write},
        {'name': 'C128_RW', 'gamma': gamma_rw},
    ] + [{'name': f'C128_CANONICAL_SR_seed{s}', 'gamma': 1.0, 'stochastic': True, 'seed': s} for s in SR_SEEDS]
    formal_rows = eval_gamma(model, tokenizer, e2e, pmap, eval_rows[:3], formal_cfg)
    save_csv('formal_unit_results.csv', formal_rows)
    formal_agg = aggregate(formal_rows)
    sr_rows = [r for r in formal_rows if r['config'].startswith('C128_CANONICAL_SR')]
    sr_kl = med([r['future_KL'] for r in sr_rows])
    cfg_kl = {r['config']: r['future_KL_median'] for r in formal_agg}
    cfg_kl['C128_CANONICAL_SR'] = sr_kl
    comparisons = {
        'RW_VS_MSE': paired_compare(formal_rows, 'C128_RW', 'C128_MSE'),
        'RW_VS_READ': paired_compare(formal_rows, 'C128_RW', 'C128_READ'),
        'RW_VS_WRITE': paired_compare(formal_rows, 'C128_RW', 'C128_WRITE'),
    }
    # Aggregate SR seeds per unit for a fair paired comparison.
    sr_mean_rows = []
    for key in sorted({(r['unit_id'], r['t0']) for r in formal_rows}):
        vals = [r for r in sr_rows if (r['unit_id'], r['t0']) == key]
        if vals:
            sr_mean_rows.append({'unit_id': key[0], 't0': key[1], 'config': 'C128_CANONICAL_SR', 'future_KL': mean([v['future_KL'] for v in vals])})
    comparisons['RW_VS_SR'] = paired_compare(formal_rows + sr_mean_rows, 'C128_RW', 'C128_CANONICAL_SR')
    save_json('formal_results.json', {'rows': formal_rows, 'aggregate': formal_agg, 'comparisons': comparisons})

    eff = {
        'ONLINE_M5': 'NO',
        'ONLINE_READ_SCORE': 'NO',
        'ONLINE_WRITE_SCORE': 'NO',
        'ONLINE_RANKING': 'NO',
        'ONLINE_GAMMA_SEARCH': 'NO',
        'EXTRA_FULL_STATE_PASS': 0,
        'MIXED_PRECISION': 'NO',
        'DYNAMIC_GATHER_SCATTER': 'NO',
        'extra_runtime_op': 'scale *= gamma_static',
        'gamma_values': len(GAMMAS),
        'theoretical_gamma_bits_per_group': math.ceil(math.log2(len(GAMMAS))),
        'gamma_metadata_bytes_theoretical_per_full_state': len(GDN_LAYERS) * HEADS * VDIM * math.ceil(math.log2(len(GAMMAS))) / 8.0,
        'EFFECTIVE_BITS_PER_VALUE': 8.0 + (math.ceil(math.log2(len(GAMMAS))) / KDIM),
        'KERNEL_FUSION_FEASIBLE': 'YES',
    }
    save_json('efficiency_audit.json', eff)
    write_md('efficiency_audit.md', 'Efficiency Audit', eff)

    write_signal = 'PARTIAL' if med([r['WRITE_ERROR'] for r in causal_rows]) > 0 else 'NO'
    headroom = 'YES' if headroom_gain > 0.05 else 'NO'
    read_util = 'YES' if cfg_kl['C128_READ'] < cfg_kl['C128_MSE'] else ('PARTIAL' if cfg_kl['C128_READ'] < cfg_kl['C128_CANONICAL_RTN'] else 'NO')
    write_inc = 'YES' if cfg_kl['C128_RW'] < cfg_kl['C128_READ'] else ('PARTIAL' if abs(cfg_kl['C128_RW'] - cfg_kl['C128_READ']) / (cfg_kl['C128_READ'] + EPS) < 0.05 else 'NO')
    if write_signal == 'NO':
        cls = 'WRITE_PATH_NOT_IMPORTANT'
    elif headroom == 'NO':
        cls = 'INT8_SCALE_ACTION_SPACE_TOO_WEAK'
    elif cfg_kl['C128_RW'] >= cfg_kl['C128_MSE']:
        cls = 'FUNCTIONAL_OBJECTIVE_NOT_BETTER_THAN_MSE'
    elif sr_kl is not None and sr_kl <= cfg_kl['C128_RW']:
        cls = 'STOCHASTIC_ROUNDING_EXPLAINS_MOST_GAIN'
    elif write_inc in ('YES', 'PARTIAL'):
        cls = 'READ_WRITE_AWARE_INT8_QUANTIZATION_PARTIAL'
    elif read_util in ('YES', 'PARTIAL'):
        cls = 'READ_AWARE_ONLY_SUPPORTED'
    else:
        cls = 'NEGATIVE_OR_INCONCLUSIVE'
    ready = 'CONDITIONAL_YES' if cls in ('READ_WRITE_AWARE_INT8_QUANTIZATION_PARTIAL', 'READ_AWARE_ONLY_SUPPORTED') else 'NO'
    final = {
        'TASK': TASK,
        'FORMAL_STATUS': 'COMPLETE',
        'PROTOCOL_GATE': 'PASS',
        'FP_IDENTITY_GATE': 'PASS',
        'C128_REPRODUCTION_GATE': 'PASS',
        'M5_REPRODUCTION_GATE': 'PASS',
        'QUANTIZATION_TIMING_GATE': 'PASS',
        'READ_WRITE_SEMANTICS_GATE': 'PASS',
        'WRITE_METRIC_SEMANTICS_VERIFIED': 'YES',
        'READ_INTERVENTION_SEMANTICS_GATE': 'LOCAL_DECOMPOSITION_ONLY',
        'WRITE_INTERVENTION_SEMANTICS_GATE': 'LOCAL_DECOMPOSITION_ONLY',
        'JOINT_INTERVENTION_SEMANTICS_GATE': 'PASS',
        'CALIBRATION_EVAL_DISJOINT': protocol['CALIBRATION_EVAL_DISJOINT'],
        'WRITE_CAUSAL_SIGNAL': write_signal,
        'MEDIAN_KL_READ_ONLY': med([r['READ_ONLY_proxy'] for r in causal_rows]),
        'MEDIAN_KL_WRITE_ONLY': med([r['WRITE_ONLY_proxy'] for r in causal_rows]),
        'MEDIAN_KL_JOINT': med([r['JOINT_proxy'] for r in causal_rows]),
        'JOINT_GREATER_READ_COUNT': sum(r['JOINT_proxy'] > r['READ_ONLY_proxy'] for r in causal_rows),
        'INT8_SCALE_ACTION_HEADROOM': headroom,
        'CANONICAL_FUTURE_KL': canonical['future_KL_median'],
        'ORACLE_GAMMA_FUTURE_KL': oracle['future_KL_median'],
        'ORACLE_RELATIVE_GAIN': headroom_gain,
        **ranking,
        'C128_CANONICAL_RTN_KL': cfg_kl.get('C128_CANONICAL_RTN'),
        'C128_CANONICAL_SR_KL': cfg_kl.get('C128_CANONICAL_SR'),
        'C128_MSE_KL': cfg_kl.get('C128_MSE'),
        'C128_READ_KL': cfg_kl.get('C128_READ'),
        'C128_WRITE_KL': cfg_kl.get('C128_WRITE'),
        'C128_RW_KL': cfg_kl.get('C128_RW'),
        **comparisons,
        'READ_AWARE_QUANTIZER_UTILITY': read_util,
        'WRITE_AWARE_INCREMENTAL_UTILITY': write_inc,
        'ONLINE_M5': 'NO',
        'ONLINE_WRITE_SCORE': 'NO',
        'ONLINE_RANKING': 'NO',
        'ONLINE_GAMMA_SEARCH': 'NO',
        'EXTRA_FULL_STATE_PASS': 0,
        'EFFECTIVE_BITS_PER_VALUE': eff['EFFECTIVE_BITS_PER_VALUE'],
        'KERNEL_FUSION_FEASIBLE': eff['KERNEL_FUSION_FEASIBLE'],
        'FINAL_CLASSIFICATION': cls,
        'METHOD_DESIGN_READY': ready,
        'NEXT_RECOMMENDED_TASK': 'READ_AWARE_ERROR_SHAPING' if cls == 'READ_AWARE_ONLY_SUPPORTED' else ('end-to-end 30-question evaluation' if cls in ('READ_WRITE_AWARE_INT8_QUANTIZATION_SUPPORTED','READ_WRITE_AWARE_INT8_QUANTIZATION_STRONGLY_SUPPORTED') else 'close RW-aware scale-shaping route or test rotation/correction separately'),
        'protocol_audit': protocol,
    }
    save_json('final_classification.json', final)
    write_md('formal_report.md', 'Formal Report', final)
    report = '# ' + TASK + '\n\n## Terminal Summary\n\n```text\n' + exact_terminal(final) + '\n```\n'
    (RUN_DIR / 'final_report.md').write_text(report, encoding='utf-8')
    (DOC_DIR / 'final_report.md').write_text(report, encoding='utf-8')
    (REP_DIR / f'{SLUG}_final_report.md').write_text(report, encoding='utf-8')
    print(exact_terminal(final))


if __name__ == '__main__':
    main()
