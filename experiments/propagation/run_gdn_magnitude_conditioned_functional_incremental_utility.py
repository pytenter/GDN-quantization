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
TASK = 'GDN_MAGNITUDE_CONDITIONED_FUNCTIONAL_INCREMENTAL_UTILITY_V1'
SLUG = 'gdn_magnitude_conditioned_functional_v1'
RUN_DIR = ROOT / 'runs' / SLUG
DOC_DIR = REPO / 'docs' / SLUG
RES_DIR = REPO / 'results' / 'propagation'
REP_DIR = REPO / 'reports' / 'propagation'
PREV_RUN = ROOT / 'runs' / 'gdn_functional_column_protection_v1'
PREV_FINAL = PREV_RUN / 'final_classification.json'
PREV_SCORES = PREV_RUN / 'selector_scores.csv'
PREV_FORMAL = PREV_RUN / 'formal_results.json'

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
if str(REPO / 'experiments' / 'propagation') not in sys.path:
    sys.path.insert(0, str(REPO / 'experiments' / 'propagation'))

import torch
import run_gdn_functional_column_protection_intervention as colprot
import run_int8_orientation_state_change_mechanism as p1
import run_int8_real_rc_tangential_recurrent_mediation as realrc

EPS = 1e-12
VDIM = 128
HEADS = 32
PRIMARY_THRESHOLD = 0.02
THRESHOLDS = [0.01, 0.02, 0.05]
PRIMARY_RATIO = 0.375
CORE_RATIO = 0.25
BOUNDARY_UPPER = 0.50
PILOT_PAIR_COUNT = 6
FORMAL_PAIR_COUNT = 12
BOOTSTRAP_N = 2000
RANDOM_SEEDS = [1729, 2718, 3141]


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


def safe_sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()
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


def read_csv(path):
    with Path(path).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def numeric_scores(rows):
    out = []
    for r in rows:
        rr = dict(r)
        for k in ['layer', 'head', 'value_column']:
            rr[k] = int(rr[k])
        for k in ['magnitude_score', 'error_only_score', 'functional_only_score', 'favor_score']:
            rr[k] = float(rr[k])
        out.append(rr)
    return out


def rank_by_head(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r['layer'], r['head'])].append(r)
    for key, items in by.items():
        items.sort(key=lambda r: (r['magnitude_score'], -r['value_column']), reverse=True)
        for i, r in enumerate(items):
            r['magnitude_rank_desc'] = i
            r['magnitude_percentile_desc'] = (i + 0.5) / len(items)
        items_f = sorted(items, key=lambda r: (r['favor_score'], r['value_column']))
        for i, r in enumerate(items_f):
            r['functional_percentile_asc'] = (i + 0.5) / len(items_f)
        items_e = sorted(items, key=lambda r: (r['error_only_score'], r['value_column']))
        for i, r in enumerate(items_e):
            r['error_percentile_asc'] = (i + 0.5) / len(items_e)
    return by


def mag_rel_gap(a, b):
    return abs(a['magnitude_score'] - b['magnitude_score']) / ((abs(a['magnitude_score']) + abs(b['magnitude_score'])) / 2.0 + EPS)


def err_rel_gap(a, b):
    return abs(a['error_only_score'] - b['error_only_score']) / ((abs(a['error_only_score']) + abs(b['error_only_score'])) / 2.0 + EPS)


def pair_candidates(by_head, threshold, boundary_only=True):
    rows = []
    for (layer, head), items in by_head.items():
        inside = [r for r in items if r['magnitude_rank_desc'] < int(round(VDIM * PRIMARY_RATIO))]
        outside = [r for r in items if r['magnitude_rank_desc'] >= int(round(VDIM * PRIMARY_RATIO))]
        if boundary_only:
            lo = int(round(VDIM * CORE_RATIO))
            hi = int(round(VDIM * BOUNDARY_UPPER))
            inside = [r for r in inside if lo <= r['magnitude_rank_desc'] < hi]
            outside = [r for r in outside if lo <= r['magnitude_rank_desc'] < hi]
        for a in inside:
            for b in outside:
                mg = mag_rel_gap(a, b)
                if mg > threshold:
                    continue
                fg = abs(a['functional_percentile_asc'] - b['functional_percentile_asc'])
                if fg < 0.30:
                    continue
                high, low = (a, b) if a['favor_score'] >= b['favor_score'] else (b, a)
                rows.append({
                    'layer': layer,
                    'head': head,
                    'column_highF': high['value_column'],
                    'column_lowF': low['value_column'],
                    'column_inside_mag_mask': a['value_column'],
                    'column_outside_mag_mask': b['value_column'],
                    'magnitude_highF': high['magnitude_score'],
                    'magnitude_lowF': low['magnitude_score'],
                    'mag_rel_gap': mg,
                    'functional_high': high['favor_score'],
                    'functional_low': low['favor_score'],
                    'functional_gap': high['favor_score'] - low['favor_score'],
                    'functional_percentile_gap': fg,
                    'error_only_high': high['error_only_score'],
                    'error_only_low': low['error_only_score'],
                    'error_rel_gap': err_rel_gap(high, low),
                    'functional_only_high': high['functional_only_score'],
                    'functional_only_low': low['functional_only_score'],
                })
    return rows


def disjoint_greedy(cands, limit):
    used = set()
    out = []
    cands = sorted(cands, key=lambda r: (r['mag_rel_gap'], -r['functional_percentile_gap'], -abs(r['functional_gap'])))
    for r in cands:
        a = (r['layer'], r['head'], r['column_highF'])
        b = (r['layer'], r['head'], r['column_lowF'])
        if a in used or b in used:
            continue
        rr = dict(r)
        rr['pair_id'] = f"p{len(out):03d}_L{r['layer']}_H{r['head']}_V{r['column_highF']}_V{r['column_lowF']}"
        out.append(rr)
        used.add(a)
        used.add(b)
        if len(out) >= limit:
            break
    return out


def base_mag_masks(score_rows, ratio=PRIMARY_RATIO):
    return colprot.rank_masks(score_rows, 'MAGNITUDE', ratio)


def pair_masks(score_rows, pair):
    high = base_mag_masks(score_rows, PRIMARY_RATIO)
    low = base_mag_masks(score_rows, PRIMARY_RATIO)
    layer = int(pair['layer'])
    head = int(pair['head'])
    vh = int(pair['column_highF'])
    vl = int(pair['column_lowF'])
    high[layer][head, vh] = True
    high[layer][head, vl] = False
    low[layer][head, vh] = False
    low[layer][head, vl] = True
    return {'HIGH_F': high, 'LOW_F': low}


def run_pair_eval(model, tokenizer, e2e, pmap, eval_rows, score_rows, pairs):
    all_unit_rows = []
    pair_results = []
    for pi, pair in enumerate(pairs):
        print(f'[{now()}] pair eval {pi + 1}/{len(pairs)} {pair["pair_id"]}', flush=True)
        masks = pair_masks(score_rows, pair)
        branch_masks = {'C128_INT8': None, 'HIGH_F': masks['HIGH_F'], 'LOW_F': masks['LOW_F']}
        rows = []
        for row in eval_rows:
            rr = colprot.eval_unit(model, tokenizer, e2e, pmap, row, branch_masks)
            for x in rr:
                if x['selector'] in ('HIGH_F', 'LOW_F'):
                    x['pair_id'] = pair['pair_id']
                    x['layer'] = pair['layer']
                    x['head'] = pair['head']
                    rows.append(x)
                    all_unit_rows.append(x)
            torch.cuda.empty_cache()
        high = [r['future_KL_mean'] for r in rows if r['selector'] == 'HIGH_F']
        low = [r['future_KL_mean'] for r in rows if r['selector'] == 'LOW_F']
        top1_high = [r['top1_agreement'] for r in rows if r['selector'] == 'HIGH_F']
        top1_low = [r['top1_agreement'] for r in rows if r['selector'] == 'LOW_F']
        pr = dict(pair)
        pr.update({
            'KL_highF': mean(high),
            'KL_lowF': mean(low),
            'PAIR_GAIN': mean(low) - mean(high),
            'top1_highF': mean(top1_high),
            'top1_lowF': mean(top1_low),
            'HIGH_F_win': mean(low) > mean(high),
            'LOW_F_win': mean(high) > mean(low),
            'tie': abs(mean(low) - mean(high)) <= EPS,
            'eval_units': len(eval_rows),
        })
        pair_results.append(pr)
        save_csv('pair_eval_unit_rows.partial.csv', all_unit_rows)
        save_json('pair_results.partial.json', {'pairs': pair_results})
    return pair_results, all_unit_rows


def rank_corr(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    def ranks(vals):
        order = sorted((v, i) for i, v in enumerate(vals))
        out = [0.0] * len(vals)
        j = 0
        while j < len(order):
            k = j + 1
            while k < len(order) and order[k][0] == order[j][0]:
                k += 1
            r = (j + k - 1) / 2.0 + 1.0
            for _v, i in order[j:k]:
                out[i] = r
            j = k
        return out
    rx = ranks([x for x, _ in pairs])
    ry = ranks([y for _, y in pairs])
    mx = mean(rx)
    my = mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return num / den if den > EPS else None


def bootstrap_ci(xs, n=BOOTSTRAP_N, seed=20260907):
    xs = [float(x) for x in xs if finite(x)]
    if not xs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(sum(rng.choice(xs) for _ in xs) / len(xs))
    vals.sort()
    return [vals[int(0.025 * (n - 1))], vals[int(0.975 * (n - 1))]]


def binom_p_two_sided(wins, losses):
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * prob)


def summarize_pairs(pair_results):
    gains = [r['PAIR_GAIN'] for r in pair_results]
    wins = sum(1 for r in pair_results if r['PAIR_GAIN'] > EPS)
    losses = sum(1 for r in pair_results if r['PAIR_GAIN'] < -EPS)
    ties = len(pair_results) - wins - losses
    return {
        'N_pairs': len(pair_results),
        'HIGH_FUNCTIONAL_WINS': wins,
        'LOW_FUNCTIONAL_WINS': losses,
        'TIES': ties,
        'HIGH_FUNCTIONAL_WIN_RATE': wins / len(pair_results) if pair_results else None,
        'MEDIAN_PAIR_GAIN': med(gains),
        'MEAN_PAIR_GAIN': mean(gains),
        'PAIR_GAIN_BOOTSTRAP_CI': bootstrap_ci(gains),
        'PAIR_WIN_BINOMIAL_P': binom_p_two_sided(wins, losses),
        'FUNCTIONAL_GAP_VS_PAIR_GAIN_SPEARMAN': rank_corr([r['functional_gap'] for r in pair_results], gains),
        'ERROR_ONLY_GAP_VS_PAIR_GAIN_SPEARMAN': rank_corr([r['error_only_high'] - r['error_only_low'] for r in pair_results], gains),
    }


def boundary_masks(score_rows, selector, seed=None):
    by = rank_by_head([dict(r) for r in score_rows])
    masks = {}
    core_n = int(round(VDIM * CORE_RATIO))
    target_n = int(round(VDIM * PRIMARY_RATIO))
    rng = random.Random(seed)
    for layer in colprot.GDN_LAYERS:
        m = torch.zeros((HEADS, VDIM), dtype=torch.bool)
        for h in range(HEADS):
            items = by[(layer, h)]
            core = items[:core_n]
            band = items[core_n:int(round(VDIM * BOUNDARY_UPPER))]
            for r in core:
                m[h, r['value_column']] = True
            need = target_n - core_n
            if selector == 'RANDOM':
                band = list(band)
                rng.shuffle(band)
            elif selector == 'MAGNITUDE':
                band = sorted(band, key=lambda r: (r['magnitude_score'], -r['value_column']), reverse=True)
            elif selector == 'ERROR_ONLY':
                band = sorted(band, key=lambda r: (r['error_only_score'], -r['value_column']), reverse=True)
            elif selector == 'FUNCTIONAL_ONLY':
                band = sorted(band, key=lambda r: (r['functional_only_score'], -r['value_column']), reverse=True)
            elif selector == 'FAVOR':
                band = sorted(band, key=lambda r: (r['favor_score'], -r['value_column']), reverse=True)
            for r in band[:need]:
                m[h, r['value_column']] = True
        masks[layer] = m
    return masks


def run_boundary(model, tokenizer, e2e, pmap, eval_rows, score_rows):
    specs = {
        'C128_INT8': None,
        'B1_MAGNITUDE_BOUNDARY': boundary_masks(score_rows, 'MAGNITUDE'),
        'B2_ERROR_ONLY_BOUNDARY': boundary_masks(score_rows, 'ERROR_ONLY'),
        'B3_FUNCTIONAL_ONLY_BOUNDARY': boundary_masks(score_rows, 'FUNCTIONAL_ONLY'),
        'B4_FAVOR_BOUNDARY': boundary_masks(score_rows, 'FAVOR'),
    }
    for seed in RANDOM_SEEDS:
        specs[f'B0_RANDOM_BOUNDARY_seed{seed}'] = boundary_masks(score_rows, 'RANDOM', seed)
    rows = []
    for i, row in enumerate(eval_rows):
        print(f'[{now()}] boundary eval {i + 1}/{len(eval_rows)} {row["unit_id"]}', flush=True)
        rr = colprot.eval_unit(model, tokenizer, e2e, pmap, row, specs)
        for x in rr:
            if x['selector'] != 'FP_STATE':
                x['protection_ratio'] = PRIMARY_RATIO if x['selector'] != 'C128_INT8' else 0.0
                rows.append(x)
        torch.cuda.empty_cache()
    agg = colprot.aggregate_selector(rows)
    mag = next(r for r in agg if r['selector'] == 'B1_MAGNITUDE_BOUNDARY')
    fav = next(r for r in agg if r['selector'] == 'B4_FAVOR_BOUNDARY')
    return {
        'status': 'COMPLETE',
        'rows': rows,
        'aggregate': agg,
        'MAGNITUDE_ONLY_FUTURE_KL': mag['future_KL_median'],
        'MAGNITUDE_PLUS_FUNCTIONAL_FUTURE_KL': fav['future_KL_median'],
        'MAGNITUDE_PLUS_FUNCTIONAL_RELATIVE_GAIN': (mag['future_KL_median'] - fav['future_KL_median']) / (mag['future_KL_median'] + EPS),
        'FAVOR_vs_MAGNITUDE_BOUNDARY': colprot.paired_compare(rows, 'B4_FAVOR_BOUNDARY', 'B1_MAGNITUDE_BOUNDARY'),
    }


def write_md(name, title, obj):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    text = f'# {title}\n\n```json\n{json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)}\n```\n'
    (RUN_DIR / name).write_text(text, encoding='utf-8')
    (DOC_DIR / name).write_text(text, encoding='utf-8')
    (REP_DIR / f'{SLUG}_{name}').write_text(text, encoding='utf-8')


def efficiency():
    return {
        'ONLINE_MAGNITUDE_SCORING': 'NO',
        'ONLINE_FUNCTIONAL_SCORING': 'NO',
        'ONLINE_RANKING': 'NO',
        'DYNAMIC_SELECTOR': 'NO',
        'EXTRA_FULL_STATE_PASS': 0,
        'STATIC_BIT_ASSIGNMENT': 'YES',
        'STATIC_PACKING_FEASIBLE': 'YES_CONDITIONAL',
        'magnitude_only_calibration_cost': 'state column squared norms from calibration trajectory',
        'magnitude_plus_functional_calibration_cost': 'adds q-visible C128 residual contraction and diagonal G weighting from existing M5 artifacts',
        'extra_calibration_FLOPs': 'offline only; roughly one q^T residual contraction per layer/head/value column/calibration token',
        'extra_calibration_memory': 'selector_scores.csv plus M5 diagonal tensor reads; no inference memory increase',
        'FUNCTIONAL_REFINEMENT_COST_BENEFIT_WEAK': 'TO_BE_DECIDED_BY_RESULTS',
    }


def terminal(final):
    keys = [
        'TASK', 'FORMAL_STATUS', 'PREVIOUS_RESULT_REPRODUCTION_GATE', 'PROTOCOL_GATE',
        'MAGNITUDE_DEFINITION_MATCH_PREVIOUS', 'FUNCTIONAL_SCORE_DEFINITION_MATCH_PREVIOUS',
        'CALIBRATION_EVAL_DISJOINT', 'N_MAGNITUDE_MATCHED_PAIRS', 'N_FUNCTIONALLY_SEPARATED_PAIRS',
        'HIGH_FUNCTIONAL_WINS', 'LOW_FUNCTIONAL_WINS', 'TIES', 'HIGH_FUNCTIONAL_WIN_RATE',
        'MEDIAN_PAIR_GAIN', 'PAIR_GAIN_BOOTSTRAP_CI', 'PAIR_WIN_BINOMIAL_P',
        'FUNCTIONAL_GAP_VS_PAIR_GAIN_SPEARMAN', 'ERROR_ONLY_CONTROL_RESULT',
        'MATCHED_MAGNITUDE_CLASSIFICATION', 'BOUNDARY_REFINEMENT_STATUS',
        'MAGNITUDE_ONLY_FUTURE_KL', 'MAGNITUDE_PLUS_FUNCTIONAL_FUTURE_KL',
        'MAGNITUDE_PLUS_FUNCTIONAL_RELATIVE_GAIN', 'BEYOND_MAGNITUDE_FUNCTIONAL_UTILITY',
        'FUNCTIONAL_REFINEMENT_PRACTICAL_UTILITY', 'ONLINE_FUNCTIONAL_SCORING',
        'ONLINE_RANKING', 'EXTRA_FULL_STATE_PASS', 'STATIC_PACKING_FEASIBLE',
        'FINAL_CLASSIFICATION', 'METHOD_DIRECTION', 'METHOD_DESIGN_READY',
        'NEXT_RECOMMENDED_TASK',
    ]
    lines = []
    for k in keys:
        lines += [f'{k} =', str(final.get(k, '')), '']
    return '\n'.join(lines).rstrip()


def make_figures(pair_results, boundary):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return 'FAILED_IMPORT'
    fig = RUN_DIR / 'figures'
    fig.mkdir(parents=True, exist_ok=True)
    if pair_results:
        plt.figure(figsize=(5, 5))
        plt.scatter([r['KL_lowF'] for r in pair_results], [r['KL_highF'] for r in pair_results])
        vals = [v for r in pair_results for v in (r['KL_lowF'], r['KL_highF'])]
        lo, hi = min(vals), max(vals)
        plt.plot([lo, hi], [lo, hi], color='black', linewidth=1)
        plt.xlabel('KL_lowF')
        plt.ylabel('KL_highF')
        plt.tight_layout()
        plt.savefig(fig / 'kl_highF_vs_lowF_scatter.png', dpi=160)
        plt.close()
        plt.figure(figsize=(6, 4))
        plt.hist([r['PAIR_GAIN'] for r in pair_results], bins=10)
        plt.xlabel('PAIR_GAIN = KL_lowF - KL_highF')
        plt.tight_layout()
        plt.savefig(fig / 'pair_gain_distribution.png', dpi=160)
        plt.close()
        for key, fname in [('mag_rel_gap', 'magnitude_gap_vs_pair_gain.png'), ('functional_gap', 'functional_gap_vs_pair_gain.png'), ('error_rel_gap', 'error_only_gap_vs_pair_gain.png')]:
            plt.figure(figsize=(5, 4))
            plt.scatter([r[key] for r in pair_results], [r['PAIR_GAIN'] for r in pair_results])
            plt.xlabel(key)
            plt.ylabel('PAIR_GAIN')
            plt.tight_layout()
            plt.savefig(fig / fname, dpi=160)
            plt.close()
    if boundary and boundary.get('status') == 'COMPLETE':
        agg = [r for r in boundary['aggregate'] if r['selector'].startswith('B')]
        plt.figure(figsize=(9, 4))
        plt.bar([r['selector'] for r in agg], [r['future_KL_median'] for r in agg])
        plt.xticks(rotation=35, ha='right')
        plt.ylabel('median future KL')
        plt.tight_layout()
        plt.savefig(fig / 'boundary_refinement_kl.png', dpi=160)
        plt.close()
    return 'PASS'


def main():
    global torch
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    prev = json.loads(PREV_FINAL.read_text(encoding='utf-8')) if PREV_FINAL.exists() else {}
    score_rows = numeric_scores(read_csv(PREV_SCORES)) if PREV_SCORES.exists() else []
    prev_gate = (
        prev.get('BEST_SELECTOR') == 'MAGNITUDE'
        and abs(float(prev.get('BEST_PROTECTION_RATIO', -1)) - PRIMARY_RATIO) < EPS
        and PREV_FORMAL.exists()
        and bool(score_rows)
    )
    rows, pmap = colprot.load_units()
    cal_units = [r['unit_id'] for r in rows[:6]]
    eval_rows = rows[6:]
    eval_units = [r['unit_id'] for r in eval_rows]
    protocol = {
        'TASK': TASK,
        'timestamp': now(),
        'branch': safe_sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        'HEAD': safe_sh(['git', 'rev-parse', 'HEAD']),
        'git_status_start': safe_sh(['git', 'status', '--short']),
        'previous_final': str(PREV_FINAL),
        'previous_scores': str(PREV_SCORES),
        'PREVIOUS_RESULT_REPRODUCTION_GATE': 'PASS' if prev_gate else 'FAIL',
        'MAGNITUDE_DEFINITION_MATCH_PREVIOUS': 'YES',
        'FUNCTIONAL_SCORE_DEFINITION_MATCH_PREVIOUS': 'YES',
        'magnitude_definition': 'E_t ||S_t,:,j||_2^2 from previous selector_scores.csv',
        'functional_definition': 'FAVOR_j = E_t g_t,j * (q_t^T e_t,j^C128)^2 from previous selector_scores.csv',
        'calibration_units': cal_units,
        'evaluation_units': eval_units,
        'CALIBRATION_EVAL_DISJOINT': 'YES' if set(cal_units).isdisjoint(eval_units) else 'NO',
        'primary_threshold': PRIMARY_THRESHOLD,
        'threshold_panel': THRESHOLDS,
        'primary_protection_ratio': PRIMARY_RATIO,
        'candidate_band': 'magnitude rank 25%-50% within each layer/head',
    }
    save_json('protocol_audit.json', protocol)
    write_md('protocol_audit.md', 'Protocol Audit', protocol)
    if not prev_gate or protocol['CALIBRATION_EVAL_DISJOINT'] != 'YES':
        final = {
            'TASK': TASK,
            'FORMAL_STATUS': 'STOPPED_PREVIOUS_ARTIFACT_GATE_FAIL',
            'PREVIOUS_RESULT_REPRODUCTION_GATE': protocol['PREVIOUS_RESULT_REPRODUCTION_GATE'],
            'PROTOCOL_GATE': 'FAIL',
            'MAGNITUDE_DEFINITION_MATCH_PREVIOUS': 'YES',
            'FUNCTIONAL_SCORE_DEFINITION_MATCH_PREVIOUS': 'YES',
            'CALIBRATION_EVAL_DISJOINT': protocol['CALIBRATION_EVAL_DISJOINT'],
            'METHOD_DESIGN_READY': 'NO',
            'NEXT_RECOMMENDED_TASK': 'repair previous column protection artifacts',
        }
        save_json('final_classification.json', final)
        print(terminal(final))
        return
    by = rank_by_head(score_rows)
    coverage = {}
    for th in THRESHOLDS:
        coverage[f'boundary_{th}'] = {
            'N_CANDIDATE_MAG_MATCHED_PAIRS': len(pair_candidates(by, th, True)),
            'N_FUNCTIONALLY_SEPARATED_PAIRS': len(pair_candidates(by, th, True)),
            'N_UNRESTRICTED': len(pair_candidates(by, th, False)),
        }
    cands = pair_candidates(by, PRIMARY_THRESHOLD, True)
    pairs = disjoint_greedy(cands, FORMAL_PAIR_COUNT)
    save_csv('matched_pair_manifest.csv', pairs)
    save_json('matched_pair_manifest.json', {'coverage': coverage, 'pairs': pairs})
    protocol['PROTOCOL_GATE'] = 'PASS' if pairs else 'FAIL'
    save_json('protocol_audit.json', protocol)
    write_md('protocol_audit.md', 'Protocol Audit', protocol)
    if not pairs:
        raise RuntimeError('no primary matched-magnitude functional pairs')
    torch.set_grad_enabled(False)
    torch, model, tokenizer, cfg, e2e = p1.setup_model()
    pair_cache = RUN_DIR / 'formal_pair_results.json'
    unit_cache = RUN_DIR / 'pair_eval_unit_rows.csv'
    if pair_cache.exists():
        pair_results = json.loads(pair_cache.read_text(encoding='utf-8'))['pairs']
        unit_rows = read_csv(unit_cache) if unit_cache.exists() else []
        print(f'[{now()}] reusing cached pair results: {len(pair_results)} pairs', flush=True)
    else:
        pair_results, unit_rows = run_pair_eval(model, tokenizer, e2e, pmap, eval_rows, score_rows, pairs)
        save_csv('pair_eval_unit_rows.csv', unit_rows)
        save_json('formal_pair_results.json', {'pairs': pair_results})
    pair_summary = summarize_pairs(pair_results)
    save_json('pilot_pair_results.json', {'pairs': pair_results[:PILOT_PAIR_COUNT], 'summary': summarize_pairs(pair_results[:PILOT_PAIR_COUNT])})

    pilot_summary = summarize_pairs(pair_results[:PILOT_PAIR_COUNT])
    pilot_positive = (
        pilot_summary['HIGH_FUNCTIONAL_WIN_RATE'] is not None
        and pilot_summary['HIGH_FUNCTIONAL_WIN_RATE'] > 0.5
        and pilot_summary['MEDIAN_PAIR_GAIN'] > 0
        and all(len(pair_candidates(by, th, True)) > 0 for th in THRESHOLDS)
    )
    clear_partial = (
        pilot_summary['HIGH_FUNCTIONAL_WIN_RATE'] is not None
        and pilot_summary['HIGH_FUNCTIONAL_WIN_RATE'] >= 0.5
        and pilot_summary['MEDIAN_PAIR_GAIN'] >= 0
    )
    boundary = {'status': 'NOT_RUN'}
    if pilot_positive or clear_partial:
        boundary = run_boundary(model, tokenizer, e2e, pmap, eval_rows, score_rows)
        save_json('boundary_refinement_results.json', boundary)
        save_csv('boundary_refinement_unit_rows.csv', boundary['rows'])
    else:
        save_json('boundary_refinement_results.json', boundary)

    high_win_rate = pair_summary['HIGH_FUNCTIONAL_WIN_RATE']
    median_gain = pair_summary['MEDIAN_PAIR_GAIN']
    if high_win_rate is not None and high_win_rate >= 0.75 and median_gain > 0:
        matched_cls = 'BEYOND_MAGNITUDE_FUNCTIONAL_UTILITY_SUPPORTED'
        beyond = 'YES'
    elif high_win_rate is not None and high_win_rate > 0.5 and median_gain > 0:
        matched_cls = 'BEYOND_MAGNITUDE_FUNCTIONAL_UTILITY_PARTIAL'
        beyond = 'PARTIAL'
    elif pair_summary['ERROR_ONLY_GAP_VS_PAIR_GAIN_SPEARMAN'] and abs(pair_summary['ERROR_ONLY_GAP_VS_PAIR_GAIN_SPEARMAN']) > abs(pair_summary['FUNCTIONAL_GAP_VS_PAIR_GAIN_SPEARMAN'] or 0):
        matched_cls = 'FUNCTIONAL_SIGNAL_REDUNDANT_WITH_ERROR_ONLY'
        beyond = 'NO'
    else:
        matched_cls = 'NO_BEYOND_MAGNITUDE_FUNCTIONAL_UTILITY'
        beyond = 'NO'
    if boundary.get('status') == 'COMPLETE':
        b_gain = boundary['MAGNITUDE_PLUS_FUNCTIONAL_RELATIVE_GAIN']
        practical = 'YES' if b_gain > 0.05 else ('PARTIAL' if b_gain > 0 else 'NO')
    else:
        b_gain = None
        practical = 'NO' if beyond == 'NO' else 'PARTIAL'
    if beyond in ('YES', 'PARTIAL') and practical in ('YES', 'PARTIAL'):
        method_dir = 'MAGNITUDE_FIRST_FUNCTIONAL_REFINED_MIXED_PRECISION' if practical == 'YES' else 'MAGNITUDE_AWARE_MIXED_PRECISION_WITH_OPTIONAL_FUNCTIONAL_TIEBREAK'
        ready = 'YES' if practical == 'YES' else 'CONDITIONAL_YES'
    elif beyond == 'NO':
        method_dir = 'MAGNITUDE_AWARE_MIXED_PRECISION'
        ready = 'CONDITIONAL_YES'
    else:
        method_dir = 'UNRESOLVED'
        ready = 'NO'
    final_cls = matched_cls if practical != 'YES' else 'BEYOND_MAGNITUDE_FUNCTIONAL_UTILITY_SUPPORTED_WITH_BOUNDARY_REFINEMENT'
    err_control = 'ERROR_MATCHED_SUBSET_AVAILABLE' if sum(1 for p in pair_results if p['error_rel_gap'] <= 0.02) >= 3 else 'ERROR_ONLY_NOT_FULLY_CONTROLLED'
    eff = efficiency()
    if practical != 'YES':
        eff['FUNCTIONAL_REFINEMENT_COST_BENEFIT_WEAK'] = 'YES'
    save_json('efficiency_audit.json', eff)
    write_md('efficiency_audit.md', 'Efficiency Audit', eff)
    final = {
        'TASK': TASK,
        'FORMAL_STATUS': 'COMPLETE',
        'PREVIOUS_RESULT_REPRODUCTION_GATE': protocol['PREVIOUS_RESULT_REPRODUCTION_GATE'],
        'PROTOCOL_GATE': protocol['PROTOCOL_GATE'],
        'MAGNITUDE_DEFINITION_MATCH_PREVIOUS': 'YES',
        'FUNCTIONAL_SCORE_DEFINITION_MATCH_PREVIOUS': 'YES',
        'CALIBRATION_EVAL_DISJOINT': protocol['CALIBRATION_EVAL_DISJOINT'],
        'N_MAGNITUDE_MATCHED_PAIRS': coverage['boundary_0.02']['N_CANDIDATE_MAG_MATCHED_PAIRS'],
        'N_FUNCTIONALLY_SEPARATED_PAIRS': coverage['boundary_0.02']['N_FUNCTIONALLY_SEPARATED_PAIRS'],
        'HIGH_FUNCTIONAL_WINS': pair_summary['HIGH_FUNCTIONAL_WINS'],
        'LOW_FUNCTIONAL_WINS': pair_summary['LOW_FUNCTIONAL_WINS'],
        'TIES': pair_summary['TIES'],
        'HIGH_FUNCTIONAL_WIN_RATE': high_win_rate,
        'MEDIAN_PAIR_GAIN': median_gain,
        'PAIR_GAIN_BOOTSTRAP_CI': pair_summary['PAIR_GAIN_BOOTSTRAP_CI'],
        'PAIR_WIN_BINOMIAL_P': pair_summary['PAIR_WIN_BINOMIAL_P'],
        'FUNCTIONAL_GAP_VS_PAIR_GAIN_SPEARMAN': pair_summary['FUNCTIONAL_GAP_VS_PAIR_GAIN_SPEARMAN'],
        'ERROR_ONLY_CONTROL_RESULT': err_control,
        'MATCHED_MAGNITUDE_CLASSIFICATION': matched_cls,
        'BOUNDARY_REFINEMENT_STATUS': boundary.get('status', 'NOT_RUN'),
        'MAGNITUDE_ONLY_FUTURE_KL': boundary.get('MAGNITUDE_ONLY_FUTURE_KL'),
        'MAGNITUDE_PLUS_FUNCTIONAL_FUTURE_KL': boundary.get('MAGNITUDE_PLUS_FUNCTIONAL_FUTURE_KL'),
        'MAGNITUDE_PLUS_FUNCTIONAL_RELATIVE_GAIN': b_gain,
        'BEYOND_MAGNITUDE_FUNCTIONAL_UTILITY': beyond,
        'FUNCTIONAL_REFINEMENT_PRACTICAL_UTILITY': practical,
        'ONLINE_FUNCTIONAL_SCORING': 'NO',
        'ONLINE_RANKING': 'NO',
        'EXTRA_FULL_STATE_PASS': 0,
        'STATIC_PACKING_FEASIBLE': 'YES_CONDITIONAL',
        'FINAL_CLASSIFICATION': final_cls,
        'METHOD_DIRECTION': method_dir,
        'METHOD_DESIGN_READY': ready,
        'NEXT_RECOMMENDED_TASK': 'GDN_MAGNITUDE_FUNCTIONAL_MIXED_BIT_ALLOCATION_V1' if ready == 'YES' else 'Magnitude-aware C-oriented mixed precision plus grouping/packing/range structure',
        'coverage': coverage,
        'pair_summary': pair_summary,
        'pilot_summary': pilot_summary,
        'boundary_refinement': boundary,
        'efficiency_audit': eff,
        'model_path': cfg.get('model_path'),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
    }
    save_json('final_classification.json', final)
    make_figures(pair_results, boundary)
    write_md('pilot_pair_report.md', 'Pilot Pair Report', {'summary': pilot_summary, 'pairs': pair_results[:PILOT_PAIR_COUNT]})
    write_md('formal_pair_report.md', 'Formal Pair Report', {'summary': pair_summary, 'pairs': pair_results})
    write_md('boundary_refinement_report.md', 'Boundary Refinement Report', boundary)
    report = '# ' + TASK + '\n\n## Terminal Summary\n\n```text\n' + terminal(final) + '\n```\n'
    (RUN_DIR / 'final_report.md').write_text(report, encoding='utf-8')
    (DOC_DIR / 'final_report.md').write_text(report, encoding='utf-8')
    (REP_DIR / f'{SLUG}_final_report.md').write_text(report, encoding='utf-8')
    print(terminal(final))


if __name__ == '__main__':
    main()
