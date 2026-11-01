"""Fixed full-vocab CPU-only reducer qualification; run in two fresh processes."""
import argparse
import copy
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import reducer as R


def run():
    R.configure_cpu()
    torch.use_deterministic_algorithms(True)
    v = 248320
    axis = torch.arange(v, dtype=torch.float32)
    base = torch.sin(axis * 0.013) * 6.0 + torch.cos(axis * 0.0007) * 2.0
    cases = [('identical', base, base.clone()),
             ('near_identical', base, base + torch.sin(axis * 0.031) * 0.0001),
             ('different', base, torch.cos(axis * 0.003) * 4.0 - torch.sin(axis * 0.0009)),
             ('zeros', torch.zeros_like(base), torch.zeros_like(base))]
    rows = []
    identity = {'doc_id': 'SYNTHETIC_REDUCER_ONLY', 'input_ids_sha256': 'SYNTHETIC_NOT_SCIENCE'}
    for name, a, b in cases:
        first = R.score_pair(a, b, 8192, 0, identity)
        second = R.score_pair(a, b, 8192, 0, identity)
        comparable = lambda x: {k: z for k, z in x.items() if k != 'scoring_wall_seconds'}
        if comparable(first) != comparable(second):
            raise AssertionError('CPU repeated arithmetic mismatch: ' + name)
        if name in ('identical', 'zeros') and first['KL_FP32'] != 0.0:
            raise AssertionError('identical logits primary KL nonzero')
        if name == 'different' and first['KL_FP64'] <= 0.0:
            raise AssertionError('different logits reference KL not positive')
        rows.append(dict(case=name, **comparable(first), repeat_exact=True))
    # Coverage and pairing checks use already-qualified real arithmetic rows;
    # they do not invent a scientific 384-position dataset or a scientific CSV.
    ar, dr = [], []
    for t in R.SCORE_POSITIONS:
        x, y = copy.deepcopy(rows[2]), copy.deepcopy(rows[1])
        for z in (x, y):
            z.update(token_index=t, consumed_tokens=t+1, predicted_token_index=t+1)
            z.pop('case'); z.pop('repeat_exact')
        ar.append(x); dr.append(y)
    aggregate = R.summarize_document(ar, dr)
    if aggregate['score_count'] != 384 or any(w['score_count'] != 128 for w in aggregate['windows'].values()):
        raise AssertionError('score/window coverage mismatch')
    rejected = []
    for name, bad in [('missing_score', dr[:-1]), ('duplicate_position', [dr[0]] + dr[:-1])]:
        try:
            R.summarize_document(ar, bad)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('invalid score stream was accepted: ' + name)
    bad = copy.deepcopy(dr); bad[0]['predicted_token_index'] += 1
    try:
        R.summarize_document(ar, bad)
    except ValueError:
        rejected.append('off_by_one')
    else:
        raise AssertionError('off-by-one score accepted')
    if torch.cuda.is_initialized():
        raise AssertionError('CPU qualification initialized CUDA')
    return {'status': 'PASS', 'scope': 'fixed synthetic full-vocabulary CPU scoring qualification only',
            'identity': R.identity_record(), 'fixture_vocab': v, 'cases': rows,
            'aggregation_fixture_only': aggregate, 'invalid_streams_rejected': rejected,
            'CUDA_initialized': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); report = run()
    p = pathlib.Path(args.output)
    with p.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(p),
                      'fixed_reduce_order_sha256': report['identity']['fixed_reduce_order_sha256']}))
