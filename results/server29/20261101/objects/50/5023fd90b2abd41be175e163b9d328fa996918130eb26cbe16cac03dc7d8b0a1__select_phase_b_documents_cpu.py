"""Outcome-blind, CPU-only prospective document identity/availability audit.

Reads only input files, source selection metadata and path names. Never opens
historical result/canonical/logit files or any other experiment's outcomes.
Does not launch GPU measurements, select rotation pairs or relax eligibility.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
TASK = 'LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1'
POOL = Path('/data/zypan/diagnostic_inputs/phase3_mixed_hardware_v1_20261003')
PREVIOUS_METADATA = Path('/data/zypan/experiment_runs/LING_KDA_PERSISTENT_SURROGATE_PROSPECTIVE_VALIDATION_V1/PROSPECTIVE_DOCUMENT_SELECTION.json')
MODEL = Path('/data/zypan/models/Ling-3.0-tiny')
DOC_RE = re.compile(r'pg19-test-\d+')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    sidecar = path.with_suffix('.sha256.json')
    with sidecar.open('x', encoding='utf8') as stream:
        json.dump({'path': str(path), 'sha256': sha(path)}, stream, sort_keys=True)
        stream.write('\n')


def inventory_used_paths():
    evidence = {}
    skipped = {'.git', '__pycache__', 'node_modules', '.cache', 'inputs', 'tokens',
               'raw_text', 'rotations', 'sources', 'models', 'envs'}
    for base in ['/data/zypan/experiment_runs', '/data/zypan/worktrees',
                 '/data/zypan/cleanroom', '/data/zypan/diagnostic_inputs']:
        for directory, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in skipped and Path(directory) / x != ROOT]
            # Measurement directories/files retain document identities in paths.
            # Only path strings are examined; outcome contents remain unopened.
            for item in [directory] + [str(Path(directory) / x) for x in files]:
                for document in DOC_RE.findall(item):
                    evidence.setdefault(document, set()).add(item)
    return {doc: sorted(paths) for doc, paths in sorted(evidence.items())}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_ONLY_ENVIRONMENT_REQUIRED'
    assert not (ROOT / 'PROSPECTIVE_DOCUMENT_SELECTION.json').exists(), 'NO_DUPLICATE_DOCUMENT_SELECTION'
    assert not list((ROOT / 'runs').glob('*.STOP.json')), 'FAIL_CLOSED'
    route_path = ROOT / 'phase_a_analysis_20261011/PHASE_A_ROUTE.json'
    route = json.loads(route_path.read_text())
    assert route['route'] == 'A' and not route['used_historical_or_new_future_outcomes']
    source = json.loads((ROOT / 'SOURCE_IDENTITIES.json').read_text())
    for filename, expected in [('config.json', source['model_identity']['config_sha256']),
                               ('tokenizer.json', source['model_identity']['tokenizer_sha256']),
                               ('tokenizer_config.json', source['model_identity']['tokenizer_config_sha256'])]:
        assert sha(MODEL / filename) == expected, 'MODEL_OR_TOKENIZER_IDENTITY_CHANGED'
    rule = {
        'experiment': TASK, 'seed': 0, 'frozen_before_candidate_ordering': True,
        'order': 'ascending SHA256(UTF8(experiment + "|seed=0|" + document_id + "|" + token_file_sha256)); document_id lexicographic tie-break',
        'take': 2, 'rank_gpu_map': {'0': 6, '1': 3},
        'pool': str(POOL), 'exclude_all_prior_measurement_path_ids': True,
        'exclude_prior_ineligible_metadata_candidates': True,
        'explicit_exclusions': ['pg19-test-37702', 'pg19-test-5396', 'pg19-test-28988',
                                'pg19-test-26493', 'pg19-test-48693', 'pg19-test-51410'],
        'rotation_pair_selection_not_executed': True,
        'no_long_outcome_file_opened': True, 'gpu_hours': 0,
        'route_sha256': sha(route_path), 'selection_code_sha256': sha(__file__),
    }
    publish(ROOT / 'PHASE_B_DOCUMENT_SELECTION_RULE_FREEZE.json', rule)
    metadata = json.loads(PREVIOUS_METADATA.read_text())
    used_paths = inventory_used_paths()
    forbidden = set(rule['explicit_exclusions']) | set(used_paths) | set(metadata['forbidden_documents'])
    forbidden.update(x['document_id'] for x in metadata['selected_documents'])
    registry = {'prior_measurement_path_ids': used_paths,
                'previous_document_selection_metadata_sha256': sha(PREVIOUS_METADATA),
                'no_outcome_contents_opened': True, 'registry_is_conservative': True}
    publish(ROOT / 'PRIOR_DOCUMENT_PATH_IDENTITY_AUDIT.json', registry)
    vocab_size = json.loads((MODEL / 'config.json').read_text())['vocab_size']
    candidates = []
    for old in metadata['candidate_audit']:
        document = old['document_id']
        tokens_path = POOL / 'tokens' / f'{document}.json'
        text_path = POOL / 'raw_text' / f'{document}.txt'
        tokens = json.loads(tokens_path.read_text())
        token_sha, text_sha = sha(tokens_path), sha(text_path)
        assert token_sha == old['token_file']['sha256'] and text_sha == old['raw_text']['sha256'], 'INPUT_IDENTITY_CHANGED'
        legal = isinstance(tokens, list) and len(tokens) >= 32769 and all(type(x) is int and 0 <= x < vocab_size for x in tokens)
        reason = ('PRIOR_DOCUMENT_ID_OR_MEASUREMENT_PATH' if document in forbidden else
                  'PREVIOUS_IDENTITY_AUDIT_MARKED_USED' if not old['eligible'] else
                  'ILLEGAL_OR_INSUFFICIENT_INPUT_TOKENS' if not legal else None)
        key = hashlib.sha256(f'{TASK}|seed=0|{document}|{token_sha}'.encode()).hexdigest()
        candidates.append({'document_id': document, 'eligible': reason is None,
            'exclusion_reason': reason, 'order_sha256': key, 'token_count_with_target': len(tokens),
            'token_file': {'path': str(tokens_path), 'sha256': token_sha},
            'raw_text': {'path': str(text_path), 'sha256': text_sha}})
    ordered = sorted([x for x in candidates if x['eligible']], key=lambda x: (x['order_sha256'], x['document_id']))
    if len(ordered) < 2:
        publish(ROOT / 'PHASE_B_DATA_AVAILABILITY_STOP.json', {'status': 'DATA_AVAILABILITY_BLOCKER', 'gpu_hours': 0, 'candidate_audit': candidates})
        raise RuntimeError('INSUFFICIENT_GENUINELY_UNSEEN_DOCUMENTS_NO_GPU_LAUNCH')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    chosen = []
    for rank, candidate in enumerate(ordered[:2]):
        token_ids = json.loads(Path(candidate['token_file']['path']).read_text())
        encoded = tokenizer.encode(Path(candidate['raw_text']['path']).read_text(encoding='utf8'), add_special_tokens=False)
        assert encoded[:len(token_ids)] == token_ids and len(encoded) >= 32769, 'FROZEN_TOKENIZER_RAW_TEXT_REENCODING_MISMATCH'
        chosen.append(dict(candidate, selection_rank=rank, gpu=rule['rank_gpu_map'][str(rank)],
            gpu_uuid=source_gpu_uuid(rule['rank_gpu_map'][str(rank)]),
            frozen_tokenizer_reencoding_exact=True, available_legal_raw_text_tokens=len(encoded)))
    output = {'experiment': TASK, 'status': 'INDEPENDENT_INPUT_IDENTITIES_SELECTED_NO_PAIR_OR_LONG_RUN_YET',
        'route': 'A', 'selected_documents': chosen, 'candidate_audit': candidates,
        'selection_rule_sha256': sha(ROOT / 'PHASE_B_DOCUMENT_SELECTION_RULE_FREEZE.json'),
        'prior_document_path_audit_sha256': sha(ROOT / 'PRIOR_DOCUMENT_PATH_IDENTITY_AUDIT.json'),
        'model_identity': source['model_identity'], 'outcome_blind': True,
        'new_full_persistent_outcomes_generated_or_read': False,
        'rotation_pairs_selected': False, 'phase_b_gpu_measurements_started': False,
        'no_replacement_document_if_pair_missing': True, 'gpu_hours': 0, 'epoch': time.time()}
    publish(ROOT / 'PROSPECTIVE_DOCUMENT_SELECTION.json', output)
    print(json.dumps({'status': output['status'], 'eligible_candidates': len(ordered),
                      'documents': [(x['document_id'], x['gpu']) for x in chosen],
                      'sha256': sha(ROOT / 'PROSPECTIVE_DOCUMENT_SELECTION.json'), 'gpu_hours': 0}))


def source_gpu_uuid(gpu):
    return {6: 'GPU-de53f243-b271-521c-d875-d96611ed8574',
            3: 'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'}[gpu]


if __name__ == '__main__':
    main()
