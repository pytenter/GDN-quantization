import argparse

from common import EXP, PARENT, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--document', type=int, required=True, choices=range(1, 5))
    args = parser.parse_args()
    doc = args.document
    current_dir = EXP / f'discovery/doc_{doc:02d}/drive'
    parent_dir = PARENT / f'scientific/doc_{doc:02d}'
    current = read_json(current_dir / 'drive_manifest.json')
    parent = read_json(parent_dir / 'drive_manifest.json')
    scalar_fields = ['doc_id', 'doc_text_sha256', 'fixture_token_ids_file_sha256',
                     'token_count', 'layer_ids', 'source_sha256', 'model_identity',
                     'tensors', 'dtypes', 'normalization', 'initial_state',
                     'original_production_call_counts', 'all_complete_finite',
                     'generated_tokens', 'model_forward_count',
                     'full_hidden_trajectory_saved']
    comparisons = []
    for field in scalar_fields:
        same = current[field] == parent[field]
        comparisons.append({'field': field, 'pass': same})
        if not same:
            raise RuntimeError(f'drive identity mismatch in {field}')
    if current['fp_state_hashes_file_sha256'] != parent['fp_state_hashes_file_sha256']:
        raise RuntimeError('FP state hash file differs from parent')
    if sha256(current_dir / 'fp_state_hashes.json') != parent['fp_state_hashes_file_sha256']:
        raise RuntimeError('current FP state hash file does not match declared identity')
    if len(current['shards']) != len(parent['shards']):
        raise RuntimeError('shard count mismatch')
    shards = []
    for now, old in zip(current['shards'], parent['shards']):
        for field in ['relative_path', 'file_sha256', 'file_size_bytes',
                      'first_token', 'last_token', 'tensors']:
            if now[field] != old[field]:
                raise RuntimeError(f'shard identity mismatch: {now["relative_path"]} {field}')
        path = current_dir / now['relative_path']
        actual = sha256(path)
        if actual != old['file_sha256']:
            raise RuntimeError(f'recaptured shard hash mismatch: {path}')
        shards.append({'relative_path': now['relative_path'], 'sha256': actual,
                       'bytes': path.stat().st_size})
    checkpoints = []
    for now, old in zip(current['fp_reference_checkpoints'], parent['fp_reference_checkpoints']):
        for field in ['token', 'relative_path', 'file_sha256', 'per_layer_sha256',
                      'file_size_bytes']:
            if now[field] != old[field]:
                raise RuntimeError(f'FP checkpoint identity mismatch at {now["token"]}')
        actual = sha256(current_dir / now['relative_path'])
        if actual != old['file_sha256']:
            raise RuntimeError('FP checkpoint file mismatch')
        checkpoints.append({'token': now['token'], 'sha256': actual})
    write_json(EXP / f'discovery/doc_{doc:02d}/drive_identity.json', {
        'FROZEN_DRIVE_IDENTITY_GATE': 'PASS', 'document_index': doc,
        'parent_manifest_sha256': sha256(parent_dir / 'drive_manifest.json'),
        'recaptured_manifest_sha256': sha256(current_dir / 'drive_manifest.json'),
        'scientific_content_identity': 'EXACT', 'scalar_comparisons': comparisons,
        'shards_verified': shards, 'reference_checkpoints_verified': checkpoints,
        'fp_state_hashes_sha256': current['fp_state_hashes_file_sha256']})
    print(f'DOC_{doc}_FROZEN_DRIVE_IDENTITY_PASS', flush=True)


if __name__ == '__main__':
    main()
