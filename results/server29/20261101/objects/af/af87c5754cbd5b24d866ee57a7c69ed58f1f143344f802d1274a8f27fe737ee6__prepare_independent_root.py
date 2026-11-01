"""CPU-only creation of a separate continuation workspace; never alters old seals."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

TASK = 'QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1'
OLD = Path('/data/zypan/analyses/qwen_gdn_c6r_dual_m_interaction_reuse_5doc_v1/experiments/qwen_gdn/quantizer') / TASK
ROOT = Path('/data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer') / TASK
CP = '/data/zypan/qwen_c6r_dual_m_interaction_archive/' + TASK + '/checkpoints/CURRENT.pt'
CP_SHA = '0d9c2e22a634b4327c03e34343087fd3d796276035f653b4b22ead3d8cf442ce'
SSH = ['ssh', '-S', '/tmp/qwen_replication_archive29.sock', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'zypan@202.38.247.29']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not ROOT.exists(), 'Independent root already exists; no implicit retry'
    manifest = json.loads((OLD / 'SOURCE_MANIFEST.json').read_text())
    assert sha(OLD / 'SOURCE_MANIFEST.json') == 'd76fcb4057eaaa781806ab6865e91998705ecb773449cf8851d2f0641969f9fc'
    assert (OLD / 'FIRST_FAILURE.json').is_file(), 'Expected preserved original stop absent'
    ROOT.mkdir(parents=True)
    copied = []

    def copy(rel, target=None, expected=None):
        source = OLD / rel
        dest = ROOT / (target or rel)
        assert source.is_file() and not dest.exists()
        digest = sha(source)
        if expected:
            assert digest == expected['sha256'] and source.stat().st_size == expected['bytes'], rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        assert sha(dest) == digest
        copied.append(dict(original=str(source), path=str(dest.relative_to(ROOT)), sha256=digest, bytes=dest.stat().st_size))

    for row in manifest['execution_sources']:
        p = Path(row['path'])
        assert not p.is_absolute() and '..' not in p.parts
        copy(str(p), expected=row)
    for rel in ['SOURCE_MANIFEST.json', 'source_manifest.json', 'preregistration.json', 'GPU_QUALIFICATION.json']:
        if not (ROOT / rel).exists():
            copy(rel)
    for rel in ['FIRST_FAILURE.json', 'GPU_RESOURCE_LEDGER.json', 'FINAL_GPU_STATUS.json', 'FINAL_SCIENTIFIC_REPORT.md',
                'SHA256_MANIFEST.json', 'hashes/SHA256SUMS.txt', 'hashes/FINAL_ARCHIVE_VERIFICATION.json',
                'analysis/ENV_science_doc5_C6R_C128.json', 'checkpoints/CATALOG.jsonl']:
        copy(rel, 'provenance/original/' + rel)
    for doc in range(1, 5):
        copy(f'archive/DOC_{doc:02d}_CURRENT_ASSET_LOCATIONS.json')
        for cond in ['C6R_C128', 'C6R_DUAL_M']:
            phase = f'science_doc{doc}_{cond}'
            for name in [phase, 'ENV_' + phase, 'RECEIPT_' + phase]:
                copy('analysis/' + name + '.json')
    for rel in ['inputs', 'scores', 'checkpoints', 'analysis', 'logs', 'hashes', 'archive', 'provenance']:
        (ROOT / rel).mkdir(exist_ok=True)
    dest = ROOT / 'inputs/CURRENT_DOC5_C128.pt'
    with dest.with_suffix('.partial').open('xb') as f:
        subprocess.run(SSH + ['cat ' + CP], stdout=f, check=True, timeout=600)
    assert dest.with_suffix('.partial').stat().st_size == 715984046
    assert sha(dest.with_suffix('.partial')) == CP_SHA
    dest.with_suffix('.partial').replace(dest)
    result = dict(status='PASS', no_GPU=True, original_root=str(OLD), continuation_root=str(ROOT),
                  copied_files=copied, original_stop_preserved_SHA256=sha(OLD / 'FIRST_FAILURE.json'),
                  checkpoint=dict(path=str(dest), archive_path=CP, sha256=CP_SHA, bytes=dest.stat().st_size),
                  prepared_unix=time.time())
    with (ROOT / 'provenance/INDEPENDENT_ROOT_PREPARATION.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', copied_files=len(copied), root=str(ROOT), checkpoint_SHA256=CP_SHA, no_GPU=True)))


if __name__ == '__main__':
    main()
