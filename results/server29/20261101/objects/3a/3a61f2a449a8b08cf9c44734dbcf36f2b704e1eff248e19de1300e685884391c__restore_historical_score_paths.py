"""Restore 15 exact sealed score files to a /data cache and real local aliases.

No forward, no tensor transformation, and no replacement of existing files.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

TASK = 'QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1'
ROOT = Path('/data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer') / TASK
OLD = Path('/data/zypan/analyses/qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1')
ARCHIVE = Path('/data/zypan/qwen_dual_axis_replication_archive/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1')
SSH = ['ssh', '-S', '/tmp/qwen_replication_archive29.sock', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'zypan@202.38.247.29']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    report = ROOT / 'provenance/HISTORICAL_SCORE_PATH_RESTORATION.json'
    assert not report.exists()
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    rows = [a for a in manifest['external_dependencies'] if str(a['path']).startswith(str(OLD / 'scores') + '/')]
    assert len(rows) == 15
    cache = ROOT / 'inputs/historical_scores'
    cache.mkdir()
    checked = []
    for a in rows:
        original = Path(a['path'])
        assert original.parent == OLD / 'scores' and original.suffix == '.pt'
        resolved = original.resolve(strict=False)
        assert resolved.parent == Path('/tmp/qwen_5doc_scores_20261016')
        assert not original.exists() and not resolved.exists() and not resolved.is_symlink()
        dest = cache / original.name
        remote = ARCHIVE / 'scores' / original.name
        with dest.with_suffix('.partial').open('xb') as f:
            subprocess.run(SSH + [shlex.join(['cat', str(remote)])], stdout=f, check=True, timeout=600)
        assert dest.with_suffix('.partial').stat().st_size == a['bytes']
        assert sha(dest.with_suffix('.partial')) == a['sha256']
        dest.with_suffix('.partial').replace(dest)
        os.symlink(str(dest), resolved)
        assert original.stat().st_size == a['bytes'] and sha(original) == a['sha256']
        checked.append(dict(original_path=str(original), local_alias=str(resolved), cache_path=str(dest), archive_host='202.38.247.29', archive_path=str(remote), sha256=a['sha256'], bytes=a['bytes'], status='EXACT_CONTENT_RESTORED'))
        with (ROOT / 'provenance/HISTORICAL_SCORE_PATH_RESTORATION_PARTIAL.jsonl').open('a') as f:
            f.write(json.dumps(checked[-1]) + '\n')
        print(json.dumps(dict(restored=len(checked), total=15, path=str(original))), flush=True)
    with report.open('x') as f:
        json.dump(dict(status='PASS', assets=checked, restored_bytes=sum(a['bytes'] for a in checked),
                       no_GPU=True, no_FP_H_forward=True, no_frozen_metadata_modified=True, restored_unix=time.time()), f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
