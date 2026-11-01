"""Final CPU-only source/content/resource check. Does not dispatch a worker."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path('/data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    need = lambda v, s: None if v else (_ for _ in ()).throw(RuntimeError(s))
    need(not (ROOT / 'COORDINATOR_STARTED.json').exists(), 'No GPU phase may have started')
    spec = importlib.util.spec_from_file_location('recovery_control_preflight', ROOT / 'control/controller.py')
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    O = C.original_control()
    admission, budget, manifest, original, parent = C.preflight(ROOT, O)
    checked = []
    for a in original['execution_sources'] + original['external_dependencies']:
        p = Path(a['path']) if Path(a['path']).is_absolute() else ROOT / a['path']
        need(p.is_file() and p.stat().st_size == a['bytes'] and sha(p) == a['sha256'], 'Actual frozen dependency failed ' + str(p))
        checked.append(dict(path=str(p), sha256=a['sha256'], bytes=a['bytes']))
    proof = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'control/recovery_worker_entry.py'), '--audit'], text=True, timeout=30))
    need(proof['status'] == 'PASS_CPU_AST_ONLY' and proof['run']['token_loop_body_AST_identical'], 'Recovery AST proof failed')
    probe = O.inventory(parent['GPU_UUIDS'])
    O.idle(probe, 0)
    storage = C.storage(ROOT, budget, O)
    archive = O.archive_preflight(budget)
    out = dict(status='PASS', scope='CPU-only original/current path content, recovery AST, source, budget and resource preflight; actual GPU restore still required',
               all_original_frozen_dependencies_fresh_SHA=checked, original_dependency_count=len(checked),
               continuation_source_manifest_sha256=sha(ROOT / 'CONTINUATION_SOURCE_MANIFEST.json'),
               original_source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'),
               checkpoint_sha256=sha(ROOT / admission['input_checkpoint']['path']),
               recovery_AST_proof=proof, physical_GPU_probe=probe, storage=storage, permanent_archive=archive,
               conservative_cumulative_GPUh=C.projection(budget, []) / 3600,
               no_GPU_forward=True, checked_unix=time.time())
    p = ROOT / 'provenance/PRE_GPU_RECOVERY_QUALIFICATION.json'
    with p.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='PASS', no_GPU_forward=True, original_dependencies_fresh_SHA=len(checked),
                         conservative_cumulative_GPUh=out['conservative_cumulative_GPUh'], source_sha256=out['continuation_source_manifest_sha256'])))


if __name__ == '__main__':
    main()
