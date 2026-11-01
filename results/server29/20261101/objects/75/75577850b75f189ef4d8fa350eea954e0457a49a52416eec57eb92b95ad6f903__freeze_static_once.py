"""Create the one static source manifest and GPU0 permit after CPU admission.

This local operation never starts a child, imports torch, or calls a GPU API.
Remote staging/identity admission and the actual tmux launch are separate.
"""
import hashlib
import json
from pathlib import Path
import sys

L = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(L / 'src'))
import pace_control as C


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write('\n')


for name in ('source_manifest.json', 'GPU_PERMIT.json', 'SESSION_STARTED.json',
             'OFFLINE_PARENT_STARTED.json', 'SCIENCE_FREEZE.json'):
    assert not (L / name).exists(), 'Already frozen/started: ' + name
reports = [read(L / ('operations/ACTUAL_TARGET_CPU_FRESH' + str(i) + '.json'))
           for i in (6, 7)]
for item in reports:
    report = item['report']
    assert item['returncode'] == 0 and report['status'] == 'PASS'
    assert report['GPU_qualification_credit'] is False and report['model_weight_loads'] == 0
    assert report['first_chunk_adapter_parity']['status'] == 'PASS'
    assert report['CUDA_context_operation_calls'] == 0 and report['frozen_prompts'] == 30
    assert report['tokenizer_prefixes'] == 8
    for name, digest in report['source_sha256'].items():
        assert sha(L / name) == digest, 'CPU-qualified source changed: ' + name
assert reports[0]['report']['source_sha256'] == reports[1]['report']['source_sha256']
controls = read(L / 'operations/ACTUAL_TARGET_FINAL_TWO_FRESH_CPU_CONTROL.json')
assert controls['status'] == 'PASS_TWO_FRESH_TARGET_CPU_CONTROL_PIPELINE'
for item in controls['runs']:
    assert item['returncode'] == 0
for item in controls['files']:
    assert sha(L / item['path']) == item['sha256']
assert sha(L / 'operations/pace_independent_delivery.py') == 'b86bfb0d0273b9cb7b0e0090c0a528a89636abe1ee5fbfceacc7a208972ba97c'
assert read(L / 'FORMAL_JOB_MANIFEST.json') == C.build_jobs(read(L / 'AIME26_DATASET_FREEZE.json'))
assert not (L / 'calibration/calibration_manifest.json').exists()

rows = []
excluded = {'source_manifest.json', 'GPU_PERMIT.json'}
for path in sorted(L.rglob('*')):
    assert not path.is_symlink(), 'A source symlink is not allowed'
    if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
        continue
    name = path.relative_to(L).as_posix()
    if name in excluded:
        continue
    assert not any(x in name for x in ('\r', '\n', '\x00'))
    rows.append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path)))
save(L / 'source_manifest.json', dict(task_id=C.TASK,
     status='STATIC_CODE_INPUTS_METHOD_FROZEN_BEFORE_ANY_GPU_STAGE',
     count=len(rows), files=rows, code_or_input_changes_after_START_FORBIDDEN=True))
permit = dict(task_id=C.TASK, ALLOW_GPU='YES', execution_root=C.REMOTE,
              source_manifest_sha256=sha(L / 'source_manifest.json'),
              method_spec_sha256=sha(L / 'method_spec.json'),
              human_authorization_sha256=sha(L / 'operations/HUMAN_PACE48_AUTHORIZATION.json'),
              human_GPU0_clarification_sha256=sha(L / 'operations/HUMAN_GPU0_ONLY_CLARIFICATION.json'),
              physical_indices=[0], devices=[C.UUID], seed=1,
              formal_new_samples=60, max_new_tokens=81920,
              watchdog_seconds=1800, cleanup_seconds=120,
              GPUh_hard_cap=None, total_wall_cap=None, unit_elapsed_cap=None,
              no_foreign_GPU_signals=True, server29_CPU_archive_only=True,
              stage_order=['A_B_C_actual_CUDA', 'calibration_once_8x2048',
                           'freeze_one_same_head_mask', 'actual_mask_C_and_D',
                           'R2_R3_canonical30_seed1_60', 'once_CPU_finisher'])
save(L / 'GPU_PERMIT.json', permit)
C.validate_permit_values(read(L / 'ROOT.json'), permit,
     read(L / 'operations/HUMAN_PACE48_AUTHORIZATION.json'),
     sha(L / 'source_manifest.json'), sha(L / 'method_spec.json'),
     sha(L / 'operations/HUMAN_PACE48_AUTHORIZATION.json'))
print(json.dumps(dict(status='FROZEN_ONCE_NO_GPU_OR_CHILD_STARTED', count=len(rows),
                     source_sha256=sha(L / 'source_manifest.json'),
                     permit_sha256=sha(L / 'GPU_PERMIT.json'))))
