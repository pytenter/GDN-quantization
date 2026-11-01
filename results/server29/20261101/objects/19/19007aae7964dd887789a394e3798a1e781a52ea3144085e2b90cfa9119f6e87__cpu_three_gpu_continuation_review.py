"""Finite CPU review of actual production entrypoints and strict origin admission."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'src'))
import execution_control as C
import continuation_admission as A
from qualification_input_admission import validate as input_admission
from cross_host_contract import validate_global_layout

checks = []
layout = validate_global_layout(R)
assert layout['qualification'] == 57 and layout['formal'] == 120
checks.append('ACTUAL_THREE_GPU57_UNIQUE120_LAYOUT')
inputs = input_admission(R)
assert inputs['qualification_units'] == 19 * len(C.UUIDS)
checks.append('ACTUAL_FULL_REPLAY_BOUNDS_ALL_HOST_DEVICES')
reuse = A.admit_local_reuse(R)
assert len(reuse['qualification_ids']) == 19
assert len(reuse['formal_ids']) == (1 if C.NODE['host'] == '202.38.247.29' else 0)
checks.append('ACTUAL_ORIGIN19_AND_COMPLETED_FORMAL_FRESH_SOURCE_IDENTITY_REVIEW')

with tempfile.TemporaryDirectory(prefix='g64_three_gpu_negative_cpu_') as tmp:
    F = Path(tmp)
    for n in ['CROSS_HOST_PLAN.json', 'GLOBAL_FORMAL_JOB_MANIFEST.json', 'QUESTION_GPU_ASSIGNMENT.json', 'AIME26_DATASET_FREEZE.json', 'REUSE_ADMISSION_STATIC.json', 'GPU_EXECUTION_PLAN.json']:
        (F / n).write_bytes((R / n).read_bytes())
    jobs = C.read(F / 'GLOBAL_FORMAL_JOB_MANIFEST.json')
    for kind in ['duplicate', 'wrong_uuid', 'wrong_prompt', 'wrong_seed', 'wrong_cap', 'cross_question_bundle']:
        value = copy.deepcopy(jobs)
        if kind == 'duplicate': value['jobs'][1] = copy.deepcopy(value['jobs'][0])
        if kind == 'wrong_uuid': value['jobs'][0]['gpu_uuid'] = 'GPU-foreign'
        if kind == 'wrong_prompt': value['jobs'][0]['input_ids_sha256'] = '0' * 64
        if kind == 'wrong_seed': value['jobs'][0]['seed'] = 2
        if kind == 'wrong_cap': value['jobs'][0]['max_new_tokens'] = 81921
        if kind == 'cross_question_bundle': value['jobs'][0]['node_id'] = 'server29_gpu5'
        (F / 'GLOBAL_FORMAL_JOB_MANIFEST.json').write_text(json.dumps(value))
        try: validate_global_layout(F)
        except (RuntimeError, KeyError): checks.append('REJECT_LAYOUT_' + kind)
        else: raise AssertionError('Invalid layout admitted: ' + kind)
    (F / 'GLOBAL_FORMAL_JOB_MANIFEST.json').write_text(json.dumps(jobs))
    original = C.read(F / 'REUSE_ADMISSION_STATIC.json')
    selected = reuse['qualification_ids'][0]
    for kind in ['wrong_authorization', 'wrong_origin_root', 'wrong_source', 'wrong_result_sha', 'wrong_interval_sha']:
        value = copy.deepcopy(original)
        if kind == 'wrong_authorization': value['human_authorization_sha256'] = '0' * 64
        if kind == 'wrong_origin_root': value['origins'][C.NODE['id']]['root'] += '_foreign'
        if kind == 'wrong_source': value['origins'][C.NODE['id']]['source_sha256'] = '0' * 64
        if kind == 'wrong_result_sha': value['qualification'][selected]['result_sha256'] = '0' * 64
        if kind == 'wrong_interval_sha': value['qualification'][selected]['interval_sha256'] = '0' * 64
        (F / 'REUSE_ADMISSION_STATIC.json').write_text(json.dumps(value))
        try: A.qualification_source(F, selected)
        except RuntimeError: checks.append('REJECT_REUSE_' + kind)
        else: raise AssertionError('Invalid reused origin admitted: ' + kind)
    (F / 'REUSE_ADMISSION_STATIC.json').write_text(json.dumps(original))
    try: A.qualification_source(F, 'unknown_or_duplicate_unit')
    except RuntimeError: checks.append('REJECT_UNKNOWN_QUALIFICATION')
    else: raise AssertionError('Unknown qualification admitted')
    for formal in A.host_reused_formal(R).values():
        for field, value in [('gpu_uuid', 'GPU-foreign'), ('seed', 2), ('input_tokens', 999)]:
            job = dict(formal['job'], **{field: value})
            try: A.formal_source(R, job)
            except RuntimeError: checks.append('REJECT_FORMAL_REUSE_' + field)
            else: raise AssertionError('Moved/modified old formal admitted')

def run_harness(name, source):
    program = 'from pathlib import Path\n__file__=' + repr(str(R / 'scripts' / name)) + '\n' + source
    p = subprocess.run([sys.executable, '-B', '-c', program], capture_output=True, text=True, env=os.environ, timeout=300)
    if p.returncode:
        raise RuntimeError(name + ': ' + p.stdout[-3000:] + p.stderr[-8000:])
    value = json.loads(p.stdout.splitlines()[-1])
    checks.append(dict(actual_entrypoint_cpu_harness=name, result=value))

worker = (R / 'scripts/cpu_worker_entrypoint_review.py').read_text()
assert worker.count("fake.UUIDS={real.NODE['physical_gpu']:''}") == 1
worker = worker.replace("fake.UUIDS={real.NODE['physical_gpu']:''}", "fake.UUIDS={g:'' for g in real.UUIDS}")
worker = worker.replace('assert len(checks)==79 and len(set(checks))==79', 'assert len(checks)==len(plan)+len(jobs) and len(set(checks))==len(plan)+len(jobs)')
run_harness('cpu_worker_entrypoint_review.py', worker)

coordinator = (R / 'scripts/cpu_global_coordinator_review.py').read_text()
coordinator = coordinator.replace('GPU_COMPLETE_GROUP64_HOST19_FORMAL60', 'GPU_COMPLETE_GROUP64_NODE_ALL_PLANNED')
coordinator = coordinator.replace('PASS_BOTH_HOST19_TRUE_QUALIFICATIONS_ALLOW_FORMAL120_ONCE', 'PASS_THREE_GPU57_MIXED_ORIGIN_QUALIFICATIONS_ALLOW_UNIQUE120')
needle = "gpu_uuid=n['uuid'],qualification_count=19,"
assert coordinator.count(needle) == 1
coordinator = coordinator.replace(needle, "gpu_uuid=n['uuid'],gpu_uuids=[x['uuid']for x in C.read(F/'CROSS_HOST_PLAN.json')['nodes']if x['host']==n['host']],qualification_count=(19 if node=='server55_gpu0'else 38),reused_qualification_count=19,fresh_qualification_count=(0 if node=='server55_gpu0'else 19),")
run_harness('cpu_global_coordinator_review.py', coordinator)

print(json.dumps(dict(status='PASS_THREE_GPU_ACTUAL_CONTROL_AND_STRICT_REUSE_CPU',
                     count=len(checks), checks=checks, source_sha256=C.sha(R / 'src/continuation_admission.py'),
                     GPU_calls=0, model_forwards=0, native_driver_and_worker_harness_devices='EXPLICIT_CPU_MOCKS',
                     original_reuse_review='ACTUAL_FRESH_FILES_AND_PROCESS_EXIT_READ_ONLY',
                     old_payloads_not_scored=True, new_GPU_qualification_credit=0), ensure_ascii=False))
