#!/usr/bin/env python3
"""One authorized PACE engineering unit or one original-protocol trajectory."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pace_control as C


def run(root, stage, unit):
    root = Path(root).resolve()
    permit = C.authorize(root)
    target = C.unit_target(root, stage, unit)
    dispatch = C.read(target / 'DISPATCH.json')
    identity = C.process_identity(os.getpid())
    C.require(dispatch['task_id'] == C.TASK and dispatch['stage'] == stage and dispatch['unit'] == unit and
              dispatch['source_manifest_sha256'] == C.sha(root / C.SOURCE) and
              dispatch['permit_sha256'] == C.sha(root / 'GPU_PERMIT.json') and
              dispatch['controller']['pid'] == os.getppid() and C.same_process(dispatch['controller']),
              'Exact live controller dispatch is required')
    C.require(os.environ.get('CUDA_VISIBLE_DEVICES') == C.UUID, 'Exact GPU0 UUID visibility required')
    C.atomic_new(target / 'STARTED.json', dict(task_id=C.TASK, identity=identity, stage=stage,
                                              unit=unit, unix=time.time(), dispatch_sha256=C.sha(target / 'DISPATCH.json')))
    loaded_guard = []
    def progress(phase, **values):
        C.progress(target, phase, **values)
        if phase in ('MODEL_LOADED', 'NATIVE_REFERENCE_READY'):
            proof = C.library_check(root)
            if phase == 'MODEL_LOADED':
                proof['NVML'] = C.device(os.getpid())
            loaded_guard.append(proof)
            C.atomic_new(target / ('LOADED_LIBRARY_' + str(len(loaded_guard)) + '.json'), proof)
        print(C.canonical(dict(phase=phase, **values)).decode(), flush=True)
    try:
        progress('WORKER_SOURCE_AND_NATIVE_ADMISSION')
        C.model_guard(root)
        external = C.verify_external(root, full_model=False)
        C.atomic_new(target / 'DRIVER_IDENTITY.json', C.driver_identity(root))
        science = C.verify_science(root) if stage in ('gate_d', 'formal') else None
        import pace_runtime as R
        if stage == 'math_gpu_gates':
            result = R.run_math_device_gates(root, progress=progress)
            C.require(result['status'] == 'PASS' and result['actual_cuda'] is True and
                      result['GPU_qualification_credit'] is True and set(result['gates']) == {'A', 'B', 'C'} and
                      all(x['status'] == 'PASS' and x['actual_cuda'] is True for x in result['gates'].values()),
                      'Actual device A/B/C did not all pass')
        elif stage == 'calibration':
            result = R.run_calibration(root, progress=progress)
            C.verify_head_mask(root)
        elif stage == 'gate_d':
            result = R.run_gate(root, 'gate_d', dict(prompt_tokens=32, max_new_tokens=32), progress=progress)
            C.require(result['status'] == 'PASS' and result['actual_cuda'] is True and
                      set(result['configs']) == {'R2', 'R3'} and result['numerical_81K_claim'] is False,
                      'Actual paired full-model smoke failed')
            actual_mask = result['actual_mask_gate']
            C.require(actual_mask['status'] == 'PASS' and actual_mask['actual_cuda'] is True and
                      actual_mask['head_mask_sha256'] == C.sha(root / 'calibration/head_mask.json') and
                      actual_mask['layers'] == 24 and len(actual_mask['cases']) == 48 and
                      all(x['status'] == 'PASS' for x in actual_mask['cases']),
                      'Post-calibration actual24mask/paired-period device GateC failed')
            for condition, row in result['configs'].items():
                C.require(row['status'] == 'COMPLETE' and row['nonfinite'] is False and
                          row['runtime_error'] is None and row['condition'] == condition and
                          row['head_mask_sha256'] == C.sha(root / 'calibration/head_mask.json') and
                          row['quantizer_audit']['dense_recurrent_carrier'] is False,
                          'Actual-mask full-model smoke identity/numeric failure')
            C.atomic_new(root / 'validation/full_model_smoke.json', result)
        else:
            job = next(x for x in C.validate_jobs(root) if x['id'] == unit)
            result = R.run_formal(root, job, progress=progress)
            C.validate_formal(result, job, root)
        C.require(loaded_guard, 'Actual native/model load boundary was never observed')
        maps = C.library_check(root)
        C.atomic_new(target / 'POSTFLIGHT_LIBRARY.json', maps)
        nvml = C.device(os.getpid())
        C.model_guard(root); C.verify_sources(root)
        if science is not None:
            C.require(C.verify_science(root) == science, 'Science freeze changed during unit')
        value = dict(status='PASS', task_id=C.TASK, stage=stage, unit=unit, identity=identity,
                     gpu=0, gpu_uuid=C.UUID, source_manifest_sha256=C.sha(root / C.SOURCE),
                     permit_sha256=C.sha(root / 'GPU_PERMIT.json'), science_freeze_sha256=science,
                     external_admission=external, postflight_NVML=nvml, result=result)
        value['payload_sha256'] = C.objsha(value)
        C.atomic_new(target / 'RESULT.json', value)
        progress('UNIT_RESULT_SAVED', result_sha256=C.sha(target / 'RESULT.json'))
        return 0
    except BaseException as exc:
        failure = dict(status='FAIL_CLOSED_STOP', task_id=C.TASK, stage=stage, unit=unit,
                       identity=identity, error=type(exc).__name__ + ':' + str(exc),
                       traceback=traceback.format_exc(), unix=time.time(),
                       no_automatic_retry=True, formal_completion_credit=0)
        if not (target / 'FIRST_FAILURE.json').exists():
            C.atomic_new(target / 'FIRST_FAILURE.json', failure)
        print(C.canonical(failure).decode(), flush=True)
        return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--stage', required=True, choices=C.STAGES)
    p.add_argument('--unit', required=True)
    a = p.parse_args()
    return run(a.root, a.stage, a.unit)

if __name__ == '__main__':
    raise SystemExit(main())
