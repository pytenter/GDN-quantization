"""CPU regression of runtime/init ordering, immutable engine, isolation and guards."""
import ast
import json
import os
from pathlib import Path
from v2_common import *

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    from v2_runtime import initialize,snapshot
    assert not torch.cuda.is_initialized()
    torch.use_deterministic_algorithms(False)
    init=initialize(torch,0)
    assert init['deterministic_algorithms_enabled'] and not init['warn_only']
    snapshot(torch,'CPU_after_init',0)
    torch.use_deterministic_algorithms(False)
    detected=False
    try: snapshot(torch,'CPU_negative_reset_test',0)
    except AssertionError: detected=True
    assert detected
    initialize(torch,0)
    engine_sha=digest(CODE/'persistent_engine_v1.py')
    assert engine_sha==digest(OLD/'persistent_engine_v1.py')=='d448121735fd5ed7eff3a62689d359f17eca4b5467972ff69d299073a2b865c8'
    for name in ('objective.py','run_matched_l7.py','run_phase_b_calibration_v2.py','phase_b_calibration_common_v2.py'):
        assert digest(CODE/name)==digest(OLD/name),name
    src=(CODE/'v2_worker.py').read_text(); tree=ast.parse(src)
    assert src.index('initialize(torch,cfg')<src.index('direction.gpu_guard(torch')<src.index('capture_module.Capture()')<src.index('result=stream(')
    assert 'assert torch.are_deterministic_algorithms_enabled()' in src
    assert 'persistent_stage_common_v1' not in src
    runtime=(CODE/'v2_runtime.py').read_text()
    assert 'torch.use_deterministic_algorithms(True,warn_only=False)' in runtime
    assert 'warn_only=True' not in runtime
    for name in ('v2_common.py','v2_worker.py','v2_controller.py'):
        code=(CODE/name).read_text()
        assert 'kill(' not in code and 'SIGTERM' not in code and 'optimizer.step' not in code
        assert '<= 16' not in code and '<=16' not in code and '<= .85' not in code and '<=.85' not in code
        compile(code,name,'exec')
    # Reuse original synthetic mathematical tests verbatim, redirect only their
    # new publication namespace. The original test/engine bytes remain unchanged.
    import run_matched_l7
    run_matched_l7.ROOT=OUT
    numerical=load_module(CODE/'test_persistent_engine_cpu_v1.py','v2_original_numerical_CPU_tests')
    numerical.main()
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS',gpu_qualification='NOT_RUN',gpu_work=False,
        checks=['strict_child_initialization_order','global_flag_true_warn_only_false','reset_detected_not_silently_reenabled',
        'CUBLAS_set_before_CUDA_init','TF32_disabled','immutable_original_engine_sha',
        'original_input_loss_helpers_byte_identical','original_CPU_engine_tests_reused',
        'full_vocabulary_KL_arithmetic_and_windows_unchanged','true_INT8_inverse_writeback_synthetic',
        'consumption_and_cache_corruption_rejected','no_administrative_16h_85pct_gate',
        'new_result_namespace_only','fresh_process_roster_not_retry'],
        runtime_CPU_evidence=init,engine_sha256=engine_sha,
        old_stop_preserved=digest(OLD/'PERSISTENT_FAIL_CLOSED_STOP.json')=='3abfb452c8c6c8232e4cd683d66582006e8455c6603bc38bbda35c27f78f3a38')
    assert result['old_stop_preserved']
    emit(OUT/'CPU_REGRESSION_TEST_RESULTS.json',result)
    print(json.dumps(dict(status='V2_CPU_REGRESSION_PASS_REAL_GPU_PENDING')))

if __name__=='__main__': main()
