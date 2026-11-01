"""CPU AST/identity only; no repeated scientific scoring or old asset audit."""
import ast
import subprocess
import time
from v4_stage1_common import *

def main():
    assert checked(OUT/'CROSS_DEVICE_FP_REPEATABILITY.json')['status']=='PASS'
    assert not LATCH.exists()
    previous=checked(FREEZE)
    for p,h in previous['code_sha256'].items(): assert digest(p)==h
    files=['v4_common.py','v4_portability_worker.py','v4_portability_controller.py','freeze_v4_portability.py',
        'persistent_engine_v1.py','v2_worker.py','v2_runtime.py','v3_guard.py','run_matched_l7.py','objective.py',
        'run_phase_b_calibration_v2.py','v2_common.py','v4_stage1_common.py']
    for name in files:
        ast.parse((CODE/name).read_text())
    commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
    emit(OUT/'V4_PORTABILITY_CODE_FREEZE.json',dict(code_commit=commit,epoch=time.time(),
        code_sha256={str(CODE/n):digest(CODE/n) for n in files},stage1_freeze_sha256=digest(FREEZE),
        Stage1_PASS_sha256=digest(OUT/'CROSS_DEVICE_FP_REPEATABILITY.json'),
        portability_criteria_predeclared_sha256=digest(OUT/'V4_EXECUTION_PLAN.md'),
        original_selected_pairs_no_reselection=True,CPU_only_AST_identity_check_not_GPU_qualification=True))
    print('V4_PORTABILITY_FREEZE',commit,digest(OUT/'V4_PORTABILITY_CODE_FREEZE.json'))

if __name__=='__main__': main()
