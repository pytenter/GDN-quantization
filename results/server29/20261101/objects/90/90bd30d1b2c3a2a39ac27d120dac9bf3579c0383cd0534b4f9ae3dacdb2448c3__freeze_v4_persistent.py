"""CPU syntax, unchanged identities and original reducer checks only."""
import ast
import math
import subprocess
from v4_stage1_common import *

def main():
    assert not LATCH.exists() and checked(OUT/'P1_L7_PORTABILITY_GATE.json')['status']=='PASS'
    assert checked(OUT/'HARDWARE_MAPPING_AMENDMENT.json')['status']=='APPROVED_PROSPECTIVE_V4_ONLY'
    prior=checked(OUT/'V4_PORTABILITY_CODE_FREEZE.json')
    for p,h in prior['code_sha256'].items(): assert digest(p)==h
    assert digest(CODE/'persistent_engine_v1.py')==ENGINE_SHA
    files=['v4_persistent_common.py','v4_persistent_worker.py','v4_persistent_controller.py','v4_persistent_stages.py',
        'v4_persistent_pipeline.py','v4_terminal_report.py','freeze_v4_persistent.py']
    for n in files: ast.parse((CODE/n).read_text())
    from persistent_engine_v1 import summarize
    values=[(i%997+1)/100003 for i in range(30720)]
    summary=summarize(values,2048,32768)
    assert summary['mean']==sum(values)/30720 and summary['auc']==sum(values)
    rounding=abs(sum(values)/30720-math.fsum(values)/30720)
    emit(OUT/'V4_PERSISTENT_CPU_CHECK.json',dict(status='PASS_CPU_ONLY',original_engine_sha256=ENGINE_SHA,
        original_V2_worker_runtime_unchanged=True,synthetic_reducer_rounding_difference=rounding,
        GPU_qualification_not_implied=True,original_windows_indices_reducer_unchanged=True))
    commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
    emit(OUT/'V4_PERSISTENT_CODE_FREEZE.json',dict(code_commit=commit,epoch=time.time(),
        code_sha256={**prior['code_sha256'],**{str(CODE/n):digest(CODE/n) for n in files}},
        portability_gate_sha256=digest(OUT/'P1_L7_PORTABILITY_GATE.json'),mapping_amendment_sha256=digest(OUT/'HARDWARE_MAPPING_AMENDMENT.json'),
        original_engine_sha256=ENGINE_SHA,original_V2_numerical_worker_runtime_bytes_unchanged=True,
        uncertainty_rule_prefrozen_before_qualification='10*max(sampled qualified same-logit GPU32/CPU64 arithmetic, FP32 epsilon, synthetic Python sum rounding); empirical not statistical/universal',
        independent_science_outcome_blind_full_roster=True))
    print('V4_PERSISTENT_FREEZE',commit,digest(OUT/'V4_PERSISTENT_CODE_FREEZE.json'))

if __name__=='__main__': main()
