import subprocess
import time
from v3_common import *

def main():
    assert checked(OUT/'CPU_REGRESSION_RESULTS.json')['status']=='PASS'
    assert not list(RUNS.glob('*.LAUNCH_AUTHORIZATION.json'))
    changes=subprocess.check_output(['git','-C',str(CODE),'diff','--name-only','4748c03a3ebd427a9dce7e6e0c6b120c72af4ad5'],text=True).splitlines()
    assert all(x.startswith(('v3_','prepare_v3','test_v3','freeze_v3','V3_EXECUTION')) for x in changes)
    commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
    emit(FREEZE,dict(status='FROZEN',code_commit=commit,base_commit='4748c03a3ebd427a9dce7e6e0c6b120c72af4ad5',
        code_sha256={str(p):digest(p) for p in sorted(CODE.glob('*.py'))},changed_files=changes,
        original_engine_sha256=ENGINE_SHA,original_worker_runtime_byte_identical=True,
        science_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
        source_identity_sha256=digest(OUT/'SOURCE_AND_FROZEN_SELECTION_IDENTITY.json'),
        plan_sha256=digest(CODE/'V3_EXECUTION_PLAN.md'),epoch=time.time(),GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None,
        qualification_order=[6,3],fresh_repeats=2,no_retry=True,guardprobe_model_loading=False,
        nearzero_rule='Before formal: 10*max(qualified same-logit CPU64/GPU32 sampled arithmetic difference, FP32 epsilon, original reducer rounding difference); empirical, not statistical significance',
        micro_and_formal_require_all_prerequisite_gates=True))
    print('V3_CODE_FROZEN',commit,digest(FREEZE))

if __name__=='__main__': main()
