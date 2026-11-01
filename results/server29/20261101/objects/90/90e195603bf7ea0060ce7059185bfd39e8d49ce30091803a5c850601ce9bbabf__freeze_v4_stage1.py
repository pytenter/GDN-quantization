"""CPU premeasurement freeze, reusing completed V3 asset audit without deep rerun."""
import ast
import subprocess
import time
from v4_stage1_common import *


def main():
    assert not FREEZE.exists() and not list(RUNS.glob('*.LAUNCH_AUTHORIZATION.json'))
    source=checked(V3/'SOURCE_AND_FROZEN_SELECTION_IDENTITY.json')
    ids=dict(source)
    ids['historical_STOP_sha256']=dict(source['historical_STOP_sha256'])
    for p in [V3/'V3_FAIL_CLOSED_STOP.json',V3/'V3_CODE_FREEZE.json',
              V3/'V3_TERMINAL_STATUS_CORRECTION.json',V3/'V3_FRESH_PROCESS_REPEAT_FAILURE_ANALYSIS.json']:
        checked(p); ids['historical_STOP_sha256'][str(p)]=digest(p)
    ids.update(reused_V3_asset_audit=True,no_historical_deep_audit_repeated=True,
        V3_identity_sha256=digest(V3/'SOURCE_AND_FROZEN_SELECTION_IDENTITY.json'),V4_epoch=time.time())
    for path,row in ids['assets'].items():
        p=Path(path); s=p.stat(); assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns']
        if p.suffix!='.safetensors': assert digest(p)==row['sha256']
    for path,sha in ids['historical_STOP_sha256'].items(): assert digest(path)==sha
    emit(OUT/'V4_REUSED_SOURCE_IDENTITY.json',ids)
    assert digest(CODE/'persistent_engine_v1.py')==ENGINE_SHA
    base=Path('/data/zypan/worktrees/ling-kda-p1-l7-fast-recovery-and-formal-32k-v3')
    unchanged=['persistent_engine_v1.py','v2_worker.py','v2_runtime.py','v3_guard.py']
    for name in unchanged: assert digest(CODE/name)==digest(base/name)
    files=[str(CODE/x) for x in unchanged+['v4_stage1_common.py','v4_fp512_worker.py','v4_fp512_controller.py','freeze_v4_stage1.py','V4_EXECUTION_PLAN.md']]
    for p in files:
        if p.endswith('.py'): ast.parse(Path(p).read_text())
    commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
    emit(FREEZE,dict(status='FROZEN_STAGE1_ONLY',code_commit=commit,
        base_V3_commit='31646483521ce6732be80b7cd64675b3a7b0663a',code_sha256={p:digest(p) for p in files},
        engine_sha256=ENGINE_SHA,source_identity_sha256=digest(OUT/'V4_REUSED_SOURCE_IDENTITY.json'),
        objective_or_quantizer_or_model_changed=False,actual_FP_token_geometry=512,fresh_processes=2,
        exact_repeat_tolerance_changed=False,student_calls_omitted_only=True,
        raw_FP_logits_archived_for_direct_difference=True,
        CPU_syntax_PASS_not_GPU_qualification=True,resource_caps_none=True,epoch=time.time()))
    with (OUT/'V4_EXECUTION_PLAN.md').open('xb') as f: f.write((CODE/'V4_EXECUTION_PLAN.md').read_bytes())
    print('V4_STAGE1_FROZEN',commit,digest(FREEZE))


if __name__=='__main__': main()
