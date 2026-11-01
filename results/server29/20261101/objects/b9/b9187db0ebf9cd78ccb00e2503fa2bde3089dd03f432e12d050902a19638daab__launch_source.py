"""V2-A only; never freeze/commit/advance on failure and never Discovery."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from capsule_v2 import sha, write

A=Path(__file__).resolve().parent;E=A.parents[1];ROOT=E.parents[1]
PY='/data/zypan/envs/ling-kda/bin/python'
T=Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/transport.py')


def main():
    assert Path.cwd()==ROOT,'EXECUTION_CWD_CHANGED'
    freeze=json.loads((A/'architecture_freeze.json').read_text())
    assert sha(A/'AMENDMENT_V2.md')==freeze['AMENDMENT_SHA256']
    assert all(freeze['architecture_tests']['V2_TEST_'+str(i)].startswith('PASS') for i in range(1,11))
    for row in freeze['source_files']:assert sha(A/row['name'])==row['sha256']
    s=importlib.util.spec_from_file_location('frozen_transport_setup',T);t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
    runner=E/'provenance/executed_dependency_amendment_v1/capsule_runner_v2.py'
    assert sha(runner)==sha(A/'capsule_runner_v2.py')
    env=os.environ.copy()
    profile=json.loads((E/'runtime_manifest.json').read_text())['expected']['launch_profile']
    for k,value in profile.items():
        if value is None:env.pop(k,None)
        else:env[k]=str(t.mapped(value)) if isinstance(value,str) and value.startswith('/data01/user2/') else str(value)
    env.update(CUDA_VISIBLE_DEVICES='6',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',GIT_OPTIONAL_LOCKS='0')
    run=A/'V2_A';run.mkdir(exist_ok=False)
    # Roots are fixed before observation; nothing new is auto-classified as
    # derived. The inherited source/asset graph remains the existing authority.
    roots=[dict(path=str(ROOT),classification='FROZEN_EXPERIMENT_SOURCE_AND_INPUT'),
        dict(path='/data/zypan/envs/ling-kda',classification='EXISTING_PYTHON_NUMERICAL_RUNTIME'),
        dict(path='/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1',classification='FROZEN_INHERITED_PARENT_ASSET'),
        dict(path='/data/zypan/worktrees/ling-error-coherence-diagnosis-v1',classification='FROZEN_PARENT_HARNESS'),
        dict(path='/data/zypan/models/Ling-3.0-tiny',classification='FROZEN_OFFICIAL_MODEL_ASSET'),
        dict(path='/data/zypan/.cache/huggingface/modules',classification='FILE_BACKED_CACHED_MODEL_CODE'),
        dict(path='/data/zypan/.triton',classification='EXISTING_JIT_FILE_BACKED_RUNTIME_ARTIFACT',volatile=True),
        dict(path='/usr/lib',classification='SYSTEM_NATIVE_RUNTIME'),
        dict(path='/usr/bin',classification='SYSTEM_EXECUTABLE')]
    exact=[dict(path='/etc/ld.so.cache',classification='DYNAMIC_LOADER_CONFIGURATION')]
    policy=dict(roots=roots,exact=exact,write_roots=[str(A),'/tmp','/data/zypan/.triton'],
        advisory_exact=['/proc/cpuinfo','/proc/meminfo','/proc/stat','/proc/loadavg',
            '/etc/passwd','/etc/group','/etc/nsswitch.conf'],
        advisory_prefixes=['/usr/share/zoneinfo/','/usr/lib/locale/',
            '/proc/driver/nvidia/','/sys/devices/','/sys/class/',
            '/data/zypan/worktrees/ling-observability-provenance-amendment-v1/authority.git/'],
        inherited_scientific_manifest=str(E/'source_identity.json'),
        science_semantics='UNCHANGED',derived_file_backed_code_requires_content_binding=True)
    write(A/'policy_v2.json',policy)
    write(A/'launch_identity.json',dict(python=PY,cwd=str(ROOT),runner=str(runner),
        numerical_environment={k:env.get(k) for k in set(profile)|{'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT'}},
        policy_sha256=sha(A/'policy_v2.json'),architecture_sha256=freeze['AMENDMENT_SHA256']))
    for name,args in [('verify',['verify']),('smoke0',['smoke','--repeat','0']),('smoke1',['smoke','--repeat','1']),('parity',['parity'])]:
        cmd=[PY,str(A/'syscall_supervisor_v2.py'),'--policy',str(A/'policy_v2.json'),
            '--output',str(run/name),'--cwd',str(ROOT),'--',PY,str(runner),*args,'--run-root',str(run/'work')]
        with (run/(name+'.log')).open('xb') as f:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        report=json.loads((run/name/'runtime_escape_report.json').read_text())
        print('V2_A_STAGE',name,report['gate'],report['error'],flush=True)
        if result.returncode or report['gate']!='PASS':
            write(run/'STOP.json',dict(stage=name,first_blocker=report['error'],FINAL_ACTION='FAIL_AND_STOP',
                FORMAL_MEASUREMENTS=0,automatic_retry=False,manifest_frozen=False))
            return 1
    write(run/'A_COMPLETE.json',dict(gate='PASS',formal_measurements=0,stage_receipts=[str(run/n/'runtime_escape_report.json') for n in ('verify','smoke0','smoke1','parity')]))
    print('V2_A_COMPLETE_NO_AUTO_B_NO_COMMIT_NO_DISCOVERY',flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
