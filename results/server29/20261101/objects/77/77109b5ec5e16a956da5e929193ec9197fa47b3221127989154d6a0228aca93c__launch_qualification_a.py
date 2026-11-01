"""One V2-A engineering qualification only; unknown ingress STOP, no retries."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from capsule_v2 import sha, write

C=Path(__file__).resolve().parent; A=C.parent; E=A.parents[1]; ROOT=E.parents[1]
PY='/data/zypan/envs/ling-kda/bin/python'
TRANSPORT=Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/transport.py')


def main():
    assert Path.cwd()==ROOT
    freeze=json.loads((C/'candidate_architecture_freeze.json').read_text())
    for row in freeze['source_files']:
        assert sha(C/row['name'])==row['sha256'],row['name']
    assert all(freeze['original_10_tests']['V2_TEST_'+str(i)].startswith('PASS') for i in range(1,11))
    assert freeze['semantic_tests']['gate']=='PASS'
    assert sha(C/'test_v2.py')==sha(A/'test_v2.py')
    for row in json.loads((C/'prior_v2_protection.json').read_text())['files']:
        assert sha(A/row['path'])==row['sha256'],row['path']
    assert sha(TRANSPORT)=='7ca78b3786fe10d4e55a395560dcb452c72e494bdd49472e51d2affcd937f424'
    s=importlib.util.spec_from_file_location('frozen_qualification_transport',TRANSPORT)
    t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
    runner=E/'provenance/executed_dependency_amendment_v1/capsule_runner_semantic.py'
    assert sha(runner)==sha(C/'capsule_runner_semantic.py')
    env=os.environ.copy()
    profile=json.loads((E/'runtime_manifest.json').read_text())['expected']['launch_profile']
    for k,value in profile.items():
        if value is None:env.pop(k,None)
        else:env[k]=str(t.mapped(value)) if isinstance(value,str) and value.startswith('/data01/user2/') else str(value)
    env.update(CUDA_VISIBLE_DEVICES='6',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',GIT_OPTIONAL_LOCKS='0')
    # Migrate existing predeclared authority metadata from silent advisory
    # bypass to B source-identity closure. No observed unknown path is added.
    policy=json.loads((A/'policy_v2.json').read_text())
    authority='/data/zypan/worktrees/ling-observability-provenance-amendment-v1/authority.git'
    assert authority+'/' in policy['advisory_prefixes']
    policy['roots']=policy['roots']+[dict(path=authority,classification='FROZEN_SOURCE_AUTHORITY_METADATA')]
    policy['advisory_prefixes']=[]; policy['advisory_exact']=[]
    policy['write_roots']=[str(C),'/tmp','/data/zypan/.triton']
    policy['semantic_classifier']='A_B_C_D_CONTENT_CONSUMER_USAGE'
    policy['synthetic_ambient_consumers_enabled']=False
    assert not policy.get('synthetic_ambient_test_only'), 'SYNTHETIC_CLASSIFIER_FORBIDDEN_IN_QUALIFICATION'
    write(C/'qualification_policy.json',policy)
    write(C/'launch_identity.json',dict(python=PY,cwd=str(ROOT),runner=str(runner),
        scientific_environment={k:env.get(k) for k in set(profile)|{'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT'}},
        locale_environment={k:env.get(k) for k in ('LANG','LC_ALL','LC_CTYPE','LC_NUMERIC','LC_COLLATE')},
        original_launch_identity_sha256=sha(A/'launch_identity.json'),
        policy_sha256=sha(C/'qualification_policy.json'),
        numerical_configuration_changed=False,GPU6_FORMAL_AUTHORIZED='NO'))
    run=C/'V2_A';run.mkdir(exist_ok=False)
    for name,args in [('verify',['verify']),('smoke0',['smoke','--repeat','0']),('smoke1',['smoke','--repeat','1']),('parity',['parity'])]:
        cmd=[PY,str(C/'syscall_supervisor_v2.py'),'--policy',str(C/'qualification_policy.json'),
            '--output',str(run/name),'--cwd',str(ROOT),'--audit-channel','--',PY,str(runner),
            *args,'--run-root',str(run/'work')]
        with (run/(name+'.log')).open('xb') as f:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        report=json.loads((run/name/'runtime_escape_report.json').read_text())
        print('V2_A_STAGE',name,report['gate'],report['error'],flush=True)
        if result.returncode or report['gate']!='PASS':
            write(run/'STOP.json',dict(stage=name,first_blocker=report['error'],FINAL_ACTION='FAIL_AND_STOP',
                FORMAL_MEASUREMENTS=0,automatic_retry=False,manifest_frozen=False,
                original_V2_FAIL_STOP_preserved=True,GPU6_FORMAL_AUTHORIZED='NO'))
            return 1
    write(run/'A_COMPLETE.json',dict(gate='PASS',formal_measurements=0,
        stage_receipts=[str(run/n/'runtime_escape_report.json') for n in ('verify','smoke0','smoke1','parity')]))
    print('V2_A_COMPLETE_NO_AUTOMATIC_DISCOVERY',flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
