"""Formal launcher -> unchanged lane expression -> real supervisor -> initializer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from execution04_launcher import W,worker_environment,actual_lane_environment,selected_environment,ENV_KEYS
from recovery_state import sha,atomic_json

def case(root,stage,repeat,missing=None):
    path=root/(stage+'_'+str(repeat)+('_negative_'+missing if missing else ''));path.mkdir();base=path/'probe'
    env=worker_environment(os.environ,0,'CPU_ONLY',2);env.update(CUDA_VISIBLE_DEVICES='',LING_PROSPECTIVE_RESULTS=str(path))
    if missing=='seed':env.pop('PYTHONHASHSEED',None)
    if missing=='profile_mode':env.pop('FLA_CACHE_MODE',None)
    if missing=='profile_dir':env.pop('FLA_CONFIG_DIR',None)
    lane=actual_lane_environment(env,'CPU_ONLY','')
    expected=path/'EXPECTED_ENV.json';atomic_json(expected,selected_environment(lane))
    ident=dict(document='CPU_FIXTURE_ONLY',rotation='CPU_FIXTURE_ONLY',stage=stage,target_gpu_uuid='CPU_ONLY',execution=str(path))
    command=[sys.executable,'-B',str(W/'runtime_launch_probe.py'),'--mode','cpu','--stage',stage,'--result',str(base)+'.json','--expected',str(expected)]
    intent=path/'intent.json';atomic_json(intent,dict(command=command,base=str(base),unit_identity=ident,source_commit='CPU_FIXTURE_ONLY_NOT_GPU_SCIENCE',tag='probe',stage=stage))
    r=subprocess.run([sys.executable,'-B',str(W/'execution04_launcher.py'),'--cpu-hop','master','--intent',str(intent)],env=env,capture_output=True,text=True,timeout=90)
    (path/'chain.stdout.txt').write_text(r.stdout);(path/'chain.stderr.txt').write_text(r.stderr)
    master=json.loads(Path(str(base)+'.MASTER_ENV.json').read_text());observed_lane=json.loads(Path(str(base)+'.LANE_ENV.json').read_text());observed=json.loads(base.with_suffix('.OBSERVED_ENV.json').read_text())
    assert master['environment']==selected_environment(env)
    assert observed_lane['environment']==selected_environment(lane)==observed['environment']
    assert observed['environment']==json.loads(expected.read_text())
    assert all(x['environment']['CUDA_VISIBLE_DEVICES']=='' for x in [master,observed_lane,observed])
    if missing:
        assert r.returncode!=0 and Path(str(base)+'.FAILED_EXIT.json').exists()
        failure=json.loads(base.with_suffix('.PROBE_FAILURE.json').read_text())
        if missing=='seed':
            assert 'PYTHONHASHSEED' in failure['traceback']
            assert all(x['environment']['PYTHONHASHSEED'] is None and x['hash_randomization']==1 for x in [master,observed_lane,observed])
        else:
            assert 'FROZEN_FLA_PROFILE_ENV_INVALID' in failure['traceback']
            assert all(x['environment']['PYTHONHASHSEED']=='0' and x['hash_randomization']==0 for x in [master,observed_lane,observed])
        result=None
    else:
        assert r.returncode==0,(r.stdout,r.stderr)
        result=json.loads(Path(str(base)+'.json').read_text());receipt=json.loads(Path(str(base)+'.NATURAL_EXIT.json').read_text())
        assert receipt['natural_exit'] and receipt['result_sha256']==sha(Path(str(base)+'.json'))
        assert result['runtime'][0]['cuda_initialized'] is False and result['science_trajectory_count']==0
        assert all(x['environment']['PYTHONHASHSEED']=='0' and x['hash_randomization']==0 for x in [master,observed_lane,observed])
        assert len({x['hash_probe'] for x in [master,observed_lane,observed]})==1
        assert result['frozen_FLA_profile']['status']=='PASS_FROZEN_RUNTIME_PROFILE_INPUTS'
        assert result['actual_environment']['FLA_CACHE_MODE']=='default'
    return dict(stage=stage,repeat=repeat,missing_seed_negative=missing=='seed',negative_kind=missing,status='PASS_EXPECTED_REJECTION' if missing else 'PASS_CPU_ONLY',returncode=r.returncode,master=master,lane=observed_lane,worker=observed,worker_runtime=result['runtime'] if result else None,worker_command=command,supervisor_source_sha256=sha(W/'unit_runner.py'),source_environment_builder_sha256=sha(W/'execution04_launcher.py'))

if __name__=='__main__':
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);rows=[]
        for stage in ['parent','resumed_parity']:
            rows.extend(case(root,stage,r) for r in [0,1])
            for missing in ['seed','profile_mode','profile_dir']:rows.append(case(root,stage,0,missing=missing))
        hashes=[r['worker']['hash_probe'] for r in rows if r['negative_kind'] is None];assert len(set(hashes))==1
        archive=output/'CPU_TRUE_LAUNCH_CHAIN_EVIDENCE.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            for p in root.rglob('*'):
                if p.is_file():t.add(p,arcname=str(p.relative_to(root)),recursive=False)
    atomic_json(output/'CPU_LAUNCH_CHAIN_REGRESSION.json',dict(status='PASS_CPU_TRUE_LAUNCH_CHAIN',cases=rows,case_count=len(rows),parent_and_fresh64_paths=True,actual_formal_environment_builder=True,unchanged_master_lane_expression=True,frozen_initializer_called=True,interpreter_start_hashseed_verified=True,all_positive_fresh_interpreters_hash_equal=True,missing_seed_negative_cases=2,missing_FLA_profile_negative_cases=4,CUDA_exposed=False,model_loads=0,GPU_scientific_units=0,source_sha256={n:sha(W/n) for n in ['execution04_launcher.py','runtime_launch_probe.py','test_true_launch_chain.py','unit_runner.py','qualified_completion.py','v2_runtime.py','frozen_runtime_profile.py','runtime_worker_entry.py']},evidence_archive_sha256=sha(archive),elapsed_CPU_wall_seconds=time.time()-started))
    print('PASS_CPU_TRUE_LAUNCH_CHAIN_4_POSITIVES_6_NEGATIVES_NO_MODEL_NO_CUDA')
