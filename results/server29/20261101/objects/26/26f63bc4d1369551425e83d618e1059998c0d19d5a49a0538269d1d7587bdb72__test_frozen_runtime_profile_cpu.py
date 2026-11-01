"""Fail-closed asset/env and envelope checks; CPU stubs are not GPU evidence."""
import contextlib
import json
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
import frozen_runtime_profile as profile
import runtime_worker_entry as entry

def rejected(fn,label):
    try:fn()
    except AssertionError:return label
    raise AssertionError(('EXPECTED_REJECTION_MISSING',label))

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert 'torch' not in sys.modules
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
             LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',**profile.profile_environment())
    assert profile.validate_environment(env)['status']=='PASS_FROZEN_RUNTIME_PROFILE_INPUTS'
    checks=['six_original_config_SHA_positive']
    for key,value in [('FLA_CACHE_MODE',None),('FLA_CACHE_MODE','disabled'),('FLA_CONFIG_DIR',None),('FLA_CONFIG_DIR','/nonexistent')]:
        bad=dict(env)
        if value is None:bad.pop(key)
        else:bad[key]=value
        checks.append(rejected(lambda:profile.validate_environment(bad),key+'_'+str(value)))
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);configs=root/'configs';shutil.copytree(profile.CONFIG_DIR,configs)
        assert profile.validate_files(configs)==profile.validate_files()
        conv=configs/'causal_conv1d_update_kernel.json';raw=conv.read_bytes()
        conv.unlink();checks.append(rejected(lambda:profile.validate_files(configs),'missing_kernel_file'))
        conv.write_bytes(raw+b' ');checks.append(rejected(lambda:profile.validate_files(configs),'modified_kernel_file_SHA'))
        conv.write_bytes(raw);extra=configs/'unknown_kernel.json';extra.write_text('{}')
        checks.append(rejected(lambda:profile.validate_files(configs),'unexpected_kernel_file'))
        extra.unlink()
        work=root/'entry_success';work.mkdir();seen=[]
        args=['runtime_worker_entry.py',str(profile.W/'functional_worker.py'),str(work/'profile.json'),'--','--document','pg19-test-26183','--condition','FP','--stage','resumed_parity']
        def fake_run(path,run_name):seen.append((path,run_name,list(sys.argv)))
        with patch.dict(os.environ,dict(env,LING_PROSPECTIVE_RESULTS=str(work)),clear=True),patch.object(sys,'argv',args),patch.object(entry.runpy,'run_path',fake_run),patch.object(entry,'validate_loaded_profile',return_value={'status':'PASS_CPU_STUB_ONLY_NOT_GPU_PROOF'}):
            entry.main()
        assert len(seen)==1 and seen[0][0]==str(profile.W/'functional_worker.py')
        assert seen[0][2][1:]==args[4:]
        assert (work/'profile.json').exists()
        checks.append('unchanged_worker_argument_delegation_CPU_stub_only')
        work=root/'entry_profile_failure';work.mkdir();args[2]=str(work/'profile.json')
        with patch.dict(os.environ,dict(env,LING_PROSPECTIVE_RESULTS=str(work)),clear=True),patch.object(sys,'argv',args),patch.object(entry.runpy,'run_path',fake_run),patch.object(entry,'validate_loaded_profile',side_effect=AssertionError('POST_FORWARD_PROFILE_BAD')):
            checks.append(rejected(entry.main,'post_forward_profile_failure_closed'))
        assert 'POST_FORWARD_PROFILE_BAD' in json.loads((work/'STOP.json').read_text())['reason']
        first=(work/'STOP.json').read_bytes()
        with patch.dict(os.environ,dict(env,LING_PROSPECTIVE_RESULTS=str(work)),clear=True),patch.object(sys,'argv',args),patch.object(entry.runpy,'run_path',fake_run):
            checks.append(rejected(entry.main,'first_STOP_not_replaced'))
        assert (work/'STOP.json').read_bytes()==first
    assert 'torch' not in sys.modules
    (output/'CPU_FROZEN_PROFILE_TESTS.json').write_text(json.dumps(dict(status='PASS_CPU_FROZEN_PROFILE_AND_ENVELOPE_CHECKS',
        checks=checks,check_count=len(checks),torch_imported=False,CUDA_calls=0,model_loads=0,
        envelope_tests='CPU stubs only; actual scientific delegation still requires GPU qualification',
        source_SHA256={n:hashlib.sha256((profile.W/n).read_bytes()).hexdigest() for n in ['frozen_runtime_profile.py','runtime_worker_entry.py','test_frozen_runtime_profile_cpu.py']},
        config_SHA=profile.CONFIG_SHA256),indent=2)+'\n')
    print('PASS_CPU_FROZEN_PROFILE_CHECKS',len(checks),'NO_MODEL_NO_CUDA')

if __name__=='__main__':main()
