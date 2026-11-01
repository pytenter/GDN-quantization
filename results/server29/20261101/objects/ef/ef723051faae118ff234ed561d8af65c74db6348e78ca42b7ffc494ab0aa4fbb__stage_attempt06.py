"""Prepare frozen instrumentation and tests. Does not import Torch or use CUDA."""
from pathlib import Path
import ast, hashlib, importlib.util, json, os, subprocess, sys, tarfile
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3');C=R/'provenance_capsule_v3';A=C/'attempt_06_generated_code';AUTH=Path(str(R)+'-authority')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def main():
    assert not (A/'STOP.json').exists()
    assert read(A/'GATE0_REVERIFICATION.json')['OLD_EVIDENCE_INTEGRITY_GATE']=='PASS'
    names=['generated_code_guard_attempt06.py','test_generated_guard_attempt06.py','sealed_runner_attempt06.py','stage_attempt06.py']
    for name in names:
        data=(Path('/data/zypan')/name).read_bytes()
        with (A/name).open('xb') as f:f.write(data)
        compile(data,str(A/name),'exec')
    result=subprocess.run([sys.executable,'-B',str(A/'test_generated_guard_attempt06.py')],capture_output=True,text=True)
    with (A/'SYNTHETIC_TEST_OUTPUT.log').open('x') as f:f.write(result.stdout+result.stderr)
    assert result.returncode==0,('NEW_SYNTHETIC_TESTS_FAILED',result.stdout,result.stderr)
    # Re-run every unchanged old architecture assertion without running its preparation main.
    spec=importlib.util.spec_from_file_location('old_frozen_static_adapter',C/'sealed_runner_attempt05_gpu6.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cap=module.Capsule.__new__(module.Capsule);dep=read(C/'attempts/file_runtime_attempt04/FILE_RUNTIME_DEPENDENCIES.json')
    executable=dep['executable_files'][0];cap.file_runtime=dep;cap.exe_hash={executable['realpath']:executable['sha256']};cap.events=[];cap.git_count=0
    tests=[]
    def check(name,fn):fn();tests.append(dict(name=name,gate='PASS'))
    def deny(fn):
        try:fn()
        except (AssertionError,RuntimeError):return
        raise AssertionError('EXPECTED_FAIL_CLOSED')
    check('exact file executable and dependency hashes accepted',lambda:cap.audit('subprocess.Popen',('file',['file','-b','/data/zypan/envs/ling-kda/bin/python3.11'],None,os.environ.copy())))
    check('unknown executable denied',lambda:deny(lambda:cap.audit('subprocess.Popen',('/usr/bin/true',['/usr/bin/true'],None,None))))
    cap.exe_hash[executable['realpath']]='0'*64
    check('changed executable denied',lambda:deny(lambda:cap.audit('subprocess.Popen',('file',['file'],None,None))))
    cap.exe_hash[executable['realpath']]=executable['sha256']
    for key,name in [('shared_libraries','changed child native dependency denied'),('support_data','changed explicit magic support denied')]:
        old=dep[key][0]['sha256'];dep[key][0]['sha256']='0'*64;check(name,lambda:deny(cap.check_file_runtime));dep[key][0]['sha256']=old
    dep['MAGIC_environment']='synthetic changed selector';check('changed MAGIC denied',lambda:deny(cap.check_file_runtime));dep['MAGIC_environment']=None
    check('nonexistent PATH probe is not ingress',lambda:cap.audit('subprocess.Popen',('ling-nonexistent-file-probe',['ling-nonexistent-file-probe'],None,None)))
    check('Git denied',lambda:deny(lambda:cap.audit('subprocess.Popen',('git',['git','status'],None,None))))
    check('shell denied',lambda:deny(lambda:cap.audit('os.system',('true',))))
    assert len(cap.exe_hash)==1;tests.append(dict(name='no prefix or ambient file whitelist',gate='PASS'))
    cap.files={};cap.envfiles={};cap.shared_hash={};cap.generated=set();cap.model_source_hashes=set()
    check('unknown module denied',lambda:deny(lambda:cap.module_file(A/'sealed_runner_attempt06.py','synthetic_unknown')))
    old=read(C/'attempts/gpu6_file_recovery_attempt05/FAILED_LAUNCH_IDENTITY.json')
    assert all(sha(C/r['path'])==r['sha256'] for r in old['files'])
    tests.append(dict(name='all prior candidate and failure bytes preserved',gate='PASS'))
    assert len(tests)==12
    emit(A/'ARCHITECTURE_TESTS.json',dict(old12=tests,new14='PASS',new_test_log_sha256=sha(A/'SYNTHETIC_TEST_OUTPUT.log'),MODEL_LOADED=False,CUDA_INITIALIZED=False,FORMAL_MEASUREMENTS=0))
    runtime=read(C/'runtime_environment_manifest.json');frozen={r['path']:r['sha256'] for r in runtime['environment_files']}
    prefix=Path('/data/zypan/envs/ling-kda/lib/python3.11/site-packages/torch/distributed/nn/jit')
    producer=prefix/'instantiator.py';template=prefix/'templates/remote_module_template.py'
    assert sha(producer)==frozen[str(producer)] and sha(template)==frozen[str(template)]
    emit(A/'PRODUCER_TEMPLATE_AUDIT.json',dict(producer_source_path=str(producer),producer_source_sha256=sha(producer),producer_functions=['_do_instantiate_remote_module_template','_write'],template_source_path=str(template),template_source_sha256=sha(template),template_function='get_remote_module_template',template_constants=['_TEMPLATE_PREFIX','_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA','_REMOTE_FORWARD_TEMPLATE'],generation='get_remote_module_template(flag).format(**str_dict); _write(out_path, exact text); importlib.import_module(generated name)',template_modified=False,Torch_modified=False,ATTEMPT05_bytes='UNAVAILABLE; never reconstructed',identity_policy='generated final exact bytes are primary; observed writer stack attributes frozen producer; random path audit-only'))
    emit(A/'STATIC_FROZEN_SOURCE_ROOTS.json',dict(roots=[str(R),str(C/'source_snapshot'),'/data/zypan/envs/ling-kda/lib/python3.11','/data/zypan/envs/ling-kda/lib/python3.11/site-packages'],policy='Descriptive roots only. Exact per-file SHA manifest controls static admission; no prefix whitelist.',source_manifest_sha256=sha(C/'historical_source_snapshot_manifest.json'),runtime_manifest_sha256=sha(C/'runtime_environment_manifest.json')))
    files=[dict(path=str(p.relative_to(C)),sha256=sha(p)) for p in sorted(A.iterdir()) if p.is_file()]
    emit(A/'INPUT_SOURCE_IDENTITY.json',dict(files=files,classification='ENGINEERING_INSTRUMENTATION_INPUT_NOT_QUALIFICATION',FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
    names=[str(p.relative_to(R)) for p in sorted(A.iterdir()) if p.is_file()]
    with tarfile.open(AUTH/'attempt06-instrumentation.tar.gz','x:gz') as tar:
        for name in names:tar.add(R/name,arcname=name,recursive=False)
    print('ATTEMPT06_PREPARATION_STATIC_TESTS_PASS',len(names),flush=True)
if __name__=='__main__':main()
