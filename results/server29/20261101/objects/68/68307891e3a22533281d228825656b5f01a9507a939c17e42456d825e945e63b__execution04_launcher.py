"""Strict environment launch and diagnostic canaries; no science formula changes."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent
D=Path(os.environ.get('LING_PROSPECTIVE_RESULTS','/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05'))
os.environ['LING_PROSPECTIVE_RESULTS']=str(D)
from recovery_state import atomic_json,lock,sha,complete
from frozen_runtime_profile import profile_environment,validate_files

ENV_KEYS=['PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','LING_GATED_RMSNORM_FIXED_CONFIG','CUDA_VISIBLE_DEVICES','LING_TARGET_GPU_UUID','LING_EXECUTION_LANE','PYTHONDONTWRITEBYTECODE','LING_PROSPECTIVE_RESULTS','FLA_CACHE_MODE','FLA_CONFIG_DIR']
def selected_environment(env):return {k:env.get(k) for k in ENV_KEYS}

def worker_environment(base,seed,uuid,index):
    assert seed==0,'FROZEN_SEED_CHANGED'
    env=dict(base)
    env.update(PYTHONHASHSEED=str(seed),CUBLAS_WORKSPACE_CONFIG=':4096:8',LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(index),LING_TARGET_GPU_UUID=uuid)
    env.update(profile_environment())
    env.pop('LING_EXECUTION_LANE',None)
    return env

def actual_lane_environment(base,uuid,index):
    # Evaluate the exact unchanged expression used by the real master.
    tree=ast.parse((W/'qualified_completion.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and {k.arg for k in n.keywords}=={'LING_EXECUTION_LANE','LING_TARGET_GPU_UUID','CUDA_VISIBLE_DEVICES'}]
    assert len(calls)==1,'MASTER_ENVIRONMENT_BOUNDARY_CHANGED'
    expression=ast.Expression(calls[0]);ast.fix_missing_locations(expression)
    return eval(compile(expression,'unchanged_master_lane_environment','eval'),{'dict':dict,'str':str,'os':types.SimpleNamespace(environ=base),'uuid':uuid,'index':index})

def process_observation():
    return dict(environment=selected_environment(os.environ),hash_randomization=sys.flags.hash_randomization,hash_probe=hash('Ling-KDA execution04 launch chain'),pid=os.getpid())

def cpu_chain_hop(role,intent):
    value=json.loads(intent.read_text());base=Path(value['base'])
    atomic_json(str(base)+'.'+role.upper()+'_ENV.json',process_observation())
    if role=='master':
        env=actual_lane_environment(dict(os.environ),value['unit_identity']['target_gpu_uuid'],'')
        return subprocess.run([sys.executable,'-B',str(__file__),'--cpu-hop','lane','--intent',str(intent)],env=env).returncode
    return subprocess.run([sys.executable,'-B',str(W/'unit_runner.py'),str(intent)]).returncode

def readiness():
    def read(p):return json.loads(p.read_text())
    assert not (D/'STOP.json').exists(),'FIRST_FAILURE_NO_RETRY'
    validate_files()
    f=read(D/'SOURCE_CODE_FREEZE.json')
    assert subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()==f['source_commit']
    assert subprocess.check_output(['git','-C',str(W),'status','--porcelain'],text=True)==''
    for n,h in f['files'].items():assert sha(W/n)==h,('SOURCE_CHANGED',n)
    for n,k in [('CALIBRATION_CONFIG.json','config_sha256'),('EXECUTION_ROUTING.json','routing_sha256'),('FROZEN_SPEC.json','scientific_spec_sha256')]:assert sha(D/n)==f[k]
    audit=read(D/'FROZEN_RUNTIME_ENVIRONMENT_AUDIT.json')
    assert audit['required_environment']=={k:worker_environment({},0,'',0)[k] for k in ENV_KEYS[:3]}
    assert sha(W/'v2_runtime.py')==audit['runtime_sha256']
    assert sha(audit['runtime_qualified_path'])==audit['runtime_qualified_sha256']
    tests=read(D/'CPU_LAUNCH_CHAIN_REGRESSION.json')
    assert tests['status']=='PASS_CPU_TRUE_LAUNCH_CHAIN'
    for n,h in tests['source_sha256'].items():assert sha(W/n)==h,('TESTED_LAUNCH_SOURCE_CHANGED',n)
    profile_tests=read(D/'CPU_FROZEN_PROFILE_TESTS.json')
    assert profile_tests['status']=='PASS_CPU_FROZEN_PROFILE_AND_ENVELOPE_CHECKS'
    for n,h in profile_tests['source_SHA256'].items():assert sha(W/n)==h,('TESTED_PROFILE_SOURCE_CHANGED',n)
    return f

def run_canary(uuid,index,routing,freeze):
    from prospective_common import verify
    from gpu_admission import admit
    from gpu_telemetry import sample
    os.environ['LING_TARGET_GPU_UUID']=uuid
    cfg=verify();tag='gpu'+str(index)+'__runtime_canary';base=D/'runtime_canary'/tag;base.parent.mkdir(exist_ok=True)
    admit(cfg,D/'admission'/(tag+'.window.jsonl'))
    env=worker_environment(os.environ,cfg['seed'],uuid,index)
    expected=base.with_name(tag+'.EXPECTED_ENV.json');atomic_json(expected,selected_environment(env))
    document=next(d for d,u in routing['document_devices'].items() if u==uuid)
    identity=dict(document=document,rotation='RUNTIME_CANARY_ONLY',stage='runtime_canary',source_commit=freeze['source_commit'],execution=str(D),config_sha256=freeze['config_sha256'],target_gpu_uuid=uuid,scientific_trajectory=False)
    intent=base.with_name(tag+'.INTENT.json')
    command=[sys.executable,'-B','-u',str(W/'runtime_launch_probe.py'),'--mode','gpu','--stage','runtime_canary','--result',str(base)+'.json','--expected',str(expected)]
    atomic_json(intent,dict(command=command,base=str(base),unit_identity=identity,source_commit=freeze['source_commit'],tag=tag,stage='runtime_canary'))
    p=subprocess.Popen([sys.executable,'-B','-u',str(W/'unit_runner.py'),str(intent)],env=env)
    rc=p.wait(timeout=900)
    assert rc==0,('GPU_RUNTIME_CANARY_FAILED',uuid,rc)
    result=complete(Path(str(base)+'.json'),Path(str(base)+'.NATURAL_EXIT.json'),identity)
    assert result['science_trajectory_count']==0 and result['numerical_qualification'] is False and result['gpu_uuid']==uuid
    current=sample(uuid)
    atomic_json(base.with_name(tag+'.POST_EXIT_TELEMETRY.json'),current)
    return dict(uuid=uuid,index=index,status='PASS',result_path=str(base)+'.json',result_sha256=sha(str(base)+'.json'),natural_exit_sha256=sha(str(base)+'.NATURAL_EXIT.json'),scientific_trajectory_count=0,numerical_qualification=False)

def launch():
    freeze=readiness();held=lock(D/'launch.lock');test=lock(D/'controller.lock');test.close()
    read=lambda p:json.loads(p.read_text())
    routing=read(D/'EXECUTION_ROUTING.json');devices=sorted(routing['authorized_devices'].items(),key=lambda x:x[1])
    for b in read(D/'REUSED_PARENT_BINDINGS.json')['units'].values():assert sha(b['original_manifest_path'])==b['original_manifest_sha256']
    for p in (D/'runs').glob('*__parent.json'):
        r=read(p);assert sha(D/'runs'/p.name.replace('__parent.json','__8192.pt'))==r['checkpoint_sha256']
    from gpu_telemetry import observe
    from gpu_admission import classify,classify_window
    ready=False
    for number in [1,2,3]:
        path=D/f'GPU_PREFLIGHT_WINDOW_{number:02d}.jsonl';assert not path.exists(),'PREFLIGHT_ALREADY_EXISTS_NO_RETRY'
        rows=observe([u for u,_ in devices],path,60,5)
        results=[]
        for u,_ in devices:
            values=[r for r in rows if r['uuid']==u];results.append(dict(uuid=u,status=classify_window(values),states=[classify(r) for r in values],sample_count=len(values)))
        atomic_json(D/f'GPU_PREFLIGHT_RESULT_{number:02d}.json',dict(epoch=time.time(),window_sha256=sha(path),results=results,maximum_windows=3))
        if all(r['status']=='AVAILABLE_VERIFIED' for r in results):ready=True;break
    assert ready,'RESOURCE_UNRESOLVED_STOP'
    (D/'admission').mkdir(exist_ok=True)
    units=[run_canary(u,i,routing,freeze) for u,i in devices]
    atomic_json(D/'GPU_RUNTIME_CANARY.json',dict(status='PASS_BOTH_AUTHORIZED_DEVICES',units=units,scientific_trajectory_count=0,numerical_qualification=False,epoch=time.time()))
    uuid,index=devices[0];env=worker_environment(os.environ,0,uuid,index)
    atomic_json(D/'USER_AUTHORIZED_RUN_LAUNCH.json',dict(epoch=time.time(),source_commit=freeze['source_commit'],launcher_sha256=sha(__file__),runtime_canary='PASS',actual_master_environment=selected_environment(env),formal_order_unchanged=True))
    os.set_inheritable(held.fileno(),True)
    os.execve(sys.executable,[sys.executable,'-B','-u',str(W/'recovery_controller.py')],env)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-authorized',action='store_true');parser.add_argument('--cpu-hop',choices=['master','lane']);parser.add_argument('--intent',type=Path);a=parser.parse_args()
    if a.cpu_hop:return cpu_chain_hop(a.cpu_hop,a.intent)
    if not a.run_authorized:
        readiness();print('READY_CPU_CHAIN_PASS_GPU_NOT_STARTED');return 0
    try:launch()
    except BlockingIOError:print('EXISTING_LAUNCH_NO_DUPLICATE');return 0
    except BaseException as e:
        if not (D/'STOP.json').exists():atomic_json(D/'STOP.json',dict(status='STOP',stage='EXECUTION05_LAUNCH_OR_CANARY',reason=repr(e),traceback=traceback.format_exc(),epoch=time.time(),no_retry=True))
        env=worker_environment(os.environ,0,'',0);env['CUDA_VISIBLE_DEVICES']=''
        rc=subprocess.run([sys.executable,'-B','-u',str(W/'finalize_cpu.py')],env=env).returncode
        assert rc==0,'CPU_DELIVERY_ERROR_PRESERVE_SCIENCE'
        return 1

if __name__=='__main__':sys.exit(main())
