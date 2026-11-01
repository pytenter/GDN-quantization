"""CPU55 coordinator: two host-local owned-Popen nodes, global38 barrier, original120. No retry."""
import argparse,concurrent.futures,contextlib,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();R=a.root.resolve();sys.path.insert(0,str(R/'src'))
import execution_control as C
from cross_host_contract import validate_global_layout,inspect_qualification,require_global_formal_admission
import node_rpc as RPC
C.require(C.NODE['id']=='server55_gpu0','Global coordinator may run only55CPU profile');C.authorize(R,R/'GPU_PERMIT.json');validate_global_layout(R)
C.require(not(R/'GLOBAL_COORDINATOR_STARTED.json').exists(),'Coordinator already started; no restart')
mine=C.process_identity(os.getpid());C.atomic_new(R/'GLOBAL_COORDINATOR_STARTED.json',dict(identity=mine,source_sha256=C.sha(R/'SOURCE_FREEZE.json'),global_plan_sha256=C.sha(R/'CROSS_HOST_PLAN.json'),monotonic=time.monotonic()))
local=None;log=None;first=None;last={};receipts={};remote_launched=False
def local_query(code):
    p=subprocess.run([sys.executable,'-B','-'],input=code,text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=55)
    C.require(p.returncode==0,'Local CPU observer failed:'+p.stderr[-4000:]);return json.loads(p.stdout)
def observe(name):return local_query(RPC.state_code())if name=='server55_gpu0'else RPC.query29(RPC.state_code())
def save_phase(stage):
    C.status(R/'GLOBAL_PHASE_STATUS.json',dict(status=stage,monotonic=time.monotonic(),nodes={n:dict(controller_alive=v.get('controller_alive'),formal=v.get('FORMAL_PROGRESS.json'),qualified=bool(v.get('LOCAL_QUALIFICATION_READY.json')),active=v.get('active',[]),session=v.get('SESSION_RESULT.json'),failure=v.get('FIRST_FAILURE.json'))for n,v in last.items()},answers_read_for_dispatch=False))
def remote_new(name,value):
    C.require(name in ['GLOBAL_FORMAL_ADMISSION.json','GLOBAL_STOP_REQUEST.json'],'Only exact frozen coordination outputs may be written')
    code=RPC.prefix()+"value="+repr(value)+"\np=R/"+repr(name)+"\nif p.exists():assert C.read(p)==value\nelse:C.atomic_new(p,value)\nprint(json.dumps(dict(status='PASS_EXACT_COORDINATION_RECEIPT',sha256=C.sha(p))))\n"
    return RPC.query29(code)
def stop_other(reason):
    value=dict(status='GLOBAL_TRUE_STOP_REQUEST',first_observed_global_cause=reason,coordinator_identity=mine,global_plan_sha256=C.sha(R/'CROSS_HOST_PLAN.json'),monotonic_on55=time.monotonic(),no_cross_host_time_order_claim=True)
    if not(R/'GLOBAL_STOP_REQUEST.json').exists():C.atomic_new(R/'GLOBAL_STOP_REQUEST.json',value)
    else:value=C.read(R/'GLOBAL_STOP_REQUEST.json')
    if remote_launched:remote_new('GLOBAL_STOP_REQUEST.json',value)
def connected_observe():
    global last
    out={}
    for name in ['server55_gpu0','server29_gpu6']:
        try:out[name]=observe(name)
        except RPC.ConnectionPending as e:
            C.event(R/'GLOBAL_CONNECTION_OBSERVATIONS.jsonl','READONLY_CONNECTION_WAIT',node=name,error=str(e));return False
    last=out;return True
def await_exits():
    while True:
        if not connected_observe():time.sleep(5);continue
        if all(not v['controller_alive']and not v['active']and not v.get('node_parent_alive')and not v.get('node_tmux_alive')for v in last.values()):break
        time.sleep(2)
    if local is not None:local.wait()
try:
    # Both actual UUIDs must be free before either node GPU dispatcher is created.
    C.device(C.NODE['physical_gpu'])
    check=RPC.prefix()+"C.authorize(R,R/'GPU_PERMIT.json');C.device(C.NODE['physical_gpu']);assert not(R/'GPU_RUN_STARTED.json').exists()and not(R/'operations/NODE_PARENT_STARTED.json').exists();print(json.dumps(dict(status='PASS_REMOTE_NODE_IDLE_SOURCE_PERMIT',source_sha256=C.sha(R/'SOURCE_FREEZE.json'))))\n"
    RPC.query29(check)
    launch=RPC.prefix()+r'''
import shlex,subprocess
C.authorize(R,R/'GPU_PERMIT.json');C.device(C.NODE['physical_gpu']);assert not(R/'GPU_RUN_STARTED.json').exists()and not(R/'operations/NODE_PARENT_STARTED.json').exists()
name='qwen-g64-ledger-v6-node29';assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode!=0
cmd=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONNOUSERSITE=1','PYTHONPATH='+C.NODE['user_site'],C.NODE['runtime_python'],'-u','-B',str(R/'scripts/run_node_once.py'),str(R)]
subprocess.run(['tmux','new-session','-d','-s',name,shlex.join(cmd)],check=True)
C.atomic_new(R/'operations/NODE_UNIQUE_LAUNCH_RECEIPT.json',dict(status='NODE29_UNIQUE_LAUNCH',tmux=name,source_sha256=C.sha(R/'SOURCE_FREEZE.json'),monotonic=time.monotonic()))
print(json.dumps(dict(status='PASS_REMOTE_NODE_ACTUALLY_LAUNCHED',tmux=name)))
'''
    remote_launched=True # A lost SSH reply never authorizes a second launch; stop/read actual state.
    RPC.query29(launch)
    log=(R/'node_controller.log').open('x');local=subprocess.Popen([sys.executable,'-u','-B',str(R/'scripts/gpu_controller.py'),'--root',str(R),'--execute'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    C.atomic_new(R/'operations/NODE_CONTROLLER_LAUNCH.json',dict(identity=C.process_identity(local.pid),parent=mine,monotonic=time.monotonic()))
    while len(receipts)<2:
        if not connected_observe():time.sleep(5);continue
        for name,v in last.items():
            if name=='server55_gpu0'and local.poll()is not None and not v.get('SESSION_RESULT.json'):raise RuntimeError('Node55 actual controller exited without terminal')
            if name=='server29_gpu6'and not v.get('GPU_RUN_STARTED.json')and not v.get('node_parent_alive')and not v.get('node_tmux_alive'):raise RuntimeError('Node29 unique parent exited before GPU controller start')
            if v.get('FIRST_FAILURE.json')or v.get('SESSION_RESULT.json',{}).get('status')=='FAIL_CLOSED_STOP':raise RuntimeError('Node scientific first failure: '+name+' '+repr(v.get('FIRST_FAILURE.json')))
            if not v['controller_alive']and v.get('GPU_RUN_STARTED.json')and not v.get('LOCAL_QUALIFICATION_READY.json'):raise RuntimeError('Node controller exited before qualification-ready: '+name)
            if v.get('LOCAL_QUALIFICATION_READY.json')and name not in receipts:
                code=RPC.prefix()+"from cross_host_contract import inspect_qualification\nprint(json.dumps(inspect_qualification(R)))\n"
                receipts[name]=local_query(code)if name=='server55_gpu0'else RPC.query29(code)
        save_phase('QUALIFICATION_RUNNING_WAIT_BOTH19')
        if len(receipts)<2:time.sleep(10)
    admission=dict(status='PASS_THREE_GPU57_MIXED_ORIGIN_QUALIFICATIONS_ALLOW_UNIQUE120',global_plan_sha256=C.sha(R/'CROSS_HOST_PLAN.json'),global_jobs_sha256=C.sha(R/'GLOBAL_FORMAL_JOB_MANIFEST.json'),human_grant_sha256=C.sha(R/'HUMAN_GPU_AUTHORIZATION.json'),nodes=receipts,coordinator_identity=mine,coordinator_source_sha256=C.sha(R/'SOURCE_FREEZE.json'),monotonic_on55=time.monotonic())
    require_global_formal_admission(R,admission)
    # Qualification workers have exited. A newly occupied selected GPU is never preempted.
    while True:
        try:
            C.device(C.NODE['physical_gpu']);RPC.query29(RPC.prefix()+"C.device(C.NODE['physical_gpu']);print(json.dumps(dict(status='IDLE')))\n");break
        except RPC.ConnectionPending:time.sleep(5)
        except RuntimeError as e:
            if 'Foreign process on assigned GPU'in str(e)or'Insufficient exclusive VRAM'in str(e):
                C.event(R/'GLOBAL_CONNECTION_OBSERVATIONS.jsonl','FORMAL_RESOURCE_WAIT_AFTER_QUALIFIED_RELEASE',error=str(e));save_phase('QUALIFIED_RESOURCE_WAIT_NO_GPU_WORKER');time.sleep(15)
            else:raise
    remote_new('GLOBAL_FORMAL_ADMISSION.json',admission);C.atomic_new(R/'GLOBAL_FORMAL_ADMISSION.json',admission)
    while True:
        if not connected_observe():time.sleep(5);continue
        for name,v in last.items():
            if name=='server55_gpu0'and local.poll()is not None and not v.get('SESSION_RESULT.json'):raise RuntimeError('Node55 actual controller exited without terminal')
            if name=='server29_gpu6'and not v.get('GPU_RUN_STARTED.json')and not v.get('node_parent_alive')and not v.get('node_tmux_alive'):raise RuntimeError('Node29 unique parent exited before GPU controller start')
            if v.get('FIRST_FAILURE.json')or v.get('SESSION_RESULT.json',{}).get('status')=='FAIL_CLOSED_STOP':raise RuntimeError('Node scientific first failure: '+name+' '+repr(v.get('FIRST_FAILURE.json')))
            if not v['controller_alive']and v.get('GPU_RUN_STARTED.json')and not v.get('SESSION_RESULT.json'):raise RuntimeError('Node controller exited without terminal: '+name)
        save_phase('FORMAL_RUNNING')
        if all(v.get('SESSION_RESULT.json',{}).get('status')=='GPU_COMPLETE_GROUP64_NODE_ALL_PLANNED'for v in last.values()):break
        time.sleep(10)
    await_exits();C.require(local.returncode==0,'Node55 actual exit must0');C.require(last['server29_gpu6']['operations/NODE_ACTUAL_EXIT.json']['exit']==0,'Node29 actual owned controller exit must0')
    C.atomic_new(R/'operations/NODE_ACTUAL_EXIT.json',dict(exit=local.returncode,monotonic=time.monotonic()))
    C.atomic_new(R/'GLOBAL_SESSION_RESULT.json',dict(status='GPU_COMPLETE_CROSS_HOST_GROUP64_120_FORMAL',nodes=last,qualification=57,reused_qualification=38,fresh_qualification=19,formal=120,source_sha256=C.sha(R/'SOURCE_FREEZE.json'),all_node_controllers_and_GPU_workers_exited=True,global_qualification_admission_sha256=C.sha(R/'GLOBAL_FORMAL_ADMISSION.json')))
except BaseException as e:
    first=dict(status='GLOBAL_TRUE_STOP',error=repr(e),traceback=traceback.format_exc(),identity=mine,monotonic_on55=time.monotonic(),earliest_physical_cross_host_cause_order='UNRESOLVED; retain each node actual first cause')
    if not(R/'GLOBAL_FIRST_FAILURE.json').exists():C.atomic_new(R/'GLOBAL_FIRST_FAILURE.json',first)
    while True:
        try:stop_other(first);break
        except RPC.ConnectionPending:time.sleep(5)
    if local is not None or remote_launched:
        # Only the started node controllers can terminate/reap their own ownedPopen workers.
        await_exits()
    if local is not None and not(R/'operations/NODE_ACTUAL_EXIT.json').exists():C.atomic_new(R/'operations/NODE_ACTUAL_EXIT.json',dict(exit=local.returncode,monotonic=time.monotonic()))
    C.atomic_new(R/'GLOBAL_SESSION_RESULT.json',dict(status='FAIL_CLOSED_CROSS_HOST_STOP',nodes=last,first_observed_failure=first,all_started_node_controllers_and_GPU_workers_exited=True));raise
finally:
    if log is not None:log.close()
