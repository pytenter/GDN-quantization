"""Qualified-only advancement, source-side Git freeze, no numerical fallback."""
from pathlib import Path
import json,shlex,subprocess,time
S=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v3')
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
RUNS=Path('/data/zypan/experiment_runs/ling_historical_runtime_rebase_v1')
SSH=['ssh','-n','-i','/data01/user2/.ssh/id_ed25519_ling_remote','-o','BatchMode=yes','zypan@202.38.247.29']
PY='/data/zypan/envs/ling-kda/bin/python'
def remote(s):return subprocess.check_output(SSH+[s],text=True).strip()
def py(code):return remote(PY+' -B -c '+shlex.quote(code))
def freeze(label,message):
    old=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
    subprocess.run(['python','-B','/data01/user2/repro_freeze_v2.py',label,old,message],check=True)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
def main():
    deadline=time.time()+12*3600
    while time.time()<deadline:
        state=py('from pathlib import Path; import json; p=Path('+repr(str(RUNS))+'); s=p/"SUPERVISOR_STOP.json"; log=Path("/data/zypan/historical-runtime-rebase-qualification.log"); ready=log.exists() and "PHASE_B_QUALIFICATION_PASS_READY_FOR_PREMEASUREMENT" in log.read_text(); print("STOP" if s.exists() else "PASS" if ready and (p/"FRESH_PROCESS_NUMERIC_DETERMINISM_GATE.json").exists() and Path('+repr(str(Q/'phase_b_qualification_evidence'))+').exists() else "WAIT")')
        if state=='STOP':raise RuntimeError('PHASE_B_STOP_NO_ADVANCEMENT')
        if state=='PASS':break
        time.sleep(30)
    else:raise RuntimeError('QUALIFICATION_WAIT_TIMEOUT_NO_ADVANCEMENT')
    # Wait for the supervisor's complete archive, not merely a mid-write receipt.
    remote('test -f '+shlex.quote(str(Path(str(R)+'-authority')/'scientific-repro-historical-phase-b.tar.gz')))
    qualification=freeze('scientific-repro-historical-phase-b','ling: freeze whole historical runtime transparent rotated numerical qualification')
    remote(PY+' -B /data/zypan/historical_stage_premeasurement.py')
    premeasurement=freeze('scientific-repro-historical-premeasurement','ling: freeze known-good historical runtime scientific Discovery premeasurement')
    code='from pathlib import Path; import json; p=Path('+repr(str(Q/'SCIENTIFIC_PREMEASUREMENT_COMMIT.json'))+'); p.open("x").write(json.dumps(dict(commit='+repr(premeasurement)+',qualification_commit='+repr(qualification)+',FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED="NO"),indent=2)+"\\n")'
    py(code)
    print('SCIENTIFIC_PREMEASUREMENT_COMMIT',premeasurement,flush=True)
    runner=Q/'advancement_inputs/historical_backend_discovery_runner.py'
    env=py('from pathlib import Path; import json; x=json.loads(Path('+repr(str(Q/'HISTORICAL_RUNNER_LAUNCH_IDENTITY.json'))+').read_text())["environment"]; print(json.dumps(x))')
    values=json.loads(env);launch=['env']
    for k,v in values.items():
        if v is None:launch+=['-u',k]
    launch += [k+'='+str(v) for k,v in values.items() if v is not None]
    def command(mode):return shlex.join(launch+[PY,'-B',str(runner),mode])
    remote(command('postcommit')+' > /data/zypan/historical-rebase-postcommit.log 2>&1')
    py('from pathlib import Path; import json; d=Path('+repr(str(R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'))+'); assert json.loads((d/"postcommit/MINIMAL_POSTCOMMIT_IDENTITY_PARITY_GATE.json").read_text())["gate"]=="PASS"; p=Path('+repr(str(Q/'GPU6_FORMAL_AUTHORIZATION.json'))+'); p.open("x").write(json.dumps(dict(GPU6_FORMAL_AUTHORIZED="YES",premeasurement_commit='+repr(premeasurement)+',postcommit_gate="PASS"),indent=2)+"\\n")')
    print('GPU6_FORMAL_AUTHORIZED_YES',flush=True)
    remote(command('discovery')+' > /data/zypan/historical-rebase-discovery.log 2>&1')
    remote(command('analyze')+' > /data/zypan/historical-rebase-analysis.log 2>&1')
    print('DISCOVERY_COMPLETE_REPORT_ONLY',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        print('ADVANCEMENT_STOP',repr(e),flush=True);raise
