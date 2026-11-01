"""Outside scientific process: freeze after real PASS, then dispatch postcommit checks."""
from pathlib import Path
import json,shlex,subprocess,time,sys
S=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v3')
T='/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3'
U='/data/zypan/experiment_runs/ling_observability_scientific_repro_v2'
KEY='/data01/user2/.ssh/id_ed25519_ling_remote'
SSH=['ssh','-n','-i',KEY,'-o','BatchMode=yes','zypan@202.38.247.29'];PY='/data/zypan/envs/ling-kda/bin/python'
INPUT=sys.argv[1]
def remote(code):return subprocess.check_output(SSH+[PY+' -B -c '+shlex.quote(code)],text=True).strip()
def main():
    while True:
        value=json.loads(remote("from pathlib import Path;import json;Q=Path('"+T+"/provenance_capsule_v3/scientific_reproducibility_v2_loader_alias');U=Path('"+U+"');print(json.dumps(dict(stop=(Q/'STOP.json').exists() or (U/'QUALIFICATION_SUPERVISOR_STOP.json').exists(),complete=(U/'ALL_SCIENTIFIC_QUALIFICATION_GATES.json').exists())))"))
        assert not value['stop'],'QUALIFICATION_STOP_NO_ADVANCEMENT'
        if value['complete']:break
        time.sleep(30)
    print('REAL_QUALIFICATION_COMPLETE_ADVANCE_TO_COMMIT',flush=True)
    subprocess.run(SSH+[PY+' -B /data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/provenance_capsule_v3/scientific_reproducibility_v2_loader_alias/orchestration/repro_stage_premeasurement.py'],check=True)
    subprocess.run(['python','-B','/data01/user2/repro_freeze_v2.py','scientific-premeasurement',INPUT,
         'ling: freeze scientific premeasurement after real rotated-path and three fresh-process qualification'],check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
    cmd='nohup '+PY+' -B /data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/provenance_capsule_v3/scientific_reproducibility_v2_loader_alias/orchestration/repro_postcommit_advance.py '+shlex.quote(commit)+' > /data/zypan/scientific-repro-postcommit-advance.log 2>&1 < /dev/null & echo $!'
    print('POSTCOMMIT_ADVANCEMENT_PID',subprocess.check_output(SSH+[cmd],text=True).strip(),commit,flush=True)
if __name__=='__main__':main()
