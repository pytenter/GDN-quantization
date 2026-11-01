"""Append-only receipt serialization correction; original resume FAIL preserved."""
from pathlib import Path
import hashlib,importlib.util,json,shlex,subprocess
original=Path('/data01/user2/historical_auto_advance_source.py')
assert hashlib.sha256(original.read_bytes()).hexdigest()=='132a0bb79c2cf8fdc3c91b3397a8e954da32f23d5e62338ee3ab6c6bea6fbbb0'
spec=importlib.util.spec_from_file_location('frozen_advancement',original)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.S,text=True).strip()
assert commit=='e428b5deb0579b926b2ad098a8699b3f4f2233f0'
assert a.remote('git -C '+shlex.quote(str(a.R))+' rev-parse HEAD')==commit
assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=a.S,text=True).strip()
assert a.remote('git -C '+shlex.quote(str(a.R))+' status --porcelain=v1')==''
receipt=a.Q/'SCIENTIFIC_PREMEASUREMENT_COMMIT.json'
a.py('from pathlib import Path; import json; p=Path('+repr(str(receipt))+'); p.open("x").write(json.dumps(dict(commit='+repr(commit)+',qualification_commit="aca8fdc4617cafd341e446307fe76a9ff4653200",FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED="NO"),indent=2)+chr(10))')
print('SCIENTIFIC_PREMEASUREMENT_COMMIT',commit,flush=True)
runner=a.Q/'advancement_inputs/historical_backend_discovery_runner.py'
values=json.loads(a.py('from pathlib import Path; import json; print(json.dumps(json.loads(Path('+repr(str(a.Q/'HISTORICAL_RUNNER_LAUNCH_IDENTITY.json'))+').read_text())["environment"]))'))
launch=['env']
for k,v in values.items():
    if v is None:launch+=['-u',k]
launch += [k+'='+str(v) for k,v in values.items() if v is not None]
def command(mode):return shlex.join(launch+[a.PY,'-B',str(runner),mode])
try:
    a.remote(command('postcommit')+' > /data/zypan/historical-rebase-postcommit.log 2>&1')
    a.py('from pathlib import Path; import json; d=Path('+repr(str(a.R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'))+'); assert json.loads((d/"postcommit/MINIMAL_POSTCOMMIT_IDENTITY_PARITY_GATE.json").read_text())["gate"]=="PASS"; p=Path('+repr(str(a.Q/'GPU6_FORMAL_AUTHORIZATION.json'))+'); p.open("x").write(json.dumps(dict(GPU6_FORMAL_AUTHORIZED="YES",premeasurement_commit='+repr(commit)+',postcommit_gate="PASS"),indent=2)+chr(10))')
    print('GPU6_FORMAL_AUTHORIZED_YES',flush=True)
    a.remote(command('discovery')+' > /data/zypan/historical-rebase-discovery.log 2>&1')
    a.remote(command('analyze')+' > /data/zypan/historical-rebase-analysis.log 2>&1')
    print('DISCOVERY_COMPLETE_REPORT_ONLY',flush=True)
except BaseException as e:
    print('ADVANCEMENT_RESUME_STOP',repr(e),flush=True);raise
