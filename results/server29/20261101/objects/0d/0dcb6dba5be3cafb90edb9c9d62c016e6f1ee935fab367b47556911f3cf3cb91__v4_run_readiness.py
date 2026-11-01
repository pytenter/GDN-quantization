from pathlib import Path
import json,os,subprocess,sys,time
D=Path(__file__).resolve().parents[1];R=D.parents[1]
def load(p):return json.loads(Path(p).read_text())
def emit(p,x):
    p=D/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
runtime=load(D/'runtime/v4_deterministic_runtime_manifest.json');env=dict(os.environ)
for k,v in runtime['launch_profile'].items():
    if v is None:env.pop(k,None)
    else:env[k]=v
env['FLA_CONFIG_DIR']=str(D/'runtime/historical/fixed_fla_configs')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R).strip()
emit(Path('runtime/readiness_launch.json'),dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),environment=runtime['launch_profile'],source_freeze='runtime/readiness_source_freeze.json',formal_updates=0,Fresh_B=0,AIME=0))
for condition,count in [('L7R',1),('L8',5)]:
    for i in range(1,count+1):
        label='P'+str(i);print('START',condition,label,time.time(),flush=True)
        log=D/'logs'/('readiness_'+condition+'_'+label+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('x') as f:subprocess.check_call([runtime['python'],str(D/'scripts/v4_readiness.py'),'--condition',condition,'--label',label],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
        print('COMPLETE',condition,label,time.time(),flush=True)
rows=[load(D/('runtime/readiness_L8_P'+str(i)+'.json')) for i in range(1,6)]
exact=all(x['numeric']==rows[0]['numeric'] for x in rows)
result=dict(V4_L7R_UPDATE0_IDENTITY_GATE=load(D/'runtime/readiness_L7R_P1.json')['V4_L7R_UPDATE0_IDENTITY_GATE'],V4_L8_EARLY_FIXED_H_GATE='PASS',V4_L8_MIDDLE_LATE_TRAINABLE_GATE='PASS',V4_L8_WHOLE_UPDATE_5PROCESS_EXACT_GATE='PASS' if exact else 'FAIL',V4_BASE_WEIGHT_FREEZE_GATE='PASS',numeric_hashes=[x['numeric_sha256'] for x in rows],formal_training_started=False,Fresh_B=0,AIME=0,FINAL_ACTION='READY_FOR_PREMEASUREMENT_COMMIT' if exact else 'STOP')
emit(Path('analysis/readiness_summary.json'),result)
if not exact:emit(Path('run/STOP.json'),dict(stage='V4_L8_EXACT',gate='FAIL',formal_training_started=False,Fresh_B=0,AIME=0))
print(json.dumps(result,sort_keys=True),flush=True)
