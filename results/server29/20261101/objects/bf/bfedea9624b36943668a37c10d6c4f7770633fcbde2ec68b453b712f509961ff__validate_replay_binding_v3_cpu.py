"""Install pre-freeze input guard and run independent fresh CPU suites, no GPU operations."""
import base64,json
from pathlib import Path
import remote
L=remote.ROOT/'revisions/replay_binding_v3'
names=['src/qualification_input_admission.py','scripts/gpu_controller.py','scripts/cpu_worker_entrypoint_review.py','scripts/cpu_input_admission_review.py']
updates={n:base64.b64encode((L/n).read_bytes()).decode()for n in names}
code=r'''
import base64,concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/replay_binding_v3');assert not(R/'SOURCE_FREEZE.json').exists()and not(R/'GPU_RUN_STARTED.json').exists()
for n,b in UPDATES.items():
 p=R/n
 if p.exists():
  q=R/'provenance/PRE_INPUT_ADMISSION_CONTROL'/n;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb')as f:f.write(p.read_bytes())
 p.write_bytes(base64.b64decode(b))
folder=R/'analysis/FINAL_CPU_CHECKS';folder.mkdir(exist_ok=False)
scripts=['cpu_qualification.py','cpu_control_review.py','cpu_native_admission_review.py','cpu_production_loader_review.py','cpu_input_admission_review.py','cpu_worker_entrypoint_review.py']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH='/data/zypan/envs/qwen-gdn-group64-historical-user-site-v1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
py='/data/zypan/envs/qwen-gdn-group64-historical-port-v1/bin/python'
def run(item):
 s,rep=item;start=time.monotonic();p=subprocess.run([py,'-B',str(R/'scripts'/s)],env=env,capture_output=True,text=True,timeout=240)
 prefix=folder/(s[:-3]+'_R'+str(rep));prefix.with_suffix('.stdout').write_text(p.stdout);prefix.with_suffix('.stderr').write_text(p.stderr)
 row=dict(script=s,repeat=rep,exit=p.returncode,seconds=time.monotonic()-start)
 if p.returncode==0:row['result']=json.loads(p.stdout.split('RESULT_JSON=')[-1])
 else:row.update(stdout=p.stdout[-1000:],stderr=p.stderr[-8000:])
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:runs=list(pool.map(run,[(s,r)for s in scripts for r in (0,1)]))
passed=all(x['exit']==0 for x in runs)
if passed:
 for s in scripts:
  a,b=[x for x in runs if x['script']==s];assert a['result']==b['result'],s
result=dict(status='PASS_TWO_FRESH_REPLAY_BINDING_CPU_SUITES'if passed else'PRE_FREEZE_CPU_REVIEW_FAILED_NO_GPU',runs=runs,GPU_dispatches=0)
with(R/'analysis/REPAIRED_REVISION_CPU_VALIDATION.json').open('x')as f:json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
if passed:
 group=[x for x in runs if x['script']=='cpu_qualification.py']
 with(R/'GROUP64_CPU_CODEC_QUALIFICATION.json').open('x')as f:json.dump(dict(status='PASS_TWO_FRESH_CPU_PROCESSES',runs=group,GPU_dispatches=0),f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(result))
'''.replace('UPDATES',repr(updates))
compile(code,'<remote-cpu-suite>','exec')
r=remote.run(code,timeout=1800)
with(L/'analysis/REPAIRED_REVISION_CPU_VALIDATION.json').open('x')as f:json.dump(r,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(status=r['status'],runs=[{k:v for k,v in x.items()if k!='result'}|({'count':x['result']['count']}if'result'in x else{})for x in r['runs']]),ensure_ascii=False))
