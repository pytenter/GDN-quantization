"""Run the independent delivery auditor on real58 proof bytes; no GPU."""
import argparse,copy,importlib.util,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
s=importlib.util.spec_from_file_location('actual_readonly_delivery',R/'operations/download_verified_delivery.py');D=importlib.util.module_from_spec(s);s.loader.exec_module(D)
ns={};exec(D.REMOTE_HELPERS,ns);ns['role']='55'
original=D.read(R/'QUALIFICATION_REUSE_ADMISSION.json');by={'QUALIFICATION_REUSE_ADMISSION.json':{}}
for row in original['reused']:
 for x in row['files']:by[row['cached_evidence_prefix']+'/'+x['path']]=dict(x,path=row['cached_evidence_prefix']+'/'+x['path'])
result=ns['qualification_reuse_scope'](R,by);assert result['reused_count']==58 and len(result['units'])==58
checks=['actual_independent58_original_cached_proofs_exit_release_and_six_traces']
read=ns['read']
for name,mutate in [('reused57',lambda d:d.update(reused_count=57)),('remaining1',lambda d:d.update(remaining_count=1)),('wrong_origin_source',lambda d:d['origins'][0].update(source_sha256='0'*64))]:
 value=copy.deepcopy(original);mutate(value)
 ns['read']=lambda p: value if Path(p)==R/'QUALIFICATION_REUSE_ADMISSION.json'else read(p)
 try:
  try:ns['qualification_reuse_scope'](R,by)
  except (AssertionError,RuntimeError,KeyError,FileNotFoundError):pass
  else:raise AssertionError('Auditor accepted '+name)
 finally:ns['read']=read
 checks.append('reject_'+name)
assert 'torch'not in sys.modules
with args.output.open('x')as f:json.dump(dict(status='PASS_ACTUAL_INDEPENDENT58_AUDIT_AND_BAD_PROVENANCE_CPU_ONLY',check_count=len(checks),checks=checks,tool_sha256=D.sha(R/'operations/download_verified_delivery.py'),audit=result,GPU_dispatches=0,Torch_imported=False),f,sort_keys=True,indent=2)
print(json.dumps(dict(status='PASS',check_count=len(checks))))
