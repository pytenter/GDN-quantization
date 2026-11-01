"""Actual full/partial production CPU finisher and29 publisher isolated fixtures.

Reads completed explicit-CPU-mock control fixtures, never scientific payloads.
Original finisher and scorer execute unchanged; tokenizer/model are CPU mocks.
Real transport reuses existing SSHmasters and real perfile SHA/renameat2NOREPLACE
publish on29 under a new CPU fixture path. No GPU/real finisher is relaunched.
"""
import argparse,ast,hashlib,importlib.util,json,os,shlex,subprocess,sys,tempfile,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def require(v,m):
 if not v:raise RuntimeError(m)
SSH55=['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ControlMaster=no','-o','ConnectTimeout=15','zypan@202.38.247.55']
SSH29=['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','-o','ControlMaster=no','-o','ConnectTimeout=15','zypan@202.38.247.29']
PRELUDE=r'''
import json,os,runpy,shlex,subprocess,sys
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text());original_run=subprocess.run
SSH55=cfg['SSH55'];SSH29=cfg['SSH29']
def transport_run(command,*args,**kwargs):
 if isinstance(command,(list,tuple)) and command and command[0]=='ssh' and 'zypan@202.38.247.29' in command:
  command=SSH55+[shlex.join(SSH29+[command[-1]])]
 elif isinstance(command,(list,tuple)) and command and command[0]=='rsync':
  command=list(command);command[command.index('-e')+1]=cfg['rsync_shell']
 return original_run(command,*args,**kwargs)
subprocess.run=transport_run
sys.argv=[cfg['production_finisher'],cfg['fixture_root']]
runpy.run_path(cfg['production_finisher'],run_name='__main__')
'''

def run_remote(code,timeout=120):
 p=subprocess.run(SSH55+[shlex.join(SSH29+[shlex.join(['/usr/bin/python3','-B','-c',code])])],capture_output=True,text=True,timeout=timeout)
 require(p.returncode==0,'IsolatedCPU29verificationfailed:'+p.stderr[-3000:]);return json.loads(p.stdout)

def root_archive_contract():
 # Evaluate only the unchanged production ROOT-read expression, not a finisher
 # invocation, scorer, subprocess, publisher, or model import.
 import types
 source=R/'scripts/finalize_offline.py';tree=ast.parse(source.read_text())
 expression=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ARCHIVE' for t in n.targets))
 reads=[]
 def recorded_read(path):
  require(path==R/'ROOT.json','Finisher archive expression read outside newROOT')
  reads.append(str(path));return read(path)
 actual=eval(compile(ast.Expression(body=expression),'exact_production_archive_ROOT_expression','eval'),{'C':types.SimpleNamespace(read=recorded_read),'R':R})
 expected='/data/zypan/qwen_gdn_wscale_rowbal_archive/'+R.name
 require(actual==expected and len(reads)==1 and R.name.endswith('_BYTEVIEW_REPAIR_20261025'),'New finisher ROOT archive identity differs')
 parent=R.parent/(R.name.replace('_BYTEVIEW_REPAIR_20261025','_REDUCTION_REPAIR_20261025'))
 require(sha(source)==sha(parent/'scripts/finalize_offline.py') and sha(R/'scripts/score_pass_cpu.py')==sha(parent/'scripts/score_pass_cpu.py'),'Unexpected finisher/scorer mathematical/control edit')
 return dict(status='PASS_CPU_EXACT_FINISHER_ROOT_ARCHIVE_EXPRESSION_NOT_FINISHER_EXECUTION',root_file_sha256=sha(R/'ROOT.json'),production_finisher_sha256=sha(source),production_scorer_sha256=sha(R/'scripts/score_pass_cpu.py'),actual_archive=actual,ROOT_reads=reads,parent_full_partial_double_fresh_evidence_reused_byte_unchanged=True,finisher_calls=0,scoring_calls=0,SSH_calls=0,GPU_calls=0)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--control-fixture-root',type=Path);ap.add_argument('--root-contract-only',action='store_true');a=ap.parse_args()
 if a.root_contract_only:print(json.dumps(root_archive_contract(),sort_keys=True));return
 require(a.control_fixture_root is not None,'Explicit isolated CPU control fixture is required')
 base=a.control_fixture_root.resolve();require((base/'success/SESSION_RESULT.json').exists(),'Need successful explicitCPUcontrolfixture')
 output=Path(tempfile.mkdtemp(prefix='wscale_byteview_repair_finalizer_cpu_')).resolve();prelude=output/'run_actual_finisher.py';prelude.write_text(PRELUDE)
 # Standard rsync remote-shell interface: discard its hostarg and forward its
 # unchanged remote rsyncserver command through the two existing masters.
 helper=output/'rsync_shell.py';helper.write_text('#!'+sys.executable+'\nimport os,shlex,sys\nSSH55='+repr(SSH55)+'\nSSH29='+repr(SSH29)+'\ncommand=SSH55+[shlex.join(SSH29+[shlex.join(sys.argv[2:])])]\nos.execvp(command[0],command)\n');helper.chmod(0o700)
 suffix=output.name;archive_parent='/data/zypan/analyses/wscale_byteview_repair_cpu_fixture_evidence/'+suffix;results=[]
 source=R/'scripts/finalize_offline.py';source_sha=sha(source);score=R/'scripts/score_pass_cpu.py';score_sha=sha(score)
 for mode,case,expected in [('full','success','FORMAL_COMPLETE_90_90_AND_STOP'),('partial','qualification_failure','PARTIAL_RESULTS_PRESERVED_AND_STOP')]:
  f=base/case;require(read(f/'CASE.json')['CPU_FIXTURE_ONLY'],'Refuse nonfixture source')
  require(not (f/'FINALIZER_STARTED.json').exists(),'Fixture finisher alreadyused; select anotherindependentCPUfixture')
  (f/'scripts/score_pass_cpu.py').write_bytes(score.read_bytes());(f/'protocol/aime26_frozen.jsonl').write_bytes((R/'protocol/aime26_frozen.jsonl').read_bytes())
  # New files are fixture-only support for the actual finalizer. Original
  # controller payload/source pins stay unchanged and are rechecked byscorer.
  (f/'src/transformers.py').write_text("class _Tokenizer:\n def decode(self,tokens,skip_special_tokens=True):return 'CPU_FIXTURE_ONLY_NOT_AIME_GENERATION'\nclass AutoTokenizer:\n @staticmethod\n def from_pretrained(*args,**kwargs):return _Tokenizer()\n")
  archive=archive_parent+'/'+mode;dump(f/'ROOT.json',dict(task_id='CPU_FIXTURE_ONLY_NOT_SCIENCE',archive_root=archive,actual_model_weights_copied=0))
  dump(f/'CPU_FIXTURE_ONLY_NOT_SCIENTIFIC_RESULT.json',dict(CPU_FIXTURE_ONLY=True,GPU_calls=0,AIME_samples=0,production_finisher_sha256=source_sha,production_scorer_sha256=score_sha,scope='Actual full/partial finisher controlflow onexplicitmockpayloads only'))
  cfg=output/(mode+'_runner.json');dump(cfg,dict(SSH55=SSH55,SSH29=SSH29,rsync_shell=str(helper),production_finisher=str(source),fixture_root=str(f)))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(f/'src'),WSCALE_CPU_FIXTURE_ONLY='YES',WSCALE_CPU_FIXTURE_ROOT=str(f));log=f/'finisher.log'
  with log.open('wb') as stream:p=subprocess.run([sys.executable,'-u','-B',str(prelude),str(cfg)],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
  require(p.returncode==0,'ActualCPU'+mode+'finisherfailed:'+log.read_text()[-5000:]);require(read(f/'FINAL_TASK_STATUS.json')['status']==expected,'Actualfinalscientificstatusbranchwrong')
  require(read(f/'FINAL_SEAL_RECEIPT.json')['scientific_status']==expected,'Actualsealstatuswrong')
  if mode=='full':require((f/'FROZEN_V4_SCORES.json').exists() and (f/'AIME26_PER_PROBLEM_RESULTS.csv').exists() and read(f/'SCORE_PASS_0.json')==read(f/'SCORE_PASS_1.json'),'ActualtwoCPUscore/fullCSVbranchmissing')
  else:require(not (f/'FROZEN_V4_SCORES.json').exists() and not (f/'AIME26_PER_PROBLEM_RESULTS.csv').exists(),'Partialcreatedfalsefullaccuracy/emptyCSV')
  verify=run_remote('''import json,hashlib\nfrom pathlib import Path\nR=Path('''+repr(archive)+''');m=json.loads((R/'SHA256_MANIFEST.json').read_text());count=0\nfor x in m['files']:\n p=R/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'];h=hashlib.sha256()\n with p.open('rb') as f:\n  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)\n assert h.hexdigest()==x['sha256'];count+=1\nprint(json.dumps(dict(status='PASS_INDEPENDENT29_CPU_FIXTURE_ALL_SHA_BYTES',count=count,manifest_sha256=hashlib.sha256((R/'SHA256_MANIFEST.json').read_bytes()).hexdigest(),controls={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['FINAL_TASK_STATUS.json','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json']},GPU_calls=0,AIME_samples=0)))''')
  for n,h in verify['controls'].items():require(sha(f/n)==h,'New55local/29 controlbytesdiffer:'+n)
  # Actual source's restart guard must reject before anotherpublishercall.
  with (output/(mode+'_restart.log')).open('wb') as stream:repeat=subprocess.run([sys.executable,'-u','-B',str(prelude),str(cfg)],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=20)
  require(repeat.returncode!=0 and 'Finisher already started' in (output/(mode+'_restart.log')).read_text(),'Actualfinisherrestartnotrejected')
  results.append(dict(mode=mode,status='PASS',fixture_root=str(f),archive_root=archive,expected_branch=expected,independent29=verify,restart_rejected=True,production_finisher_sha256=source_sha,production_scorer_sha256=score_sha,CPU_MOCK_ONLY=True,GPU_calls=0,AIME_samples=0));print(json.dumps(dict(mode=mode,status='PASS_ACTUAL_CPU_FINISHER_AND29_PUBLISHER',GPU_calls=0,AIME_samples=0)),flush=True)
 # Reuse the exact production verification+renameat2NOREPLACE source with a
 # separate stage against the already-existing full fixture finaldirectory.
 tree=ast.parse(source.read_text());publisher=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='remote' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and 'fn=lib.renameat2' in n.args[0].value)
 final=archive_parent+'/full';collision=archive_parent+'/full_collision_stage'
 collision_result=run_remote('''import json,hashlib,shutil,sys\nfrom pathlib import Path\nfinal=Path('''+repr(final)+''');stage=Path('''+repr(collision)+''');before=hashlib.sha256((final/'SHA256_MANIFEST.json').read_bytes()).hexdigest();shutil.copytree(final,stage);sys.argv=['actual_cpu_publisher',str(stage),str(final)]\ntry:\n exec(compile('''+repr(publisher)+''','actual_production_publisher','exec'))\nexcept AssertionError as e:\n assert 'exist' in str(e).lower() and stage.exists() and hashlib.sha256((final/'SHA256_MANIFEST.json').read_bytes()).hexdigest()==before\n print(json.dumps(dict(status='PASS_ACTUAL_RENAMEAT2_NOREPLACE_REJECTED_EXISTING_FINAL',source_sha256='''+repr(hashlib.sha256(publisher.encode()).hexdigest())+''',final_unchanged=True,stage_preserved=True,GPU_calls=0)))\nelse:raise AssertionError('NOREPLACE overwroteexistingfinal')''')
 require(sha(source)==source_sha and sha(score)==score_sha,'Productionfinalizersourcechangedduringfixture')
 report=dict(status='PASS_ACTUAL_FULL_PARTIAL_CPU_FINISHER_AND29_SHA_BYTES_NOREPLACE',cases=results,collision=collision_result,production_finisher_sha256=source_sha,production_scorer_sha256=score_sha,CPU_MOCK_ONLY=True,GPU_calls=0,GPU_qualification_credit=0,AIME_samples=0,remote_writes='OnlynewisolatedCPUfixturestages+NOREPLACEarchives',existing_science_roots_untouched=True,evidence_root=str(output));dump(output/'FINALIZER_CPU_INTEGRATION_REPORT.json',report);print(json.dumps(dict(status=report['status'],report=str(output/'FINALIZER_CPU_INTEGRATION_REPORT.json'),GPU_calls=0,AIME_samples=0)),flush=True)
if __name__=='__main__':main()
