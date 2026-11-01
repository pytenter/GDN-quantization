"""Actual production planner/roster and frozen-loop regression; no CUDA imports."""
import ast,copy,hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import execution_control as C
from control_plan import build_formal_plan,build_qualification_plan,validate_formal_jobs
OLD=R.parent/'QWEN_GDN_GROUP32_FOUR_CODEC_AIME26_SEED1_V1';dataset=C.read(OLD/'AIME26_DATASET_FREEZE.json');assignment,formal=build_formal_plan(dataset,C.UUIDS);plan=build_qualification_plan(C.UUIDS)
checks=[]
def checked(name,fn):
 fn();checks.append(dict(name=name,status='PASS'))
def reject(name,change):
 a=copy.deepcopy(assignment);j=copy.deepcopy(formal['jobs']);d=copy.deepcopy(dataset);change(a,j,d)
 try:validate_formal_jobs(d,a,j,C.UUIDS)
 except (RuntimeError,KeyError,ValueError):checks.append(dict(name=name,status='PASS_REJECTED'))
 else:raise AssertionError('Negative accepted:'+name)
checked('production90jobs10questions_per_sameUUID3methods',lambda:validate_formal_jobs(dataset,assignment,formal['jobs'],C.UUIDS))
reject('GPU0foreignjob_forbidden',lambda a,j,d:j[0].update(gpu=0))
reject('wrongUUID',lambda a,j,d:j[0].update(gpu_uuid=C.UUIDS[0]))
reject('seed2_forbidden',lambda a,j,d:j[0].update(seed=2))
reject('shorter_cap_forbidden',lambda a,j,d:j[0].update(max_new_tokens=4096))
reject('missingmethod',lambda a,j,d:j.pop())
reject('duplicatemethod',lambda a,j,d:j.__setitem__(1,copy.deepcopy(j[0])))
reject('inputSHAchanged',lambda a,j,d:j[0].update(input_ids_sha256='0'*64))
reject('wrongpromptcount',lambda a,j,d:j[0].update(input_tokens=1))
reject('rewritten_methodorder',lambda a,j,d:a['assignments'][0].update(conditions=list(reversed(a['assignments'][0]['conditions']))))
reject('rewritten_hashrank',lambda a,j,d:a['assignments'][0].update(hash_rank=29))
reject('rewritten_questionrank',lambda a,j,d:a['assignments'].reverse())
reject('noncanonical_roster',lambda a,j,d:d['records'][0].update(problem_id='aime26_31'))
reject('prompt_longerthan373',lambda a,j,d:d['records'][0].update(input_ids=[1]*374,input_tokens=374))
reject('duplicatefrozenquestion',lambda a,j,d:d['records'].__setitem__(1,copy.deepcopy(d['records'][0])))
assert len(plan['qualification_units'])==52 and len({u['id'] for u in plan['qualification_units']})==52
checks.append(dict(name='finite52_currentqualification_no_science_MSEgate',status='PASS'))
def methods(p):
 text=p.read_text();tree=ast.parse(text);return {n.name:ast.get_source_segment(text,n) for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Engine' for n in c.body if isinstance(n,ast.FunctionDef)}
old=methods(OLD/'src/runtime_engine.py');new=methods(R/'src/runtime_engine.py')
for name in ('__init__','forward','generate'):
 assert old[name]==new[name],name
 checks.append(dict(name='originalEngine_'+name+'_exact_source',status='PASS',sha256=hashlib.sha256(new[name].encode()).hexdigest()))
for name in ('sampler.py','common.py','aime26_scorer_v4.py'):
 assert (R/'protocol'/name).read_bytes()==(OLD/'protocol'/name).read_bytes(),name
 checks.append(dict(name='original_protocol_'+name+'_exactbytes',status='PASS',sha256=C.sha(R/'protocol'/name)))
for p in list((R/'src').glob('*.py'))+list((R/'scripts').glob('*.py')):ast.parse(p.read_text(),filename=str(p))
checks.append(dict(name='all_current_source_AST_parse',status='PASS'))
assert 'torch' not in sys.modules,'CPUcontrol importedtorch'
print(json.dumps(dict(status='PASS_ACTUAL_PRODUCTION_CONTROL_CPU_ONLY',checks=checks,count=len(checks),GPU_calls=0,GPU_qualification_credit=0,AIME_samples=0),indent=2))
