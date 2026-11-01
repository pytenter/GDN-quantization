"""Exercise the production pre-model input admission with exact inputs and finite strict negatives."""
import ast, copy, json, os, shutil, sys, tempfile
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from qualification_input_admission import validate,ORIGIN
checks=[]
assert validate(R)['qualification_units']==38;checks.append('Actual_production_input_admission_current_root')
with tempfile.TemporaryDirectory(prefix='g64_input_admission_cpu_')as tmp:
    F=Path(tmp)
    for n in [ORIGIN,'protocol/non_aime_fixture.json','history/group32/protocol/non_aime_fixture.json','GPU_EXECUTION_PLAN.json']:
        p=F/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/n,p)
    original=json.loads((F/'protocol/non_aime_fixture.json').read_text());plan=json.loads((F/'GPU_EXECUTION_PLAN.json').read_text())
    def put(n,v):(F/n).write_text(json.dumps(v))
    def reject(name,call):
        try:call()
        except (RuntimeError,KeyError,TypeError):checks.append('Reject_'+name);return
        raise AssertionError('Invalid input admitted: '+name)
    assert validate(F)['frozen_teacher_token_count']==32768;checks.append('Actual_admission_exact_CPU_fixture')
    ids=original['tokens']
    for name,bad in [
        ('old64',dict(original,tokens=ids[:64])),('short511',dict(original,tokens=ids[:511])),
        ('short32767',dict(original,tokens=ids[:-1])),('repeat64_padding',dict(original,tokens=ids[:64]*512)),
        ('wrong_origin_SHA',dict(original,source_frozen_fixture_raw_sha256='0'*64)),
        ('changed511',dict(original,tokens=ids[:511]+[(ids[511]+1)%248320]+ids[512:])),
        ('negative_ID',dict(original,tokens=[-1]+ids[1:])),('vocab_ID',dict(original,tokens=[248320]+ids[1:])),
        ('bool_ID',dict(original,tokens=[True]+ids[1:])),('float_ID',dict(original,tokens=[1.0]+ids[1:])),
        ('unknown_schema',dict(original,ignored=True))]:
        put('protocol/non_aime_fixture.json',bad);reject(name,lambda:validate(F))
    put('protocol/non_aime_fixture.json',original)
    for name,mut in [
        ('missing_unit',lambda p:p['qualification_units'].pop()),
        ('duplicated_unit',lambda p:p['qualification_units'].append(p['qualification_units'][0])),
        ('wrong_GPU',lambda p:p['qualification_units'][0].update(gpu=0)),
        ('wrong512_length',lambda p:p['qualification_units'][1].update(tokens=64)),
        ('wrong_checkpoint',lambda p:p['qualification_units'][1].update(checkpoints=[0,31,127,63])),
        ('wrong_condition',lambda p:p['qualification_units'][1].update(condition='Q0')),
        ('wrong_repeat',lambda p:p['qualification_units'][1].update(repeat=3)),
        ('wrong_free_cap',lambda p:p['qualification_units'][3].update(max_new_tokens=81920))]:
        value=copy.deepcopy(plan);mut(value);put('GPU_EXECUTION_PLAN.json',value);reject(name,lambda:validate(F))
    put('GPU_EXECUTION_PLAN.json',plan)
    p=F/ORIGIN;raw=p.read_bytes();p.write_bytes(raw+b' ');reject('historical_raw_byte_change',lambda:validate(F));p.write_bytes(raw)
    assert validate(F)['status']=='PASS_FULL_HISTORICAL_REPLAY_AND_ALL38_INPUT_BOUNDS_CPU';checks.append('No_false_rejection_after_restoring_exact_fixture')
tree=ast.parse((R/'scripts/gpu_controller.py').read_text())
block=next(x for x in tree.body if isinstance(x,ast.Try))
assert isinstance(block.body[0],ast.ImportFrom)and block.body[0].module=='qualification_input_admission'
assert 'EXECUTION_QUALIFICATION_INPUT_ADMISSION.json'in ast.unparse(block.body[1])
assert 'verify_external'in ast.unparse(block.body[2])
checks.append('Actual_controller_calls_input_admission_before_external_hash_and_GPU_dispatch')
assert 'torch'not in sys.modules
print(json.dumps(dict(status='PASS_PRODUCTION_PREMODEL_INPUT_ADMISSION_CPU',count=len(checks),checks=checks,GPU_calls=0,model_forward_calls=0)))
