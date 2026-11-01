"""Execute the actual worker entry point for all19+60 IDs with explicitly CPU mocked model/device APIs."""
import ast, copy, hashlib, importlib.util, json, os, runpy, sys, tempfile, types
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import execution_control as real
from qualification_input_admission import validate
admission=validate(R)
checks=[]
plan=real.read(R/'GPU_EXECUTION_PLAN.json')['qualification_units']
jobs=real.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']
records=real.read(R/'AIME26_DATASET_FREEZE.json')['records']
canary=real.read(R/'protocol/FREE_CANARY_INPUT.json')
all_records={x['problem']:x for x in records+[canary]}
assert len(all_records)==31 and real.validate_jobs(R,jobs)
common=ast.parse((R/'protocol/common.py').read_text());fn=next(x for x in common.body if isinstance(x,ast.FunctionDef)and x.name=='ids_sha256')
ns={'hashlib':hashlib,'json':json};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-ids-hash>','exec'),ns);ids_sha=ns['ids_sha256']
full_ids=real.read(R/'protocol/non_aime_fixture.json')['tokens']
class Vector:
    def __init__(self,ids):self.ids=ids
    def tolist(self):return self.ids
class F:
    @staticmethod
    def render_input(tokenizer,problem):return 'CPU_RENDER_MOCK', [Vector(copy.deepcopy(all_records[problem]['input_ids']))]
    ids_sha256=staticmethod(ids_sha)
class FakeEngine:
    def __init__(self,root,gpu,target):
        assert root==fixture and gpu in real.UUIDS
        self.root=root;self.gpu=gpu;self.target=target;self.f=F;self.tokenizer=None
        self.environment={'CPU_MODEL_AND_CUDA_MOCK':True};self.torch=types.SimpleNamespace(cuda=types.SimpleNamespace(max_memory_allocated=lambda:0,max_memory_reserved=lambda:0))
    def teacher(self,unit,ids,reference):
        assert unit['kind']=='group64_persistent_fresh'and ids==full_ids and len(ids[:unit['tokens']])==512 and reference is not None
        return {'CPU_MODEL_MOCK':True,'tokens':512}
    def generate(self,condition,ids,cap,unit,reference):
        assert condition in ('G64-0','G64-1','G64-2','G64-3')
        row=canary if unit['kind']=='nonAIME_free_generation_fresh'else next(x for x in records if x['problem_id']==unit['problem_id'])
        assert ids==row['input_ids']and cap==(64 if reference is not None else 81920)
        return dict(CPU_MODEL_MOCK=True,generated_token_ids=[1,2],input_ids_sha256=ids_sha(ids),initial_generator_sha256='CPU_RNG_INITIAL',final_generator_sha256='CPU_RNG_FINAL',finish_reason='eos')
    def benchmark(self,unit):
        assert unit['kind']=='nonAIME_throughput'and len(full_ids[:384])==384
        return dict(CPU_MODEL_MOCK=True,benchmark_tokens=384)
    def pressure(self):
        assert len(full_ids[:16])==16
        return dict(CPU_MODEL_MOCK=True,context_slots=82293)
class Ref:
    def __init__(self,root,target,unit):self.cleaned=False
    def cleanup_success(self):self.cleaned=True
mock_runtime=types.ModuleType('runtime_engine');mock_runtime.Engine=FakeEngine;mock_runtime.ByteReference=Ref
sys.modules['runtime_engine']=mock_runtime
micro=types.ModuleType('gpu_codec_micro');micro.run_micro=lambda root,target:dict(CPU_MICRO_MOCK=True)
sys.modules['gpu_codec_micro']=micro
fake=types.ModuleType('execution_control')
for n in dir(real):setattr(fake,n,getattr(real,n))
fake.UUIDS={real.NODE['physical_gpu']:''} # Explicit CPU test selector; empty CVD throughout, never a device grant.
fake.authorize=lambda *a:None;fake.model_guard=lambda *a:None;fake.storage_guard=lambda *a:None
fake.driver_identity=lambda *a:dict(status='PASS',CPU_DRIVER_MOCK=True)
fake.library_check=lambda *a:dict(status='PASS',CPU_LIBRARY_MOCK=True)
fake.same_process=lambda *a:True
fake.process_identity=lambda pid:dict(pid=pid,CPU_PROCESS_MOCK=True)
def atomic(p,value):
    assert p.is_relative_to(fixture)
    with p.open('x')as f:json.dump(value,f)
fake.atomic_new=atomic
fake.sha=lambda p:'CPU_SHA_IDENTITY_MOCK'if p.name in ('SOURCE_FREEZE.json','GPU_PERMIT.json','GROUP64_GPU_CODEC_QUALIFICATION.json')else real.sha(p)
def progress(p,phase,**kw):
    assert p.is_relative_to(fixture)
    (p/'PROGRESS.json').write_text(json.dumps(dict(phase=phase,**kw)))
fake.progress=progress
def read(path):
    rel=path.relative_to(fixture)
    if path.name=='DISPATCH.json':return dict(unit=current['id'],permit_sha256='CPU_SHA_IDENTITY_MOCK',controller_identity=dict(pid=os.getppid()))
    if path.name=='AUTHORIZATION_FORMAL.json':return dict(source_freeze_sha256='CPU_SHA_IDENTITY_MOCK',GPU_qualification_sha256='CPU_SHA_IDENTITY_MOCK')
    if path.is_file():return real.read(path)
    return real.read(R/rel)
fake.read=read;sys.modules['execution_control']=fake
with tempfile.TemporaryDirectory(prefix='g64_actual_worker_cpu_entrypoint_')as tmp:
    fixture=Path(tmp)
    for current in plan+jobs:
        formal=current['kind']=='formal';target=fixture/('formal'if formal else'execution')/current['id'];target.mkdir(parents=True)
        sys.argv=[str(R/'scripts/gpu_unit_worker.py'),'--root',str(fixture),'--unit',current['id']]+(['--formal']if formal else[])
        try:runpy.run_path(str(R/'scripts/gpu_unit_worker.py'),run_name='__main__')
        except SystemExit as e:assert e.code in (None,0)
        assert (target/'RESULT.json').is_file()and not(target/'FIRST_FAILURE.json').exists()
        saved=real.read(target/'RESULT.json');payload=dict(saved);h=payload.pop('payload_sha256');assert real.objsha(payload)==h
        assert saved['status']=='PASS'and saved['unit']==current['id']and real.read(target/'PROGRESS.json')['phase']=='UNIT_PASS'
        checks.append(current['id'])
assert len(checks)==79 and len(set(checks))==79
# A control-input regression independently reproduces the old failure from the actual source expression.
line=next(x for x in(R/'scripts/gpu_unit_worker.py').read_text().splitlines()if x.lstrip().startswith("ids=C.read(R/'protocol/non_aime_fixture.json')"))
guard=compile(ast.parse(line.lstrip()),'<actual-worker-reader>','exec')
class GuardRead:
    require=staticmethod(real.require)
    def __init__(self,value):self.value=value
    def read(self,p):return self.value
for name,value in [('original64',{'tokens':full_ids[:64]}),('truncated511',{'tokens':full_ids[:511]})]:
    try:exec(guard,dict(C=GuardRead(value),R=R))
    except RuntimeError:checks.append('reject_'+name)
    else:raise AssertionError('Short teacher stream accepted')
assert 'torch'not in sys.modules and os.environ['CUDA_VISIBLE_DEVICES']==''
print(json.dumps(dict(status='PASS_ACTUAL_WORKER_ENTRYPOINT_NODE19_AND60_CPU_MOCKS',count=len(checks),checks=checks,input_admission=admission,actual_worker_source_sha256=real.sha(R/'scripts/gpu_unit_worker.py'),model_driver_native_identity_and_tokenizer_render='EXPLICIT_CPU_MOCKS; production real tokenizer independently covered by loader suite',GPU_calls=0,model_forward_calls=0,qualification_credit=0,formal_generation_credit=0)))
