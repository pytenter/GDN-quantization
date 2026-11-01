"""Actual loader path, args and real tokenizer; 9B creation is explicitly CPU mocked."""
import os,sys,json,hashlib,builtins
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import canonical_h128 as h
T=h.torch;T.cuda._lazy_init=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('CPU loader proof forbids GPU init'))
assert 'transformers'not in sys.modules
original_import=builtins.__import__;calls=[]
class FakeModel:
 def eval(self):calls.append(dict(eval_called=True));return self
model=FakeModel()
def intercept(name,globals=None,locals=None,fromlist=(),level=0):
 module=original_import(name,globals,locals,fromlist,level)
 if name=='transformers'and set(fromlist or ())=={'AutoModelForCausalLM','AutoTokenizer'}:
  assert Path(module.__file__).resolve()==(R/'runtime/transformers-qwen35/src/transformers/__init__.py').resolve(),'Actual production loader selected wrong transformers tree'
  def fake_load(path,**kw):
   assert str(path)=='/data/zypan/modelscope_models/Qwen3.5-9B'and kw==dict(torch_dtype=T.bfloat16,device_map={'':0},trust_remote_code=True,local_files_only=True)
   calls.append(dict(path=str(path),kwargs={k:str(v)for k,v in kw.items()},CPU_model_creation_mock=True));return model
  module.AutoModelForCausalLM.from_pretrained=fake_load
 return module
builtins.__import__=intercept
try:m,tok=h.load_model_and_tokenizer()
finally:builtins.__import__=original_import
assert m is model and calls[-1]['eval_called']
from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
assert hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest()=='90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
from protocol_binding import sampler
f,ast=sampler(R,T);records=json.loads((R/'AIME26_DATASET_FREEZE.json').read_text())['records'];checks=[]
for x in records+[json.loads((R/'protocol/FREE_CANARY_INPUT.json').read_text())]:
 _,ids=f.render_input(tok,x['problem']);assert ids[0].tolist()==x['input_ids'];checks.append(dict(problem_id=x.get('problem_id','NON_AIME_CANARY'),input_tokens=len(x['input_ids']),input_ids_sha256=f.ids_sha256(x['input_ids'])))
assert not T.cuda.is_initialized()
print(json.dumps(dict(status='PASS_ACTUAL_LOADER_ROUTE_AND_ALL31_TOKENIZER_INPUTS_CPU',count=34,real_tokenizer_records=checks,actual_loader_invoked=True,model_creation='MOCKED_NO9B_NO_GPU',architecture_sha256=hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest(),actual_transformers_origin=str(Path(sys.modules['transformers'].__file__).resolve()),model_call=calls,CUDA_initialized=False)))
