"""New execution/driver adapters on CPU and fake driver only; actual tiny Qwen flow."""
import os,sys,json,tempfile,hashlib,uuid,contextlib,subprocess,types,importlib.util
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'inherited/group32/src'),'/data/zypan/transformers-qwen35/src']
import execution_control as C
import torch
real_lazy=torch.cuda._lazy_init;torch.cuda._lazy_init=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Real GPU forbidden in CPU test'))
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from runtime_engine import Engine,ByteReference
from int4_adapter import Int4QDQ as FourCodecQDQ,NativeLayerTracker
from protocol_binding import sampler
from canonical_h128 import get_state
from group_codecs import tensor_bytes
from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
checks=[]
def passed(name):checks.append(name)
class Fn:
 def __init__(self,fn):self.fn=fn
 def __call__(self,*a):return self.fn(*a)
class FakeDriver:
 def __init__(self,g,count=1):
  self.cuInit=Fn(lambda _:0);self.cuDeviceGetCount=Fn(lambda p:self.put(p,count));self.cuDeviceGet=Fn(lambda p,i:self.put(p,i));self.cuDeviceGetUuid_v2=Fn(lambda p,d:self.uid(p,g))
 def put(self,p,v):p._obj.value=v;return 0
 def uid(self,p,g):
  for i,b in enumerate(uuid.UUID(C.UUIDS[g][4:]).bytes):p._obj.bytes[i]=b
  return 0
with tempfile.TemporaryDirectory(dir='/dev/shm') as d:
 base=Path(d);lib=base/'libcuda.so.fake';lib.write_bytes(b'fake_driver_for_CPU_only');old=C.bindings;C.bindings=lambda r:{str(lib):dict(path=str(lib),bytes=lib.stat().st_size,sha256=C.sha(lib))}
 try:
  for g in [3]:
   v=C.driver_identity(R,g,driver_factory=lambda p,g=g:FakeDriver(g),environment={'CUDA_VISIBLE_DEVICES':C.UUIDS[g]});assert v['actual_uuid']==C.UUIDS[g] and v['context_creation_API_calls']==0
   try:C.driver_identity(R,g,driver_factory=lambda p:FakeDriver((g+1)%4),environment={'CUDA_VISIBLE_DEVICES':C.UUIDS[g]});raise AssertionError('Wrong UUID accepted')
   except RuntimeError:pass
   try:C.driver_identity(R,g,driver_factory=lambda p,g=g:FakeDriver(g,count=2),environment={'CUDA_VISIBLE_DEVICES':C.UUIDS[g]});raise AssertionError('Multidevice admitted')
   except RuntimeError:pass
   passed('FAKE_DRIVER_SELECTED_GPU_'+str(g)+'_POSITIVE_WRONG_UUID_AND_COUNT_NEGATIVE')
 finally:C.bindings=old
 ident=C.process_identity(os.getpid());assert C.same_process(ident) and not C.same_process(dict(ident,starttick=ident['starttick']+1));passed('PID_STARTTICK_ARGV_IDENTITY')
 original=C.query_gpu
 try:
  for g in [3]:
   obs=dict(devices=[dict(physical_index=i,uuid=C.UUIDS[i],name='RTX3090',memory_total_MiB=24576,memory_free_MiB=24256) for i in range(4)],apps=[]);C.query_gpu=lambda:obs;C.device(g)
   obs['apps']=[dict(pid=999999,uuid=C.UUIDS[g],memory_MiB=1)]
   try:C.device(g);raise AssertionError('Foreign task admitted')
   except RuntimeError:pass
  passed('FOUR_DEVICE_FRESH_AND_FOREIGN_OWNER_NEGATIVE')
 finally:C.query_gpu=original
 class CPUProxy:
  cuda=types.SimpleNamespace(synchronize=lambda:None,max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,empty_cache=lambda:None,manual_seed_all=lambda s:None)
  def __getattr__(self,k):return getattr(torch,k)
  def tensor(self,*a,**k):k.pop('device',None);return torch.tensor(*a,**k)
  def zeros(self,*a,**k):k.pop('device',None);return torch.zeros(*a,**k)
  def Generator(self,*a,**k):return torch.Generator(device='cpu')
 T=CPUProxy()
 class Tok:
  eos_token_id=63;unk_token_id=1000
  def convert_tokens_to_ids(self,s):return 63
  def decode(self,ids,skip_special_tokens=True):return ' '.join(map(str,ids))
 def tiny():
  torch.random.default_generator.manual_seed(20261020)
  c=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=128,linear_value_head_dim=128,linear_conv_kernel_dim=4,layer_types=['linear_attention','linear_attention','full_attention'],rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[2,1,1]});c._attn_implementation='sdpa';return Q.Qwen3_5ForCausalLM(c).eval()
 def engine(target):
  e=Engine.__new__(Engine);e.root=R;e.gpu=0;e.target=target;target.mkdir();e.torch=T;e.model=tiny();e.QDQ=FourCodecQDQ;e.Native=NativeLayerTracker;e.layers=(0,1);e.get_state=get_state;e.bytes=tensor_bytes;e.vocab=64;e.tokenizer=Tok();e.f,e.sampler_ast=sampler(R,T);e.environment=dict(scope='TINY_CPU_ONLY');return e
 oldref=C.reference_root;C.reference_root=lambda:base/'references'
 try:
  for condition in ['I4-N','I4-H']:
   results=[]
   for rep in [0,1]:
    target=base/(condition+'_R'+str(rep));unit=dict(reference_unit=condition+'_R0',repeat=rep);e=engine(target);ref=ByteReference(base,target,unit)
    # ByteReference references immutable previous RESULT/TRACE from an actual distinct CPU engine.
    if rep==1:
     pass
    result=e.generate(condition,[3,7,11],8,dict(problem_id='NON_AIME_CPU_FIXTURE'),ref);results.append(result)
    out=base/'execution'/unit['reference_unit'];out.mkdir(parents=True,exist_ok=True)
    if rep==0:(out/'TRACE.json').write_bytes((target/'TRACE.json').read_bytes())
    else:ref.cleanup_success()
   for key in ['generated_token_ids','initial_generator_sha256','final_generator_sha256','actual_model_forward_calls','quantizer_audit']:assert results[0][key]==results[1][key],key
   assert results[0]['prompt_layer_QDQ']==6 and results[0]['actual_model_forward_calls']==3+results[0]['generated_token_count']-1
   passed('ACTUAL_TINY_QWEN_STEPWISE_PROMPT_FREE_SAMPLER_AND_FRESH_FULL_BYTES_'+condition)
  # Actual tiny GDN flow, with explicit controlled logits only for termination branch fixtures.
  for condition in ['I4-N','I4-H']:
   for ending in ['eos','length']:
    target=base/(condition+'_termination_'+ending);e=engine(target)
    def force_logits(_m,_a,out,ending=ending):
     out.logits.fill_(-1000.);out.logits[...,63 if ending=='eos' else 5]=1000.;return out
    handle=e.model.register_forward_hook(force_logits)
    value=e.generate(condition,[3,7,11],8,dict(problem_id='NON_AIME_CONTROLLED_TERMINATION'))
    handle.remove();assert value['finish_reason']==ending and value['generated_token_count']==(1 if ending=='eos' else 8)
    passed('ACTUAL_TINY_CPU_INT4_'+condition+'_SAMPLER_TERMINATION_'+ending)
  # Native unquantized mathematical control uses identical models with/without hook.
  for rep in [0,1]:
   target=base/('native_R'+str(rep));e=engine(target);unit=dict(reference_unit='native_R0',repeat=rep,kind='native_unquantized_math_control',tokens=4);ref=ByteReference(base,target,unit);e.teacher(unit,[3,7,11,17],ref)
   out=base/'execution/native_R0';out.mkdir(parents=True,exist_ok=True)
   if rep==0:(out/'TRACE.json').write_bytes((target/'TRACE.json').read_bytes())
   else:ref.cleanup_success()
  passed('ACTUAL_NATIVE_NO_QDQ_AND_HOOK_CPU_FULL_BYTES')
 finally:C.reference_root=oldref
 # Exact original scorer fixture tests, not any AIME generation/outcome.
 sp=importlib.util.spec_from_file_location('cpu_frozen_v4_fixture',R/'protocol/aime26_scorer_v4.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
 for text in ['Therefore the final answer is \\boxed{42}.','The answer is 17. That was incorrect. The final answer is \\boxed{42}.']:
  x=m.extract_v4(text,'eos','NonAIME CPU test.');assert m.compare_with_gold(x,'42')
 passed('EXACT_FROZEN_V4_NON_AIME_FIXTURES')
 f,_=sampler(R,T);_,survivors=f.sample_next(f.build_warpers(),torch.zeros((1,64)),torch.zeros((1,64),dtype=torch.bool),torch.Generator().manual_seed(1));assert survivors>20;passed('ORIGINAL_TOPK_TIE_SEMANTICS_NOT_REPLACED_OR_FALSELY_REJECTED')
 C.admit_inherited(R);C.validate_jobs(R,C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']);passed('BOUND_EXISTING_GPU_QUALIFICATION_AND60_SINGLE_UUID_BUNDLES')
assert not torch.cuda.is_initialized()
print('RESULT_JSON='+json.dumps(dict(status='PASS_NEW_EXECUTION_CPU_INTEGRATION',checks=checks,count=len(checks),actual_tiny_Qwen_and_full_bytes=True,fake_driver_only=True,GPU_dispatches=0,scope='NOT9B_NOTGPU_NOT81K'),sort_keys=True))
