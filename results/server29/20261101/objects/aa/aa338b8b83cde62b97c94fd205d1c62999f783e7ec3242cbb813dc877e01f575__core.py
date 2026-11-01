"""Read-only original formal Hadamard/C128 + qualified V3 arithmetic bindings."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import ast,gc,hashlib,inspect,json,math,pathlib,random,sys,time
from types import SimpleNamespace
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
OLD=pathlib.Path('/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1')
V3=pathlib.Path('/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3')
ARCH=pathlib.Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
CONDITIONS=('FP_NATIVE','H_C128','H_DUAL_M')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def save(rel,obj):
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def check(ok,message):
 if isinstance(ok,torch.Tensor):ok=bool(ok.item())
 if not ok:raise RuntimeError(message)
def extract(path,names,ns):
 nodes=[n for n in ast.parse(pathlib.Path(path).read_text()).body if getattr(n,'name',None) in names or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
 found={getattr(n,'name',None) for n in nodes}|{t.id for n in nodes if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
 check(names<=found,'Missing original AST dependency: '+str(names-found));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return SimpleNamespace(**ns)
def formal(memoize=False):
 ns=dict(torch=torch,math=math,os=os,sys=sys,Path=pathlib.Path,MODEL_PATH=pathlib.Path('/data/zypan/modelscope_models/Qwen3.5-9B'),TRANSFORMERS_SRC=pathlib.Path('/data/zypan/transformers-qwen35'),STATE_QUANT_SRC=pathlib.Path('/data/zypan/experiments/qwen35_gdn_quant'))
 f=extract(OLD/'runtime_sources/formal_c128_runner.py',{'EPS','GDN_LAYERS','get_state','quantize_c128','hadamard','normalize_qk','QwenKeyHadamardPatch','load_model_and_tokenizer'},ns)
 if memoize:
  original=ns['hadamard'];matrices={}
  def constant(n=128,dtype=torch.float32,device=None):
   key=(n,dtype,str(device))
   if key not in matrices:matrices[key]=original(n,dtype,device)
   return matrices[key]
  ns['hadamard']=constant;f.hadamard=constant
 return f

def numerical():
 return extract(OLD/'runtime_sources/numerical_recovery.py',{'compensated_scan','backend','chunk_binding'},dict(torch=torch,ast=ast,Path=pathlib.Path,ROOT=V3,ARCH=ARCH,ARCH_SHA=sha(OLD/'runtime_sources/current_modeling_qwen3_5.py'),sha=sha,check=check))
def configure():
 check(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Frozen visible GPU mapping changed')
 check(torch.__version__=='2.5.1+cu121' and torch.version.cuda=='12.1','Torch/CUDA identity changed')
 torch.set_num_threads(1)
 if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
 torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 random.seed(20261015);torch.manual_seed(20261015);torch.cuda.manual_seed_all(20261015);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
 check('3090' in torch.cuda.get_device_properties(0).name,'Wrong GPU')
def load():
 start=time.monotonic();f=formal(True);model,tok=f.load_model_and_tokenizer();model.eval()
 for p in model.parameters():p.requires_grad_(False)
 check(all(p.device.type=='cuda' and p.device.index==0 for p in model.parameters()),'Device map/offload changed')
 import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
 check(pathlib.Path(qmod.__file__).resolve()==ARCH.resolve(),'Live architecture changed')
 check(pathlib.Path(inspect.unwrap(qmod.torch_recurrent_gated_delta_rule).__code__.co_filename).resolve()==ARCH.resolve(),'Live recurrent helper changed')
 qmod.torch_chunk_gated_delta_rule=numerical().chunk_binding(qmod,'compensated')
 return model,qmod,dict(load_seconds=time.monotonic()-start,tokenizer_class=type(tok).__name__,parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),architecture_sha256=sha(ARCH),recurrent_source=inspect.unwrap(qmod.torch_recurrent_gated_delta_rule).__code__.co_filename)
def states(cache,f):return [f.get_state(cache,l) for l in f.GDN_LAYERS]
def nested_tensors(value,prefix):
 if isinstance(value,torch.Tensor):yield prefix,value
 elif isinstance(value,dict):
  for key in sorted(value,key=str):yield from nested_tensors(value[key],prefix+'/'+str(key))
 elif isinstance(value,(list,tuple)):
  for j,t in enumerate(value):yield from nested_tensors(t,prefix+'/'+str(j))
def cache_tensors(cache):
 for i,l in enumerate(cache.layers):
  for name in ('keys','values','conv_states','recurrent_states'):
   yield from nested_tensors(getattr(l,name,None),str(i)+'/'+name)

def cache_hash(cache):return hashlib.sha256(json.dumps([(n,list(t.shape),str(t.dtype),tsha(t)) for n,t in cache_tensors(cache)],sort_keys=True).encode()).hexdigest()
def gpu_memory():return dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
