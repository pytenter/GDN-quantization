"""Actual native Qwen classes, tiny random CPU model; never 9B/GPU proof."""
import os,sys,json,hashlib
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));sys.path.insert(0,'/data/zypan/transformers-qwen35/src')
import torch
torch.cuda._lazy_init=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('GPU forbidden'))
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from persistent_adapter import PersistentQDQ
from group_codecs import tensor_bytes
assert hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest()=='90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
def sha(x):return hashlib.sha256(tensor_bytes(x)).hexdigest()
def model(dtype):
    torch.random.default_generator.manual_seed(20261019)
    c=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
      num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_num_key_heads=2,linear_num_value_heads=2,
      linear_key_head_dim=128,linear_value_head_dim=128,linear_conv_kernel_dim=4,
      layer_types=['linear_attention','linear_attention','full_attention'],
      rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[2,1,1]})
    c._attn_implementation='sdpa'
    return Q.Qwen3_5ForCausalLM(c).to(dtype=dtype,device='cpu').eval()
traces=[];negative=[]
for dtype in (torch.float32,torch.bfloat16):
 for method in ('B0','B1','B2'):
    m=model(dtype);cache=None;trace=[]
    with torch.inference_mode(),PersistentQDQ(m,method,(0,1),observe=True) as p:
      for i,token in enumerate((3,7,11,13,17,19,23,29)):
        out=m(input_ids=torch.tensor([[token]]),past_key_values=cache,use_cache=True);cache=out.past_key_values
        p.after_token(cache)
        assert bool(torch.isfinite(out.logits).all())
        trace.append(dict(token=i,logits=sha(out.logits),states=[sha(cache.layers[j].recurrent_states[0]) for j in (0,1)]))
      audit=p.audit();assert audit['next_consumption_checks']==14 and audit['qdq_layer_updates']==16
      assert p.h.phase_counts=={'chunk_prefill':2,'recurrent_decode':14}
    traces.append(dict(dtype=str(dtype),method=method,audit=audit,trace=trace))
    # Deliberately corrupt actual cache after legal QDQ; next helper must reject it.
    m=model(dtype)
    with torch.inference_mode(),PersistentQDQ(m,method,(0,1),observe=True) as p:
      out=m(input_ids=torch.tensor([[3]]),use_cache=True);p.after_token(out.past_key_values)
      out.past_key_values.layers[0].recurrent_states[0].add_(.03125)
      try:m(input_ids=torch.tensor([[7]]),past_key_values=out.past_key_values,use_cache=True);detected=False
      except RuntimeError as e:detected='did not consume' in str(e)
      assert detected
      negative.append(dict(dtype=str(dtype),method=method,actual_cache_corruption_detected=True))
result=dict(status='PASS_TINY_REAL_CLASS_CPU_ONLY',scope='TINY_RANDOM_CPU_NOT_9B_NOT_GPU',traces=traces,negative_controls=negative,
    architecture_sha256=hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest(),GPU_queries=0,GPU_dispatches=0,
    H128_unchanged=True,native_FP_math_unchanged=True)
print('RESULT_JSON='+json.dumps(result,sort_keys=True))
