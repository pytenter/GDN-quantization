"""A single planned 16384 checkpoint/64-step overlap; frozen math is unchanged."""
from pathlib import Path
import ast,copy,gc,hashlib,os,pickle,random,sys
from cleanroom_common import D,W,read,publish,sha
METRICS=('scale','codes','injection','kl','state','refs','memory','suffix_parity_pass','runtime')

def rng_hash(torch):
 import numpy as np
 return hashlib.sha256(pickle.dumps((torch.get_rng_state().numpy().tobytes(),torch.cuda.get_rng_state().cpu().numpy().tobytes(),random.getstate(),np.random.get_state()),protocol=4)).hexdigest()

def atomic_torch_save(torch,value,path):
 path=Path(path);tmp=path.with_name(path.name+'.partial');assert not tmp.exists()
 with tmp.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
 if path.exists():
  # A planned 64-step overlap may encounter one prewritten FP reference.
  # It must have identical full tensor/metadata bytes; retain the original file.
  import functional_worker as fw
  previous=torch.load(path,map_location='cpu',weights_only=False)
  assert fw.object_fingerprint(torch,previous)==fw.object_fingerprint(torch,value),'PLANNED_FP_REFERENCE_REPLAY_MISMATCH'
  tmp.unlink()
 else:os.link(tmp,path);tmp.unlink()

class FormalControl:
 def __init__(self,document,condition,segment):
  assert segment in [1,2];self.document=document;self.condition=condition;self.segment=segment
  self.start=8192 if segment==1 else 16384;self.end=16448 if segment==1 else 32768
  self.key=document+'__'+condition;self.cp=D/'checkpoints'/(self.key+'__16384.pt')
  self.proof=D/'segments'/(self.key+'__overlap64.json');self.digest=hashlib.sha256();self.restored=None;self.long_exact_pass=False
 def identity(self):
  routing=read(D/'EXECUTION_ROUTING.json')
  return dict(document=self.document,condition=self.condition,UUID=os.environ['LING_TARGET_GPU_UUID'],source_commit=read(D/'SOURCE_CODE_FREEZE.json')['source_commit'],source_manifest_sha256=sha(D/'SOURCE_CODE_FREEZE.json'),frozen_spec_sha256=sha(D/'FROZEN_SPEC.json'),parent_manifest_sha256=sha(D/'runs'/(self.key+'__parent.json')),prefix8192_sha256=sha(D/'runs'/(self.key+'__8192.pt')),model=read(D/'FROZEN_SPEC.json')['model'])
 def initial(self,torch,original_cp):
  if self.segment==1:return torch.load(original_cp,map_location='cpu',weights_only=False)
  index=read(self.cp.with_suffix('.json'));assert index['sha256']==sha(self.cp)
  value=torch.load(self.cp,map_location='cpu',weights_only=False);assert value['identity']==self.identity(),'LONG_CHECKPOINT_IDENTITY_CHANGED'
  native=value.pop('native');assert native['next_position']==16384 and native['frozen_spec_sha256']==sha(D/'FROZEN_SPEC.json')
  self.restored=value;return native
 def restore_metrics(self,torch,values,sample_bytes):
  if self.segment==1:return values,sample_bytes
  assert self.restored is not None
  saved=self.restored['metrics'];assert saved['suffix_parity_pass'] is True
  assert [x['position'] for x in saved['kl']]==self.restored['processed_score_positions']
  assert sum(x['KL'] for x in saved['kl'])==self.restored['prefix_Python_sum']
  saved['runtime']=saved['runtime']+values['runtime']
  if self.condition=='FP':
   import functional_worker as f
   for ref in saved['refs']:
    assert sha(ref['path'])==ref['sha256']
    if ref['logits']:
     payload=torch.load(ref['path'],map_location='cpu',weights_only=False)
     sample_bytes.update(ref['position'].to_bytes(8,'little'));sample_bytes.update(f.engine.tensor_bytes(torch,payload['logits']))
  return saved,sample_bytes
 def observe(self,torch,position,logits):
  if 16384<=position<16448:
   import functional_worker as f
   self.digest.update(position.to_bytes(8,'little'));self.digest.update(f.engine.tensor_bytes(torch,logits))
 def after_token(self,position,local):
  import functional_worker as f
  torch=local['torch'];cache=local['cache'];audit=local['audit']
  if position==16383:
   assert self.segment==1 and local['suffix_parity_pass'] and local['deferred'] is None
   native=f.checkpoint(torch,cache,audit,self.condition,position,local['cfg'],sha(D/'FROZEN_SPEC.json'),f.sha(local['token_path']),None if self.condition=='FP' else local['rotation_ids'][self.condition]['sha256'],None)
   metrics={k:local[k] for k in METRICS}
   payload=dict(schema='cleanroom_planned_full_native_checkpoint_v1',native=native,identity=self.identity(),metrics=metrics,processed_score_positions=[x['position'] for x in local['kl']],prefix_Python_sum=sum(x['KL'] for x in local['kl']),cursor=16384,original_prefix_full_logit_hash='NOT_COLLECTED_AFTER8192_NOT_INVENTED',reducer='original list of KL samples, final original Python sum/384')
   atomic_torch_save(torch,payload,self.cp)
   publish(self.cp.with_suffix('.json'),dict(status='PASS_ATOMIC_CHECKPOINT_SAVED',sha256=sha(self.cp),bytes=self.cp.stat().st_size,cursor=16384,identity=self.identity()))
   del payload,native;gc.collect()
  if position==16447:
   value=dict(logits64_sha256=self.digest.hexdigest(),native_cache_sha256=f.object_fingerprint(torch,cache),audit_state_sha256=None if self.condition=='FP' else f.object_fingerprint(torch,f.audit_state(audit,self.condition)),RNG_sha256=rng_hash(torch),identity=self.identity(),positions=[16384,16448],checkpoint_sha256=sha(self.cp))
   if self.segment==1:publish(self.proof,dict(status='PASS_UNINTERRUPTED_OVERLAP_COLLECTED',comparison=value))
   else:
    assert read(self.proof)['comparison']==value,'EXACT_PLANNED_LONG_CHECKPOINT_REPLAY_FAILURE'
    self.long_exact_pass=True
    publish(D/'segments'/(self.key+'__long_resume_exact.json'),dict(status='PASS_EXACT_64_LOGITS_NATIVE_CACHE_AUDIT_RNG',comparison=value))
 def finish_first(self):
  assert self.proof.exists() and self.cp.exists()
  publish(D/'segments'/(self.key+'__part1.json'),dict(status='PASS',document=self.document,condition=self.condition,logical_saved_cursor=16384,actual_endpoint=16448,overlap_replayed_for_qualification=64,checkpoint_sha256=sha(self.cp),overlap_proof_sha256=sha(self.proof),identity=self.identity(),formal_complete=False))

def adapted_source():
 original=(W/'formal_worker.py').read_text();s=original
 replacements=[
  ("root/(tag+'.STARTED.json')","root/(tag+f'.part{CR.segment}.STARTED.json')"),
  ("refdir.mkdir(); fp_manifest = None","refdir.mkdir(exist_ok=CR.segment==2); fp_manifest = None"),
  ("saved = torch.load(cp,map_location='cpu',weights_only=False)","saved = CR.initial(torch,cp)"),
  ("assert saved['next_position'] == 8192 and saved['token_sha256'] == w.sha(token_path)","assert saved['next_position'] == CR.start and saved['token_sha256'] == w.sha(token_path)"),
  ("        deferred = None\n        def collect_metrics", "        deferred = None\n        control_values,sample_bytes = CR.restore_metrics(torch,{k:locals()[k] for k in ()},sample_bytes)\n        def collect_metrics"),
  ("for position in range(8192,32768):","for position in range(CR.start,CR.end):"),
  ("            del logits\n        assert suffix_parity_pass and deferred is None", "            CR.observe(torch,position,logits)\n            del logits\n            CR.after_token(position,locals())\n        if CR.segment == 1:\n            CR.finish_first()\n            return\n        assert CR.long_exact_pass\n        assert suffix_parity_pass and deferred is None"),
 ]
 # Explicit names avoid the separate scope of a locals() comprehension.
 restored="        control_values,sample_bytes = CR.restore_metrics(torch,dict(scale=scale,codes=codes,injection=injection,kl=kl,state=state,refs=refs,memory=memory,suffix_parity_pass=suffix_parity_pass,runtime=runtime),sample_bytes)\n"
 restored+="        scale,codes,injection,kl,state,refs,memory,suffix_parity_pass,runtime = [control_values[k] for k in CR_METRICS]\n"
 for old,new in replacements:
  assert s.count(old)==1,('FROZEN_CONTROL_INSERTION_BOUNDARY_CHANGED',old);s=s.replace(old,new)
 s=s.replace("        control_values,sample_bytes = CR.restore_metrics(torch,{k:locals()[k] for k in ()},sample_bytes)\n",restored)
 # Revert every registered edit and require the entire frozen source string.
 reverse=s.replace(restored,"        control_values,sample_bytes = CR.restore_metrics(torch,{k:locals()[k] for k in ()},sample_bytes)\n")
 for old,new in reversed(replacements):reverse=reverse.replace(new,old)
 assert reverse==original,'UNREGISTERED_FORMAL_MATH_SOURCE_EDIT'
 ast.parse(s)
 return s

def execute(document,condition,segment):
 cr=FormalControl(document,condition,segment)
 scope=dict(__name__='__main__',__file__=str(W/'formal_worker.py'),CR=cr,CR_METRICS=METRICS)
 # Atomic payload orchestration only. The original torch serialization
 # function is restored even if the scientific worker raises.
 import torch
 saved_torch_save=torch.save
 def atomic(value,path,*args,**kw):
  # Our atomic helper uses a file object; preserve the original serializer.
  if hasattr(path,'write') or not Path(path).resolve().is_relative_to(D):return saved_torch_save(value,path,*args,**kw)
  return atomic_torch_save(torch,value,path)
 torch.save=atomic
 try:exec(compile(adapted_source(),str(W/'formal_worker.py'),'exec'),scope)
 finally:torch.save=saved_torch_save
