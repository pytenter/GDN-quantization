#!/usr/bin/env python3
"""AST and CPU fakes only; never import Torch, controller, model or GPU libraries."""
import ast
import hashlib
import json
import math
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
R=Path('/Users/pytenter/Documents/临时项目1/QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1')
O=R/'analysis/protocol'

def run():
 worker=R/'worker/restoration_worker.py'
 source=worker.read_text();source_sha=hashlib.sha256(source.encode()).hexdigest()
 tree=ast.parse(source)
 arrays=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='arrays')
 class DType:
  def __init__(self,name):self.name=name
  def __str__(self):return self.name
 class Device:
  def __init__(self,kind='cuda',index=0):self.type=kind;self.index=index
  def __str__(self):return self.type+(':'+str(self.index) if self.index is not None else '')
 class Storage:
  def __init__(self,p):self.p=p
  def data_ptr(self):return self.p
 class Tensor:
  def __init__(self,name,p):self.name=name;self.device=Device();self.p=p
  def untyped_storage(self):return Storage(self.p)
 class FakeTorch:
  dtype=DType
  device=Device
  long=DType('torch.int64')
  @staticmethod
  def tensor(value,dtype=None):return Tensor('input_ids',999)
 FakeTorch.Tensor=Tensor
 class Cache:
  pass
 class Layer:
  pass
 Cache.__module__='transformers.cache_utils';Layer.__module__='transformers.cache_utils'
 namespace={'torch':FakeTorch,'ids':[1]*32768}
 exec(compile(ast.Module(body=[arrays],type_ignores=[]),'<cache-array-cpu-fake>','exec'),namespace)
 array=namespace['arrays']
 def fixture():
  c=Cache();c.offloading=False;c.layer_class_to_replicate=None;c.layers=[]
  L=Layer();L.keys=Tensor('keys',100);L.values=Tensor('values',101);L.conv_states={0:Tensor('conv',102)};L.recurrent_states={0:Tensor('state',103)}
  L.is_conv_states_initialized={0:True};L.is_recurrent_states_initialized={0:True};L.has_previous_state={0:True};L.conv_kernel_size={0:4};L.record_past=False;L.dtype=DType('torch.float32');L.device=Device()
  c.layers.append(L)
  return c
 class Out:
  logits=Tensor('logits',998)
 checks=[]
 def check(name,ok,**kw):
  assert ok,name
  checks.append(dict(check=name,status='PASS',**kw))
 c=fixture();v,m,a=array(c,Out(),0);names=[n for n,x in v]
 check('dict_recurrent_and_conv_state_tensors_selected',{'cache/layers/0/conv_states/int:0','cache/layers/0/recurrent_states/int:0'}<=set(names),selected_names=names)
 check('metadata_JSON_roundtrip_identical',m==json.loads(json.dumps(m)))
 check('cache_flags_dtype_device_in_metadata',all(k in m for k in ['cache/offloading','cache/layers/0/record_past','cache/layers/0/device','cache/layers/0/dtype','cache/layers/0/has_previous_state/int:0']))
 before=m.copy();c.layers[0].record_past=True;_,after,_=array(c,Out(),0)
 check('metadata_change_is_observable',before!=after and before['cache/layers/0/record_past'] is False and after['cache/layers/0/record_past'] is True)
 c=fixture();c.self_reference=c;_,m,_=array(c,Out(),0)
 check('cycle_reference_recorded',m['cache/self_reference']=={'reference_to':'cache'})
 c=fixture();c.layers[0].values.p=c.layers[0].keys.p;_,_,aliases=array(c,Out(),0)
 check('shared_tensor_storage_alias_recorded',['cache/layers/0/keys','cache/layers/0/values'] in aliases)
 c=fixture();c.unknown=object()
 try:array(c,Out(),0)
 except RuntimeError as e:rejected='Unsupported cache metadata field' in str(e)
 else:rejected=False
 check('unknown_cache_metadata_fails_closed',rejected)
 c=fixture();c.extra_dict={1:'b',0:'a'};_,m1,_=array(c,Out(),0);c.extra_dict={0:'a',1:'b'};_,m2,_=array(c,Out(),0)
 check('dictionary_insertion_order_canonicalized',m1==m2)
 arch=R/'candidate/source_tree/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py'
 at=ast.parse(arch.read_text());cl=next(n for n in at.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5GatedDeltaNet');fr=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 calls=[n for n in ast.walk(fr) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']]
 check('actual_helper_kwargs_match_adapter',len(calls)==2 and all({'g','beta','initial_state','use_qk_l2norm_in_kernel'}<={k.arg for k in n.keywords} for n in calls))
 check('FP64_oracle_carried_per_layer',"oracle_states[idx]=S" in source and "else oracle_states[idx]" in source)
 check('native_normalization_separately_bounded',"normalization_atol" in source and "normalization_scale_rtol" in source and "source.detach().cpu().double()" in source)
 check('historical_benchmark_source_unchanged',hashlib.sha256((R/'candidate/benchmark_replay/source/matched_worker.py').read_bytes()).hexdigest()=='08b62839e88fa6bb49f10388a19e152e04beb0ab36b347ffcfdf2d9259e2ff8c')
 check('GDN_raw_minimum_3GiB_for_two_32_token_runs',2*32*24*32*128*128*4==3*1024**3,raw_GDN_bytes_only=3221225472)

 # Execute only the pure CPU raw-diff adapter with a tiny stdlib tensor substitute.
 class Vector:
  def __init__(self,values):self.v=list(values)
  def double(self):return self
  def __sub__(self,other):return Vector(a-b for a,b in zip(self.v,other.v))
  def __truediv__(self,other):return Vector(a/b for a,b in zip(self.v,other.v))
  def abs(self):return Vector(abs(a) for a in self.v)
  def clamp_min(self,value):return Vector(max(a,value) for a in self.v)
  def max(self):return max(self.v)
  def __getitem__(self,k):return self.v[k]
 class RawTorch:
  float32='float32';bfloat16='bfloat16'
  @staticmethod
  def frombuffer(data,dtype):
   if dtype=='float32':return Vector(x[0] for x in struct.iter_unpack('<f',data))
   if dtype=='bfloat16':return Vector(struct.unpack('<f',struct.pack('<I',x[0]<<16))[0] for x in struct.iter_unpack('<H',data))
   raise AssertionError(dtype)
  @staticmethod
  def empty(shape,dtype):
   class Empty:
    def element_size(self):return 4 if dtype=='float32' else 2
   return Empty()
 diffnode=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='difference')
 diffns={'torch':RawTorch,'math':math}
 exec(compile(ast.Module(body=[diffnode],type_ignores=[]),'<raw-diff-cpu-fake>','exec'),diffns)
 diff=diffns['difference']
 def desc(data,dtype='float32'):
  return {'name':'tiny','dtype':'torch.'+dtype,'shape':[len(data)//(4 if dtype=='float32' else 2)],'stride':[1],'offset':0,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 with tempfile.TemporaryDirectory(prefix='restoration_worker_review_') as tmp:
  rp=Path(tmp)/'ref.bin';ap=Path(tmp)/'actual.bin'
  d0=struct.pack('<ff',0.,1.);d1=struct.pack('<ff',-0.,1.25);rp.write_bytes(d0);ap.write_bytes(d1)
  ref={'tensors':[desc(d0)],'cache_metadata':{'flag':False},'tensor_storage_aliases':[]}
  actual={'tensors':[desc(d1)],'cache_metadata':{'flag':False},'tensor_storage_aliases':[]}
  got=diff(ref,actual,rp,ap);first=got['first_tensor_difference']
  check('raw_diff_preserves_signed_zero_byte_failure',first['kind']=='ACTUAL_BYTES_DIFFER' and first['first_element']==0 and first['reference_value']==first['actual_value']==0 and math.copysign(1,first['actual_value'])==-1)
  check('raw_diff_reports_actual_maxabs_after_earliest_byte',first['max_absolute']==.25 and first['max_relative']==.25)
  actual={'tensors':[desc(d0)],'cache_metadata':{'flag':True},'tensor_storage_aliases':[]};ap.write_bytes(d0)
  got=diff(ref,actual,rp,ap)
  check('metadata_only_change_is_not_reported_as_tensor_numeric_diff',got['first_tensor_difference'] is None and not got['cache_metadata_equal'])
  actual={'tensors':[dict(desc(d0),stride=[2])],'cache_metadata':{'flag':False},'tensor_storage_aliases':[]}
  got=diff(ref,actual,rp,ap)
  check('layout_change_reported_separately',got['first_tensor_difference']['kind']=='LAYOUT')
  d0=struct.pack('<HH',0x3f80,0x4000);d1=struct.pack('<HH',0x3f81,0x4000);rp.write_bytes(d0);ap.write_bytes(d1)
  ref={'tensors':[desc(d0,'bfloat16')],'cache_metadata':{},'tensor_storage_aliases':[]};actual={'tensors':[desc(d1,'bfloat16')],'cache_metadata':{},'tensor_storage_aliases':[]}
  got=diff(ref,actual,rp,ap)
  check('BF16_raw_scalar_decode_has_expected_value',got['first_tensor_difference']['max_absolute']==.0078125 and got['first_tensor_difference']['reference_value']==1 and got['first_tensor_difference']['actual_value']==1.0078125)
  ap.write_bytes(b'')
 # The real worker's GDN alias gate uses full byte intervals, not merely start pointers.
 intervals=[(100,200,0),(150,250,1)]
 check('overlapping_contiguous_state_views_rejected_by_interval_rule',not all(x[1]<=y[0] for x,y in zip(intervals,intervals[1:])))
 check('no_top_level_GPU_engine_import',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any((getattr(alias,'name','') or '').split('.')[0] in ['torch','transformers','pynvml'] for alias in getattr(n,'names',[])) for n in tree.body))
 main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 permit_line=next(n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='require_worker_permit')
 torch_line=next(n.lineno for n in ast.walk(main) if isinstance(n,ast.Import) and any(x.name=='torch' for x in n.names))
 check('parent_permit_before_torch_import',permit_line<torch_line)
 check('first_failure_metrics_and_raw_saved_before_raise',"EARLIEST_NUMERICAL_FAILURE.json" in source and "FIRST_FAILURE_" in source and "normalization_reference_FP64" in source and "state_reference_FP64" in source)
 check('alias_gate_checks_contiguous_full_address_intervals',"s.is_contiguous()" in source and "s.data_ptr()+s.numel()*s.element_size()" in source and "x[1]<=y[0]" in source)
 check('numeric_scope_explicit_prefix32_not_long_or_INT8',"N==32" in source and "32_TOKEN_PREFIX_ONLY_NOT_32K_81K_OR_INT8" in source)

 # Execute only the signature selection/control statements; no file/maps/engine I/O.
 lib=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='library_check')
 def assigns(n,name):return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
 start=next(i for i,n in enumerate(lib.body) if assigns(n,'signature'))
 end=next(i for i,n in enumerate(lib.body) if assigns(n,'receipt'))
 signature_code=compile(ast.Module(body=lib.body[start:end],type_ignores=[]),'<same-phase-signature-cpu>','exec')
 def signature_run(unit,phase,reference):
  reads=[]
  def readref(p):reads.append(str(p));return {'critical_selection_signature':reference}
  ns={'selected':[{'path':'/lib/B','sha256':'b'},{'path':'/lib/A','sha256':'a'},{'path':'/lib/A','sha256':'a'}], 'a':SimpleNamespace(unit=unit),'r':Path('/candidate'),'phase':phase,'read':readref}
  exec(signature_code,ns)
  return ns,reads
 expected=[{'path':'/lib/A','sha256':'a'},{'path':'/lib/B','sha256':'b'}]
 n,reads=signature_run('NUMERIC_FRESH0','PREFLIGHT',None)
 check('library_signature_canonical_dedup_sorted',n['signature']==expected)
 check('first_numeric_signature_does_not_read_own_missing_reference',n['selection_matches'] and n['reference_signature'] is None and not reads)
 n,reads=signature_run('NUMERIC_FRESH1','PREFLIGHT',expected)
 check('second_numeric_requires_matching_same_phase_library_signature',n['selection_matches'] and reads==['/candidate/execution/NUMERIC_FRESH0/PREFLIGHT_LIBRARY_IDENTITY.json'])
 n,reads=signature_run('OLD_FP_R0','POSTFLIGHT',[{'path':'/lib/A','sha256':'CHANGED'}])
 check('OLD_benchmark_changed_bound_library_selection_rejected',not n['selection_matches'] and reads==['/candidate/execution/NUMERIC_FRESH0/POSTFLIGHT_LIBRARY_IDENTITY.json'])
 n,reads=signature_run('NEW_FP_R0','POSTFLIGHT',None)
 check('NEW_signature_not_forced_equal_to_OLD',n['selection_matches'] and n['reference_signature'] is None and not reads)
 receipt_node=lib.body[end]
 check('mapped_library_match_and_signature_both_required_for_PASS',all(x in ast.unparse(receipt_node) for x in ['not missing',"x['match']",'selection_matches']))

 previous=O/'review_history/worker_1c67f7ef_v1/WORKER_REVIEW_FINAL_SOURCE.py.txt'
 class HideLibraryCheck(ast.NodeTransformer):
  def visit_FunctionDef(self,node):
   if node.name=='library_check':node.body=[ast.Pass()];return node
   return self.generic_visit(node)
 old_ast=HideLibraryCheck().visit(ast.parse(previous.read_text()))
 new_ast=HideLibraryCheck().visit(ast.parse(source))
 check('only_library_check_AST_changed_since_previous_review',ast.dump(old_ast,include_attributes=False)==ast.dump(new_ast,include_attributes=False))
 check('worker_source_stable_during_review',hashlib.sha256(worker.read_bytes()).hexdigest()==source_sha)
 result={'status':'PASS_CPU_FAKE_AND_AST_CHECKS_NOT_GPU_QUALIFICATION','reviewed_worker_path':str(worker),'reviewed_worker_sha256':source_sha,'CPU_checks':checks,'CPU_checks_passed':len(checks),'GPU_dispatches':0,'GPU_queries':0,'limitations':'AST/fake tests validate data selection and control properties only. No audited module imported; no Torch/GPU numerical correctness, real cache implementation, performance or device release is established.'}
 out=O/'WORKER_PREPARATION_CPU_MOCKS.json'
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'worker_sha256':source_sha,'CPU_checks':len(checks)}))
if __name__=='__main__':run()
