#!/usr/bin/env python3
"""Read-only frozen AST replay; no candidate, model, CUDA or research data."""
import ast,dataclasses,hashlib,json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import torch
BASE=Path('/Users/pytenter/Documents/临时项目1')
OUT=BASE/'qwen_gdn_dual_m_equivalent_speedup_v1/audit_reference'
P=BASE/'qwen_gdn_dual_axis_prospective_replication_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
DEP=BASE/'qwen_gdn_c6r_dual_m_interaction_reuse_5doc_v1/frozen_source_dependencies'
ADAPTER=P/'src/codec.py'
ORIGINAL=DEP/'cd494824c242b2ce967dc470b50160aff25cf1776e1d45344d547a30fb6253e6_dual_axis_codec.py'
OFFICIAL=DEP/'0c39fe6da2b5ad68e2a081471b1ec22465a9c9f914c367ed984a78ad3717ee69_stepquant_quantization.py'
EXPECTED={ADAPTER:'ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f',ORIGINAL:'cd494824c242b2ce967dc470b50160aff25cf1776e1d45344d547a30fb6253e6',OFFICIAL:'0c39fe6da2b5ad68e2a081471b1ec22465a9c9f914c367ed984a78ad3717ee69'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def nodes(p,names):
 t=ast.parse(p.read_text()); ns=[n for n in t.body if getattr(n,'name',None) in names]
 assert {n.name for n in ns}==set(names)
 return ns
def run_ast(p,names,ns):exec(compile(ast.Module(body=nodes(p,names),type_ignores=[]),str(p),'exec'),ns)
def thash(t):return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def bitsame(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
for p,h in EXPECTED.items():assert sha(p)==h,(p,'SOURCE_SHA_MISMATCH')
NAMES=('C128_CANONICAL','C128_FP16_SCALE','C128_LS_FP16','DUAL_M_INT8')
a=dict(torch=torch,dataclass=dataclasses.dataclass,NAMES=NAMES,SYNC_COUNT=0,SCALE_TRACE=None)
run_ast(ADAPTER,['require','source','EncodedState'],a)
e=dict(torch=torch,CANON={},NAMES=NAMES,LOW=2**-24,HIGH=65504.,source=a['source'],EncodedState=a['EncodedState'])
run_ast(ORIGINAL,['safe_half','levels','encode'],e)
a.update(_original_safe=e['safe_half'],_original_levels=e['levels'],encode=e['encode'])
run_ast(ADAPTER,['safe_half','levels','trace_encode'],a)
e.update(safe_half=a['safe_half'],levels=a['levels'])
f=dict(torch=torch,dataclass=dataclasses.dataclass)
run_ast(OFFICIAL,['QuantizationPlan','_fp16_scale','fit_state'],f)
torch.set_num_threads(1);torch.manual_seed(20261017)
shape=(1,32,128,128)
cases=[('zero_B1',torch.zeros(shape)),('constant_positive_B1',torch.full(shape,.25)),('constant_negative_B1',torch.full(shape,-.25)),('random_normal_B1',torch.randn(shape)),('random_normal_B2',torch.randn((2,32,128,128))),('outlier_B1',torch.randn(shape)*.01),('noncontiguous_B1',torch.randn(shape).transpose(-1,-2)),('tiny_sub_fp16_scale_B1',torch.randn(shape)*1e-24),('large_clipped_scale_B1',torch.randn(shape)*1e16),('signed_tie_and_zero_B1',torch.tensor([-.0,.0,-127.5,-126.5,-2.5,-1.5,-.5,.5,1.5,2.5,126.5,127.5],dtype=torch.float32).repeat((524288+11)//12)[:524288].reshape(shape))]
cases[5][1][0,0,0,0]=1e8
results=[]
for name,x in cases:
 start=time.monotonic(); before=thash(x);enc,raw=a['trace_encode'](x,'DUAL_M_INT8')
 plan=f['QuantizationPlan']('gdn',torch.full((32,128),8,dtype=torch.long),torch.ones((32,128)))
 z,r,c,two,y=f['fit_state'](x,plan)
 dequant=enc.decode(); zero_pair=(y==0)&(dequant==0); diag={'torch_equal_dequant':torch.equal(y,dequant),'different_bits':int((y.contiguous().view(torch.int32)!=dequant.contiguous().view(torch.int32)).sum()),'signed_zero_differences':int((zero_pair&(torch.signbit(y)!=torch.signbit(dequant))).sum()),'nonzero_value_differences':int(((y!=dequant)&(~zero_pair)).sum()),'max_absolute_value_difference':float((y-dequant).abs().max())};checks={'codes_exact':torch.equal(z,enc.codes.to(torch.int16)),'row_scale_fp16_bits_exact':bitsame(r,enc.rows.squeeze(-1)),'column_scale_fp16_bits_exact':bitsame(c,enc.columns.squeeze(-2)),'dequant_fp32_bits_exact':bitsame(y,enc.decode()),'pivots_absent':two is None,'input_unmodified':before==thash(x),'signed_int8_no_minus128':enc.codes.dtype==torch.int8 and not bool((enc.codes==-128).any()),'three_original_raw_scale_stages':len(raw)==3,'owned_codes':enc.codes._base is None,'scales_finite_positive':bool((torch.isfinite(enc.rows)&(enc.rows>0)).all()&(torch.isfinite(enc.columns)&(enc.columns>0)).all()),'decode_finite':bool(torch.isfinite(enc.decode()).all())}
 results.append({'name':name,'scope':'SYNTHETIC_NOT_REAL_GDN_STATE','shape':list(x.shape),'noncontiguous':not x.is_contiguous(),'input_sha256':before,'checks':checks,'dequant_difference_diagnostics':diag,'status':'PASS' if all(checks.values()) else 'FAIL','seconds':time.monotonic()-start})
# Direct scalar-contract tests exercise tie-even and scale edges, not generated model states.
levels_in=torch.tensor([-127.5,-126.5,-2.5,-1.5,-.5,.5,1.5,2.5,126.5,127.5])
levels_expected=torch.tensor([-127.,-126.,-2.,-2.,-0.,0.,2.,2.,126.,127.])
levelcheck=torch.equal(e['levels'](levels_in),levels_expected)
edge=torch.tensor([-float('inf'),-1.,0.,2**-25,2**-24,65504.,65505.,float('inf')])
edge_expected=edge.clamp(2**-24,65504.).to(torch.float16)
scalecheck=bitsame(e['safe_half'](edge),edge_expected)
negative=[]
for name,x in [('nan',torch.full((1,1,1,1),float('nan'))),('inf',torch.full((1,1,1,1),float('inf'))),('empty',torch.empty(1,1,0,1)),('integer',torch.ones(1,1,1,1,dtype=torch.long))]:
 try:a['encode'](x,'DUAL_M_INT8');negative.append({'name':name,'rejected':False})
 except (RuntimeError,ValueError,FloatingPointError) as exc:negative.append({'name':name,'rejected':True,'exception':type(exc).__name__,'reason':str(exc)})
report={'task_id':'QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1','status':'PASS' if all(r['status']=='PASS' for r in results) and levelcheck and scalecheck and all(x['rejected'] for x in negative) else 'FAIL','scope':'CPU_SYNTHETIC_REFERENCE_ONLY_NO_CANDIDATE_NO_MODEL_NO_GPU','torch_version':torch.__version__,'python':sys.version,'platform':sys.platform,'CUDA_initialized':torch.cuda.is_initialized(),'source_files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in EXPECTED],'AST_reuse':'Original encode/safe_half/levels plus exact device-adapter interfaces/tracing; frozen official QuantizationPlan/_fp16_scale/fit_state. No rederived codec formula. CANON empty because C128 branches never dispatched.','comparison':'Actual frozen official all8 w=1 fit_state; exact INT8 codes, FP16 scale bytes and FP32 dequant bytes; not StateCodec.decode association.','cases':results,'scalar_contracts':{'round_ties_even_signed_saturation':levelcheck,'safe_half_floor_ceiling_infinities':scalecheck},'invalid_input_rejections':negative,'limitations':['All fixtures synthetic CPU tensors, including B1/B2/noncontiguous. No real GDN fixture locally available.','Mac arm64 CPU parity does not prove CUDA reduction parity, launch count, throughput, runtime carrier or optimized-candidate equivalence.','No GPU or model initialization performed.']}
report['historical_contract_status']='PASS' if all(r['dequant_difference_diagnostics']['torch_equal_dequant'] and r['dequant_difference_diagnostics']['nonzero_value_differences']==0 and all(v for k,v in r['checks'].items() if k!='dequant_fp32_bits_exact') for r in results) and levelcheck and scalecheck and all(x['rejected'] for x in negative) else 'FAIL';report['official_byte_contract']='NOT_BYTE_EXACT_SIGNED_ZERO_ONLY' if report['historical_contract_status']=='PASS' and report['status']=='FAIL' else report['status'];report['prior_attempt_path']='REFERENCE_CPU_CODEC_TESTS.json';(OUT/'REFERENCE_CPU_CODEC_DIAGNOSTICS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'cases':len(results),'checks':sum(len(r['checks']) for r in results)+2+len(negative),'CUDA_initialized':report['CUDA_initialized'],'result_sha256':sha(OUT/'REFERENCE_CPU_CODEC_DIAGNOSTICS.json')},ensure_ascii=False))
assert report['historical_contract_status']=='PASS'
