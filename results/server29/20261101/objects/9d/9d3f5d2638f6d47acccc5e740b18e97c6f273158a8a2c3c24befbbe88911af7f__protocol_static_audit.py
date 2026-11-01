#!/usr/bin/env python3
"""Local stdlib-only source/protocol audit. Does not import audited modules."""
from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone
B = Path('/Users/pytenter/Documents/临时项目1')
ROOT = B/'QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1'
OUT = ROOT/'analysis/protocol'
H = B/'qwen_gdn_historical_throughput_reconciliation_v1'
S = H/'snapshots'
D = H/'delivery'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def snap(digest, name): return S/digest/name
paths = {
 'old_runner': snap('9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd','run_qwen_aime26_formal.py'),
 'new_formal':snap('77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69','formal_c128_runner.py'),
 'architecture':snap('90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a','modeling_qwen3_5.py'),
 'new_core':snap('aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575','core.py'),
 'new_worker':snap('e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773','worker.py'),
 'baseline_worker':snap('0961d96bdafa92d6c3bca7d1980b74e9d70fe3aabff404417878ced20c7f1e41','baseline_worker.py'),
 'numerical_recovery':snap('b13152f81ecd87428936dc5eced5a2a9f1e06ff992e8ba5bbb81703206dec26c','numerical_recovery.py'),
 'new_codec':snap('ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f','codec.py'),
 'matched_worker':H/'source/matched_worker.py',
 'cache_utils':D/'dependency_snapshot/by_path_sha256/82797af7e16219ccb9a1456710f327f2a4af85e5a372e6ad4c729944c9313019/cache_utils.py',
 'historical_manifest':D/'SOURCE_SHA256_MANIFEST.json',
 'historical_scan_policy':D/'dependency_snapshot/by_path_sha256/733984c97bcfad4f19f9126785197c7baccb075aba0b9297c8db0bfe4cb70570/QUALIFICATION_POLICY.json',
 'new_numerical_contract':D/'dependency_snapshot/by_path_sha256/89e9dad5a6b5fb4134ae9c87b21a798a539427027d2d25d349bb4dbab8416c22/NEW_NUMERICAL_CONTRACT.json',
 'prior_fast_cpu_contract':B/'QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1/analysis/STRICT_BASELINE_CONTRACT.json',
 'user_request':ROOT/'USER_REQUEST.txt',
}
for unit in ['OLD_FP_R0','NEW_FP_R0','OLD_H_C128_R0','NEW_H_C128_R0']:
 for kind in ['RESULT','ENVIRONMENT']:
  paths[unit+'_'+kind]=D/'execution'/unit/(kind+'.json')
rows={k:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for k,p in paths.items()}
for key in ['old_runner','new_formal','architecture','new_core','new_worker','baseline_worker','numerical_recovery','new_codec']:
 assert sha(paths[key])==paths[key].parent.name,(key,'snapshot SHA mismatch')
assert rows['matched_worker']['sha256']=='08b62839e88fa6bb49f10388a19e152e04beb0ab36b347ffcfdf2d9259e2ff8c'
assert rows['cache_utils']['sha256']=='b63732985b520a62aa4b06fc2af8ca6e506912679e0f0479b706534d041fb1bd'
manifest=read(paths['historical_manifest'])
pinned={r['sha256'] for r in manifest['execution_sources']+manifest['external_dependencies']}
for key in ['old_runner','new_formal','architecture','new_core','new_worker','baseline_worker','numerical_recovery','new_codec','matched_worker','cache_utils']:
 assert rows[key]['sha256'] in pinned,(key,'not bound by historical manifest')

def node(key,name): return next(n for n in ast.parse(paths[key].read_text()).body if getattr(n,'name',None)==name)
def loc(key,name):
 n=node(key,name)
 return {'source_id':key,'lines':[n.lineno,n.end_lineno]}
def claim(key,text,evidence,qualification='SOURCE_VERIFIED'):
 return {'id':key,'fact':text,'evidence':evidence,'evidence_type':qualification}
asts=[]
for name in ['hadamard','normalize_qk','QwenKeyHadamardPatch','get_state','quantize_c128','load_model_and_tokenizer','setup_seed']:
 a,b=node('old_runner',name),node('new_formal',name)
 ah=hashlib.sha256(ast.dump(a,include_attributes=False).encode()).hexdigest()
 bh=hashlib.sha256(ast.dump(b,include_attributes=False).encode()).hexdigest()
 assert ah==bh
 asts.append({'function':name,'AST_identical':True,'AST_sha256':ah,'old':loc('old_runner',name),'new':loc('new_formal',name),'limitation':'Function AST identity is not orchestration/protocol/device-output equivalence.'})
old_h=read(paths['OLD_H_C128_R0_RESULT']);new_h=read(paths['NEW_H_C128_R0_RESULT'])
assert old_h['QDQ_calls']==255 and new_h['QDQ_calls']==256
assert all(read(paths[x+'_RESULT'])['scope']=='MATCHED_NON_AIME_PERFORMANCE_NOT_NUMERICAL_EQUIVALENCE' for x in ['OLD_FP_R0','NEW_FP_R0','OLD_H_C128_R0','NEW_H_C128_R0'])
claims=[
 claim('P02','Original OLD AIME generate_one uses a whole-prompt model call with an all-ones attention_mask. It samples the first output from unquantized prefill cache. Only each following cached decode is followed by C128 QDQ before the next decode; prefill_quantized=False.',[loc('old_runner','generate_one')]),
 claim('P03','Matched OLD_H_C128 applies QDQ only at t>0 (255 applications); matched NEW_H_C128 applies it also at t=0 (256 applications). Identical measured indices do not repair the distinct warmup-cache lineage.',[loc('matched_worker','main'),{'source_id':'OLD_H_C128_R0_RESULT'},{'source_id':'NEW_H_C128_R0_RESULT'}],'SOURCE_AND_HISTORICAL_RECORD_VERIFIED'),
 claim('P04','The newer prospective worker token-steps every input from an empty cache and performs H_C128 or Dual-M QDQ after every model forward, including token0. Saved current logits precede this outer QDQ and quantify the current transition driven by earlier cache; next-token effects begin on the next forward.',[loc('new_worker','run')]),
 claim('P05','OLD architecture chunk fallback uses native g.cumsum(dim=-1); NEW core installs numerical().chunk_binding(...,compensated), changing exactly that scan site to FP32 Neumaier compensated scan. The recurrent helper body is not replaced by this binding.',[loc('architecture','torch_chunk_gated_delta_rule'),loc('architecture','torch_recurrent_gated_delta_rule'),loc('numerical_recovery','chunk_binding'),loc('numerical_recovery','compensated_scan'),loc('new_core','load')]),
 claim('P06','Architecture converts q,k,v,beta,g to FP32 inside GDN and returns FP32 recurrent state; current readout casts to the original query dtype. Historical matched records observe actual state shape[1,32,128,128] FP32 for both chunk and recurrent; BF16 model parameters are shared.',[loc('architecture','torch_chunk_gated_delta_rule'),loc('architecture','torch_recurrent_gated_delta_rule'),{'source_id':'OLD_FP_R0_RESULT'},{'source_id':'NEW_FP_R0_RESULT'}],'SOURCE_AND_HISTORICAL_RECORD_VERIFIED'),
 claim('P07','The model obtains initial_state from cache.layers[layer].recurrent_states[0], and update_recurrent_state copies each new state into the persistent cache buffer. C128 then overwrites that buffer with dequantized values in place. Source data flow supports next-token QDQ consumption, but old matched records did not retain a per-token byte-consumption proof.',[{'source_id':'architecture','lines':[505,535]},{'source_id':'cache_utils','lines':[1072,1086]},loc('old_runner','quantize_c128')]),
 claim('P08','Hadamard patch normalizes q/k when requested, then transforms each by float32 H128/sqrt(128) immediately before both chunk and recurrent GDN core; v is unchanged. OLD calls construct H per transform; NEW formal(True) memoizes the constant H matrix by shape/dtype/device.',[loc('old_runner','hadamard'),loc('old_runner','QwenKeyHadamardPatch'),loc('new_core','formal')]),
 claim('P09','C128 source uses source=state.detach().float(); scale=max_K(abs(source)).clamp_min(1e-12)/127; codes=round(source/scale).clamp(-127,127); dequant=codes*scale; state.copy_(dequant.to(state.dtype)). For state[B,32,128,128], scale is[B,32,1,128] in FP32. Codes here remain floating-point temporaries; persistent state is FP32 QDQ carrier, not a packed INT8 runtime.',[loc('old_runner','quantize_c128'),loc('new_formal','quantize_c128')]),
 claim('P10','Visible C128/OLD cache source stores one reconstructed recurrent buffer per GDN layer, without a separate persistent unquantized state. Transient source/dequant variables are not evidence of a persistent shadow. Actual alias graph and decorated/hub-selected implementation remain a future execution gate.',[loc('old_runner','quantize_c128'),{'source_id':'cache_utils','lines':[1019,1030]},{'source_id':'cache_utils','lines':[1072,1086]}]),
 claim('P11','NEW Observe checks each updated state finite, verifies initial-state pointer against the prior QDQ cache, and optionally checks byte SHA. OLD matched wrapper checks actual state shape/dtype and full-logit finite; it does not implement the same state-finite or QDQ pointer/byte gate. Adding such gates in a separate numerical qualification is necessary, and their overhead must not enter the historical TPS window.',[loc('new_worker','Observe'),loc('matched_worker','main')]),
 claim('P12','A successful NEW compensated-route contract explicitly leaves old_historical_parity=FAIL_UNCHANGED and new_matched_panel_only=True. It cannot qualify restored native OLD or supply an OLD Future-KL teacher.',[{'source_id':'new_numerical_contract'}],'HISTORICAL_RECORD_VERIFIED'),
]
claims=[x for x in claims if x]
claims.insert(0,claim('P01','Historical matched comparison is 256 fixed-input forwards: token0 one-token prefill; tokens1..255 cached; warmup0..127; measurement128..255. All four RESULT scope fields explicitly state performance, not numerical equivalence.',[loc('matched_worker','main')]+[{'source_id':u+'_RESULT'} for u in ['OLD_FP_R0','NEW_FP_R0','OLD_H_C128_R0','NEW_H_C128_R0']],'SOURCE_AND_HISTORICAL_RECORD_VERIFIED'))
env={u:read(paths[u+'_ENVIRONMENT']) for u in ['OLD_FP_R0','NEW_FP_R0','OLD_H_C128_R0','NEW_H_C128_R0']}
fixture=read(paths['prior_fast_cpu_contract'])['fixture']
policy=read(paths['historical_scan_policy'])
unknown=[
 'No historical OLD full per-token logits/state/cache byte reference has been established by the inspected matched RESULTs; their scope explicitly disclaims numerical equivalence.',
 'Historical OLD actual loaded DSO maps and complete optional hub/FLA fallback selection were not captured here. Shared architecture source/versions do not prove historical binary or actual kernel implementation identity.',
 'Native CUDA cumsum or any other nondeterministic operation current repeatability/long-horizon error is NOT_RUN in this CPU-only task. Source presence is not measured nondeterminism.',
 'Restored OLD safety against uninitialized/illegal alias reads and real next-token consumption is NOT_RUN. No deterministic-fill disabling is authorized or justified.',
 'Future group32 axis, scale dtype/epsilon/rounding/range, asymmetric zero-point convention/storage/rounding, and all actual codec implementations are not frozen or numerically qualified by this audit.',
 'Current restored FP/H/G32 TPS, VRAM peaks, actual math fidelity and fresh-process repeatability are NOT_RUN; earlier short measurements do not establish current performance.',
 'One-token-prefill 256-forward qualification would not qualify multi-token prompt chunk semantics, 32K/81K trajectories, Future-KL math resolution, or any stochastic AIME protocol.',
]
future={
 'status':'PREREGISTRATION_RECOMMENDATION_ONLY_NOT_EXECUTABLE_AUTHORIZATION',
 'ALLOW_GPU_RESTORE':'NO',
 'first_stage':'FP_ONLY',
 'required_new_human_authorization':'ALLOW_GPU_RESTORE=YES',
 'device':'one exclusive RTX3090, actual UUID frozen before dispatch; no automatic four-card validation',
 'fixed_input':fixture,
 'schedule_recommendation':[
  {'unit':'OLD_FP_NUMERIC_FRESH0','forwards':256,'prefill_tokens':1,'cached_tokens':255,'maximum_dispatch_reservation_seconds':600,'math_check':'Capture all 24 actual GDN helper calls/intermediates and cache boundaries, full248320 logits and input/cursor identity each token; apply independently frozen FP32 math/oracle tests, finite and alias/consumption gates. Hash/export outside benchmark only.'},
  {'unit':'OLD_FP_NUMERIC_FRESH1','forwards':256,'prefill_tokens':1,'cached_tokens':255,'maximum_dispatch_reservation_seconds':600,'comparison':'Independent process, identical device/source/runtime/input; compare each token logits/all recurrent+conv+attention cache tensors and metadata, update counts, consumption identity; report exact bytes plus earliest numeric mismatch. No OLD-vs-NEW byte requirement.'},
  {'unit':'OLD_FP_TPS_R0','warmup':128,'measure':128,'maximum_dispatch_reservation_seconds':120},
  {'unit':'OLD_FP_TPS_R1','warmup':128,'measure':128,'maximum_dispatch_reservation_seconds':120},
  {'unit':'OLD_FP_TPS_R2','warmup':128,'measure':128,'maximum_dispatch_reservation_seconds':120},
 ],
 'resource_recommendation':{'hard_reservation_seconds':1800,'hard_GPUh':0.5,'protective_stop_seconds':1650,'cleanup_reserve_seconds':150,'mandatory_sum_unit_caps_seconds':1560,'optional_NEW_compare':'Only one separately frozen same-input NEW FP128+128 unit if still fits total protective budget; not needed if budget insufficient. Historical NEW7.776711 remains historical-only reference.'},
 'performance_gates':['All numerical/finite/cache/identity gates pass first; no performance after failure.','Independent fresh process each repeat; preserve actual OLD code path and historical common counters/hooks or explicitly label any instrumentation difference.','Same BF16 model and FP32 state, fixed input and context; only CUDA sync immediately before and after128 measured forwards. Retain OLD original finite gates.','No profiler, extra per-token tensor export/hash or safety-deletion in timed windows; model load, prefill setup, tensor saves and CPU scoring excluded.','Report all repeat walls/TPS and median;13.0TPS is engineering goal,12..13 partial, not numerical significance. Do not tune after observing results.'],
 'mathematical_contract':'Freeze independent equation/reference policy and raw source before any future GPU. Prior policy83e6432 caps are available only as explicit prospectively adopted reference, not inherited native-OLD PASS. Original prior oracle was deterministic compensated-vs-rounded FP64 scan at multi-token prefill, not old-native qualification. Do not call old-vs-new parity a fidelity oracle.',
 'available_prior_policy':{'source_id':'historical_scan_policy','sha256':rows['historical_scan_policy']['sha256'],'scan_error_rule':policy['scan_error_rule'],'model_caps':policy['full_model']['caps'],'adoption_status':'NOT_ADOPTED_BY_THIS_CPU_AUDIT'},
 'fresh_gate':'Strict byte repeatability of this restored protocol required for formal deterministic scientific use; if it fails, preserve exact earliest difference and numerical magnitude, mark runtime reproducibility fail rather than relax tolerance. Mathematical correctness and repeatability are separate gates.',
 'historical_reference_gate':'Only compare to actual saved OLD per-token reference if identified and bound; otherwise historical exact-byte equivalence remains UNKNOWN.',
 'long_horizon_limit':'No long32K/81K, AIME or quantized GPU work authorized. Native cumsum multi-token prefill must receive its own future numerical qualification if that use is proposed.'
}
compat={
 'status':'CPU_SOURCE_FEASIBILITY_ONLY_INT8_NOT_QUALIFIED',
 'H_C128_Sym':{'source_math':'RECOVERED_AST_IDENTICAL','source_writeback':'RECOVERED_IN_PLACE_FP32_QDQ','actual_restored_GPU_qualification':'NOT_RUN','historical_protocol':'Unquantized full prompt prefill, QDQ after cached decode; matched window has equivalent t0-skip lineage.'},
 'H_C32_Sym':{'implementation':'NOT_DEFINED_OR_QUALIFIED_BY_THIS_AUDIT','required_frozen_items':['Exact32-element grouping axis (K axis if intended) and4-group layout, including reshape/contiguity','Scale dimensions/dtype/epsilon and recomputation frequency','Code integer range, round ties, clamp/saturation andzero handling','Post-update same-position QDQ/cache-copy timing andnext-consumption proof']},
 'H_C32_Asym':{'implementation':'NOT_DEFINED_OR_QUALIFIED_BY_THIS_AUDIT','required_frozen_items':['All symmetric-group layout/timing items','Exact code signedness andrange','Scale/zero-point formula, clamp, integer zero-point/storage dtype androunding','All-zero/constant group behavior, finite positive scales andlegal codes/zero-points','No retained unquantized state or fallback that bypassesQDQ']},
 'common_future_panel':['FP Native on the same restored runtime/protocol','H+C128 Sym newly run on that same protocol','H+C32 Sym newly run on that same protocol','H+C32 Asym newly run on that same protocol'],
 'must_match':['Model/weight/config/tokenizer SHA','Fixed input anddoc/cursor identities','Fixed Hadamard definition/placement for the H arms','GDN native-vs-compensated numerical path andbackend','One-token/multi-token prefill semantics andmask','Outer QDQ timing includingwhether prefill cache isQDQ','Cache lifetime andresume semantics','BF16/FP32 math/storage andactual selected libraries','ForAIME:prompt/sampler/seed/EOS/scorer','ForFutureKL:FP teacher trajectory,positions/full-vocabularyFP32KL/reducer/resolution anddoc-levelstatistics'],
 'rebuild_required':'If OLD restored protocol is adopted, generate a new matched FP teacher and all quantized trajectories for future comparisons. Historical NEW_STRICT Future-KL/C6R/C7/H/Dual-M numerical values may be descriptive references only, never injected as matched OLD-arm measurements.',
 'prohibited_inferences':['FP13TPS impliesINT8speed','OLDH9.064610 iscurrentrestoredHmeasurement','SameC128AST makesOLD/NEWFutureKLcomparable','NewcompensatedfreshPASS qualifiesnativeOLD','ShortsameGPUbytePASS qualifieslonghorizon oranotherGPU','FPqualification qualifiesH/C32/Asymwithoutactualcodecgates'],
}
result={'task_id':'QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1','status':'CPU_ONLY_PROTOCOL_AUDIT_COMPLETE_GPU_NOT_AUTHORIZED','created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Local stdlib AST/source and saved metadata only; no imported audited modules, SSH, torch, CUDA, GPUquery or dispatch.','new_GPU_dispatches':0,'new_GPU_queries':0,'source_evidence':rows,'AST_comparisons':asts,'claims':claims,'historical_effective_environment_records':env,'unknowns_and_limits':unknown,'INT8_compatibility':compat,'future_minimal_preregistration_recommendation':future,'auditor_script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))}}
OUT.mkdir(parents=True,exist_ok=True)
p=OUT/'PROTOCOL_COMPATIBILITY_AUDIT.json'
with p.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':result['status'],'source_rows':len(rows),'AST_equalities':len(asts),'claims':len(claims),'output_sha256':sha(p),'bytes':p.stat().st_size}))
