"""Independent synthetic CPU-only analysis fixtures. Does not read science outputs.

Only writes the named CPU_ANALYSIS_TESTS audit JSON; never calls seal().
"""
import ast
import copy
import datetime
import hashlib
import importlib
import math
import sys
import traceback
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
W=HERE.parent
D=Path('/data/zypan/experiment_runs/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1')
sys.path.insert(0,str(HERE))
import metrics_cpu as M
import analysis as A
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
SOURCE_PATHS=[HERE/'metrics_cpu.py',HERE/'analysis.py',W/'persistent_engine_v1.py']
SOURCE_BEFORE={str(x):sha(x) for x in SOURCE_PATHS}
UUID='CPU_FIXTURE_UUID_NOT_A_GPU_DEVICE'
DOC='CPU_SYNTHETIC_DOC'
SCOPE='same_FP32_logits_scoring_only_not_forward_error_bound'
results=[]
def check(name,fn):
 try:
  detail=fn();results.append({'name':name,'status':'PASS','detail':detail})
 except BaseException as e:
  results.append({'name':name,'status':'FAIL','type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
def rejected(fn):
 try:fn()
 except (AssertionError,KeyError,TypeError,ValueError):return {'rejected':True}
 raise AssertionError('INVALID_INPUT_ACCEPTED')
def value(k,r=1e-10):return {'K':float(k),'arithmetic_error_radius':float(r)}
def values(h=8,hd=7,l=6,ld=3,r=1e-10):return {'FP_NATIVE':value(0,r),'H_R128':value(h,r),'H_DUAL_M':value(hd,r),'L6_R128':value(l,r),'L6_DUAL_M':value(ld,r)}
def document(i,v):return {'document':'CPU_SYNTHETIC_DOC_'+str(i),'complete_pairing':len(v)==5,'metrics':A.derive(v)}
def score_fixture(k=.25,condition='H_R128'):
 rows=[{'position':8192+i,'kl_fp32':float(k),'kl_reference_extended':float(k),'kl_reference_extended_decimal':str(k),'arithmetic_error_radius':1e-10,'reference_rounding_allowance':1e-20,'vocabulary_size':157184,'extended_reference_mantissa_bits':int(np.finfo(np.longdouble).nmant),'negative_fp32_value_preserved':k<0,'arithmetic_scope':SCOPE} for i in range(256)]
 return {'status':'PASS','kind':'formal','document':DOC,'condition':condition,'UUID':UUID,'begin':0,'endpoint':8448,'scores':rows,'aggregate':M.aggregate(rows)}
def mutate(field,change,condition='H_R128'):
 r=score_fixture(condition=condition);change(r)
 return rejected(lambda:A.score_result(r,DOC,condition,UUID))

def ast_contract():
 left=ast.parse((W/'persistent_engine_v1.py').read_text());right=ast.parse((HERE/'metrics_cpu.py').read_text())
 original=next(x for x in left.body if isinstance(x,ast.FunctionDef) and x.name=='full_kl')
 copied=next(x for x in right.body if isinstance(x,ast.FunctionDef) and x.name=='original_full_kl')
 strip=lambda f:[x for x in f.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
 assert ast.dump(ast.Module(body=strip(original),type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=strip(copied),type_ignores=[]),include_attributes=False)
 assert ast.dump(original.args)==ast.dump(copied.args)
 return {'original_full_kl_AST_operation_sequence':'EXACT','engine_path':str(W/'persistent_engine_v1.py'),'production_math_changed':False}
check('original_FP32_KL_body_AST_exact_to_frozen_engine',ast_contract)

def fullv_actual():
 a=np.full(157184,-50,dtype=np.float32);b=a.copy();a[:2]=[2,-1];b[:2]=[-1,1]
 out=M.score_pair(a,b);reverse=M.score_pair(b,a);self_out=M.score_pair(a,a)
 expected=M.original_full_kl(torch,torch.from_numpy(a).reshape(1,1,-1),torch.from_numpy(b).reshape(1,1,-1))
 assert out['kl_fp32']==expected and out['vocabulary_size']==157184
 assert abs(out['kl_fp32']-reverse['kl_fp32'])>1e-3
 assert self_out['kl_fp32']==self_out['kl_reference_extended']==0
 assert abs(out['kl_fp32']-out['kl_reference_extended'])<=out['arithmetic_error_radius']
 assert out['arithmetic_scope']==SCOPE and out['extended_reference_mantissa_bits']>53
 return {'KL_FP32_P_to_Q':out['kl_fp32'],'reverse_KL_FP32_Q_to_P':reverse['kl_fp32'],'fullV':157184,'same_logit_reference_difference_covered':True,'scope':SCOPE,'forward_uncertainty_bound_claim':False}
check('actual_fullV_CPU_FP32_KL_direction_and_independent_extended_reference',fullv_actual)
check('production_scoring_CPU_selftests',M.cpu_tests)

def reduction():
 rows=[{'kl_fp32':x,'kl_reference_extended':x,'arithmetic_error_radius':0.0} for x in ([1e16,1.0,-1e16]+[0.0]*253)]
 out=M.aggregate(rows)
 assert out['K']==math.fsum(x['kl_fp32'] for x in rows)/256==1/256
 assert out['K']!=sum(x['kl_fp32'] for x in rows)/256
 assert out['position_count']==256 and out['reducer']=='math.fsum(FP32_fullV_KL_values)/256'
 return {'synthetic_cancellation_fixture':True,'K_fsum':out['K'],'naive_Python_sum':sum(x['kl_fp32'] for x in rows)/256}
check('exact256_fsum_reducer_not_naive_sum',reduction)
check('aggregate_rejects255',lambda:rejected(lambda:M.aggregate([{'kl_fp32':0.0,'kl_reference_extended':0.0,'arithmetic_error_radius':0.0}]*255)))
check('aggregate_rejects257',lambda:rejected(lambda:M.aggregate([{'kl_fp32':0.0,'kl_reference_extended':0.0,'arithmetic_error_radius':0.0}]*257)))

def signs():
 out=A.derive(values())
 expected={'Delta_L6':3.0,'Delta_H':1.0,'Interaction':2.0,'LearnedGain_R128':2.0,'LearnedGain_DualM':4.0}
 assert all(out[k]['value']==v and out[k]['label']=='POSITIVE' for k,v in expected.items())
 assert out['Delta_L6']['relative_gain']==.5
 assert out['Delta_H']['relative_gain']==1/8
 assert out['actual_best_student_by_point_estimate']=='L6_DUAL_M'
 return expected
check('all_five_scientific_metric_signs_and_relative_gains',signs)

def six_positive():
 out=A.summarize([document(i,values()) for i in range(6)])
 assert out['classification']=='PRIMARY_DIRECTIONALLY_REPLICATED' and out['complete_six_document_pairing']
 assert out['metrics']['Delta_L6']['positive']==6 and out['metrics']['Delta_L6']['missing']==0
 assert out['project_independence']=='UNKNOWN' and out['confirmatory_sign_test_p'] is None
 assert out['metrics']['Delta_L6']['mean']==out['metrics']['Delta_L6']['median']==3
 assert out['no_token_layer_head_pseudoreplication']
 return out
check('six_of_six_primary_positive_only_directional_no_confirmatory_p',six_positive)

def six_negative():
 out=A.summarize([document(i,values(h=3,hd=4,l=2,ld=5)) for i in range(6)])
 assert out['classification']=='PRIMARY_DIRECTIONALLY_NEGATIVE'
 assert out['metrics']['Delta_L6']['negative']==6 and out['confirmatory_sign_test_p'] is None
 return out
check('six_of_six_primary_negative',six_negative)

def five_positive():
 docs=[document(i,values()) for i in range(5)]+[document(5,values(h=3,hd=4,l=2,ld=5))]
 out=A.summarize(docs)
 assert out['classification']=='MIXED_OR_NUMERICALLY_UNRESOLVED'
 assert out['metrics']['Delta_L6']['positive']==5 and out['metrics']['Delta_L6']['negative']==1
 assert out['complete_six_document_pairing'] and out['confirmatory_sign_test_p'] is None
 return out
check('five_of_six_primary_positive_not_replicated',five_positive)

def missing_condition():
 missing=values();missing.pop('L6_DUAL_M')
 doc=document(5,missing);out=A.summarize([document(i,values()) for i in range(5)]+[doc])
 assert out['classification']=='INCOMPLETE' and not out['complete_six_document_pairing']
 p=out['metrics']['Delta_L6'];assert p['paired_documents']==5 and p['positive']==5 and p['missing']==1
 assert doc['metrics']['Delta_L6'] is None and doc['metrics']['Interaction'] is None
 assert doc['metrics']['LearnedGain_DualM'] is None and doc['metrics']['actual_best_student_by_point_estimate'] is None
 assert p['mean']==p['median']==3.0
 return out
check('missing_sixth_L6_DualM_primary_paired_n5_not_zero_or_tie',missing_condition)

def only_five_docs():
 out=A.summarize([document(i,values()) for i in range(5)])
 assert out['classification']=='INCOMPLETE' and out['metrics']['Delta_L6']['paired_documents']==5 and out['metrics']['Delta_L6']['missing']==1
 return out
check('only5_complete_documents_incomplete_planned_n6',only_five_docs)

def unresolved():
 v=values(h=1,hd=1,l=1,ld=1,r=1e-6)
 out=A.summarize([document(i,v) for i in range(6)])
 assert out['classification']=='MIXED_OR_NUMERICALLY_UNRESOLVED'
 assert all(out['metrics'][k]['numerically_unresolved']==6 for k in A.METRICS)
 assert A.derive(v)['best_separation']=='NUMERICALLY_UNRESOLVED'
 v2=values(l=1+1e-10,ld=1,r=1e-6)
 assert A.derive(v2)['Delta_L6']['value']>0 and A.derive(v2)['Delta_L6']['label']=='NUMERICALLY_UNRESOLVED'
 return out
check('zero_and_tiny_effects_unresolved_not_positive_or_negative',unresolved)

def no_primary_best_inference():
 v=values(h=1,hd=.5,l=5,ld=4)
 out=A.derive(v);summary=A.summarize([document(i,v) for i in range(6)])
 assert out['Delta_L6']['label']=='POSITIVE' and out['Interaction']['label']=='POSITIVE'
 assert out['actual_best_student_by_point_estimate']=='H_DUAL_M'
 assert out['LearnedGain_DualM']['label']=='NEGATIVE'
 assert summary['classification']=='PRIMARY_DIRECTIONALLY_REPLICATED'
 return {'primary_positive':True,'interaction_positive':True,'actual_best':'H_DUAL_M','L6_DualM_absolutely_best':False}
check('primary_and_positive_interaction_do_not_imply_L6_DualM_absolute_best',no_primary_best_inference)

def zero_baseline():
 out=A.derive(values(h=0,hd=.2,l=0,ld=.1))
 assert out['Delta_L6']['relative_gain'] is None and out['Delta_H']['relative_gain'] is None
 assert out['Delta_L6']['relative_gain_status']=='UNDEFINED_NONPOSITIVE_BASELINE'
 return {'no_divide_by_zero':True,'relative_gain':None}
check('nonpositive_baseline_relative_gain_undefined',zero_baseline)

check('legal256_score_result_accepted',lambda:A.score_result(score_fixture(),DOC,'H_R128',UUID))
check('legal_FP_native_zeros_accepted',lambda:A.score_result(score_fixture(k=0,condition='FP_NATIVE'),DOC,'FP_NATIVE',UUID))
check('reject_score_count255',lambda:mutate('scores',lambda r:r['scores'].pop()))
check('reject_score_count257',lambda:mutate('scores',lambda r:r['scores'].append(copy.deepcopy(r['scores'][-1]))))
check('reject_shifted_scoring_positions',lambda:mutate('positions',lambda r:[x.update(position=x['position']+1) for x in r['scores']]))
check('reject_duplicate_scoring_position',lambda:mutate('positions',lambda r:r['scores'][1].update(position=8192)))
check('reject_reversed_scoring_order',lambda:mutate('positions',lambda r:r['scores'].reverse()))
check('reject_saved_wrong_K',lambda:mutate('K',lambda r:r['aggregate'].update(K=123.0)))
check('reject_saved_wrong_reference_K',lambda:mutate('reference_K',lambda r:r['aggregate'].update(reference_K=123.0)))
check('reject_saved_wrong_reducer',lambda:mutate('reducer',lambda r:r['aggregate'].update(reducer='Python_sum/256')))
check('reject_saved_wrong_position_count',lambda:mutate('position_count',lambda r:r['aggregate'].update(position_count=255)))
check('reject_saved_wrong_arithmetic_scope',lambda:mutate('scope',lambda r:r['aggregate'].update(arithmetic_scope='full_model_forward_bound')))
check('reject_saved_wrong_radius',lambda:mutate('radius',lambda r:r['aggregate'].update(arithmetic_error_radius=1.0)))
check('reject_wrong_per_position_vocabulary_size',lambda:mutate('vocabulary_size',lambda r:r['scores'][0].update(vocabulary_size=12)))
check('reject_wrong_per_position_scoring_scope',lambda:mutate('scope',lambda r:r['scores'][0].update(arithmetic_scope='full_model_forward_bound')))
check('reject_negative_row_radius',lambda:mutate('radius',lambda r:r['scores'][0].update(arithmetic_error_radius=-1e-3)))
check('reject_nonfinite_FP32_KL',lambda:mutate('finite',lambda r:r['scores'][0].update(kl_fp32=float('nan'))))
check('reject_nonfinite_extended_reference',lambda:mutate('finite',lambda r:r['scores'][0].update(kl_reference_extended=float('inf'))))
check('reject_boolean_as_numeric_KL',lambda:mutate('type',lambda r:r['scores'][0].update(kl_fp32=True)))
check('reject_wrong_document',lambda:mutate('document',lambda r:r.update(document='OTHER_SYNTHETIC')))
check('reject_wrong_UUID',lambda:mutate('UUID',lambda r:r.update(UUID='OTHER_CPU_FIXTURE_UUID')))
check('reject_wrong_condition',lambda:mutate('condition',lambda r:r.update(condition='L6_DUAL_M')))
check('reject_borrowed_prefix_begin8192',lambda:mutate('begin',lambda r:r.update(begin=8192)))
check('reject_short_endpoint8447',lambda:mutate('endpoint',lambda r:r.update(endpoint=8447)))
check('reject_nonformal_kind',lambda:mutate('kind',lambda r:r.update(kind='qualification')))
check('reject_status_notPASS',lambda:mutate('status',lambda r:r.update(status='INCOMPLETE')))
check('reject_FP_native_nonzero_KL',lambda:rejected(lambda:A.score_result(score_fixture(k=.1,condition='FP_NATIVE'),DOC,'FP_NATIVE',UUID)))

# These are explicit input-contract boundary tests, not real science evidence.
def reject_inconsistent_radius():
 r=score_fixture();r['scores'][0]['kl_reference_extended']=1.0;r['aggregate']=M.aggregate(r['scores'])
 return rejected(lambda:A.score_result(r,DOC,'H_R128',UUID))
check('reject_reference_difference_outside_declared_arithmetic_radius',reject_inconsistent_radius)
def reject_FP_extended_nonzero():
 r=score_fixture(k=0,condition='FP_NATIVE');r['scores'][0]['kl_reference_extended']=1e-12;r['scores'][0]['arithmetic_error_radius']=1e-8;r['aggregate']=M.aggregate(r['scores'])
 return rejected(lambda:A.score_result(r,DOC,'FP_NATIVE',UUID))
check('reject_FP_native_nonzero_extended_self_KL',reject_FP_extended_nonzero)

# Read only frozen data metadata, never raw scientific results or model states.
def validation_freeze():
 f=__import__('json').loads((D/'FROZEN_DOCUMENTS.json').read_text())
 p=__import__('json').loads((D/'audits/DATA_SELECTION_PROTOCOL.json').read_text())
 assert f['source_dataset']=='PG19' and f['source_split']=='validation' and not f['RepoBench_P_used']
 assert len(f['formal_documents'])==6 and len({x['document'] for x in f['formal_documents']+[f['canary_document']]})==7
 assert all(x['source_split']=='validation' and x['document'].startswith('pg19-validation-') and x['full_token_count']>=8449 for x in f['formal_documents']+[f['canary_document']])
 assert p['eligibility']['excluded_all_previous_test100_even_if_CPU_only'] and p['seed']==0
 assert f['PROJECT_INDEPENDENCE']=='UNKNOWN'
 return {'dataset':'PG19 validation','formal6_plus_independent_canary':True,'old_test100_excluded':True,'project_independence':'UNKNOWN','science_scores_accessed':False}
check('strict_validation_split_frozen_data_identity_not_old_test100',validation_freeze)

SOURCE_AFTER={str(x):sha(x) for x in SOURCE_PATHS}
checks_pass=all(x['status']=='PASS' for x in results)
report={'TASK_ID':'LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1','status':'PASS_CPU_ANALYSIS_TESTS' if checks_pass and SOURCE_BEFORE==SOURCE_AFTER and not torch.cuda.is_initialized() else 'FAIL_CPU_ANALYSIS_REVIEW','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_CPU_cases':len(results),'passed':sum(x['status']=='PASS' for x in results),'failed':sum(x['status']=='FAIL' for x in results),'results':results,'CPU_ONLY':True,'GPU_initialized':torch.cuda.is_initialized(),'model_forwards':0,'actual_scientific_result_files_read':False,'production_sources_modified':False,'production_sources_before':SOURCE_BEFORE,'production_sources_after':SOURCE_AFTER,'source_stable_during_tests':SOURCE_BEFORE==SOURCE_AFTER,'test_source':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'review_notes':['Initial score_result review found missing reducer/reference_K/position_count/fullV/scope checks; root applied metadata-contract correction before this fixture execution, without changing scientific formulas.','Arithmetic radius is scoring arithmetic on the identical FP32 logits; not a model-forward uncertainty bound.','All aggregate/derive/summarize examples are explicitly synthetic CPU fixtures, not scientist results or GPU qualification.'],'not_tested_boundaries':['GPU worker natural EXIT/resource receipt/source/profile binding will be added by root and is outside these pure functions.','Real model logits, real formal trajectories and model-forward equivalence are NOT tested.','No final seal() filesystem path exercised; only pure scoring/derive/summarize input contracts.','Project independence remains UNKNOWN; no confirmatory p-value validated.']}
output=D/'audits/CPU_ANALYSIS_TESTS.json'
with output.open('x') as stream:__import__('json').dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
print(__import__('json').dumps({'status':report['status'],'cases':len(results),'passed':report['passed'],'failed':report['failed'],'failures':[x['name'] for x in results if x['status']=='FAIL'],'source_stable':SOURCE_BEFORE==SOURCE_AFTER,'GPU_initialized':torch.cuda.is_initialized(),'report_sha256':sha(output),'test_source_sha256':sha(__file__)},ensure_ascii=False))
if report['status']!='PASS_CPU_ANALYSIS_TESTS':raise SystemExit(1)
