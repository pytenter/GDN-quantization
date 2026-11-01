"""Finite production global layout/barrier/qualification/formal-record CPU checks. GPU proofs mocked explicitly."""
import ast,copy,json,os,shutil,sys,tempfile
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import execution_control as C
import cross_host_contract as X
checks=[]
def ok(name):checks.append(name)
def reject(name,fn):
 try:fn()
 except (RuntimeError,KeyError,TypeError,ValueError,FileNotFoundError):ok('Reject '+name);return
 raise AssertionError('Invalid input admitted: '+name)
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
original_read=C.read;original_process=C.same_process;original_query=C.query_gpu
layout=X.validate_global_layout(R);ok('Actual global38/120 two-host15/15 fixed question layout')
with tempfile.TemporaryDirectory(prefix='g64_crosshost_CPU_mock_only_')as tmp:
 F=Path(tmp)
 for name in ['CROSS_HOST_PLAN.json','GLOBAL_FORMAL_JOB_MANIFEST.json','QUESTION_GPU_ASSIGNMENT.json','AIME26_DATASET_FREEZE.json','HUMAN_GPU_AUTHORIZATION.json','GPU_EXECUTION_PLAN.json','NODE.json','analysis/MODEL_TOKENIZER_FRESH_SHA.json']:
  put(F/name,C.read(R/name))
 jobs=C.read(F/'GLOBAL_FORMAL_JOB_MANIFEST.json')
 for name,mut in [('missing',lambda x:x['jobs'].pop()),('duplicate',lambda x:x['jobs'].append(x['jobs'][0])),('wrong node',lambda x:x['jobs'][0].update(node_id='other')),('wrong host',lambda x:x['jobs'][0].update(compute_host='other')),('wrong UUID',lambda x:x['jobs'][0].update(gpu_uuid='OTHER')),('wrong physical',lambda x:x['jobs'][0].update(gpu=99)),('wrong seed',lambda x:x['jobs'][0].update(seed=2)),('wrong cap',lambda x:x['jobs'][0].update(max_new_tokens=128)),('wrong input',lambda x:x['jobs'][0].update(input_ids_sha256='bad'))]:
  bad=copy.deepcopy(jobs);mut(bad);put(F/'GLOBAL_FORMAL_JOB_MANIFEST.json',bad);reject('global '+name,lambda:X.validate_global_layout(F))
 put(F/'GLOBAL_FORMAL_JOB_MANIFEST.json',jobs)
 for name in ['SOURCE_FREEZE.json','GPU_PERMIT.json']:put(F/name,dict(CPU_FIXTURE_ONLY=True))
 plan=C.read(F/'GPU_EXECUTION_PLAN.json')['qualification_units'];units=[u['id']for u in plan];led=[]
 put(F/'GROUP64_GPU_CODEC_QUALIFICATION.json',dict(units=units))
 put(F/'LOCAL_QUALIFICATION_READY.json',dict(count=19,source_freeze_sha256=C.sha(F/'SOURCE_FREEZE.json'),permit_sha256=C.sha(F/'GPU_PERMIT.json'),qualification_sha256=C.sha(F/'GROUP64_GPU_CODEC_QUALIFICATION.json')))
 for idx,u in enumerate(plan):
  identity=dict(pid=900000+idx,CPU_MOCK_NO_GPU=True);d=F/'execution'/u['id'];led.append(dict(unit=u['id'],identity=identity,exit_returncode=0,release='PASS'))
  if u['kind']=='gpu_codec_micro':result=dict(count=74,checks=[dict(CPU_MOCK=True)]*74)
  elif u['kind']=='group64_persistent_fresh':result=dict(tokens=512,milestones=[dict(tokens=n)for n in [32,128,512]],audit=dict(qdq_layer_updates=24*512,next_consumption_checks=24*511),writebacks_by_layer={str(i):512 for i in range(24)})
  elif u['kind']=='nonAIME_free_generation_fresh':result=dict(status='COMPLETE',generated_token_ids=[1,2],generated_token_count=2,quantizer_audit=dict(full_actual_values_and_bytes_each_next=True),input_ids_sha256='CPU_MOCK',initial_generator_sha256='CPU_MOCK',final_generator_sha256='CPU_MOCK',finish_reason='eos')
  elif u['kind']=='nonAIME_throughput':result=dict(conditions={c:dict(warmup_tokens=128,measured_tokens=256,TPS=1.,observer=False)for c in X.CONDITIONS})
  else:result=dict(cases=[dict(total_context=82293,actual_short_QDQ_tokens=16)]*4,numerical_81K_proof=False)
  w=dict(status='PASS',unit=u['id'],source_freeze_sha256=C.sha(F/'SOURCE_FREEZE.json'),permit_sha256=C.sha(F/'GPU_PERMIT.json'),identity=identity,gpu=C.NODE['physical_gpu'],gpu_uuid=C.NODE['uuid'],result=result);w['payload_sha256']=C.objsha(w)
  for name,value in [('RESULT.json',w),('STARTED.json',dict(identity=identity)),('DRIVER_IDENTITY.json',dict(status='PASS',actual_uuid=C.NODE['uuid'])),('PREFLIGHT_LIBRARY.json',dict(status='PASS')),('POSTFLIGHT_LIBRARY.json',dict(status='PASS')),('TRACE.json',dict(checkpoints=[dict(token=n)for n in [0,31,127,511]]))]:put(d/name,value)
 put(F/'RESOURCE_LEDGER.json',dict(intervals=led));C.same_process=lambda *_:False;C.query_gpu=lambda:dict(apps=[])
 receipt=X.inspect_qualification(F);ok('Actual19 validator consumes production result/TRACE schemas; GPU results are CPU mock fixtures')
 original=C.read(F/'RESOURCE_LEDGER.json')
 for name,mut in [('actualexit minus11',lambda x:x['intervals'][0].update(exit_returncode=-11)),('release unresolved',lambda x:x['intervals'][0].update(release='UNRESOLVED')),('identity mismatch',lambda x:x['intervals'][0].update(identity=dict(pid=1)))]:
  bad=copy.deepcopy(original);mut(bad);put(F/'RESOURCE_LEDGER.json',bad);reject(name,lambda:X.inspect_qualification(F))
 put(F/'RESOURCE_LEDGER.json',original)
 C.same_process=lambda *_:True;reject('worker still alive',lambda:X.inspect_qualification(F));C.same_process=lambda *_:False
 C.query_gpu=lambda:dict(apps=[dict(pid=led[0]['identity']['pid'])]);reject('own NVML unreleased',lambda:X.inspect_qualification(F));C.query_gpu=lambda:dict(apps=[])
 names=layout['nodes'];receipts={}
 for name,node in names.items():receipts[name]=dict(receipt,node_id=name,host=node['host'],gpu_uuid=node['uuid'])
 a=dict(status='PASS_BOTH_HOST19_TRUE_QUALIFICATIONS_ALLOW_FORMAL120_ONCE',global_plan_sha256=C.sha(F/'CROSS_HOST_PLAN.json'),global_jobs_sha256=C.sha(F/'GLOBAL_FORMAL_JOB_MANIFEST.json'),human_grant_sha256=C.sha(F/'HUMAN_GPU_AUTHORIZATION.json'),nodes=receipts)
 X.require_global_formal_admission(F,a);ok('Both19 global barrier positive CPU fixture')
 for name,mut in [('missing peer',lambda x:x['nodes'].pop(next(n for n in names if n!=C.NODE['id']))),('bad plan',lambda x:x.update(global_plan_sha256='bad')),('bad grant',lambda x:x.update(human_grant_sha256='bad')),('bad jobs',lambda x:x.update(global_jobs_sha256='bad')),('only18',lambda x:x['nodes'][C.NODE['id']].update(qualification_count=18)),('not actual exit0',lambda x:x['nodes'][C.NODE['id']].update(all19_actual_exit0_release_PASS=False)),('wrong UUID',lambda x:x['nodes'][C.NODE['id']].update(gpu_uuid='OTHER')),('own source changed',lambda x:x['nodes'][C.NODE['id']].update(source_sha256='0'*64)),('peer model differs',lambda x:x['nodes'][next(n for n in names if n!=C.NODE['id'])].update(model_bytes_identity_sha256='0'*64))]:
  bad=copy.deepcopy(a);mut(bad);reject('barrier '+name,lambda:X.require_global_formal_admission(F,bad))
 # Same original formal validator, explicit source-node UUID, including peer55 under29 CPU.
 for name in ['protocol/sampler.py','protocol/aime26_scorer_v4.py']:p=F/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CPU_ONLY_HASH_FIXTURE')
 for job in jobs['jobs']:
  n=2;total=job['input_tokens']+n-1;q=dict(group_size=64,group_axis='Key',condition=job['condition'],tokens=total,qdq_layer_updates=24*total,next_consumption_checks=24*sum(t>0 and(t in range(1,128)or t%2048==0)for t in range(total)))
  row=dict(status='COMPLETE',nonfinite=False,runtime_error=None,condition=job['condition'],problem_id=job['problem_id'],seed=1,gpu=job['gpu'],gpu_uuid=job['gpu_uuid'],input_ids_sha256=job['input_ids_sha256'],input_tokens=job['input_tokens'],generated_token_ids=[1,2],generated_token_count=n,generated_token_ids_sha256=C.objsha([1,2]),stop_token_ids=[2],finish_reason='eos',actual_model_forward_calls=total,quantizer_audit=q,prompt_layer_QDQ=24*job['input_tokens'],decode_layer_QDQ=24*(n-1),writebacks_by_layer={str(i):total for i in range(24)},sampler_calls=n,sampler_source_sha256=C.sha(F/'protocol/sampler.py'),scorer_source_sha256=C.sha(F/'protocol/aime26_scorer_v4.py'),dataset_freeze_sha256=C.sha(F/'AIME26_DATASET_FREEZE.json'))
  w=dict(status='PASS',unit=job['id'],result=row,source_freeze_sha256=C.sha(F/'SOURCE_FREEZE.json'),permit_sha256=C.sha(F/'GPU_PERMIT.json'));w['payload_sha256']=C.objsha(w)
  X.validate_formal_record(w,job,F,names[job['node_id']])
 ok('All120 explicit-host production formal validator CPU EOS fixtures, no actual AIME generation')
 def altered(**kw):
  b=copy.deepcopy(w);b['result'].update(**kw);b.pop('payload_sha256');b['payload_sha256']=C.objsha(b);return b
 for name,kw in [('wrongUUID',dict(gpu_uuid='FOREIGN')),('wrongGPU',dict(gpu=99)),('wrongseed',dict(seed=2)),('shortlength',dict(finish_reason='length')),('forwardoffby1',dict(actual_model_forward_calls=total+1)),('samplerreset',dict(sampler_calls=3)),('nonfinite',dict(nonfinite=True))]:reject('formal '+name,lambda kw=kw:X.validate_formal_record(altered(**kw),job,F,names[job['node_id']]))
 # Mutations/launch must not silently retry after an SSH lost reply.
 coordinator=(R/'scripts/global_coordinator.py').read_text();assert coordinator.index('remote_launched=True #')<coordinator.index('RPC.query29(launch)')
 assert 'GLOBAL_STOP_REQUEST.json'in(R/'scripts/gpu_controller.py').read_text().split('GPU_RUN_STARTED.json')[0]
 assert 'global_collector.log'in(R/'scripts/collect_peer55_to29_cpu.py').read_text().split('rows.append')[0]
 ok('Production unknown-launch STOP ownership, initial STOP guard, no active collector-log snapshot')
C.read=original_read;C.same_process=original_process;C.query_gpu=original_query
assert 'torch'not in sys.modules
print(json.dumps(dict(status='PASS_CROSS_HOST_PRODUCTION_CPU_CONTRACT',node_id=C.NODE['id'],count=len(checks),checks=checks,GPU_dispatches=0,qualification_credit=0,formal_credit=0,proof_scope='CPU_INPUT_AND_CONTROL_WITH_EXPLICIT_MOCK_GPU_RESULTS_ONLY')))
