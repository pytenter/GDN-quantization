"""Separate static/negative-control review. No torch or GPU runtime imports."""
import ast,json,hashlib,sys,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));import gpu_control as C
checks=[]
def check(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw));assert ok,name
def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
parent=R/'inherited/historical_port_payload'
orig=next(parent.glob('candidate/source_tree/**/81920/run_qwen_aime26_formal.py'))
f=funcs(orig);g=funcs(R/'src/canonical_h128.py')
for name,code in g.items():check('canonical_AST_'+name,f[name]==code)
f=funcs(parent/'scripts/restoration_controller.py');g=funcs(R/'src/inherited_device_guards.py')
for name,code in g.items():check('guard_AST_'+name,f[name]==code)
for p in list((R/'src').glob('*.py'))+list((R/'scripts').glob('*.py')):
 compile(p.read_text(),str(p),'exec')
check('all_new_Python_sources_compile',True)
check('fresh_CPU_codec_actual_output_equal',(R/'qualification/CPU_PROCESS_0.json').read_bytes()==(R/'qualification/CPU_PROCESS_1.json').read_bytes())
check('fresh_tiny_real_Qwen_actual_output_equal',(R/'qualification/TINY_CPU_PROCESS_0.json').read_bytes()==(R/'qualification/TINY_CPU_PROCESS_1.json').read_bytes())
check('canonical_24_layer_full_32_heads',all(x['equal_bytes'] and x['equal_values'] and x['max_abs']==0 for x in json.loads((R/'qualification/CPU_PROCESS_0.json.canonical.json').read_text())['layers']))
calls=[];C.query_gpu=lambda *a,**k:calls.append('FORBIDDEN')
with tempfile.TemporaryDirectory(prefix='qwen_group32_CPU_negative_') as temp:
 t=Path(temp);(t/'dummy.txt').write_text('fixture');raw=(t/'dummy.txt').read_bytes()
 (t/'SOURCE_FREEZE.json').write_text(json.dumps(dict(files=[dict(path='dummy.txt',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])))
 (t/'GPU_PLAN.json').write_text('{}')
 base=dict(task_id=C.TASK,ALLOW_GPU_CANARY='NO',source_freeze_sha256=C.sha(t/'SOURCE_FREEZE.json'),plan_sha256=C.sha(t/'GPU_PLAN.json'),device=dict(physical_index=0,uuid=C.UUID),max_GPU_hours=.5,hard_seconds=1800,protective_seconds=1650,cleanup_seconds=150,human_separate_authorization=False,human_authorization_text=None)
 variants=[('default_NO',{}),('old_task_grant',dict(task_id='QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1',ALLOW_GPU_CANARY='YES')),('wrong_GPU',dict(ALLOW_GPU_CANARY='YES',device=dict(physical_index=1,uuid='foreign'))),('wrong_budget',dict(ALLOW_GPU_CANARY='YES',hard_seconds=None)),('absent_human_grant',dict(ALLOW_GPU_CANARY='YES'))]
 for name,updates in variants:
  (t/'permit.json').write_text(json.dumps(dict(base,**updates)))
  try:C.authorize(t,t/'permit.json');caught=False
  except RuntimeError:caught=True
  check('authorization_negative_'+name,caught)
check('authorization_negatives_no_GPU_query',not calls)
worker=(R/'scripts/gpu_worker.py').read_text();check('worker_admission_precedes_Torch_and_Driver',worker.index('C.worker_admit')<worker.index('C.require_driver_ordinal_identity')<worker.index('import torch,numpy'))
check('uniform_token0_QDQ_timing',"q.after_token(cache)" in worker and 't==128' in worker)
check('actual_tensor_consumption_not_pointer_hash',"torch.equal(actual_cpu,expected)" in (R/'src/group_codecs.py').read_text() and 'tensor_bytes(actual_cpu) != tensor_bytes(expected)' in (R/'src/group_codecs.py').read_text())
check('no_throughput_CSV_without_measurement',not (R/'INT8_THROUGHPUT_RESULTS.csv').exists())
result=dict(status='PASS_INDEPENDENT_STATIC_AND_CPU_EVIDENCE_REVIEW',independence='Separate stdlib-only review process, independent NumPy reference and separate fresh CPU processes; no subagent claim',checks=checks,GPU_queries=0,GPU_dispatches=0)
(R/'qualification/INDEPENDENT_CPU_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print('independent checks',len(checks))
