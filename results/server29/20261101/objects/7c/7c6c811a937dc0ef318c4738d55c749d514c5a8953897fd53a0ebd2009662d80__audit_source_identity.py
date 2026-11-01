#!/usr/bin/env python3
"""Read-only, standard-library audit of retained historical small assets. No ML imports."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import statistics
import subprocess

OUT = Path(__file__).resolve().parent
H = Path('/Users/pytenter/Documents/临时项目1/qwen_gdn_historical_throughput_reconciliation_v1/delivery')
S = json.loads((H / 'SOURCE_SHA256_MANIFEST.json').read_text())
D = json.loads((H / 'DELIVERY_ASSET_MANIFEST.json').read_text())

def sha(data):
    return hashlib.sha256(data).hexdigest()

def check(path, row, kind):
    r = dict(row, local_path=str(path), kind=kind)
    if not path.is_file():
        return dict(r, status='MISSING')
    b = path.read_bytes()
    return dict(r, actual_bytes=len(b), actual_sha256=sha(b), status='PASS' if len(b) == row['bytes'] and sha(b) == row['sha256'] else 'MISMATCH')

def dep_local(row):
    return H / 'dependency_snapshot/by_path_sha256' / sha(row['path'].encode()) / Path(row['path']).name

rows = [check(H / r['path'], r, 'execution_source') for r in S['execution_sources']]
rows += [check(dep_local(r), r, 'external_dependency') for r in S['external_dependencies']]
names = {'SOURCE_SHA256_MANIFEST.json', 'GPU_DIAGNOSTIC_PLAN.json', 'SHARED_CHECKPOINT_IDENTITY.json', 'MATCHED_RUNTIME_BENCHMARK.json', 'HISTORICAL_EXECUTION_IDENTITY.json'}
names.update(f'execution/{condition}_R{repeat}/{name}' for condition in ['OLD_FP', 'OLD_H_C128'] for repeat in range(3) for name in ['RESULT.json', 'ENVIRONMENT.json', 'STARTED.json'])
evidence_rows = [check(H / r['path'], r, 'retained_execution_evidence') for r in D['small_assets'] if r['path'] in names]
assert {r['path'] for r in evidence_rows} == names
manifest_sha = sha((H / 'SOURCE_SHA256_MANIFEST.json').read_bytes())
plan_sha = sha((H / 'GPU_DIAGNOSTIC_PLAN.json').read_bytes())
fixture_sha = sha((H / 'fixtures/doc1_tokens.json').read_bytes())
results = []
for condition in ['OLD_FP', 'OLD_H_C128']:
    for repeat in range(3):
        unit = f'{condition}_R{repeat}'
        r = json.loads((H / f'execution/{unit}/RESULT.json').read_text())
        e = json.loads((H / f'execution/{unit}/ENVIRONMENT.json').read_text())
        keep = ['status', 'condition', 'repeat', 'profile', 'scope', 'model_forward_calls', 'decoder_layer_calls', 'GDN_chunk_calls', 'GDN_recurrent_calls', 'measured_GDN_recurrent_calls', 'measured_tokens', 'warmup_tokens', 'prefill_tokens', 'cache_retained_at_measurement_start', 'cache_final_length', 'QDQ_calls', 'source_manifest_sha256', 'plan_sha256', 'fixture_sha256', 'decode_wall_seconds', 'tokens_per_second', 'GPU_UUID', 'sampling', 'audit_overhead']
        valid_bindings = r['source_manifest_sha256'] == manifest_sha and r['plan_sha256'] == plan_sha and r['fixture_sha256'] == fixture_sha
        valid_positions = r['actual_positions'] == [[[[t]], [[t]], [[t]]] for t in range(256)]
        valid_kv = all(x['before'] == t and x['after'] == t + 1 and x['all8attention_KV_lengths'] == [t + 1] * 8 for t, x in enumerate(r['actual_KV_growth'])) and len(r['actual_KV_growth']) == 256
        results.append(dict(unit=unit, result={k:r[k] for k in keep}, environment=e, source_plan_fixture_binding='PASS' if valid_bindings else 'FAIL', actual_position_check='PASS' if valid_positions else 'FAIL', actual_KV_metadata_check='PASS' if valid_kv else 'FAIL'))
fixture = json.loads((H / 'fixtures/doc1_tokens.json').read_text())
functions=[]
for name in ['run_qwen_aime26_formal.py', 'modeling_qwen3_5.py', 'cache_utils.py', 'hub_kernels.py']:
    selected = next(r for r in S['external_dependencies'] if Path(r['path']).name == name and ('transformers-qwen35' in r['path'] if name != 'run_qwen_aime26_formal.py' else True))
    tree=ast.parse(dep_local(selected).read_text())
    wanted={'load_model_and_tokenizer','generate_one','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule','causal_conv1d_fn','causal_conv1d_update','use_kernel_func_from_hub_with_fallback','update_recurrent_state'}
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in wanted:
            functions.append(dict(path=selected['path'],source_sha256=selected['sha256'],name=n.name,line=n.lineno,end_line=n.end_lineno,AST_sha256=sha(ast.dump(n,include_attributes=False).encode()),body_AST_sha256=sha(ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False).encode())))
local=dict(task_id='QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1',status='PASS' if all(r['status']=='PASS' for r in rows+evidence_rows) and all(all(r[k]=='PASS' for k in ['source_plan_fixture_binding','actual_position_check','actual_KV_metadata_check']) for r in results) else 'FAIL',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='CPU_STDLIB_READ_ONLY_RETAINED_SMALL_ASSETS',GPU_queries=0,GPU_dispatches=0,torch_imports=0,source_manifest_sha256=manifest_sha,execution_sources_count=7,external_dependencies_count=194,closure_rows=rows,evidence_rows=evidence_rows,results=results,median_TPS={c:statistics.median(r['result']['tokens_per_second'] for r in results if r['result']['condition']==c) for c in ['OLD_FP','OLD_H_C128']},fixture=dict(raw_sha256=fixture_sha,token_count=len(fixture),first256_canonical_json_sha256=sha(json.dumps(fixture[:256],separators=(',',':')).encode()),first256_tokens=fixture[:256],interpretation='Fixed input IDs, not AIME free generation; first 256 of retained 32768-token non-AIME Doc1 fixture'),functions=functions)
(OUT/'LOCAL_HISTORICAL_SMALL_ASSET_VERIFICATION.json').write_text(json.dumps(local,ensure_ascii=False,indent=2)+'\n')

# Current filesystem small hashes corroborate source availability, never reconstruct old binary maps.
base='/data/zypan/analyses/qwen_gdn_historical_throughput_reconciliation_v1/'
remote_paths = [base+x for x in ['SOURCE_SHA256_MANIFEST.json','source/matched_worker.py','source/matched_controller.py','fixtures/doc1_tokens.json']]
remote_paths += ['/data/zypan/worktrees/aime26-sglang-rotation-v1/experiments/aime26/run_qwen_aime26_formal.py','/data/zypan/worktrees/aime26-sglang-rotation-v1/experiments/aime26/aime26_common.py']
remote_paths += [r['path'] for r in S['external_dependencies'] if r['path'] in ['/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py','/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/aime26/81920/aime26_common.py','/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py','/data/zypan/transformers-qwen35/src/transformers/cache_utils.py','/data/zypan/transformers-qwen35/src/transformers/integrations/hub_kernels.py','/data/zypan/modelscope_models/Qwen3.5-9B/config.json','/data/zypan/modelscope_models/Qwen3.5-9B/tokenizer_config.json']]
expected={r['path']:r for r in S['external_dependencies']}
expected.update({base+r['path']:r for r in S['execution_sources']})
expected[base+'SOURCE_SHA256_MANIFEST.json']={'bytes':(H/'SOURCE_SHA256_MANIFEST.json').stat().st_size,'sha256':manifest_sha}
expected[remote_paths[4]]={'bytes':26510,'sha256':'9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd'}
code='''import pathlib,json,hashlib,datetime\npaths=PATHS\nrows=[]\nfor p in paths:\n try:\n  q=pathlib.Path(p); a=q.stat(); b=q.read_bytes(); z=q.stat()\n  rows.append(dict(path=p,resolved_path=str(q.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),metadata_before=dict(bytes=a.st_size,mtime_ns=a.st_mtime_ns,inode=a.st_ino),metadata_after=dict(bytes=z.st_size,mtime_ns=z.st_mtime_ns,inode=z.st_ino),stable=a.st_size==z.st_size and a.st_mtime_ns==z.st_mtime_ns and a.st_ino==z.st_ino))\n except Exception as e: rows.append(dict(path=p,error=repr(e)))\nprint(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows)))\n'''.replace('PATHS',repr(remote_paths))
proc=subprocess.run(['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','zypan@202.38.247.55','/data/ydai/miniconda3/envs/bitdecode/bin/python','-'],input=code,text=True,capture_output=True,timeout=45)
remote={'scope':'CPU_STDLIB_READ_ONLY_SMALL_SOURCE_HASH_NO_REMOTE_WRITES','GPU_queries':0,'GPU_dispatches':0,'torch_imports':0,'returncode':proc.returncode}
if proc.returncode==0:
    remote.update(json.loads(proc.stdout))
    for row in remote['rows']:
        ex=expected.get(row['path'])
        row['historical_expected']=ex
        row['historical_binding_scope']='DIRECT_MATCHED_WORKER_RAW_SHA_ASSERTION' if row['path']==remote_paths[4] else 'RETAINED_MANIFEST' if ex else 'NOT_IN_RETAINED_SOURCE_MANIFEST_CURRENT_ONLY'
        row['verification']='PASS_CURRENT_MATCHES_BOUND_SMALL_ASSET' if ex and row.get('stable') and row.get('sha256')==ex['sha256'] and row.get('bytes')==ex['bytes'] else 'CURRENT_ONLY_NO_HISTORICAL_PIN' if not ex and row.get('stable') else 'FAIL_OR_UNAVAILABLE'
    remote['status']='PASS_CURRENT_BOUND_SMALL_ASSETS' if all(r['verification']!='FAIL_OR_UNAVAILABLE' for r in remote['rows']) else 'UNRESOLVED'
else:
    remote.update(status='UNAVAILABLE',stderr=proc.stderr)
(OUT/'REMOTE_CURRENT_SMALL_SOURCE_VERIFICATION.json').write_text(json.dumps(remote,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'local_status':local['status'],'local_closure_count':len(rows),'execution_evidence_count':len(evidence_rows),'median_TPS':local['median_TPS'],'remote_status':remote['status'],'remote_rows':[{k:r.get(k) for k in ['path','sha256','verification']} for r in remote.get('rows',[])]},ensure_ascii=False))
