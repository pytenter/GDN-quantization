"""Static-only surgical patch audit and append-only failed-attempt preservation."""
import ast
import copy
import difflib
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO=Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1')
ROOT=REPO/'experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1'
OLD=ROOT/'amendments/STORAGE_ACCOUNTING_CORRECTION_V1'
BUG=ROOT/'amendments/CAPTURE_KEY_SHADOWING_BUGFIX_V1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    p.chmod(0o444)
def text(p,s):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(s)
    p.chmod(0o444)
def exact_copy(source,target):
    assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target);assert sha(source)==sha(target);target.chmod(0o444)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    failed=read(OLD/'engineering/extended_capture.error.json')
    assert failed['exception']=='TypeError' and failed['message']=="can't multiply sequence by non-int of type 'str'"
    assert read(OLD/'final_verdict.json')['completed_units']==0
    assert read(OLD/'engineering/gates/determinism_256.json')['gate']=='PASS'
    original=OLD/'scripts/capture_extended_drive.py';fixed=BUG/'scripts/capture_extended_drive.py'
    assert sha(original)==read(OLD/'long_qualification_freeze.json')['code_sha256']['scripts/capture_extended_drive.py']
    old_source=original.read_text();new_source=fixed.read_text()
    old_tree=ast.parse(old_source);new_tree=ast.parse(new_source)
    names=[n for n in ast.walk(new_tree) if isinstance(n,ast.Name) and n.id=='field_name']
    assert len(names)==4,'Only the metadata-loop binding and its three uses may change'
    class Alpha(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='field_name':node.id='key'
            return node
    assert ast.dump(old_tree,include_attributes=False)==ast.dump(Alpha().visit(copy.deepcopy(new_tree)),include_attributes=False)
    captured=next(n for n in ast.walk(new_tree) if isinstance(n,ast.FunctionDef) and n.name=='captured')
    assert [a.arg for a in captured.args.args]==['query','key','value']
    assert not any(isinstance(n,ast.Name) and n.id=='key' and isinstance(n.ctx,ast.Store) for n in ast.walk(captured))
    dispatch=next(n for n in ast.walk(captured) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='function')
    assert [ast.dump(n,include_attributes=False) for n in dispatch.args]==[ast.dump(ast.Name(id=k,ctx=ast.Load()),include_attributes=False) for k in ['query','key','value']]
    diff=''.join(difflib.unified_diff(old_source.splitlines(True),new_source.splitlines(True),
        fromfile=str(original),tofile=str(fixed)))
    assert diff and diff.count('@@')==2
    text(BUG/'source_diff.patch',diff)
    # Validate every previously sealed failed-attempt artifact; also snapshot
    # terminal exitcode/log files not present when the old final seal was made.
    for line in (OLD/'FINAL_SHA256SUMS').read_text().splitlines():
        expected,rel=line.split(None,1);assert sha(OLD/rel)==expected
    files=[p for p in sorted(OLD.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    rows=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    save(BUG/'failed_attempt_snapshot.json',dict(FAILED_ATTEMPT_PRESERVED='YES',files=rows,
        file_count=len(rows),total_bytes=sum(r['bytes'] for r in rows),cache_excluded=True,
        failed_output_directory_was_empty=not any((OLD/'engineering/capture_2048').iterdir()),
        old_final_report_sha256=sha(OLD/'FINAL_REPORT.md'),old_final_seal_sha256=sha(OLD/'FINAL_SHA256SUMS')))
    for filename in ['capture_original_drive.py','isolated_replay_engineering.py','persistent_anchor_replay.py',
        'qualify_long_and_streaming.py','scientific_lifecycle.py']:
        exact_copy(OLD/'scripts'/filename,BUG/'scripts'/filename)
    # These are byte-identical reused PASS records, not rerun qualifications.
    for name in ['factorization','schema','fp','invariance','decomposition','isolation','determinism_256']:
        assert read(OLD/'engineering/gates'/(name+'.json'))['gate']=='PASS'
        exact_copy(OLD/'engineering/gates'/(name+'.json'),BUG/'engineering/gates'/(name+'.json'))
    for name in ['drive_dependency_graph.json','drive_schema.json','drive_factorization_report.md','engineering_energy_observations.json']:
        exact_copy(OLD/'engineering/gates'/name,BUG/'engineering/gates'/name)
    for name in ['capture_engineering_preregistration.json','remaining_engineering_gate_policy.json','engineering_fixture_256_token_ids.json']:
        exact_copy(OLD/'protocol'/name,BUG/'protocol'/name)
    for name in ['capture_freeze_receipt.json','short_qualification_freeze.json','storage_lifecycle_report.json','engineering2048_storage_supplement.json']:
        exact_copy(OLD/name,BUG/name)
    exact_copy(OLD/'engineering/capture_runtime_variability.json',BUG/'engineering/capture_runtime_variability.json')
    for name in ['capture_01','capture_02']:
        (BUG/'engineering'/name).symlink_to(OLD/'engineering'/name,target_is_directory=True)
    directory=BUG/'engineering/repeatability';directory.mkdir(parents=True,exist_ok=True)
    for run in ['01','02','03']:
        name='length_00256_run_'+run
        (directory/name).symlink_to(OLD/'engineering/repeatability'/name,target_is_directory=True)
    policy=read(OLD/'protocol/extended_capture_policy.json');new=copy.deepcopy(policy)
    new['engineering2048']['output_directory']=str(BUG/'engineering/capture_2048')
    for i,d in enumerate(new['documents'],1):d['output_directory']=str(BUG/'scientific'/('doc_%02d'%i))
    normalized=copy.deepcopy(new)
    normalized['engineering2048']['output_directory']=policy['engineering2048']['output_directory']
    for d,old in zip(normalized['documents'],policy['documents']):d['output_directory']=old['output_directory']
    assert normalized==policy,'Only output routing may differ; no scientific/storage policy change'
    save(BUG/'protocol/extended_capture_policy.json',new)
    for path,expected in policy['production_policy']['source_sha256'].items():assert sha(path)==expected
    for d in read(ROOT/'protocol/frozen_discovery_documents.json')['documents']:
        assert sha(ROOT/d['text_path'])==d['text_sha256'] and sha(ROOT/d['token_path'])==d['token_sha256']
    assert sha(ROOT/'protocol/frozen_anchor_conditions.json')==policy['production_policy']['anchor_manifest_sha256']
    audit=dict(STATIC_SOURCE_AUDIT='PASS',BUG_CLASSIFICATION='IMPLEMENTATION_VARIABLE_SHADOWING',
        BUGFIX_SCOPE='VARIABLE_RENAME_OR_SCOPE_CORRECTION_ONLY',old_source_path=str(original),new_source_path=str(fixed),
        old_sha256=sha(original),fixed_sha256=sha(fixed),diff_sha256=sha(BUG/'source_diff.patch'),
        failing_binding_line=77,dispatch_line=81,old_binding="for key,t in record.items(): key becomes 'beta' after the loop",
        corrected_binding='field_name is the metadata-loop label; key remains the original function argument Tensor',
        alpha_normalized_entire_source_AST_exact=True,production_operations_changed=0,
        corrected_wrapper_assigns_Tensor_key=False,dispatch_query_key_value_unchanged=True,
        dtype_shape_device_and_dataflow='Unmodified original argument, no cast/copy/operation inserted into production dispatch',
        output_routing_only='New amendment namespace to preserve every old failed artifact; identical input/metric/storage policies',
        G0_G10_PREVIOUS_PASS_PRESERVED='YES',G0_G10_REEXECUTED=False,G11_256_TOKEN_STATUS='PASS',
        G11_256_REEXECUTED=False,G11_2048_TOKEN_PREVIOUS_STATUS='FAIL_IMPLEMENTATION_BUG',
        SCIENTIFIC_MEASUREMENT_BEFORE_FIX=0,DISCOVERY_COMPLETE='0/16',
        policy_comparison_excluding_only_output_directory_paths='EXACT',
        capture_schema_sha256=sha(BUG/'engineering/gates/drive_schema.json'),
        unchanged_rules_sha256=sha(BUG/'protocol/remaining_engineering_gate_policy.json'))
    save(BUG/'static_source_audit.json',audit)
    st=os.statvfs(ROOT);available=st.f_bavail*st.f_frsize
    required=read(BUG/'engineering2048_storage_supplement.json')['PEAK_REQUIRED_STORAGE_BYTES']
    save(BUG/'current_capacity_check.json',dict(AVAILABLE_DATA_BYTES=available,PEAK_REQUIRED_STORAGE_BYTES=required,
        SAFETY_MARGIN_BYTES=policy['minimum_runtime_free_bytes'],margin_counted_once=True,
        STORAGE_CAPACITY_GATE='PASS' if required<=available else 'FAIL'))
    assert required<=available,'Current capacity failed; STOP'
    print(json.dumps(dict(STATIC_SOURCE_AUDIT='PASS',failed_files_preserved=len(rows),fixed_sha256=sha(fixed),
        G0_G10_REEXECUTED=False,G11_256_REEXECUTED=False,available_bytes=available)),flush=True)

if __name__=='__main__':main()
