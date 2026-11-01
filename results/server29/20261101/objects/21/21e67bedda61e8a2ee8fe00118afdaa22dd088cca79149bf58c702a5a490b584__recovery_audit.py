"""CPU-only qualification of a single audit expression and exact reuse.

Never loads a model or writes frozen scientific assets. The corrected source
is a one-expression snapshot; the adapter binds only that function into the
otherwise unchanged, hash-qualified scientific module.
"""
import ast
import difflib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

PARENT = Path('/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3')
sys.path.insert(0, str(PARENT/'scripts'))
import numerical_recovery as N
import scale_science as S

REC = Path(__file__).resolve().parent.parent
OLD_COMMIT = '4035e15178195193b1fa06ee47ed4321647d58c1'
OLD_EXPR = "Path(d['text_path']).read_text()"
NEW_EXPR = "Path(d['text_path']).read_bytes().decode('utf-8')"


def save(name, data):
    path=REC/name;path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())


def patch_proof():
    spec=N.read(PARENT/'SCIENCE_PREMEASUREMENT_SPEC.json')
    original=PARENT/'scripts/scale_science.py'
    corrected=REC/'scripts/scale_science_audit_repaired.py'
    before=original.read_bytes().decode('utf-8');after=corrected.read_bytes().decode('utf-8')
    N.check(N.sha(original)==spec['science_code_sha256'],'original scientific source unchanged')
    N.check(before.count(OLD_EXPR)==1,'one proven faulty read expression')
    N.check(before.replace(OLD_EXPR,NEW_EXPR)==after,'exactly one audit-only expression correction')
    old_tree=ast.parse(before);new_tree=ast.parse(after)
    old_fn=next(x for x in old_tree.body if isinstance(x,ast.FunctionDef) and x.name=='track_b')
    new_fn=next(x for x in new_tree.body if isinstance(x,ast.FunctionDef) and x.name=='track_b')
    other_old=[ast.dump(x,include_attributes=False) for x in old_tree.body if x is not old_fn]
    other_new=[ast.dump(x,include_attributes=False) for x in new_tree.body if x is not new_fn]
    N.check(other_old==other_new,'all other functions/classes/guards/numerical/observer code AST identical')
    # Both versions must be identical after reversing only the approved read.
    N.check(ast.dump(ast.parse(after.replace(NEW_EXPR,OLD_EXPR)))==ast.dump(old_tree),'no hidden AST change')
    diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(original),tofile=str(corrected)))
    checks={str(original):N.sha(original),str(corrected):N.sha(corrected)}
    contract=N.read(PARENT/'NEW_NUMERICAL_CONTRACT.json')
    N.check(N.sha(PARENT/'scripts/numerical_recovery.py')==contract['source_code_sha256'],'numerical core unchanged')
    N.check(N.sha(PARENT/'QUALIFICATION_POLICY.json')==contract['policy_sha256'],'numerical qualification policy unchanged')
    N.check(N.sha(PARENT/'NEW_NUMERICAL_CONTRACT.json')==spec['numerical_contract_sha256'],'numerical contract unchanged')
    for path,wanted in spec['source_sha256'].items():
        N.check(N.sha(path)==wanted,'unchanged core dependency '+path);checks[path]=wanted
    audit=N.read(PARENT/'EXISTING_ASSET_REUSE_AUDIT.json')
    for row in audit['rotations']['anchors']:
        if 'export_path' in row:
            N.check(N.sha(row['export_path'])==row['export_file_sha256'],'rotation export identity')
            checks[row['export_path']]=row['export_file_sha256']
    return {'status':'PASS','original_revision':OLD_COMMIT,'original_source_sha256':N.sha(original),
            'repaired_snapshot_sha256':N.sha(corrected),'diff':diff,'source_identity_checks':checks,
            'only_changed_expression':[OLD_EXPR,NEW_EXPR],
            'second_document_text_normalization_in_direct_audit_call_chain':False,
            'JSON_read_text_not_rewritten':'JSON whitespace parsing does not alter frozen token ID values',
            'runtime_adapter':'Only compile corrected track_b function into original module globals; all original setup/Harness/collector/metrics/hash guards remain unchanged. Adapter and corrected function carry independent amendment hashes.'}


def regression():
    N.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only qualification')
    os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
    sys.path.insert(0,'/data/zypan/transformers-qwen35/src')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',trust_remote_code=True,local_files_only=True)
    tests={'LF':'a\nb\n','CRLF':'a\r\nb\r\n','mixed':'a\r\nb\nc\r\n',
           'no_final_newline':'a\r\nb','empty_lines':'a\r\n\r\n\n\nb\n','UTF8':'中文éλ\r\n日本語\n'}
    results=[]
    with tempfile.TemporaryDirectory(prefix='v3-audit-regression-') as tmp:
        for name,text in tests.items():
            path=Path(tmp)/name;raw=text.encode('utf-8');path.write_bytes(raw)
            read=path.read_bytes().decode('utf-8')
            N.check(read==text and read.encode('utf-8')==raw,'exact synthetic UTF8/newline bytes '+name)
            N.check(tokenizer(read,add_special_tokens=False).input_ids==tokenizer(text,add_special_tokens=False).input_ids,'synthetic token IDs '+name)
            results.append({'case':name,'PASS':True,'CRLF_count':raw.count(b'\r\n')})
    spec=N.read(PARENT/'SCIENCE_PREMEASUREMENT_SPEC.json');docs=[]
    for d in spec['documents']:
        path=Path(d['text_path']);raw=path.read_bytes();ids=N.read(d['token_path'])
        N.check(N.sha(path)==d['text_sha256'] and N.sha(d['token_path'])==d['token_sha256'],'document/token SHA')
        tokenized=tokenizer(raw.decode('utf-8'),add_special_tokens=False).input_ids
        N.check(len(ids)==32768==d['frozen_token_count'],'frozen token count')
        N.check(tokenized[:32768]==ids and len(tokenized)==d['token_count'],'all frozen IDs and full original length')
        if d['document_index']==3:N.check(raw.count(b'\r\n')==4962,'real Doc3 CRLF regression')
        positions=spec['track_B']['capture_positions'][str(d['document_index'])]
        core=N.module(S.CORE,'recovery_original_position_helper')
        N.check(core.sampled_positions({'raw_text_sha256':d['text_sha256'],'document_id':d['canonical_doc_id']},1024)==positions,'original document/chunk/positions')
        docs.append({'document_index':d['document_index'],'canonical_doc_id':d['canonical_doc_id'],
                     'text_sha256':N.sha(path),'token_sha256':N.sha(d['token_path']),
                     'CRLF_count':raw.count(b'\r\n'),'frozen_token_count':len(ids),'all_32768_token_IDs_exact':True,
                     'first_token':ids[0],'last_token':ids[-1],'original_positions':positions})
    return {'status':'PASS','synthetic':results,'formal_documents':docs,'model_forward_count':0,
            'frozen_asset_writes':0,'Unicode_normalization':'UNCHANGED','tokenizer_kwargs':{'add_special_tokens':False}}


def verify_poststop():
    path=PARENT/'POSTSTOP_SHA256SUMS_001.txt'
    N.check(path.exists(),'original CPU-only post-stop closure completed')
    entries={}
    for line in path.read_text().splitlines():
        digest,rel=line.split('  ',1)
        target=(PARENT/rel).resolve()
        N.check(target.is_relative_to(PARENT),'manifest relative path confinement')
        N.check(target.exists() and N.sha(target)==digest,'poststop immutable SHA '+rel)
        entries[rel]=digest
    closure=N.read(PARENT/'POSTSTOP_EXECUTION_CLOSURE_001.json')
    N.check(closure['Doc3_formal_model_forward_count']==0,'zero-forward stopped unit')
    log=PARENT/'logs/science_controller_001_after_stop_001.log'
    text=log.read_text()
    completed=[int(v) for v in re.findall(r'TRACK_B_PASS (\d+) engineering False',text)]
    N.check(sorted(completed)==[1,2,4],'natural complete worker logs doc1/2/4')
    N.check((PARENT/'execution/SCIENCE_controller_gpu0_doc1_index1_STOP.json').exists(),'original parent returned failed exit evidence')
    for pid in [4325,6004,6005,6007]:N.check(not Path(f'/proc/{pid}/cmdline').exists(),'original controller/worker fully exited')
    N.check(not N.resources()['processes_csv'],'all GPUs idle before recovery launch')
    return {'status':'PASS','checked_entries':len(entries),'manifest_sha256':N.sha(path),
            'closed_formal_docs':completed,'closure':closure,'original_controller_exit':'NONZERO_FAILURE_PRESERVED',
            'original_worker_exit_statuses':[0,0,1,0],'manifest_entries':entries}


def verify_result(doc,path,allowed_commits):
    data=N.read(path);spec=N.read(PARENT/'SCIENCE_PREMEASUREMENT_SPEC.json')
    d=spec['documents'][doc-1];positions=spec['track_B']['capture_positions'][str(doc)]
    N.check(data['status']=='PASS' and data['engineering_only'] is False,'formal atomic completed receipt')
    N.check(data['source_commit'] in allowed_commits,'compatible source revision')
    N.check(data['spec_sha256']==N.sha(PARENT/'SCIENCE_PREMEASUREMENT_SPEC.json'),'frozen metric/spec identity')
    N.check(data['strict'] and not data['TF32'] and not data['training'] and data['backward_calls']==data['generation_calls']==0,'formal runtime/forbidden')
    N.check(data['formal_QDQ_calls']=={c:1024 for c in S.CONDITIONS} and data['formal_QDQ_layers']==list(S.LAYERS),'exact QDQ counts/24layers')
    N.check([m['tokens'] for m in data['cache_memory']]==list(range(128,1025,128)),'complete 1024-token receipt')
    target=data['original_teacher_target_artifact'];target_path=PARENT/target['path']
    N.check(target_path.exists() and N.sha(target_path)==target['sha256'] and target_path.stat().st_size==target['bytes'],'original actual chunk teacher artifact')
    expected={(doc,c,h) for c in S.CONDITIONS for h in [32,64]}
    keys=[(r['document_index'],r['condition'],r['H']) for r in data['scores']]
    N.check(len(keys)==8 and set(keys)==expected,'unique expected arm/horizon rows')
    for c in S.CONDITIONS:
        N.check([r['position'] for r in data['per_capture_baselines'][c]]==positions,'capture positions/order/no duplicate')
        wanted={p for k in positions for p in range(k-63,k+1)}
        N.check({int(p) for p in data['per_token_short_observations'][c]}==wanted,'exact observer token/window coverage')
    for r in data['scores']:
        N.check(r['document_sha256']==d['text_sha256'] and r['token_sha256']==d['token_sha256'],'row input identities')
        N.check(r['GPU_UUID']==S.UUIDS[doc-1] and r['teacher_target_sha256']==target['sha256'],'row GPU/teacher pairing')
        N.check(r['window_token_exposures']==8*r['H'],'exact frozen window exposures')
        N.check(all(math.isfinite(r[k]) for k in ['S1','S2','D1','D2','B0','B1','B2']),'finite actual short/baseline scores')
    return data


def ledger(poststop):
    spec=N.read(PARENT/'SCIENCE_PREMEASUREMENT_SPEC.json');units=[]
    entries=poststop['manifest_entries']
    for stage in ['B','A','LONG']:
        for d in spec['documents']:
            doc=d['document_index'];gpu=doc-1
            rel=f'track_B/doc_{doc:02d}/result.json' if stage=='B' else f'track_A/doc_{doc:02d}/short_result.json' if stage=='A' else f'long/doc_{doc:02d}/result.json'
            path=PARENT/rel;status='NOT_STARTED';sha=None
            if stage=='B' and doc in [1,2,4]:
                N.check(rel in entries,'complete B receipt sealed in poststop manifest')
                verify_result(doc,path,[OLD_COMMIT]);status='VALID_COMPLETE';sha=N.sha(path)
            elif stage=='B' and doc==3:
                N.check(not path.exists(),'Doc3 no complete prior result');status='FAILED'
            elif path.exists():raise RuntimeError('unexpected history artifact before authorized resume '+rel)
            for c in spec['conditions']:
                units.append({'unit_id':f'{stage}_DOC{doc}_{c}','document_id':d['canonical_doc_id'],
                    'document_index':doc,'condition':c,'stage':stage,'GPU_assignment':gpu,'GPU_UUID':S.UUIDS[gpu],
                    'status':status,'reuse_eligible':status=='VALID_COMPLETE','original_revision':OLD_COMMIT,
                    'numerical_contract_sha256':N.sha(PARENT/'NEW_NUMERICAL_CONTRACT.json'),
                    'metric_spec_sha256':N.sha(PARENT/'SCIENCE_PREMEASUREMENT_SPEC.json'),
                    'text_sha256':d['text_sha256'],'token_sha256':d['token_sha256'],'artifact':rel,'artifact_sha256':sha,
                    'process_state':'EXITED' if stage=='B' else 'NOT_STARTED','horizons':[32,64] if stage!='LONG' else [32768],
                    'model_rotation_quantizer_provenance_sha256':N.sha(PARENT/'EXISTING_ASSET_REUSE_AUDIT.json'),
                    'required_checkpoint':None if stage=='B' else 'same-doc complete real anchor8256 + resume parity' if stage=='LONG' else 'continuous frozen tokens1..8256; not a reset'})
    N.check(len(units)==len({u['unit_id'] for u in units})==48,'each science unit classified once')
    return {'status':'PASS','expected_scientific_units':48,'units':units,
            'atomic_execution_bundles':'one document worker contains all four conditions; B/A has two observational horizons, not independent samples',
            'qualification_data':'separate science_qualification; never reused as formal score',
            'model_checkpoint_provenance':'existing verified identity manifest reused; original model checkpoint not modified',
            'rotation_manifest':N.read(PARENT/'EXISTING_ASSET_REUSE_AUDIT.json')['rotations'],
            'frozen_numerical_contract':N.read(PARENT/'NEW_NUMERICAL_CONTRACT.json')}


def main(phase):
    if phase=='cpu':
        proof=patch_proof();tests=regression()
        save('MINIMAL_PATCH_AUDIT.json',proof);save('CPU_REGRESSION.json',tests)
        (REC/'MINIMAL_PATCH.diff').write_text(proof['diff'],encoding='utf-8')
        print(json.dumps({'CPU_REGRESSION':'PASS','audit_diff_only':True,'formal_docs':4,'synthetic_cases':6,'GPU_forward_count':0}),flush=True)
    elif phase=='closed_audit':
        proof=patch_proof();poststop=verify_poststop();table=ledger(poststop)
        save('POSTSTOP_REUSE_AUDIT.json',poststop);save('FORMAL_UNIT_LEDGER_BEFORE_RESUME.json',table)
        qual=[]
        for rel in ['MULTITOKEN_FULL_MODEL_PARITY.json','FRESH_PROCESS_DETERMINISM.json','NEW_NUMERICAL_CONTRACT.json','OBSERVER_PARITY_REPORT.json','TRACK_B_ENGINEERING_GATE.json']:
            data=N.read(PARENT/rel);N.check(data['status']=='PASS','qualification reuse '+rel)
            N.check(rel in poststop['manifest_entries'],'sealed qualification evidence '+rel)
            qual.append({'artifact':rel,'sha256':N.sha(PARENT/rel),'action':'REUSE_NO_GPU_RERUN'})
        save('QUALIFICATION_REUSE.json',{'status':'PASS','proof':proof['repaired_snapshot_sha256'],'unchanged_numerical_and_science_functions':True,'receipts':qual})
        print(json.dumps({'POSTSTOP_AUDIT':'PASS','reuse_formal_docs':[1,2,4],'execute_formal_docs':[3],'expected_science_units':48}),flush=True)


if __name__=='__main__':main(sys.argv[1])
