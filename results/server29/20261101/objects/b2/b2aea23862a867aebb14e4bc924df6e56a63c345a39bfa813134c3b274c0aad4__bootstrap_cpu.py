#!/usr/bin/env python3
"""Create an isolated CPU preparation from the actual finalized FP delivery."""
import ast,datetime,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PARENT=ROOT.parent/'QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1/gpu_execution_20261019_v1'
REQUEST=Path('/Users/pytenter/.codex/attachments/52dc90eb-862b-4a5e-9feb-cd45d2b43258/已粘贴的文本.txt')
TASK='QWEN_GDN_HISTORICAL_PORT_INT8_GROUP32_QUALIFICATION_V1'
FREEZE='48ec3e527b28cc1a1860a72559d0e9fafad96866c9d981f6b0c80d4bf5ebe19c'
def raw(p):
    assert p.is_file()and not p.is_symlink();a=p.stat();b=p.read_bytes();z=p.stat();assert(a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns);return b
def desc(p):
    b=raw(p);return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def put(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb')as f:f.write(b)
def save(p,j):put(p,(json.dumps(j,indent=2,ensure_ascii=False,sort_keys=True)+'\n').encode())
def main():
    assert not(ROOT/'ROOT.json').exists(),'Independent task already exists: inspect, never overwrite'
    d=PARENT/'delivery';manifest=json.loads(raw(d/'SHA256_MANIFEST.json'))
    assert desc(d/'SHA256_MANIFEST.json')['sha256']=='cece32eb6c2f51d3b1316d8960f34406c9b4b0142a1a143e0de1b72c611babf5'
    checked=[]
    for row in manifest['files']:
        q=Path(row['path']);assert not q.is_absolute()and'..'not in q.parts
        z=desc(d/q);assert(z['bytes'],z['sha256'])==(row['bytes'],row['sha256']);checked.append(row)
    assert len(checked)==3753
    seal=json.loads(raw(d/'FINAL_SEAL_RECEIPT.json'));verify=json.loads(raw(d/'LOCAL_DELIVERY_VERIFICATION.json'))
    assert desc(d/'FINAL_SEAL_RECEIPT.json')['sha256']=='5be227f4b61eb61b641c559720e05a3d00159c8cc5e1d3ed53fafd712ee104b6'
    assert desc(d/'LOCAL_DELIVERY_VERIFICATION.json')['sha256']=='f16593d52b085a66093de2b5ea5dd51aa56b03ce01209d3b536bd3db1d4ce78c'
    assert verify['all_three_hosts_logical_manifest_SHA_bytes_equal']is True
    decision=json.loads(raw(d/'RUNTIME_RESTORATION_DECISION.json'));assert decision['status']=='HISTORICAL_PORT_FP_QUALIFIED_SPEED_RECOVERED'
    freeze=json.loads(raw(d/'payload/SOURCE_FREEZE.json'));assert desc(d/'payload/SOURCE_FREEZE.json')['sha256']==FREEZE and len(freeze['files'])==407
    target=ROOT/'inherited/historical_port_payload'
    for row in freeze['files']:
        b=raw(d/'payload'/row['path']);assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'];put(target/row['path'],b)
    put(target/'SOURCE_FREEZE.json',raw(d/'payload/SOURCE_FREEZE.json'))
    for name in ['SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','LOCAL_DELIVERY_VERIFICATION.json','FINAL_SCIENTIFIC_REPORT.md','RUNTIME_RESTORATION_DECISION.json','INT8_COMPATIBILITY_AUDIT.md']:
        put(ROOT/'provenance/parent_final_delivery'/name,raw(d/name))
    for name in ['operations/EXTERNAL_RECEIPT_REPLICATION.json','operations/FINAL_DELIVERY_INDEPENDENT_REVIEW.json']:
        put(ROOT/'provenance/parent_closure'/Path(name).name,raw(PARENT/name))
    old=next(x['path']for x in freeze['files']if x['path'].endswith('/81920/run_qwen_aime26_formal.py')and'/source_tree/'in x['path'])
    source=raw(target/old).decode();tree=ast.parse(source);wanted={'hadamard','normalize_qk','QwenKeyHadamardPatch','get_state','quantize_c128','load_model_and_tokenizer','setup_seed'}
    parts=[];proof=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef))and node.name in wanted:
            parts.append(ast.get_source_segment(source,node));proof.append(dict(name=node.name,AST_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()))
    assert len(parts)==7
    header='import math,os,random,sys\nfrom pathlib import Path\nimport torch\nEPS=1e-12\nGDN_LAYERS=tuple(i for i in range(32) if i%4!=3)\nMODEL_PATH=Path("/data/zypan/modelscope_models/Qwen3.5-9B")\nTRANSFORMERS_SRC=Path("/data/zypan/transformers-qwen35")\nSTATE_QUANT_SRC=Path("/data/zypan/experiments/qwen35_gdn_quant")\n\n'
    put(ROOT/'src/canonical_h128.py',(header+'\n\n'.join(parts)+'\n').encode())
    asts={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()for n in ast.parse(raw(ROOT/'src/canonical_h128.py')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert all(asts[x['name']]==x['AST_sha256']for x in proof)
    put(ROOT/'USER_REQUEST.txt',raw(REQUEST))
    status=dict(task_id=TASK,status='CPU_PREPARATION_GPU_NOT_AUTHORIZED',ALLOW_GPU_CANARY='NO',GPU_queries=0,GPU_dispatches=0,Driver_API_calls=0,local_root=str(ROOT),parent_delivery=str(d),parent_source_freeze_sha256=FREEZE,remote_python='/data/ydai/miniconda3/envs/bitdecode/bin/python',ssh_control_path='/tmp/qwen_gdn_replication_20261013.sock',remote_root='/data/zypan/analyses/qwen_gdn_historical_port_int8_group32_qualification_v1/cpu_preparation_20261019_v1',archive_root='/data/zypan/qwen_historical_port_int8_group32_qualification_archive/CPU_PREPARATION_20261019_V1',runtime_protocol='HISTORICAL_PORT_PERSISTENT_INT8_GROUP32_V1',separate_new_task_GPU_authorization_received=False,valid_GPU_permit=None,prior_four_GPU_unlimited_grant_not_reused=True,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_request_sha256=desc(ROOT/'USER_REQUEST.txt')['sha256'])
    save(ROOT/'ROOT.json',status)
    save(ROOT/'provenance/PARENT_FINAL_DELIVERY_VERIFICATION.json',dict(status='PASS_ACTUAL_FINALIZED_PARENT_407_AND3753_INDEXED_FILES',files_verified=3753,indexed_bytes=sum(x['bytes']for x in checked),parent_scientific_status=decision['status'],source_files=407,source_freeze_sha256=FREEZE,canonical_extraction_AST_parity=proof,anchors={n:desc(d/n)for n in ['SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json','LOCAL_DELIVERY_VERIFICATION.json','FINAL_SCIENTIFIC_REPORT.md']},parent_final_seal_actually_exists=True,finisher_itself_did_not_seal_but_subsequent_task_finalization_did=True,old_files_modified=False,GPU_queries=0,GPU_dispatches=0))
    print(json.dumps(dict(status=status['status'],root=str(ROOT),parent_files_verified=3753,inherited_source_files=407,canonical_ASTs=7)))
if __name__=='__main__':main()
