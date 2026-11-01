"""Readonly identity reconciliation. CPU only; never imports a model runner."""
import base64
import collections
import hashlib
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
import time
import traceback
import unicodedata

OLD = pathlib.Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_MIXED_OUTCOME_RESOLUTION_V1')
FIRST = pathlib.Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_FUNCTIONAL_VALIDATION_V1')
OUT = pathlib.Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1')
TASK = OUT.name

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(pathlib.Path(path).read_text())

def publish(name,value):
    path=OUT/name;tmp=path.with_name(path.name+'.partial')
    with tmp.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(tmp,path);tmp.unlink()

def token_sha(tokens):
    return hashlib.sha256(struct.pack('<'+'I'*len(tokens),*tokens)).hexdigest()

def main():
    start=time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU_ONLY_AUDIT_REQUIRED'
    m=read(OLD/'RESULT_AND_DELIVERY_HASHES.json')
    assert m['status'].startswith('COMPLETE'),m['status']
    for n,row in m['files'].items():
        f=OLD/n
        assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],('OLD_SEALED_ASSET_CHANGED',n)
    for line in (OLD/'SHA256SUMS.txt').read_text().splitlines():
        h,n=line.split('  ',1);assert sha(OLD/n)==h,n
    for line in (FIRST/'delivery/SHA256SUMS.txt').read_text().splitlines():
        h,n=line.split('  ',1);assert sha(FIRST/'delivery'/n)==h,n
    assert read(FIRST/'COMPLETED_ARTIFACT_AUDIT.json')['status']=='PASS'
    assert read(FIRST/'FORMAL_COMPLETE.json')['status']=='PASS'
    assert read(OLD/'CROSS_COMPLETE.json')['status']=='COMPLETE'
    assert read(OLD/'COMPLETED_CROSS_PANEL_AUDIT.json')['status']=='PASS'
    spec=read(OLD/'CROSS_ROTATION_FROZEN_SPEC.json')
    assert spec['source_commit']=='80f23cd580df3bb00b9256ab8b5b3d5d221ffdad'
    assert subprocess.check_output(['git','-C',spec['source_worktree'],'rev-parse','HEAD'],text=True).strip()==spec['source_commit']
    assert subprocess.check_output(['git','-C',spec['source_worktree'],'status','--porcelain'],text=True)==''
    assets=[]
    for name,row in spec['source_assets'].items():
        f=pathlib.Path(name);s=f.stat()
        assert (s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns']),('SOURCE_METADATA_CHANGED',name)
        reused=f.suffix=='.safetensors'
        if not reused: assert sha(f)==row['sha256'],('SOURCE_SHA_CHANGED',name)
        assets.append(dict(path=name,sha256=row['sha256'],bytes=s.st_size,mtime_ns=s.st_mtime_ns,
            verification='UNCHANGED_METADATA_WITH_PRIOR_FULL_SHA_QUALIFICATION' if reused else 'FRESH_SHA'))
    stops=spec['historical_STOP_sha256']
    for name,h in stops.items(): assert sha(name)==h,('HISTORICAL_STOP_CHANGED',name)
    freeze=read(OLD/'NEXT_PROSPECTIVE_PROTOCOL_FREEZE.json')
    for n,h in freeze['files'].items():assert sha(OLD/n)==h,('PREREGISTRATION_CHANGED',n)
    scorer=read(OLD/'PROSPECTIVE_FROZEN_SCORE_SOURCE_IDENTITIES.json')
    for name,h in scorer.items():assert sha(name)==h,('SCORER_CHANGED',name)
    rules=read(OLD/'DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json')
    order=read(OLD/'PROSPECTIVE_INPUT_ORDER_POLICY_FREEZE.json')
    assert rules['experiment']==order['task']==TASK
    assert order['seed']==0 and order['target_documents']==4 and order['maximum_calibration_screen_documents']==40
    cal=rules['calibration']
    assert cal['horizon']==128 and cal['heads']==16 and len(cal['layers'])==18 and len(cal['rotation_order'])==16
    for name,row in cal['rotation_identities'].items():assert sha(row['path'])==row['sha256'],name
    a=read(OLD/'PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json')
    assert a['pool_books']==100 and a['input_identity_eligible_count']==58
    inv=read(OLD/'historical_path_identity_inventory.json')
    now_ids=set();skip=set(inv['skipped_directories'])|{'prospective_inputs'}
    for base in inv['bases']:
        for root,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in skip]
            for name in files:
                now_ids.update(re.findall(r'pg19-test-\d+',name))
    old_ids=set(a['historical_excluded_ids'])
    assert set(inv['path_identity_evidence'])==old_ids
    assert not (now_ids-old_ids),('NEW_HISTORICAL_DOCUMENTS_PROTOCOL_CONFLICT',sorted(now_ids-old_ids))
    model=pathlib.Path(spec['model']['path'])
    for n,h in a['model_tokenizer_identity'].items():assert sha(model/n)==h,n
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=True)
    seen=[set() for _ in range(4)]; rows=[]; reasons=collections.Counter()
    entries=sorted(a['inputs'],key=lambda x:(x['document'] not in old_ids,x['order_sha256'],x['document']))
    for entry in entries:
        raw=pathlib.Path(entry['raw_text_path']).read_bytes()
        assert len(raw)==entry['size_bytes'] and hashlib.sha256(raw).hexdigest()==entry['raw_sha256']
        assert base64.b64encode(hashlib.md5(raw).digest()).decode()==entry['object_md5_base64']
        normalized=' '.join(unicodedata.normalize('NFC',raw.decode('utf-8')).split())
        normalized_sha=hashlib.sha256(normalized.encode()).hexdigest()
        assert normalized_sha==entry['normalized_text_sha256']
        tokens=tokenizer.encode(raw.decode('utf-8'),add_special_tokens=False)
        assert len(tokens)==entry['full_token_count'] and token_sha(tokens)==entry['canonical_full_token_sha256']
        prefix=tokens[:32769]
        assert token_sha(prefix)==entry['canonical_prefix_token_sha256']
        assert sha(entry['token_file_path'])==entry['token_file_sha256']
        assert read(entry['token_file_path'])==prefix
        order_sha=hashlib.sha256((TASK+'|seed=0|'+entry['document']+'|'+entry['raw_sha256']).encode()).hexdigest()
        assert order_sha==entry['order_sha256']
        hashes=[entry['raw_sha256'],normalized_sha,token_sha(tokens),token_sha(prefix)]
        if entry['document'] in old_ids: reason='HISTORICAL_MEASUREMENT_ID'
        elif len(tokens)<32769: reason='INSUFFICIENT_TOKENS'
        elif any(h in s for h,s in zip(hashes,seen)): reason='DUPLICATE_INPUT_IDENTITY'
        else: reason=None
        assert reason==entry['exclusion_reason'],('INPUT_ELIGIBILITY_CHANGED',entry['document'],reason)
        assert (reason is None)==entry['input_identity_eligible']
        for h,s in zip(hashes,seen):s.add(h)
        reasons[reason or 'ELIGIBLE']+=1;rows.append(dict(document=entry['document'],reason=reason,
            raw_sha256=entry['raw_sha256'],token_file_sha256=entry['token_file_sha256'],
            canonical_full_token_sha256=entry['canonical_full_token_sha256'],order_sha256=order_sha))
        if len(rows)%10==0:print(json.dumps(dict(event='CPU_INPUT_AUDIT_PROGRESS',documents=len(rows))),flush=True)
    assert dict(reasons)=={'HISTORICAL_MEASUREMENT_ID':20,'INSUFFICIENT_TOKENS':22,'ELIGIBLE':58},dict(reasons)
    candidates=sorted([x for x in a['inputs'] if x['input_identity_eligible']],key=lambda x:(x['order_sha256'],x['document']))
    assert [x['document'] for x in a['inputs']]==[x['document'] for x in sorted(a['inputs'],key=lambda x:(x['order_sha256'],x['document']))]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    row=next([z.strip() for z in line.split(',')] for line in gpu.splitlines() if line.startswith('6,'))
    assert row[1]=='NVIDIA GeForce RTX 3090' and row[2]=='GPU-de53f243-b271-521c-d875-d96611ed8574'
    assert row[2] not in apps and int(row[3])==24576 and int(row[4])<=128 and row[6]==spec['runtime']['driver']
    disk=shutil.disk_usage(OUT);assert disk.free>=64*1024**3
    memory=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
    assert int(memory['MemAvailable'].split()[0])*1024>=16*1024**3
    root=OUT/'preregistration';root.mkdir()
    names=list(freeze['files'])+['NEXT_PROSPECTIVE_PROTOCOL_FREEZE.json','PROSPECTIVE_FROZEN_SCORE_SOURCE_IDENTITIES.json',
        'PG19_TEST_GCS_LISTING.json','historical_path_identity_inventory.json','MODEL_WEIGHT_SHA256.txt']
    for n in names:
        f=root/n
        with f.open('xb') as dest:dest.write((OLD/n).read_bytes())
        assert sha(f)==sha(OLD/n)
    publish('INPUT_ELIGIBILITY_REPRODUCTION.json',dict(status='PASS_EXACT_100_20_22_58',counts=dict(reasons),
        actual_retokenization=True,tokenizer_identity=a['model_tokenizer_identity'],historical_refresh_additional_ids=[],
        entries=rows,eligible_order=[x['document'] for x in candidates],future_outcomes_read=False))
    publish('PREREGISTRATION_RECONCILIATION.json',dict(status='PASS',experiment=TASK,
        old_formal_manifest=dict(path=str(FIRST/'delivery/DOCUMENT_VERDICT.json'),sha256=sha(FIRST/'delivery/DOCUMENT_VERDICT.json')),
        mixed_delivery_manifest=dict(path=str(OLD/'RESULT_AND_DELIVERY_HASHES.json'),sha256=sha(OLD/'RESULT_AND_DELIVERY_HASHES.json')),
        preregistration=dict(path=str(OLD/'NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md'),sha256=sha(OLD/'NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md')),
        frozen_protocol_files=freeze['files'],scorer_source_SHA=scorer,source_assets=assets,historical_STOP_preserved=stops,
        input_counts=dict(reasons),selected_documents=0,new_long_GPU_scientific_units=0,
        authorization_reconciliation='New user instruction authorizes the previously plan-only protocol. Its scientific definitions remain unchanged; archived authorization=false is historical readiness, not a scientific conflict.',
        indexing_reconciliation='Original same-step logits are produced after consuming token[t] and predict the next token; no shift or next-token-target metric substituted.',
        physical_memory_label='Frozen plan physical_device_total_bytes is CUDA-visible total; NVML nominal board total is 24576MiB. Both are recorded without rewriting archived protocol.',
        actual_gpu_inventory=gpu,actual_compute_inventory=apps,selected_GPU=6,selected_GPU_UUID=row[2],
        selection_reason='same qualified UUID is presently idle and meets frozen runtime; no other process killed',
        runtime=spec['runtime'],disk_free_bytes=disk.free,MemAvailable=memory['MemAvailable'],
        required_storage_plan_bytes=64*1024**3,elapsed_seconds=time.time()-start,CUDA_initialized=False))
    print(json.dumps(dict(status='PHASE0_PASS',input_counts=dict(reasons),elapsed_seconds=time.time()-start)),flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:main()
    except BaseException as exc:
        publish('STOP.json',dict(status='STOP',stage='PHASE0',reason=repr(exc),traceback=traceback.format_exc(),
            epoch=time.time(),new_32K_science_units=0,no_retry=True))
        raise
