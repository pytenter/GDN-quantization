#!/usr/bin/env python3
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

manifest=json.loads((ROOT/'CANONICAL_SOURCE_MANIFEST.json').read_text())
errors=[]
keys=[]
for rec in manifest['files']:
    keys.append(tuple(rec['key']))
    for kind in ('canonical','sidecar'):
        item=rec[kind]; path=Path(item['path'])
        if not path.is_file():
            errors.append(f'missing:{path}'); continue
        if path.stat().st_size != item['bytes']:
            errors.append(f'bytes:{path}')
        if sha256(path) != item['sha256']:
            errors.append(f'sha256:{path}')
if len(keys)!=80 or len(set(keys))!=80:
    errors.append('manifest_key_cardinality')
with open(ROOT/'OBSERVABILITY_ANALYSIS_DATASET.csv',newline='') as f:
    rows=list(csv.DictReader(f))
csv_keys={(r['document'],r['condition'],int(r['anchor']),int(r['endpoint'])) for r in rows}
if len(rows)!=80 or csv_keys!=set(keys):
    errors.append('dataset_manifest_key_mismatch')
required=[
 'ANALYSIS_PLAN.json','AVAILABLE_METRICS_AUDIT.md','OBSERVABILITY_ANALYSIS_DATASET.csv',
 'CANONICAL_SOURCE_MANIFEST.json','ANALYSIS_RESULTS.json','MAGNITUDE_MATCHED_PAIRS.csv',
 'EXISTING_80_SCATTER_PANELS.svg','EXISTING_80_SCIENTIFIC_ANALYSIS.md',
 'LONG_HORIZON_ASSET_REUSE_AUDIT.md','MINIMAL_CAUSAL_INTERVENTION_PROPOSAL.md',
 'MINIMAL_GPU_VALIDATION_PLAN.md','NEXT_RESEARCH_ACTION.json','analyze_existing_80.py',
 'verify_outputs_and_sources.py','REPRODUCIBILITY_RUN.json'
]
for name in required:
    q=ROOT/name
    if not q.is_file() or q.stat().st_size==0:
        errors.append(f'required:{name}')
next_action=json.loads((ROOT/'NEXT_RESEARCH_ACTION.json').read_text())
if next_action.get('execution_authorized') is not False:
    errors.append('execution_authorized_not_false')
if next_action.get('new_gpu_processes_started') != 0:
    errors.append('new_gpu_processes_started_nonzero')
result={
 'analysis_id':'LING_OBSERVABILITY_MECHANISM_TRIAGE_V1',
 'status':'PASS' if not errors else 'FAIL',
 'canonical_files_rehashed':len(manifest['files']),
 'sidecar_files_rehashed':len(manifest['files']),
 'unique_scientific_keys':len(set(keys)),
 'dataset_rows':len(rows),
 'required_outputs_verified':len(required),
 'historical_sources_byte_identical_to_manifest':not any(e.startswith(('bytes:','sha256:','missing:')) for e in errors),
 'execution_authorized':False,
 'new_gpu_processes_started':0,
 'errors':errors
}
receipt=ROOT/'REPRODUCIBILITY_VERIFICATION.json'
if receipt.exists():
    if json.loads(receipt.read_text()) != result:
        print(json.dumps({'status':'FAIL','reason':'existing verification receipt differs','computed':result},sort_keys=True))
        raise SystemExit(1)
else:
    with receipt.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(result,sort_keys=True))
raise SystemExit(0 if not errors else 1)
