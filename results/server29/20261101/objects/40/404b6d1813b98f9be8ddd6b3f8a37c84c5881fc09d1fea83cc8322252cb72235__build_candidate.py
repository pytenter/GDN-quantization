#!/usr/bin/env python3
"""Materialize raw historical source copies; never import Torch or dispatch GPU."""
import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P=R.parent/'qwen_gdn_historical_throughput_reconciliation_v1'
M=json.loads((P/'SOURCE_SHA256_MANIFEST.json').read_bytes())
rows=[]
def copy(src,dst,expected=None):
    b=src.read_bytes();h=hashlib.sha256(b).hexdigest()
    if expected:assert len(b)==expected['bytes'] and h==expected['sha256'],str(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        assert dst.is_file() and dst.read_bytes()==b,'Existing candidate copy differs: '+str(dst)
    else:
        with dst.open('xb') as f:f.write(b)
    rows.append(dict(path=str(dst.relative_to(R)),bytes=len(b),sha256=h,
                     historical_path=expected['path'] if expected else str(src),source_copy=str(src)))
for e in M['execution_sources']:
    copy(P/e['path'],R/'candidate/benchmark_replay'/e['path'],e)
copy(P/'SOURCE_SHA256_MANIFEST.json',R/'candidate/benchmark_replay/SOURCE_SHA256_MANIFEST.json')
for e in M['external_dependencies']:
    src=P/'delivery/dependency_snapshot/by_path_sha256'/hashlib.sha256(e['path'].encode()).hexdigest()/Path(e['path']).name
    copy(src,R/'candidate/source_tree'/e['path'].lstrip('/'),e)
    copy(src,R/'candidate/dependency_by_path_sha256'/hashlib.sha256(e['path'].encode()).hexdigest()/Path(e['path']).name,e)
for n in ['SOURCE_SHA256_MANIFEST.json','SHARED_CHECKPOINT_IDENTITY.json','CURRENT_EXECUTION_IDENTITY.json',
          'HISTORICAL_EXECUTION_IDENTITY.json','MATCHED_RUNTIME_BENCHMARK.csv','MATCHED_RUNTIME_BENCHMARK.json',
          'IDENTITY_INTERPRETATION_ADDENDUM.json','GPU_RESOURCE_LEDGER.json','LOCAL_DELIVERY_VERIFICATION.json',
          'CLOSEOUT.json','CPU_EVIDENCE_SNAPSHOT_CATALOG.json','GPU_DIAGNOSTIC_PLAN.json']:
    copy(P/n,R/'evidence/reconciliation'/n)
for unit in ['OLD_FP_R0','OLD_FP_R1','OLD_FP_R2','NEW_FP_R0','NEW_FP_R1','NEW_FP_R2','OLD_H_C128_R0','NEW_H_C128_R0']:
    for name in ['RESULT.json','ENVIRONMENT.json','STARTED.json']:
        src=P/'delivery/execution'/unit/name
        if src.exists():copy(src,R/'evidence/reconciliation/execution'/unit/name)
inodes={}
for row in rows:
    s=(R/row['path']).stat();inodes.setdefault((s.st_dev,s.st_ino),[]).append(row['path'])
aliases=[v for v in inodes.values() if len(v)>1]
report=dict(status='PASS_RAW_HISTORICAL_SOURCE_MATERIALIZATION',files=rows,logical_copy_rows=len(rows),
            unique_physical_copy_files=len(inodes),case_insensitive_path_projection_aliases=aliases,
            canonical_external_copy='candidate/dependency_by_path_sha256/<sha256(original_absolute_path)>/<basename>',
            historical_execution_files=len(M['execution_sources']),historical_external_source_files=len(M['external_dependencies']),
            source_math_changes=0,GPU_dispatches=0,large_model_or_tensor_downloaded=False)
(R/'candidate/HISTORICAL_COPY_MANIFEST.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'files':len(rows),'bytes':sum(x['bytes'] for x in rows)}))
