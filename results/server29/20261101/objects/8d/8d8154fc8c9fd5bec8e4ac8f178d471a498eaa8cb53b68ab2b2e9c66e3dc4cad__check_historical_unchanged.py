#!/usr/bin/env python3
"""Fresh SHA of explicitly named prior seals and their indexed small assets."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p,h,size=None):
    assert p.is_file();actual=sha(p);ok=actual==h and(size is None or p.stat().st_size==size)
    return dict(path=str(p),bytes=p.stat().st_size,expected_sha256=h,actual_sha256=actual,passed=ok)
checks=[]
for name,expected in [('QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1','2dc2738212dcd1997a0ee6ba19838a06f4c64a662caa38e778a159bd4b12a803'),('QWEN_GDN_FAST_RUNTIME_QUALIFICATION_REPAIR_20261019_V1/delivery','a7404ae691358c2c386ef18dabdba40f459bc037c6d9285f90f5343be9296dfc')]:
    root=B/name;p=root/'SHA256_MANIFEST.json';checks.append(record(p,expected));manifest=json.loads(p.read_text())
    for row in manifest['files']:checks.append(record(root/row['path'],row['sha256'],row['bytes']))
p=B/'qwen_gdn_historical_throughput_reconciliation_v1';manifest=json.loads((p/'SOURCE_SHA256_MANIFEST.json').read_text());checks.append(record(p/'SOURCE_SHA256_MANIFEST.json','b59ee3a7417ebdae7f4f857bc3e8eb2f1644c373d6d28bf8d31da2871671c9f3'))
for row in manifest['execution_sources']:checks.append(record(p/row['path'],row['sha256'],row['bytes']))
for row in manifest['external_dependencies']:
    q=p/'delivery/dependency_snapshot/by_path_sha256'/hashlib.sha256(row['path'].encode()).hexdigest()/Path(row['path']).name;checks.append(record(q,row['sha256'],row['bytes']))
result=dict(status='PASS_EXPLICIT_PRIOR_SEALS_AND_INDEXED_SMALL_ASSETS_UNCHANGED' if all(x['passed']for x in checks)else'FAIL',checks=checks,checks_count=len(checks),scope='Two explicitly pinned prior manifests and indexed small assets plus reconciliation201 local files; not an assertion about every historical asset on servers',remote_current_194_source_proof='analysis/CURRENT_ALL_HISTORICAL_EXTERNAL_SOURCE_VERIFICATION.json',remote_model_native_current_proof='analysis/environment/CURRENT_ENVIRONMENT_CPU_VERIFICATION.json',history_written_or_deleted=False,GPU_queries=0,GPU_dispatches=0)
(R/'analysis/HISTORICAL_UNCHANGED_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');assert result['status'].startswith('PASS');print(result['status'],len(checks))
