"""Validate all original units and freeze final artifacts without changing results."""
from pathlib import Path
import hashlib,json,tarfile
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
D=R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def main():
    assert not (D/'STOP.json').exists()
    assert read(Q/'GPU6_FORMAL_AUTHORIZATION.json')['GPU6_FORMAL_AUTHORIZED']=='YES'
    units=read(D/'discovery/raw_freeze.json');assert units['complete']==128
    docs={x['document_id'] for x in read(D/'document_manifest.json')['documents']}
    conditions=set(read(D/'rotation_manifest.json')['bank']['execution_order'])
    observed=set();commit=read(Q/'SCIENTIFIC_PREMEASUREMENT_COMMIT.json')['commit']
    for row in units['files']:
        assert sha(row['path'])==row['sha256'];u=read(row['path'])
        key=(u['document'],u['condition'],u['anchor']);assert key not in observed;observed.add(key)
        assert u['gate']=='PASS' and u['endpoint']==32768 and u['premeasurement_commit']==commit
        assert u['hardware']['physical_gpu']==6 and u['hardware']['gpu_uuid']=='GPU-de53f243-b271-521c-d875-d96611ed8574'
    assert observed=={(d,c,a) for d in docs for c in conditions for a in [8192,2048]}
    for line in (D/'sha256_manifest.txt').read_text().splitlines():
        digest,rel=line.split('  ',1);assert sha(D/rel)==digest,('FINAL_SHA_FAIL',rel)
    report=read(D/'FINAL_REPORT.json')
    assert report['DISCOVERY_PRIMARY_COMPLETE']==report['DISCOVERY_SECONDARY_COMPLETE']=='64/64'
    emit(Q/'DISCOVERY_COMPLETION_VALIDATION.json',dict(gate='PASS',FORMAL_MEASUREMENTS=128,GPU6_FORMAL_AUTHORIZED='YES',
        DISCOVERY_PRIMARY_COMPLETE='64/64',DISCOVERY_SECONDARY_COMPLETE='64/64',FINAL_ACTION='DISCOVERY_COMPLETE_REPORT_ONLY',
        original_scientific_final_action=report['FINAL_ACTION'],scientific_premeasurement_commit=commit,
        final_report_sha256=sha(D/'FINAL_REPORT.json'),FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',
        L9_TRAINING_EXECUTED='NO',NEW_ROTATION_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO',FREE_GENERATION_EXECUTED='NO'))
    with tarfile.open(Path(str(R)+'-authority')/'scientific-repro-historical-discovery-final.tar.gz','w:gz') as t:
        for root in [Q,D]:
            for p in sorted(root.rglob('*')):
                if p.is_file():t.add(p,arcname=str(p.relative_to(R)))
    print('DISCOVERY_REPORT_AND_SHA_VALIDATION_PASS',flush=True)
if __name__=='__main__':main()
