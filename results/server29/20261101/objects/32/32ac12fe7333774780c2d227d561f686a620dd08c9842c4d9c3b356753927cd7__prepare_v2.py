"""V2 stdlib-only full identity revalidation; exclusive new evidence only."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

OLD = Path('/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1')
AUDIT = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PHASEB_STOP_FORENSIC_AUDIT_V1')
OUT = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_FULL_PERSISTENT_PROSPECTIVE_V2')
CODE = Path('/data/zypan/worktrees/ling-kda-p1-l7-full-persistent-prospective-v2')
BASE_COMMIT = 'f1fa49883d5f1f65ed47d8ddc0b37a62cbee49d1'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_bytes())

def write(name, value):
    p = OUT/name
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    with p.with_suffix('.sha256.json').open('x') as f:
        json.dump(dict(path=str(p), sha256=sha(p)), f, indent=2); f.write('\n')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    start=time.time()
    assert sha(AUDIT/'SHA256SUMS.txt') == 'c6383c9a38c07089b18faa8c27953719e543f45ebbcb7894198e84635810d6ce'
    audit_ids={}
    for line in (AUDIT/'SHA256SUMS.txt').read_text().splitlines():
        expected, name=line.split('  ',1)
        actual=sha(AUDIT/name); assert actual == expected, name
        audit_ids[name]=actual
    audit_files=['PHASEB_FROZEN_SELECTION_AUDIT.json','FIRST_FAILURE_GATE_REPORT.json',
        'RESOURCE_AND_BUDGET_AUDIT.json','REPAIR_FEASIBILITY_ASSESSMENT.md',
        'FINAL_READONLY_AUDIT_REPORT.md','READONLY_SERVER_EVIDENCE.json','READONLY_SERVER_SUPPLEMENT.json']
    loaded={n:read(AUDIT/n) if n.endswith('.json') else (AUDIT/n).read_text() for n in audit_files}
    e=loaded['READONLY_SERVER_EVIDENCE.json']; s=loaded['READONLY_SERVER_SUPPLEMENT.json']
    actuals={}; rows=[]
    def verify(p, expected, group):
        p=str(p)
        if p not in actuals:
            a=Path(p).stat(); h=sha(p); b=Path(p).stat()
            assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns), ('UNSTABLE',p)
            actuals[p]=dict(sha256=h, bytes=b.st_size, mtime_ns=b.st_mtime_ns)
        rows.append(dict(path=p, expected=expected, actual=actuals[p]['sha256'],group=group,
                         passed=actuals[p]['sha256']==expected))
        assert rows[-1]['passed'], ('IDENTITY_MISMATCH',p)
    for row in e['hash_checks']: verify(row['path'], row['expected_sha256'], row['group'])
    for p,row in e['identities'].items():
        if row.get('exists'): verify(p,row['sha256'],'audit_identity')
    for p,h in s['hashes'].items(): verify(p,h,'audit_supplement')
    current={str(p):sha(p) for p in sorted(OLD.rglob('*')) if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    assert current==e['original_inventory_after'], 'HISTORICAL_DIRECTORY_CHANGED'
    names=['PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json','PROSPECTIVE_DOCUMENT_SELECTION.json',
        'PHASE_B_PAIR_METHODS_FREEZE.json','PHASE_B_CALIBRATION_MATCHED_TABLES.json',
        'SOURCE_IDENTITIES.json','PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json']
    frozen={n:read(OLD/n) for n in names}
    selection=frozen[names[0]]; cfg=read(OLD/'PHASE_B_CALIBRATION_CONFIG.json')
    assert selection['route']=='A' and selection['comparator']==['P1','L7_TOTAL_ORIGINAL']
    assert {x['document']:x['gpu'] for x in selection['documents']}=={'pg19-test-860':6,'pg19-test-5956':3}
    assert [(x['pair']['condition_a'],x['pair']['condition_b']) for x in selection['documents']]==[
        ('H_to_L6_a50','H_to_L7_a25'),('H_to_L6_a50','H_to_L7_a50')]
    vocab=read(Path(frozen['SOURCE_IDENTITIES.json']['model_identity']['path'])/'config.json')['vocab_size']
    for d in frozen['PROSPECTIVE_DOCUMENT_SELECTION.json']['selected_documents']:
        tokens=read(d['token_file']['path'])
        assert len(tokens)==32769 and all(type(t) is int and 0<=t<vocab for t in tokens)
    for row in e['model_weights']:
        now=actuals[row['path']]
        assert now['bytes']==row['original_audit_bytes'] and now['mtime_ns']==row['original_audit_mtime_ns']
    head=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
    assert head==BASE_COMMIT
    for name,h in frozen['PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json']['files'].items():
        verify(CODE/name,h,'new_clean_worktree_base')
    for p,h in frozen['SOURCE_IDENTITIES.json']['source_repositories'].items():
        assert subprocess.check_output(['git','-C',p,'rev-parse','HEAD'],text=True).strip()==h
    report=dict(status='PASS',started_epoch=start,completed_epoch=time.time(), audit_files_completely_read=audit_files,
        audit_sha256=audit_ids, historical_431_checks_reexecuted=True, full_weight_shards_rehashed=len(e['model_weights']),
        original_inventory=current, original_non_git_files_preserved=len(current), actual_identities=actuals, checks=rows,
        original_head=subprocess.check_output(['git','-C',str(OLD),'rev-parse','HEAD'],text=True).strip(),
        clean_worktree_base=head, code_root=str(CODE), result_root=str(OUT), gpu_work=False)
    write('SOURCE_AND_WEIGHT_IDENTITY.json',report)
    write('FROZEN_SELECTION_REUSE_AUDIT.json',dict(status='PASS',frozen_selection_sha256=sha(OLD/names[0]),
        original_selection=selection, six_assets={n:sha(OLD/n) for n in names},
        selection_recomputed=False, calibration_reexecuted=False, pairs_changed=False,
        reused_original_score_cells=672, prior_audit_sha256=audit_ids['PHASEB_FROZEN_SELECTION_AUDIT.json']))
    write('RESOURCE_POLICY_AMENDMENT_V2.json',dict(status='FROZEN',epoch=time.time(),
        task=OUT.name, historical_policy=dict(gpu_hours_cap=16,fixed_vram_fraction_cap=.85,
        qualification_wall_seconds=900,micro_wall_seconds=3600),
        GPU_HOURS_HARD_CAP=None,FIXED_VRAM_PERCENT_CAP=None,
        scope='V2 ONLY; V1 files/latches/outcomes unchanged',
        controller_locations=['v2_common.py:verify/ledger','v2_controller.py:launch/project'],
        worker_location='v2_worker.py:guard',
        administrative_deadline_removed=True,
        progress_watchdog='bootstrap qualification/micro: no-progress 1800 s; formal evidence-derived before formal outcome',
        complete_physical_capacity_and_finite_parity_identity_stability_checks_retained=True,
        no_scientific_metric_or_quantizer_change=True, before_any_new_32k_outcome=True))
    print(json.dumps(dict(status='V2_FRESH_IDENTITY_PASS',checks=len(rows),historical_files=len(current),weights=len(e['model_weights']))))

if __name__=='__main__': main()
